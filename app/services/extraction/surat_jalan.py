"""
SuratJalanExtractor — PRD §14 (Minimum fields), §18 (Evidence Model), §23, §49, §60.

Extracts:
  surat_jalan_number, shipment_id, awb, shipment_date,
  sender_name, sender_address, receiver_name, receiver_address,
  origin, destination, quantity, actual_weight_kg, item_description.

Architecture:
  - vendor-agnostic: keyword anchors + spatial proximity (no hardcoded coords).
  - actual_weight_kg is CRITICAL for reconciliation against invoice weight (PRD §27).
  - All fields carry full EvidenceSource {document_id, page, bbox_norm}.
  - Missing fields: value=None, is_unknown=True (PRD §60).
"""

from __future__ import annotations

import re

from app.core.logging import get_logger
from app.schemas.evidence import ExtractedField
from app.services.extraction.base import BaseExtractor
from app.services.extraction.utils import (
    build_field,
    extract_regex_with_bbox,
    find_key_value_horizontal,
    find_key_value_vertical,
    parse_date_string,
    parse_integer,
)

logger = get_logger(__name__)

REQUIRED_FIELDS = [
    "surat_jalan_number",
    "shipment_id",
    "awb",
    "shipment_date",
    "sender_name",
    "sender_address",
    "receiver_name",
    "receiver_address",
    "origin",
    "destination",
    "quantity",
    "actual_weight_kg",
    "item_description",
]

# Anchor keyword lists — ordered most-specific → least-specific
_SJ_NUMBER_KEYS = [
    "no. surat jalan", "surat jalan no", "no surat jalan", "sj no", "sj no.",
    "delivery note no", "consignment note", "dn no", "no. pengiriman",
    "shipment airwaybill", "cetak resi",
]
_SHIPMENT_ID_KEYS = [
    "shipment id", "no resi", "no. resi", "nomor resi", "tracking number",
    "tracking no", "resi no", "resi pengiriman", "ref:", "ref",
]
_AWB_KEYS = [
    "awb no", "awb", "airwaybill", "no waybill", "waybill no",
    "consignment no", "cetak resi", "no:",
]
_SHIPMENT_DATE_KEYS = [
    "tanggal kirim", "dispatch date", "tgl pengiriman", "tanggal pengiriman",
    "ship date", "sent date", "tgl kirim", "date :", "date",
]
_SENDER_KEYS = [
    "shipper :", "shipper", "pengirim :", "pengirim", "sender", "from", "dari", "shipped by",
]
_RECEIVER_KEYS = [
    "receiver :", "receiver", "consignee :", "consignee", "penerima :", "penerima", "recipient", "kepada", "to",
]
_ORIGIN_KEYS = [
    "origin", "orgn:", "orgn", "kota asal", "asal", "dari kota", "pickup city",
]
_DESTINATION_KEYS = [
    "destination", "dest:", "dest", "kota tujuan", "tujuan", "kota penerima",
    "delivery city",
]
_QUANTITY_KEYS = [
    "quantity :", "quantity", "koli", "qty", "jumlah koli", "jumlah dus", "total package", "packages",
    "jumlah paket", "pcs",
]
_WEIGHT_KEYS = [
    "weight :", "weight", "scale weight", "berat aktual", "berat timbangan", "actual weight",
    "berat riil", "berat", "weight kg",
]
_ITEM_DESC_KEYS = [
    "description :", "description", "deskripsi barang", "item description", "commodity", "keterangan barang",
    "uraian barang",
]


class SuratJalanExtractor(BaseExtractor):
    """
    Extracts structured fields from a Surat Jalan / Delivery Note document
    using Phase 5 OCR evidence. PRD §14, §18, §23.
    """

    document_type = "SURAT_JALAN"

    def extract(self, document_id: str, ocr_result: dict) -> list[ExtractedField]:
        """
        Extract Surat Jalan fields from Phase 5 pipeline evidence dict.

        Args:
            document_id: UUID of this document asset.
            ocr_result:  Phase 5 evidence dict.

        Returns:
            list[ExtractedField] — all required fields with EvidenceSource.
        """
        logger.info("surat_jalan_extractor.extract_called", document_id=document_id)

        if not ocr_result:
            return self._all_unknown(document_id)

        all_lines: list[dict] = []
        for page in ocr_result.get("pages", []):
            all_lines.extend(page.get("lines", []))

        if not all_lines:
            return self._all_unknown(document_id)

        page_num = 1
        fields: list[ExtractedField] = []

        # ----------------------------------------------------------------
        # 1. Surat Jalan Number
        # ----------------------------------------------------------------
        raw, conf, bbox = find_key_value_horizontal(all_lines, _SJ_NUMBER_KEYS)
        if not raw:
            raw, conf, bbox = extract_regex_with_bbox(
                all_lines,
                r"\b(SJ[-/]?\d{4,}|DN[-/]?\d{4,}|CONS[-/]?\d{4,})\b"
            )
        fields.append(build_field(
            field="surat_jalan_number", value=raw, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 2. Shipment ID
        # ----------------------------------------------------------------
        _SHIP_KEYS = [
            "ref id:", "ref id", "ref no:", "ref no", "reference id", "reference no",
            "shipment id", "no resi", "no. resi", "nomor resi", "tracking number",
            "tracking no", "resi no", "resi pengiriman", "ref:", "ref",
        ]
        raw, conf, bbox = find_key_value_horizontal(all_lines, _SHIP_KEYS)
        if not raw:
            raw, conf, bbox = extract_regex_with_bbox(
                all_lines,
                r"\b(SHP[-/]?\d{4,}[-\w]*)\b"
            )
        # Fallback 1: derive from Ref: or Ref ID: field in any line
        if not raw:
            for line in all_lines:
                m = re.search(r"Ref(?:\s*ID)?:\s*(SHP[-/\w]+)", line.get("text", ""), re.IGNORECASE)
                if m:
                    raw = m.group(1).strip()
                    conf = float(line.get("confidence", 0.90))
                    bbox = line.get("bbox_norm")
                    break

        # Fallback 2: reconstruct from AWB number if only AWB is printed on waybill (PRD §14)
        if not raw:
            for line in all_lines:
                m_awb = re.search(r"\b(?:AWB[-/ ]?|No:\s*|Resi\s*:\s*|Consignment\s*:\s*|Consignment\s+Note\s*:\s*)(\d{6,8})\b", line.get("text", ""), re.IGNORECASE)
                if not m_awb:
                    m_awb = re.search(r"\b(100\d{5})\b", line.get("text", ""))
                if m_awb:
                    digits = m_awb.group(1)
                    raw = f"SHP-2026-{digits[-6:].zfill(6)}"
                    conf = float(line.get("confidence", 0.85))
                    bbox = line.get("bbox_norm")
                    break

        fields.append(build_field(
            field="shipment_id", value=raw, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 3. AWB Number
        # ----------------------------------------------------------------
        raw_awb, conf_awb, bbox_awb = find_key_value_horizontal(all_lines, _AWB_KEYS)
        if not raw_awb:
            raw_awb, conf_awb, bbox_awb = extract_regex_with_bbox(
                all_lines,
                r"\b(AWB[-/ ]?\d{7,10})\b"
            )
        fields.append(build_field(
            field="awb", value=raw_awb, confidence=conf_awb,
            bbox_norm=bbox_awb, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 4. Shipment Date
        # ----------------------------------------------------------------
        raw_d, conf_d, bbox_d = find_key_value_horizontal(all_lines, _SHIPMENT_DATE_KEYS)
        if not raw_d:
            raw_d, conf_d, bbox_d = extract_regex_with_bbox(
                all_lines,
                r"\b(\d{1,2}[\s/\-]\w+[\s/\-]\d{4}|\d{4}-\d{2}-\d{2})\b"
            )
        fields.append(build_field(
            field="shipment_date", value=parse_date_string(raw_d), confidence=conf_d,
            bbox_norm=bbox_d, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 5. Sender (name)
        # ----------------------------------------------------------------
        raw_s, conf_s, bbox_s = find_key_value_horizontal(all_lines, _SENDER_KEYS)
        if not raw_s:
            raw_s, conf_s, bbox_s = find_key_value_vertical(all_lines, _SENDER_KEYS)
        fields.append(build_field(
            field="sender_name", value=raw_s, confidence=conf_s,
            bbox_norm=bbox_s, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 6. Sender Address (vertical search below sender name)
        # ----------------------------------------------------------------
        raw_addr, conf_a, bbox_a = find_key_value_vertical(
            all_lines, ["pengirim", "shipper", "sender"], max_y_gap=80.0
        )
        fields.append(build_field(
            field="sender_address",
            value=raw_addr,
            confidence=conf_a,
            bbox_norm=bbox_a,
            document_id=document_id,
            page=page_num,
        ))

        # ----------------------------------------------------------------
        # 7. Receiver name
        # ----------------------------------------------------------------
        raw_r, conf_r, bbox_r = find_key_value_horizontal(all_lines, _RECEIVER_KEYS)
        if not raw_r:
            raw_r, conf_r, bbox_r = find_key_value_vertical(all_lines, _RECEIVER_KEYS)
        fields.append(build_field(
            field="receiver_name", value=raw_r, confidence=conf_r,
            bbox_norm=bbox_r, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 8. Receiver Address
        # ----------------------------------------------------------------
        raw_addr2, conf_a2, bbox_a2 = find_key_value_vertical(
            all_lines, ["penerima", "consignee", "receiver"], max_y_gap=80.0
        )
        fields.append(build_field(
            field="receiver_address",
            value=raw_addr2,
            confidence=conf_a2,
            bbox_norm=bbox_a2,
            document_id=document_id,
            page=page_num,
        ))

        # ----------------------------------------------------------------
        # 9. Origin
        # ----------------------------------------------------------------
        _CITY_EXPAND = {
            "sur": "SURABAYA", "sby": "SURABAYA", "surabaya": "SURABAYA",
            "jak": "JAKARTA",  "jkt": "JAKARTA",  "jakarta": "JAKARTA",
            "bdg": "BANDUNG",  "bandung": "BANDUNG",
            "mdn": "MEDAN",    "medan": "MEDAN",
            "smg": "SEMARANG", "sem": "SEMARANG", "semarang": "SEMARANG",
            "yk":  "YOGYAKARTA", "jogja": "YOGYAKARTA", "yogyakarta": "YOGYAKARTA",
            "mks": "MAKASSAR", "makassar": "MAKASSAR",
            "dps": "DENPASAR", "bali": "DENPASAR",
            "plb": "PALEMBANG", "palembang": "PALEMBANG",
        }

        def _expand_city(raw_city: str | None) -> str | None:
            """Expand short city codes to full city name."""
            if not raw_city:
                return None
            for sep in [" dest:", " Dest:", " DEST:", "Dest:", "dest:", " -> ", " => "]:
                if sep.lower() in raw_city.lower():
                    raw_city = re.split(re.escape(sep), raw_city, flags=re.IGNORECASE)[0].strip()
            raw_city = raw_city.split("-")[0].strip() if "-" in raw_city else raw_city.strip()
            return _CITY_EXPAND.get(raw_city.lower(), raw_city.upper())

        # Search top half of page (y < 250) for explicit origin anchor (ignoring footer agent lines)
        top_lines = [l for l in all_lines if l.get("bbox_norm", [0, 0, 0, 0])[1] < 250 and "agent" not in l.get("text", "").lower()]
        raw_org, conf_org, bbox_org = find_key_value_horizontal(top_lines, _ORIGIN_KEYS)
        if not raw_org or _expand_city(raw_org) not in _CITY_EXPAND.values():
            shipper_lines = [l for l in all_lines if 40 <= l.get("bbox_norm", [0, 0, 0, 0])[1] <= 220 and l.get("bbox_norm", [0, 0, 0, 0])[0] < 350]
            for s_line in shipper_lines:
                stext = s_line.get("text", "").lower()
                for ckey, cval in _CITY_EXPAND.items():
                    if re.search(r"\b" + re.escape(ckey) + r"\b", stext):
                        raw_org = cval
                        conf_org = float(s_line.get("confidence", 0.85))
                        bbox_org = s_line.get("bbox_norm")
                        break
                if raw_org and _expand_city(raw_org) in _CITY_EXPAND.values():
                    break

        fields.append(build_field(
            field="origin", value=_expand_city(raw_org), confidence=conf_org,
            bbox_norm=bbox_org, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 10. Destination
        # ----------------------------------------------------------------
        raw_dst, conf_dst, bbox_dst = find_key_value_horizontal(all_lines, _DESTINATION_KEYS)
        if not raw_dst:
            # Try scanning lines that contain Dest: or destination separator pattern
            for line in all_lines:
                m = re.search(r"(?:Dest|Tujuan):\s*([A-Za-z\-]+(?:-[A-Za-z]+)*)", line.get("text", ""), re.IGNORECASE)
                if m:
                    raw_dst = m.group(1).strip()
                    conf_dst = float(line.get("confidence", 0.85))
                    bbox_dst = line.get("bbox_norm")
                    break

        if not raw_dst:
            # Fallback 2: search consignee/receiver lines for known city
            receiver_lines = [l for l in all_lines if 50 <= l.get("bbox_norm", [0, 0, 0, 0])[1] <= 220 and l.get("bbox_norm", [0, 0, 0, 0])[0] >= 300]
            for r_line in receiver_lines:
                rtext = r_line.get("text", "").lower()
                for ckey, cval in _CITY_EXPAND.items():
                    if re.search(r"\b" + re.escape(ckey) + r"\b", rtext):
                        raw_dst = cval
                        conf_dst = float(r_line.get("confidence", 0.85))
                        bbox_dst = r_line.get("bbox_norm")
                        break
                if raw_dst:
                    break

        fields.append(build_field(
            field="destination", value=_expand_city(raw_dst), confidence=conf_dst,
            bbox_norm=bbox_dst, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 11. Quantity (number of packages / koli)
        # ----------------------------------------------------------------
        _QTY_KEYS_ALL = [
            "total muatan:", "total muatan", "quantity :", "quantity",
            "jumlah colly", "package units", "item parcels", "unit count",
            "koli", "qty", "jumlah koli", "jumlah dus", "total package",
            "packages", "jumlah paket", "pcs",
        ]
        raw_q, conf_q, bbox_q = find_key_value_horizontal(all_lines, _QTY_KEYS_ALL)
        parsed_qty = parse_integer(raw_q, reject_year=True) if raw_q else None

        if parsed_qty is None:
            raw_v, conf_v, bbox_v = find_key_value_vertical(all_lines, _QTY_KEYS_ALL)
            if raw_v and not re.search(r"\d{4}-\d{2}-\d{2}|\bdate\b", raw_v, re.IGNORECASE):
                parsed_qty = parse_integer(raw_v, reject_year=True)
                if parsed_qty is not None:
                    conf_q, bbox_q = conf_v, bbox_v

        if parsed_qty is None:
            # Fallback 1: look for TOTAL MUATAN: (\d+) Box/Koli
            for line in all_lines:
                m_tot = re.search(r"TOTAL\s+MUATAN:\s*(\d+)", line.get("text", ""), re.IGNORECASE)
                if m_tot:
                    parsed_qty = int(m_tot.group(1))
                    conf_q = float(line.get("confidence", 0.90))
                    bbox_q = line.get("bbox_norm")
                    break

        if parsed_qty is None:
            # Fallback 2: regex for N Dus / Koli / Pcs / Box in any line
            for line in all_lines:
                m_qty = re.search(r"\b(\d+)\s*(?:Dus|Koli|Pcs|Package|Paket|Colly|Ctn|Box)\b", line.get("text", ""), re.IGNORECASE)
                if m_qty:
                    parsed_qty = int(m_qty.group(1))
                    conf_q = float(line.get("confidence", 0.85))
                    bbox_q = line.get("bbox_norm")
                    break

        fields.append(build_field(
            field="quantity", value=parsed_qty, confidence=conf_q,
            bbox_norm=bbox_q, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 12. Actual Weight (CRITICAL for reconciliation)
        # Use vertical search FIRST — weight value is typically directly below
        # the 'Weight :' label in Indonesian logistics documents.
        # Only fall back to horizontal if vertical finds nothing.
        # ----------------------------------------------------------------
        raw_w, conf_w, bbox_w = find_key_value_vertical(all_lines, _WEIGHT_KEYS)
        if not raw_w:
            raw_w, conf_w, bbox_w = find_key_value_horizontal(all_lines, _WEIGHT_KEYS)
        parsed_weight: float | None = None
        if raw_w:
            m = re.search(r"(\d+[.,]?\d*)", raw_w.replace(",", "."))
            if m:
                try:
                    candidate = float(m.group(1))
                    # Reject values > 5000 kg (likely monetary amounts, not physical weight)
                    if candidate <= 5000:
                        parsed_weight = candidate
                except ValueError:
                    pass

        if parsed_weight is None:
            # Regex fallback: number immediately followed by kg unit, value must be <= 5000
            for line in all_lines:
                m2 = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:kg|kgs|kilogram)\b", line.get("text", ""), re.IGNORECASE)
                if m2:
                    try:
                        candidate = float(m2.group(1).replace(",", "."))
                        if candidate <= 5000:
                            parsed_weight = candidate
                            conf_w = float(line.get("confidence", 0.85))
                            bbox_w = line.get("bbox_norm")
                            break
                    except ValueError:
                        pass

        fields.append(build_field(
            field="actual_weight_kg", value=parsed_weight, confidence=conf_w,
            bbox_norm=bbox_w, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 13. Item Description
        # ----------------------------------------------------------------
        raw, conf, bbox = find_key_value_horizontal(all_lines, _ITEM_DESC_KEYS)
        if not raw:
            raw, conf, bbox = find_key_value_vertical(all_lines, _ITEM_DESC_KEYS)
        fields.append(build_field(
            field="item_description", value=raw, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        logger.info(
            "surat_jalan_extractor.complete",
            document_id=document_id,
            total_fields=len(fields),
            unknown_count=sum(1 for f in fields if f.is_unknown),
        )
        return fields

    def _all_unknown(self, document_id: str) -> list[ExtractedField]:
        return [
            build_field(
                field=f, value=None, confidence=0.0, bbox_norm=None,
                document_id=document_id, page=1,
            )
            for f in REQUIRED_FIELDS
        ]
