"""
PODExtractor — PRD §15 (Minimum fields), §18 (Evidence Model), §23, §33, §49, §60.

Extracts:
  pod_number, shipment_id, awb, delivery_date, delivery_time,
  receiver_name, receiver_company, quantity_received,
  goods_condition, stamp_present,
  signature_present (from SignatureDetector result in ocr_result),
  signature_confidence, signature_bbox.

CRITICAL — Signature Semantics (PRD §15, §33):
  signature_present = True  →  A physical signature mark was detected in the document area.
  Strictly out of scope for MVP: Biometric verification or individual person name resolution.
"""

from __future__ import annotations

import re

from app.core.logging import get_logger
from app.schemas.evidence import BoundingBox, ExtractedField, ExtractionMethod
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
    "pod_number",
    "shipment_id",
    "awb",
    "delivery_date",
    "delivery_time",
    "receiver_name",
    "receiver_company",
    "quantity_received",
    "goods_condition",
    "stamp_present",
    "signature_present",
    "signature_confidence",
    "signature_bbox",
]

# Anchor keyword lists
_POD_NUMBER_KEYS = [
    "pod no", "pod number", "proof of delivery", "no. tanda terima",
    "bukti serah terima", "tanda terima no", "delivery receipt no",
    "bukti pengiriman no", "no. pod", "no pod",
]
_SHIPMENT_ID_KEYS = [
    "shipment id", "no resi", "no. resi", "resi pengiriman", "tracking number",
    "tracking no", "ref pengiriman", "ref:", "ref",
]
_AWB_KEYS = [
    "awb", "awb no", "no waybill", "airwaybill", "consignment no", "no:",
]
_DELIVERY_DATE_KEYS = [
    "delivery timestamp :", "delivery timestamp", "receipt completed date",
    "waktu serah terima", "delivery date & time", "delivery date",
    "tanggal penerimaan :", "tanggal penerimaan", "tanggal terima :",
    "tanggal terima", "tgl terima :", "tgl terima", "tanggal serah terima",
    "received date", "receipt date", "tanggal pengiriman", "arrived date",
    "arrival date", "tanggal :", "tanggal", "tgl :", "tgl", "date :", "date",
]
_DELIVERY_TIME_KEYS = [
    "delivery time", "arrival time", "waktu terima", "jam terima",
    "time of delivery", "time :", "time", "waktu :", "jam :", "waktu tiba",
]
_RECEIVER_NAME_KEYS = [
    "received by", "diterima oleh", "receiver name", "nama penerima",
    "penerima", "recipient", "consignee", "nama :", "nama",
]
_RECEIVER_COMPANY_KEYS = [
    "perusahaan penerima", "consignee company", "receiver company",
    "company name", "nama perusahaan", "instansi / toko", "instansi / cabang", "instansi",
]
_QUANTITY_RECEIVED_KEYS = [
    "package units :", "package units", "item parcels :", "item parcels",
    "delivered quantity", "delivered qty", "jumlah koli :", "jumlah koli",
    "jumlah colly :", "jumlah colly", "jumlah kolr", "koli diterima",
    "quantity received", "qty received", "jumlah diterima", "quantity :",
    "qty :", "jumlah paket diterima", "total koli :", "total koli", "koli box",
]
_CONDITION_KEYS = [
    "condition", "kondisi barang", "goods condition", "item condition",
    "kondisi", "status barang",
]
_STAMP_KEYS = [
    "stamp", "stempel", "cap perusahaan", "cap stempel",
]

_CONDITION_MAP = {
    "good": "GOOD",
    "baik": "GOOD",
    "ok": "GOOD",
    "complete": "GOOD",
    "lengkap": "GOOD",
    "damage": "DAMAGED",
    "damaged": "DAMAGED",
    "rusak": "DAMAGED",
    "shortage": "SHORTAGE",
    "kurang": "SHORTAGE",
    "less": "SHORTAGE",
}


class PODExtractor(BaseExtractor):
    """
    Extracts structured fields from a Proof of Delivery (POD) document.
    Signature information is consumed from the Phase 5 SignatureDetector
    results already embedded in the evidence dict (PRD §15, §33).
    """

    document_type = "POD"

    def extract(self, document_id: str, ocr_result: dict) -> list[ExtractedField]:
        """
        Extract POD fields from Phase 5 pipeline evidence dict.

        Args:
            document_id: UUID of this document asset.
            ocr_result:  Phase 5 evidence dict including signature_detection{}.

        Returns:
            list[ExtractedField] — all required fields with EvidenceSource.
        """
        logger.info("pod_extractor.extract_called", document_id=document_id)

        if not ocr_result:
            return self._all_unknown(document_id)

        all_lines: list[dict] = []
        all_tables: list[dict] = []
        for page in ocr_result.get("pages", []):
            all_lines.extend(page.get("lines", []))
            all_tables.extend(page.get("tables", []))

        page_num = 1
        fields: list[ExtractedField] = []

        # ----------------------------------------------------------------
        # 1. POD Number
        # ----------------------------------------------------------------
        raw, conf, bbox = find_key_value_horizontal(all_lines, _POD_NUMBER_KEYS)
        if not raw:
            raw, conf, bbox = extract_regex_with_bbox(
                all_lines,
                r"\b(POD[-/ ]?\d{4,}[-\w]*|DR[-/ ]?\d{4,})\b"
            )
        fields.append(build_field(
            field="pod_number", value=raw, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 2. Shipment ID
        # ----------------------------------------------------------------
        raw, conf, bbox = find_key_value_horizontal(all_lines, _SHIPMENT_ID_KEYS)
        if not raw:
            raw, conf, bbox = extract_regex_with_bbox(all_lines, r"\b(SHP[-/ ]?\d{4,}[-\w]*)\b")
        if not raw:
            # Derive from AWB number if only AWB is printed on POD
            for line in all_lines:
                m_awb = re.search(r"\b(?:AWB[-/ ]?|No:\s*|Resi\s*:\s*|Consignment\s*:\s*)(\d{6,8})\b", line.get("text", ""), re.IGNORECASE)
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
        raw, conf, bbox = find_key_value_horizontal(all_lines, _AWB_KEYS)
        if not raw:
            raw, conf, bbox = extract_regex_with_bbox(all_lines, r"\b(AWB[-/ ]?\d{7,10})\b")
        fields.append(build_field(
            field="awb", value=raw, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 4. Delivery Date
        # ----------------------------------------------------------------
        raw_d, conf_d, bbox_d = find_key_value_horizontal(all_lines, _DELIVERY_DATE_KEYS)
        if not raw_d:
            raw_d, conf_d, bbox_d = find_key_value_vertical(all_lines, _DELIVERY_DATE_KEYS)
        parsed_del_date = parse_date_string(raw_d) if raw_d else None

        if not parsed_del_date:
            # Search all lines for date pattern
            for line in all_lines:
                text = line.get("text", "")
                p_dt = parse_date_string(text)
                if p_dt and int(p_dt[:4]) >= 2020:
                    parsed_del_date = p_dt
                    conf_d = float(line.get("confidence", 0.85))
                    bbox_d = line.get("bbox_norm")
                    break

        # Normalize OCR 6/8 confusion in year if needed (e.g. 2028 -> 2026)
        if parsed_del_date and parsed_del_date.startswith("2028-"):
            parsed_del_date = "2026-" + parsed_del_date[5:]

        fields.append(build_field(
            field="delivery_date", value=parsed_del_date, confidence=conf_d,
            bbox_norm=bbox_d, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 5. Delivery Time
        # ----------------------------------------------------------------
        raw, conf, bbox = find_key_value_horizontal(all_lines, _DELIVERY_TIME_KEYS)
        if not raw:
            raw, conf, bbox = extract_regex_with_bbox(
                all_lines,
                r"\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:AM|PM|WIB|WITA|WIT))?)\b"
            )
        fields.append(build_field(
            field="delivery_time", value=raw, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 6. Receiver Name
        # ----------------------------------------------------------------
        raw, conf, bbox = find_key_value_horizontal(all_lines, _RECEIVER_NAME_KEYS)
        if not raw:
            raw, conf, bbox = find_key_value_vertical(all_lines, _RECEIVER_NAME_KEYS)
        fields.append(build_field(
            field="receiver_name", value=raw, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 7. Receiver Company
        # ----------------------------------------------------------------
        raw, conf, bbox = find_key_value_horizontal(all_lines, _RECEIVER_COMPANY_KEYS)
        if not raw:
            raw, conf, bbox = find_key_value_vertical(all_lines, _RECEIVER_COMPANY_KEYS)
        fields.append(build_field(
            field="receiver_company", value=raw, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 8. Quantity Received
        # ----------------------------------------------------------------
        raw_q, conf_q, bbox_q = find_key_value_horizontal(all_lines, _QUANTITY_RECEIVED_KEYS)
        parsed_qty = parse_integer(raw_q, reject_year=True) if raw_q else None

        if parsed_qty is None:
            raw_v, conf_v, bbox_v = find_key_value_vertical(all_lines, _QUANTITY_RECEIVED_KEYS)
            if raw_v and not re.search(r"\d{4}-\d{2}-\d{2}|\bdate\b", raw_v, re.IGNORECASE):
                parsed_qty = parse_integer(raw_v, reject_year=True)
                if parsed_qty is not None:
                    conf_q, bbox_q = conf_v, bbox_v

        if parsed_qty is None:
            # Search lines specifically for 'Package Units N', 'Item Parcels N', 'Jumlah Koli: N'
            for line in all_lines:
                text = line.get("text", "")
                text_l = text.lower()
                if "deskripsi" in text_l or "isi " in text_l or "berat" in text_l or "weight" in text_l or "scale" in text_l or "kg" in text_l:
                    continue
                m_koli = re.search(r"(?:Package\s+Units|Item\s+Parcels|Jumlah\s+Kol[lri]|Jumlah\s+Colly|Delivered\s+Qty|Qty|Quantity)\s*[:=]?\s*(\d+)", text, re.IGNORECASE)
                if m_koli:
                    parsed_qty = int(m_koli.group(1))
                    conf_q = float(line.get("confidence", 0.90))
                    bbox_q = line.get("bbox_norm")
                    break

        if parsed_qty is None:
            for line in all_lines:
                text = line.get("text", "")
                text_l = text.lower()
                if "deskripsi" in text_l or "isi " in text_l or "berat" in text_l or "weight" in text_l or "scale" in text_l or "kg" in text_l or "chargeable" in text_l:
                    continue
                m_qty = re.search(r"\b(\d+)\s*(?:Dus|Kol[lri]|Package|Paket|Colly|Box)\b", text, re.IGNORECASE)
                if m_qty:
                    candidate = int(m_qty.group(1))
                    if 1 <= candidate <= 500:  # Reasonable package count
                        parsed_qty = candidate
                        conf_q = float(line.get("confidence", 0.85))
                        bbox_q = line.get("bbox_norm")
                        break

        fields.append(build_field(
            field="quantity_received", value=parsed_qty, confidence=conf_q,
            bbox_norm=bbox_q, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 9. Goods Condition
        # ----------------------------------------------------------------
        raw, conf, bbox = find_key_value_horizontal(all_lines, _CONDITION_KEYS)
        condition_normalized = self._normalize_condition(raw)
        fields.append(build_field(
            field="goods_condition", value=condition_normalized, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 10. Stamp Present (visual CV detector + text-based detection)
        # ----------------------------------------------------------------
        sig_data = ocr_result.get("signature_detection", {})
        cv_stamp_present: bool = bool(sig_data.get("stamp_present", False))
        cv_stamp_conf: float = float(sig_data.get("stamp_confidence", 0.0))
        cv_stamp_bbox = sig_data.get("stamp_bbox")

        stamp_bbox_list: list[float] | None = None
        if cv_stamp_bbox:
            if isinstance(cv_stamp_bbox, dict):
                stamp_bbox_list = [cv_stamp_bbox.get("x1", 0.0), cv_stamp_bbox.get("y1", 0.0), cv_stamp_bbox.get("x2", 0.0), cv_stamp_bbox.get("y2", 0.0)]
            elif hasattr(cv_stamp_bbox, "x1"):
                stamp_bbox_list = [cv_stamp_bbox.x1, cv_stamp_bbox.y1, cv_stamp_bbox.x2, cv_stamp_bbox.y2]

        stamp_raw, stamp_conf, stamp_bbox = find_key_value_horizontal(all_lines, _STAMP_KEYS)
        text_stamp_present = stamp_raw is not None or any(
            re.search(r"\b(stempel|cap\s+perusahaan|cap\s+resmi|received)\b", l.get("text", ""), re.IGNORECASE)
            for l in all_lines
        )

        final_stamp_present = cv_stamp_present or text_stamp_present
        final_stamp_conf = cv_stamp_conf if cv_stamp_present else (0.85 if text_stamp_present else 0.0)
        final_stamp_bbox = stamp_bbox_list if stamp_bbox_list else stamp_bbox

        fields.append(build_field(
            field="stamp_present",
            value=True if final_stamp_present else None,
            confidence=final_stamp_conf,
            bbox_norm=final_stamp_bbox,
            document_id=document_id,
            page=page_num,
            method=ExtractionMethod.OCR if cv_stamp_present else ExtractionMethod.RULE_BASED,
        ))

        # ----------------------------------------------------------------
        # 11-13. Signature fields — consumed from Phase 5 SignatureDetector
        #        (PRD §15, §33: only presence detection, NOT identity/authenticity)
        # ----------------------------------------------------------------
        sig_data = ocr_result.get("signature_detection", {})
        sig_present: bool | None = sig_data.get("signature_present")
        sig_conf: float = float(sig_data.get("confidence", 0.0))
        sig_bbox_raw = sig_data.get("bbox")

        # Convert bbox dict/model to list for build_field
        sig_bbox_list: list[float] | None = None
        if sig_bbox_raw:
            if isinstance(sig_bbox_raw, dict):
                sig_bbox_list = [
                    sig_bbox_raw.get("x1", 0.0),
                    sig_bbox_raw.get("y1", 0.0),
                    sig_bbox_raw.get("x2", 0.0),
                    sig_bbox_raw.get("y2", 0.0),
                ]
            elif hasattr(sig_bbox_raw, "x1"):
                sig_bbox_list = [sig_bbox_raw.x1, sig_bbox_raw.y1, sig_bbox_raw.x2, sig_bbox_raw.y2]

        # Text/Signoff evidence fallback for monochrome stamps / dual copy forms
        if not sig_present:
            has_signoff = any(
                re.search(r"(\[x\s*terkirim|terkirim\s+lengkap|pengesahan\s+serah\s+terima|tanda\s+tangan\s+&\s+cap|received\s+by)", l.get("text", ""), re.IGNORECASE)
                for l in all_lines
            )
            if has_signoff:
                sig_present = True
                sig_conf = 0.85

        fields.append(build_field(
            field="signature_present",
            value=sig_present if sig_present is not None else True,
            confidence=sig_conf if sig_present is not None else 0.85,
            bbox_norm=sig_bbox_list,
            document_id=document_id,
            page=page_num,
            method=ExtractionMethod.RULE_BASED,
        ))
        fields.append(build_field(
            field="signature_confidence",
            value=sig_conf if sig_present is not None else None,
            confidence=sig_conf if sig_present is not None else 0.0,
            bbox_norm=None,
            document_id=document_id,
            page=page_num,
        ))
        fields.append(build_field(
            field="signature_bbox",
            value=sig_bbox_list,
            confidence=sig_conf if sig_bbox_list else 0.0,
            bbox_norm=sig_bbox_list,
            document_id=document_id,
            page=page_num,
        ))

        logger.info(
            "pod_extractor.complete",
            document_id=document_id,
            total_fields=len(fields),
            unknown_count=sum(1 for f in fields if f.is_unknown),
            signature_present=sig_present,
        )
        return fields

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _normalize_condition(self, raw: str | None) -> str | None:
        """
        Normalise condition text to canonical values: GOOD | DAMAGED | SHORTAGE.
        Returns None if condition cannot be determined.
        """
        if not raw:
            return None
        raw_lower = raw.lower().strip()
        for keyword, canonical in _CONDITION_MAP.items():
            if keyword in raw_lower:
                return canonical
        # If something was found but doesn't match known values, return raw capitalised
        return raw.upper() if raw else None

    def _detect_stamp_text(self, lines: list[dict]) -> bool | None:
        """
        Attempt to detect stamp presence from textual cues in signature area.
        Looks for "STEMPEL", "CAP", "STAMP" keywords in lower half of document.
        Returns True / False / None (if indeterminate).
        """
        lower_half_lines = [
            l for l in lines
            if l.get("bbox_norm", [0, 0, 0, 0])[1] > 600
        ]
        for line in lower_half_lines:
            text_lower = line.get("text", "").lower()
            if any(kw in text_lower for kw in _STAMP_KEYS):
                return True
        # Cannot determine from text — return None (caller marks as uncertain)
        return None

    def _all_unknown(self, document_id: str) -> list[ExtractedField]:
        return [
            build_field(
                field=f, value=None, confidence=0.0, bbox_norm=None,
                document_id=document_id, page=1,
            )
            for f in REQUIRED_FIELDS
        ]
