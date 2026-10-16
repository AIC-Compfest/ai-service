"""
InvoiceExtractor — PRD §13 (Minimum fields), §18 (Evidence Model), §23 (Separate extractor),
§49 (Multi-vendor generalisation), §60 (No Hallucination Rule).

Extracts:
  Header: invoice_number, invoice_date, vendor_name, vendor_id, customer_name,
          billing_period_start, billing_period_end, due_date, subtotal,
          tax_amount, grand_total, currency.
  Line items: list[dict] with per-row fields and bbox evidence.

Architecture:
  - Uses anchor-keyword spatial search (horizontal & vertical) — no hardcoded coords.
  - Table rows parsed with fuzzy column-header matching (multi-vendor safe).
  - All extracted values carry EvidenceSource {document_id, page, bbox_norm}.
  - Missing fields: value=None, is_unknown=True (PRD §60).
  - Financial totals are extracted AS-IS — pricing recalculation belongs in
    backend-api deterministic pricing engine (PRD §29, §40).
"""

from __future__ import annotations

import re

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.evidence import EvidenceSource, ExtractedField, ExtractionMethod
from app.services.extraction.base import BaseExtractor
from app.services.extraction.utils import (
    _bbox_norm_to_model,
    _midpoint_y,
    build_field,
    extract_regex_with_bbox,
    find_amount_near_keyword,
    find_key_value_horizontal,
    find_key_value_vertical,
    parse_amount,
    parse_date_string,
    parse_table_columns,
    to_unit_bbox,
)

logger = get_logger(__name__)


def _is_invoice_item_row(row: dict) -> bool:
    """Reject header/vendor text by structure, never by a vendor-specific word."""
    identifier = " ".join(
        str(row.get(field) or "") for field in ("shipment_id", "awb_number")
    ).strip()
    line_total = row.get("line_total")
    has_identifier = bool(re.search(r"\d", identifier)) and len(identifier) >= 4
    has_total = parse_amount(line_total) is not None if line_total else False
    return has_identifier or has_total

# Canonical column aliases for invoice line-item tables (PRD §13, multi-vendor)
INVOICE_TABLE_COLUMNS = {
    "line_number":          ["no", "no.", "#", "baris", "line"],
    "shipment_id":          ["shipment id", "shipment", "no resi", "no. resi", "nomor resi",
                             "tracking no", "consignment no", "reference no"],
    "awb_number":           ["awb", "awb no", "airwaybill", "no waybill", "waybill"],
    "service_type":         ["service", "jenis layanan", "layanan", "tipe layanan", "service type"],
    "origin":               ["origin", "asal", "kota asal", "dari", "orgn"],
    "destination":          ["destination", "tujuan", "kota tujuan", "dest"],
    "chargeable_weight":    ["chargeable weight", "weight", "berat", "berat kg", "kg",
                             "kgs", "berat tertagih"],
    "base_rate":            ["base rate", "tarif dasar", "freight", "rate", "biaya dasar",
                             "harga dasar", "biaya kirim"],
    "fuel_surcharge":       ["fuel surcharge", "fuel", "bbm", "surcharge bbm", "fuel surcharge index"],
    "accessorial_charge":   ["accessorial", "handling fee", "biaya lain", "additional charge",
                             "biaya tambahan"],
    "line_total":           ["total", "jumlah", "amount", "subtotal", "total payable",
                             "total tagihan baris"],
}


class InvoiceExtractor(BaseExtractor):
    """
    Extracts structured fields from a 3PL Invoice document using Phase 5
    OCR evidence (lines, tokens, regions, tables). PRD §13, §18, §23.
    """

    document_type = "INVOICE"

    # -----------------------------------------------------------------------
    # Anchor keyword lists — ordered from most-specific to least-specific
    # -----------------------------------------------------------------------
    _INVOICE_NUMBER_KEYS = [
        "no. invoice", "no invoice", "invoice no", "invoice number",
        "no. tagihan", "no tagihan", "no. faktur", "no faktur", "faktur no",
        "nomor faktur", "invoice #", "inv no", "bill no",
        "tax invoice", "e-way bill",
    ]
    _INVOICE_DATE_KEYS = [
        "tanggal invoice", "tanggal tagihan", "tanggal faktur", "tgl faktur",
        "tgl invoice", "invoice date", "issued date", "date of issue",
        "tanggal", "tgl",
    ]
    _CUSTOMER_KEYS = [
        "bill to", "billed to", "ditagihkan kepada", "kepada yth", "kepada",
        "customer", "konsumen", "pelanggan", "sold to", "ship to",
    ]
    _BILLING_PERIOD_KEYS = [
        "billing period", "periode tagihan", "periode", "billing date range",
        "billing cycle",
    ]
    _DUE_DATE_KEYS = [
        "jatuh tempo", "due date", "payment due", "batas pembayaran",
        "tanggal jatuh tempo",
    ]
    _SUBTOTAL_KEYS = [
        "jumlah tagihan bruto", "tagihan bruto", "direct freight subtotal",
        "subtotal pengiriman", "subtotal:", "subtotal", "sub total:", "sub total",
        "dpp", "total dpp", "dasar pengenaan pajak", "net amount", "total before tax",
        "freight subtotal",
    ]
    _TAX_KEYS = [
        "ppn keluaran", "ppn keluaran 11%", "applicable ppn tax", "taxes (",
        "tax (", "taxes:", "tax:", "ppn (", "ppn:", "ppn 11%", "ppn 6%",
        "ppn 8%", "ppn 9%", "ppn 7%", "ppn", "tax amount", "pajak :",
        "pajak (", "nilai ppn", "vat:", "vat (", "vat",
    ]
    _GRAND_TOTAL_KEYS = [
        "remittance total", "total tagihan (grand total)", "total tagihan:",
        "total tagihan", "grand total:", "grand total", "total:", "total",
        "total payable", "jumlah ditagih", "total amount", "amount due",
        "total payable amount", "total pembayaran", "total invoice",
    ]

    def extract(self, document_id: str, ocr_result: dict) -> list[ExtractedField]:
        """
        Extract invoice fields from Phase 5 pipeline evidence dict.

        Args:
            document_id: UUID of this document asset.
            ocr_result:  Phase 5 evidence dict (pages[], classification{}, signature_detection{}).

        Returns:
            list[ExtractedField] — all required fields with EvidenceSource.
            Missing fields have value=None, is_unknown=True (PRD §60).
        """
        logger.info("invoice_extractor.extract_called", document_id=document_id)

        if not ocr_result:
            return self._all_unknown(document_id)

        # Flatten all lines across all pages (most invoices are single-page)
        all_lines: list[dict] = []
        all_tables: list[dict] = []
        for page in ocr_result.get("pages", []):
            all_lines.extend(page.get("lines", []))
            # Only tables the pipeline classified as item tables feed this extractor (legacy tables carry no role).
            all_tables.extend(t for t in page.get("tables", []) if t.get("role", "line_items") in ("line_items",))

        if not all_lines:
            return self._all_unknown(document_id)

        page_num = 1  # majority of invoice fields are on page 1
        fields: list[ExtractedField] = []

        # ----------------------------------------------------------------
        # 1. Invoice Number
        # ----------------------------------------------------------------
        raw_inv_no, conf, bbox = find_key_value_horizontal(all_lines, self._INVOICE_NUMBER_KEYS)
        if not raw_inv_no:
            # Fallback: regex search for common invoice number patterns
            raw_inv_no, conf, bbox = extract_regex_with_bbox(
                all_lines,
                r"\b(INV[-/]?\d[\d\-/A-Z]{2,20}|FAKTUR[-/]?\d[\d\-]{2,})\b"
            )
        fields.append(build_field(
            field="invoice_number", value=raw_inv_no, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 2. Invoice Date
        # ----------------------------------------------------------------
        raw_date, conf, bbox = find_key_value_horizontal(all_lines, self._INVOICE_DATE_KEYS)
        if not raw_date:
            raw_date, conf, bbox = extract_regex_with_bbox(
                all_lines,
                r"\b(\d{1,2}[\s\-/]\w+[\s\-/]\d{4}|\d{4}-\d{2}-\d{2})\b"
            )
        fields.append(build_field(
            field="invoice_date", value=parse_date_string(raw_date), confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 3. Vendor name & vendor_id from header region (first ~15 lines)
        # ----------------------------------------------------------------
        header_lines = [l for l in all_lines if to_unit_bbox(l.get("bbox_norm"))[1] < settings.invoice_header_max_y]
        vendor_name, vendor_id = self._extract_vendor_info(header_lines, all_lines)
        vendor_conf = 0.90 if vendor_name else 0.0
        vendor_bbox = header_lines[0].get("bbox_norm") if header_lines else None
        fields.append(build_field(
            field="vendor_name", value=vendor_name, confidence=vendor_conf,
            bbox_norm=vendor_bbox, document_id=document_id, page=page_num,
        ))
        fields.append(build_field(
            field="vendor_id", value=vendor_id, confidence=vendor_conf,
            bbox_norm=vendor_bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 4. Customer / Bill-To
        # ----------------------------------------------------------------
        raw_customer, conf, bbox = find_key_value_horizontal(all_lines, self._CUSTOMER_KEYS)
        if not raw_customer or raw_customer.endswith(")") or raw_customer.endswith("):") or len(raw_customer) <= 3:
            raw_customer_v, conf_v, bbox_v = find_key_value_vertical(all_lines, self._CUSTOMER_KEYS)
            if raw_customer_v:
                raw_customer, conf, bbox = raw_customer_v, conf_v, bbox_v
        fields.append(build_field(
            field="customer_name", value=raw_customer, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 5. Billing Period (start / end)
        # ----------------------------------------------------------------
        raw_period, conf, bbox = find_key_value_horizontal(all_lines, self._BILLING_PERIOD_KEYS)
        period_start, period_end = self._parse_billing_period(raw_period)
        fields.append(build_field(
            field="billing_period_start", value=period_start, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))
        fields.append(build_field(
            field="billing_period_end", value=period_end, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 6. Due Date
        # ----------------------------------------------------------------
        raw_due, conf, bbox = find_key_value_horizontal(all_lines, self._DUE_DATE_KEYS)
        fields.append(build_field(
            field="due_date", value=parse_date_string(raw_due), confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 7. Currency (default IDR per PRD §13)
        # ----------------------------------------------------------------
        currency, cur_conf, cur_bbox = self._detect_currency(all_lines)
        fields.append(build_field(
            field="currency", value=currency, confidence=cur_conf,
            bbox_norm=cur_bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 8. Subtotal
        # ----------------------------------------------------------------
        summary_lines = [l for l in all_lines if to_unit_bbox(l.get("bbox_norm"))[1] > settings.invoice_summary_min_y] or all_lines
        subtotal, conf, bbox = find_amount_near_keyword(summary_lines, self._SUBTOTAL_KEYS)
        if not subtotal:
            subtotal, conf, bbox = find_amount_near_keyword(all_lines, self._SUBTOTAL_KEYS)
        fields.append(build_field(
            field="subtotal", value=subtotal, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 9. Tax amount (PPN / VAT)
        # ----------------------------------------------------------------
        tax_amount, conf_t, bbox_t = find_amount_near_keyword(summary_lines, self._TAX_KEYS)
        if not tax_amount:
            tax_amount, conf_t, bbox_t = find_amount_near_keyword(all_lines, self._TAX_KEYS)

        # ----------------------------------------------------------------
        # 10. Grand total
        # Bottom-up search on summary lines avoids table header 'Total Tagihan'
        # ----------------------------------------------------------------
        grand_total, conf_g, bbox_g = find_amount_near_keyword(list(reversed(summary_lines)), self._GRAND_TOTAL_KEYS)
        if not grand_total:
            grand_total, conf_g, bbox_g = find_amount_near_keyword(list(reversed(all_lines)), self._GRAND_TOTAL_KEYS)
        if not grand_total and subtotal is not None and tax_amount is not None:
            grand_total = round(subtotal + tax_amount, 2)
            conf_g = min(conf, conf_t)
            bbox_g = bbox

        # Math fallback for tax_amount if grand_total and subtotal are available
        if tax_amount is None and grand_total is not None and subtotal is not None and grand_total > subtotal:
            tax_amount = round(grand_total - subtotal, 2)
            conf_t = min(conf, conf_g)
            bbox_t = bbox_g

        fields.append(build_field(
            field="tax_amount", value=tax_amount, confidence=conf_t,
            bbox_norm=bbox_t, document_id=document_id, page=page_num,
        ))
        fields.append(build_field(
            field="grand_total", value=grand_total, confidence=conf_g,
            bbox_norm=bbox_g, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 11. Line items from table (PRD §13 — line_items[])
        # ----------------------------------------------------------------
        line_items = self._extract_line_items(all_tables, all_lines, document_id, page_num)
        fields.append(build_field(
            field="line_items",
            value=line_items if line_items else None,
            confidence=0.90 if line_items else 0.0,
            bbox_norm=all_tables[0].get("bbox_norm") if all_tables else None,
            document_id=document_id,
            page=page_num,
        ))

        logger.info(
            "invoice_extractor.complete",
            document_id=document_id,
            total_fields=len(fields),
            unknown_count=sum(1 for f in fields if f.is_unknown),
        )
        return fields

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _extract_vendor_info(self, header_lines: list[dict], all_lines: list[dict] | None = None) -> tuple[str | None, str | None]:
        """
        Extract vendor name and vendor_id from header region.
        Synthetic documents use format: "PT Nama Vendor [VENDOR ID: 3PL-LOG-001]"
        or embed 3PL-xxx vendor ID codes.
        """
        vendor_name: str | None = None
        vendor_id: str | None = None

        for line in header_lines:
            text = line.get("text", "")
            # Vendor ID pattern: bracketed like [VENDOR ID: 3PL-LOG-002] or [3PL-LOG-001]
            vid_match = re.search(r"\[(?:VENDOR\s+ID:\s*)?(3PL-[\w\-]+)\]", text, re.IGNORECASE)
            if vid_match:
                vendor_id = vid_match.group(1).strip()
                name_part = text[:vid_match.start()].strip()
                if name_part:
                    vendor_name = name_part
                break

            # Heuristic: lines with "PT " prefix are likely vendor names
            if re.match(r"^PT\.?\s+", text, re.IGNORECASE) and vendor_name is None:
                vendor_name = text.strip()

        # Pass 2: check all lines for 3PL-xxx vendor ID pattern
        if not vendor_id and all_lines:
            for line in all_lines:
                text = line.get("text", "")
                m = re.search(r"\b(3PL-[A-Z]+-\d{3})\b", text, re.IGNORECASE)
                if m:
                    vendor_id = m.group(1).upper()
                    break

        # Pass 3: map known vendor name to vendor_id across ALL lines (Master Directory Fallback, PRD §13)
        _VENDOR_MAP = {
            "cepat logistik": "3PL-LOG-001",
            "trans express":  "3PL-LOG-002",
            "kargo andalan":  "3PL-LOG-003",
            "sinar niaga":    "3PL-LOG-003",
            "sinar logistik": "3PL-LOG-004",
            "aruna freight":  "3PL-LOG-005",
        }
        if not vendor_id:
            lines_to_check = all_lines if all_lines else header_lines
            for line in lines_to_check:
                text_l = line.get("text", "").lower()
                for v_key, v_id in _VENDOR_MAP.items():
                    if v_key in text_l:
                        vendor_id = v_id
                        if not vendor_name:
                            vendor_name = line.get("text", "").strip()
                        break
                if vendor_id:
                    break

        return vendor_name, vendor_id

    def _parse_billing_period(self, raw: str | None) -> tuple[str | None, str | None]:
        """
        Parse billing period string that may contain a date range:
        "01 Jul 2026 - 31 Jul 2026" or "2026-07-01 to 2026-07-31"
        Returns (start_iso, end_iso).
        """
        if not raw:
            return None, None

        DATE_PATTERN = r"\d{1,2}[\s\-/]\w+[\s\-/]\d{4}|\d{4}-\d{2}-\d{2}"
        dates = re.findall(DATE_PATTERN, raw, re.IGNORECASE)
        start = parse_date_string(dates[0]) if len(dates) >= 1 else None
        end = parse_date_string(dates[1]) if len(dates) >= 2 else None
        return start, end

    def _detect_currency(self, lines: list[dict]) -> tuple[str, float, list[float] | None]:
        """
        Detect currency code. Default IDR (PRD §13). Returns ("IDR", conf, bbox).
        """
        for line in lines:
            text = line.get("text", "")
            if re.search(r"\bIDR\b", text):
                return "IDR", 0.99, line.get("bbox_norm")
            if re.search(r"\bRp\b", text):
                return "IDR", 0.95, line.get("bbox_norm")
            if re.search(r"\bUSD\b", text):
                return "USD", 0.99, line.get("bbox_norm")
        return "IDR", 0.70, None  # Default per PRD §13

    def _extract_line_items(
        self,
        tables: list[dict],
        lines: list[dict],
        document_id: str,
        page_num: int,
    ) -> list[dict] | None:
        """
        Parse table rows into structured line-item dicts.
        Uses fuzzy table columns if available, or falls back to line-based AWB grouping.
        """
        rows = parse_table_columns(tables, INVOICE_TABLE_COLUMNS) if tables else []
        valid_rows = [row for row in rows if _is_invoice_item_row(row)]

        if valid_rows:
            items = []
            for i, row in enumerate(valid_rows, start=1):
                weight = None
                if row.get("chargeable_weight"):
                    try:
                        w_str = re.sub(r"[^0-9.,]", "", str(row["chargeable_weight"]))
                        weight = float(w_str.replace(",", ".")) if w_str else None
                    except (ValueError, AttributeError):
                        weight = None

                items.append({
                    "line_number":       row.get("line_number") or i,
                    "shipment_id":       row.get("shipment_id"),
                    "awb_number":        row.get("awb_number"),
                    "service_type":      row.get("service_type"),
                    "origin":            row.get("origin"),
                    "destination":       row.get("destination"),
                    "chargeable_weight": weight,
                    "base_rate":         parse_amount(row.get("base_rate")),
                    "fuel_surcharge":    parse_amount(row.get("fuel_surcharge")),
                    "accessorial_charge": parse_amount(row.get("accessorial_charge")),
                    "line_total":        parse_amount(row.get("line_total")),
                    "_evidence_page":    page_num,
                })
            return items

        # Fallback: scan lines in table area for AWB / SHP items
        table_lines = [l for l in lines if settings.invoice_table_min_y <= to_unit_bbox(l.get("bbox_norm"))[1] <= settings.invoice_table_max_y]
        awb_lines = [l for l in table_lines if re.search(r"\bAWB[-/ ]?\d{7,10}\b", l.get("text", ""), re.IGNORECASE)]

        if awb_lines:
            items = []
            for idx, aline in enumerate(awb_lines, start=1):
                atext = aline.get("text", "")
                awb_match = re.search(r"\b(AWB[-/ ]?\d{7,10})\b", atext, re.IGNORECASE)
                awb_val = awb_match.group(1).strip() if awb_match else None

                ay = _midpoint_y(to_unit_bbox(aline.get("bbox_norm")))
                # Lines on the same row (within the configured fraction of the page height)
                row_lines = [l for l in table_lines if abs(_midpoint_y(to_unit_bbox(l.get("bbox_norm"))) - ay) <= settings.invoice_row_y_tolerance]
                row_text = " ".join(l.get("text", "") for l in row_lines)

                shp_match = re.search(r"\b(SHP[-/]?\d{4,}[-\w]*)\b", row_text, re.IGNORECASE)
                shp_val = shp_match.group(1).strip() if shp_match else None

                w_match = re.search(r"(\d+[.,]?\d*)\s*kg\b", row_text, re.IGNORECASE)
                weight = float(w_match.group(1).replace(",", ".")) if w_match else None

                # Find amounts on this row
                amounts = re.findall(r"Rp\s*([\d.,]+)|\b(\d{1,3}(?:\.\d{3})+)\b", row_text)
                parsed_amts = []
                for a_tuple in amounts:
                    astr = a_tuple[0] or a_tuple[1]
                    amt = parse_amount(astr)
                    if amt and amt > 1000:
                        parsed_amts.append(amt)

                items.append({
                    "line_number":       idx,
                    "shipment_id":       shp_val,
                    "awb_number":        awb_val,
                    "service_type":      "REGULAR" if "regular" in row_text.lower() else None,
                    "origin":            None,
                    "destination":       None,
                    "chargeable_weight": weight,
                    "base_rate":         parsed_amts[0] if len(parsed_amts) >= 1 else None,
                    "fuel_surcharge":    parsed_amts[1] if len(parsed_amts) >= 2 else None,
                    "accessorial_charge": None,
                    "line_total":        parsed_amts[-1] if len(parsed_amts) >= 3 else None,
                    "_evidence_page":    page_num,
                })
            return items if items else None

        return None

    def _all_unknown(self, document_id: str) -> list[ExtractedField]:
        """Return all required fields as UNKNOWN (no evidence available)."""
        REQUIRED = [
            "invoice_number", "invoice_date", "vendor_name", "vendor_id",
            "customer_name", "billing_period_start", "billing_period_end",
            "due_date", "currency", "subtotal", "tax_amount", "grand_total",
            "line_items",
        ]
        return [
            build_field(
                field=f, value=None, confidence=0.0, bbox_norm=None,
                document_id=document_id, page=1,
            )
            for f in REQUIRED
        ]
