"""
RateAgreementExtractor — PRD §16 (Minimum fields), §18, §23, §28, §49, §60.

Extracts:
  agreement_id, vendor_id, vendor_name, effective_from, effective_to,
  minimum_charge, dispute_window_days,
  rate_matrix (list of rate rule dicts),
  surcharge_rules (list of surcharge dicts).

IMPORTANT — Boundary of Responsibility (PRD §16, §28):
  This extractor produces RAW extracted data from the rate agreement document.
  Conversion into EXECUTABLE pricing rules (ContractRuleCompiler) is the
  responsibility of backend-api (Go). This service must NOT compute prices.

  Do NOT store contract only as raw text or embedding (PRD §16).
  Extract structured fields so backend-api can compile them into pricing logic.
"""

from __future__ import annotations

import re

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.evidence import ExtractedField
from app.services.extraction.base import BaseExtractor
from app.services.extraction.utils import (
    build_field,
    extract_regex_with_bbox,
    find_amount_near_keyword,
    find_key_value_horizontal,
    find_key_value_vertical,
    parse_amount,
    parse_date_string,
    parse_integer,
    parse_table_columns,
    to_unit_bbox,
)

logger = get_logger(__name__)

REQUIRED_FIELDS = [
    "agreement_id",
    "vendor_id",
    "vendor_name",
    "effective_from",
    "effective_to",
    "minimum_charge",
    "dispute_window_days",
    "rate_matrix",
    "surcharge_rules",
]

# Anchor keyword lists
_AGREEMENT_ID_KEYS = [
    "no. perjanjian", "no perjanjian", "perjanjian no", "agreement id",
    "agreement no", "kontrak no", "nomor kontrak", "perjanjian kerjasama no",
    "no. kontrak", "contract number", "contract no", "msa no",
]
_VENDOR_KEYS = [
    "pihak pertama", "pihak penyedia jasa", "vendor", "logistics provider", "3pl provider",
    "service provider", "perusahaan pengiriman",
]
_EFFECTIVE_FROM_KEYS = [
    "mulai", "mulai:", "berlaku mulai", "effective from", "valid from",
    "masa berlaku mulai", "dari tanggal",
]
_EFFECTIVE_TO_KEYS = [
    "sampai", "sampai:", "berlaku sampai", "effective to", "valid to",
    "berakhir", "masa berlaku sampai", "sampai dengan tanggal", "expiry date",
]
_MIN_CHARGE_KEYS = [
    "minimum charge", "biaya minimum", "minimum fee", "min charge",
    "tarif minimum",
]
_DISPUTE_WINDOW_KEYS = [
    "dispute window", "batas waktu klaim", "dispute period",
    "klaim dalam", "masa klaim", "claim period", "masa sanggah",
]

# Canonical column aliases for rate matrix tables
RATE_MATRIX_COLUMNS = {
    "service_type":      ["service", "jenis layanan", "layanan", "tipe"],
    "origin_zone":       ["origin", "asal", "zona asal", "from", "dari"],
    "destination_zone":  ["destination", "tujuan", "zona tujuan", "to"],
    "weight_min_kg":     ["min weight", "min kg", "weight min", "berat min", "dari kg"],
    "weight_max_kg":     ["max weight", "max kg", "weight max", "berat max", "sampai kg"],
    "base_rate_idr":     ["base rate", "tarif dasar", "rate", "harga", "price", "biaya"],
}

# Surcharge keyword patterns
_SURCHARGE_PATTERNS = [
    (r"fuel surcharge[:\s]+(\d+(?:[.,]\d+)?)\s*%", "fuel_surcharge_percent"),
    (r"bahan bakar[:\s]+(\d+(?:[.,]\d+)?)\s*%", "fuel_surcharge_percent"),
    (r"bbm[:\s]+(\d+(?:[.,]\d+)?)\s*%", "fuel_surcharge_percent"),
    (r"handling fee[:\s]+(?:rp[\s.]*)?([\d.,]+)", "handling_fee_idr"),
    (r"insurance[:\s]+(\d+(?:[.,]\d+)?)\s*%", "insurance_percent"),
    (r"asuransi[:\s]+(\d+(?:[.,]\d+)?)\s*%", "insurance_percent"),
]


class RateAgreementExtractor(BaseExtractor):
    """
    Extracts structured rate agreement fields from a Rate Card / Contract document.
    Output is raw extracted data for backend-api ContractRuleCompiler (PRD §28).
    """

    document_type = "RATE_AGREEMENT"

    def extract(self, document_id: str, ocr_result: dict) -> list[ExtractedField]:
        """
        Extract Rate Agreement fields from Phase 5 pipeline evidence dict.

        Args:
            document_id: UUID of this document asset.
            ocr_result:  Phase 5 evidence dict.

        Returns:
            list[ExtractedField] — all required fields with EvidenceSource.
        """
        logger.info("rate_agreement_extractor.extract_called", document_id=document_id)

        if not ocr_result:
            return self._all_unknown(document_id)

        all_lines: list[dict] = []
        all_tables: list[dict] = []
        for page in ocr_result.get("pages", []):
            all_lines.extend(page.get("lines", []))
            all_tables.extend(page.get("tables", []))

        if not all_lines:
            return self._all_unknown(document_id)

        page_num = 1
        fields: list[ExtractedField] = []

        # ----------------------------------------------------------------
        # 1. Agreement ID
        # ----------------------------------------------------------------
        raw, conf, bbox = find_key_value_horizontal(all_lines, _AGREEMENT_ID_KEYS)
        if not raw:
            raw, conf, bbox = extract_regex_with_bbox(
                all_lines,
                r"\b(AGR[-/]?\d{3,}|CONTRACT[-/]?\d{3,}|PKS[-/]?\d{3,}|CTR-[A-Z0-9\-]+)\b"
            )
        if raw and "|" in raw:
            raw = raw.split("|")[0].strip()
        if raw:
            # Strip trailing/leading label markers
            raw = re.sub(r"^(?:no\.?\s*perjanjian|perjanjian|kontrak)\s*:\s*", "", raw, flags=re.IGNORECASE).strip()

        fields.append(build_field(
            field="agreement_id", value=raw, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 2. Vendor Name & Vendor ID
        # ----------------------------------------------------------------
        header_lines = [l for l in all_lines if to_unit_bbox(l.get("bbox_norm"))[1] < settings.rate_agreement_header_max_y]
        vendor_name, vendor_id = self._extract_vendor_info(header_lines, all_lines)
        vconf = 0.88 if vendor_name else 0.0
        fields.append(build_field(
            field="vendor_name", value=vendor_name, confidence=vconf,
            bbox_norm=header_lines[0].get("bbox_norm") if header_lines else None,
            document_id=document_id, page=page_num,
        ))
        fields.append(build_field(
            field="vendor_id", value=vendor_id, confidence=vconf,
            bbox_norm=None, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 3. Effective From
        # ----------------------------------------------------------------
        raw, conf, bbox = find_key_value_horizontal(all_lines, _EFFECTIVE_FROM_KEYS)
        if not raw:
            # Try to find date range "DD MMM YYYY - DD MMM YYYY" anywhere
            raw, conf, bbox = extract_regex_with_bbox(
                all_lines,
                r"(\d{1,2}[\s/\-]\w+[\s/\-]\d{4})\s*(?:[-–—]|s\.?d\.?|to|sampai)"
            )
        fields.append(build_field(
            field="effective_from", value=parse_date_string(raw), confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 4. Effective To
        # ----------------------------------------------------------------
        raw, conf, bbox = find_key_value_horizontal(all_lines, _EFFECTIVE_TO_KEYS)
        if not raw:
            raw, conf, bbox = extract_regex_with_bbox(
                all_lines,
                r"(?:[-–—]|s\.?d\.?|to|sampai)\s*(\d{1,2}[\s/\-]\w+[\s/\-]\d{4})"
            )
        fields.append(build_field(
            field="effective_to", value=parse_date_string(raw), confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 5. Minimum Charge
        # ----------------------------------------------------------------
        min_charge, conf, bbox = find_amount_near_keyword(all_lines, _MIN_CHARGE_KEYS)
        fields.append(build_field(
            field="minimum_charge", value=min_charge, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 6. Dispute Window Days
        # ----------------------------------------------------------------
        raw, conf, bbox = find_key_value_horizontal(all_lines, _DISPUTE_WINDOW_KEYS)
        days_val: int | None = None
        if raw:
            m = re.search(r"(\d+)\s*(?:hari|days?|hari kerja)?", raw, re.IGNORECASE)
            if m:
                try:
                    days_val = int(m.group(1))
                except ValueError:
                    pass
        if days_val is None:
            raw_fb, conf, bbox = extract_regex_with_bbox(
                all_lines,
                r"(\d+)\s*(?:hari|days?|working days?)"
            )
            if raw_fb:
                m2 = re.match(r"(\d+)", raw_fb)
                if m2:
                    try:
                        days_val = int(m2.group(1))
                    except ValueError:
                        pass
        fields.append(build_field(
            field="dispute_window_days", value=days_val, confidence=conf,
            bbox_norm=bbox, document_id=document_id, page=page_num,
        ))

        # ----------------------------------------------------------------
        # 7. Rate Matrix (from tables — core pricing data, PRD §16)
        # ----------------------------------------------------------------
        rate_rows = parse_table_columns(all_tables, RATE_MATRIX_COLUMNS)
        rate_matrix = self._process_rate_rows(rate_rows)
        fields.append(build_field(
            field="rate_matrix",
            value=rate_matrix if rate_matrix else None,
            confidence=0.90 if rate_matrix else 0.0,
            bbox_norm=all_tables[0].get("bbox_norm") if all_tables else None,
            document_id=document_id,
            page=page_num,
        ))

        # ----------------------------------------------------------------
        # 8. Surcharge Rules (regex-based multi-pattern extraction)
        # ----------------------------------------------------------------
        surcharges = self._extract_surcharges(all_lines)
        fields.append(build_field(
            field="surcharge_rules",
            value=surcharges if surcharges else None,
            confidence=0.88 if surcharges else 0.0,
            bbox_norm=None,
            document_id=document_id,
            page=page_num,
        ))

        logger.info(
            "rate_agreement_extractor.complete",
            document_id=document_id,
            total_fields=len(fields),
            unknown_count=sum(1 for f in fields if f.is_unknown),
            rate_matrix_rows=len(rate_matrix) if rate_matrix else 0,
            surcharge_count=len(surcharges) if surcharges else 0,
        )
        return fields

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _extract_vendor_info(
        self,
        header_lines: list[dict],
        all_lines: list[dict],
    ) -> tuple[str | None, str | None]:
        """Extract vendor name and vendor_id from header or via keyword search.

        vendor_id format: 3PL-<ABBR>-<NNN> (e.g. 3PL-LOG-001).
        It may appear:
          - Directly in a bracketed label:  [3PL-LOG-001]
          - Embedded in agreement number:   CTR-3PL-LOG-001-2026
          - As standalone field label
        """
        vendor_name: str | None = None
        vendor_id: str | None = None

        # Pass 1: header lines — look for bracketed vendor id or PT name
        for line in header_lines:
            text = line.get("text", "")
            vid_match = re.search(r"\[(?:VENDOR\s+ID:\s*)?(3PL-[\w\-]+)\]", text, re.IGNORECASE)
            if vid_match:
                vendor_id = vid_match.group(1).strip()
                name_part = text[:vid_match.start()].strip()
                if name_part:
                    vendor_name = name_part
                break
            if re.match(r"^PT\s+", text, re.IGNORECASE) and vendor_name is None:
                vendor_name = text.strip()

        # Pass 2: search ALL lines for a 3PL-xxx pattern (embedded in agreement id etc.)
        if not vendor_id:
            for line in all_lines:
                text = line.get("text", "")
                # Pattern: 3PL-ABC-NNN — embedded in strings like CTR-3PL-LOG-001-2026
                m = re.search(r"\b(3PL-[A-Z]+-\d{3})\b", text, re.IGNORECASE)
                if m:
                    vendor_id = m.group(1).upper()
                    break

        # Pass 3: keyword fallback for vendor name
        if not vendor_name:
            raw, _, _ = find_key_value_horizontal(all_lines, _VENDOR_KEYS)
            if raw:
                vendor_name = raw
            else:
                # Use first PT … line in the whole document
                for line in all_lines:
                    text = line.get("text", "")
                    if re.match(r"^PT\s+", text, re.IGNORECASE):
                        vendor_name = text.strip()
                        break

        return vendor_name, vendor_id


    def _process_rate_rows(self, raw_rows: list[dict]) -> list[dict] | None:
        """
        Normalise raw parsed table rows into typed rate rule dicts.
        Skips completely empty rows.
        """
        result = []
        for row in raw_rows:
            if all(v is None for v in row.values()):
                continue
            entry: dict = {
                "service_type":     row.get("service_type"),
                "origin_zone":      row.get("origin_zone"),
                "destination_zone": row.get("destination_zone"),
                "weight_min_kg":    self._parse_weight_cell(row.get("weight_min_kg")),
                "weight_max_kg":    self._parse_weight_cell(row.get("weight_max_kg")),
                "base_rate_idr":    parse_amount(row.get("base_rate_idr")),
            }
            result.append(entry)
        return result if result else None

    def _parse_weight_cell(self, raw: str | None) -> float | None:
        """Parse weight cell from rate matrix table (e.g. '0', '10', '10+')."""
        if not raw:
            return None
        # Remove non-numeric suffix (e.g. "10+", "10 kg")
        m = re.match(r"(\d+(?:[.,]\d+)?)", raw.replace(",", "."))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    def _extract_surcharges(self, lines: list[dict]) -> list[dict] | None:
        """
        Extract surcharge rules by scanning all lines for known surcharge patterns.
        Returns list of {type, value, unit} dicts or None.
        """
        full_text = "\n".join(l.get("text", "") for l in lines)
        surcharges = []
        for pattern, surcharge_type in _SURCHARGE_PATTERNS:
            m = re.search(pattern, full_text, re.IGNORECASE)
            if m:
                raw_val = m.group(1).replace(",", ".")
                try:
                    value = float(raw_val)
                except ValueError:
                    continue

                unit = "percent" if "percent" in surcharge_type else "idr"
                surcharges.append({
                    "type":  surcharge_type,
                    "value": value,
                    "unit":  unit,
                })

        return surcharges if surcharges else None

    def _all_unknown(self, document_id: str) -> list[ExtractedField]:
        return [
            build_field(
                field=f, value=None, confidence=0.0, bbox_norm=None,
                document_id=document_id, page=1,
            )
            for f in REQUIRED_FIELDS
        ]
