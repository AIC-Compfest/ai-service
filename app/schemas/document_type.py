"""Document type enum — PRD §20, §3.1."""

from enum import Enum


class DocumentType(str, Enum):
    INVOICE = "INVOICE"
    POD = "POD"
    SURAT_JALAN = "SURAT_JALAN"
    RATE_AGREEMENT = "RATE_AGREEMENT"
    UNKNOWN = "UNKNOWN"
    # BAST: future extension, belum diimplementasikan (PRD §3.1)
