# CONFIDENCE: where each confidence comes from (Fase 5)
Stempel: tidak ada konfidensi di bawah ini yang dikalibrasi terhadap data berlabel manusia. Kolom `confidence_source` (tambahan aditif pada `ExtractedField`) memberi tahu asalnya.

| Sumber | `confidence_source` | Nilai | Status |
|---|---|---|---|
| Tabel (`pages[].tables[].confidence`) | `structural_score` / `unscored` | Skor struktural 0..1 dari `scoring.table_quality` (dulu selalu 0.0) | Terukur, belum dikalibrasi. Tabel lanjutan tanpa skor: `unscored`, 0.0 |
| `line_items` (invoice), `rate_matrix` (rate agreement) dari tabel | `table_structural_score` | Skor struktural terlemah di antara tabel yang dipakai, maksimum 0.95 | Menggantikan konstanta 0.90 |
| `line_items` dari baris teks (tanpa tabel) | `line_scan_heuristic` | 0.50 (`LINE_SCAN_CONSTANT`) | Konstanta heuristik; dulu 0.90 |
| `vendor_name`, `vendor_id` (invoice 0.90, rate agreement 0.88), `surcharge_rules` (0.88) | `heuristic_constant` | Konstanta | Belum diganti: tidak ada data untuk kalibrasi |
| Field lain via helper `utils.py` (`find_key_value_*`, `find_amount_near_keyword`, regex) | belum diberi label | `line.confidence` bila ada, jika tidak 0.90 | Konstanta default 0.90 di `utils.py` (baris 108, 127, 169, 205, 224, 335, 337) |
| Mata uang (`_detect_currency`) | belum diberi label | 0.99 / 0.95 / 0.70 | Konstanta |
| Stempel dan tanda tangan POD | belum diberi label | 0.85 | Konstanta |
| `ml_extraction[].score` | (di luar `ExtractedField`) | Probabilitas LightGBM | Model belum berguna (OS-012) |

Aturan: nilai tidak ditemukan selalu `confidence=0.0`, `is_unknown=true`, `confidence_source=null`.
Validasi aritmetika (`arithmetic_validation`) dilaporkan terpisah dan TIDAK mengubah konfidensi atau nilai apa pun.
