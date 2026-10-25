# Model evaluation (out-of-fold)

Models: `models\all_v3`  |  floor threshold: from bundle (t_unknown) unless overridden

**How to read this.** `top1_hit` is the share of documents where the best-scored candidate is correct; `random_top1` is what picking a candidate at random would give; `top1_lift` = hit / random. Precision at the floor is only meaningful next to `prevalence`, and a `-` means no candidate reached the floor.

Verdicts: no_better_than_random=8, strong=5, usable=4, weak=14

Not evaluated (no bundle): POD.pihak_kedua_jabatan (skipped), POD.pihak_pertama_jabatan (skipped)

## INVOICE

| field | verdict | top1 hit (95% CI) | random | lift | prec@floor (CI) | accepted | coverage | prevalence | docs | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| currency | usable | 0.657 (0.492–0.792) | 0.301 | 2.18 | 0.643 (0.458–0.793) | 28 | 0.136 | 0.170 | 35 | 1 doc with no true candidate |
| customer_name | weak | 0.581 (0.408–0.736) | 0.035 | 16.53 | 0.875 (0.529–0.978) | 8 | 0.006 | 0.023 | 31 | 9 docs with no true candidate |
| due_date | strong | 0.900 (0.596–0.982) | 0.423 | 2.13 | 1.000 (0.610–1.000) | 6 | 0.071 | 0.118 | 10 | low_sample (10 docs); 26 docs with no true candidate |
| grand_total | weak | 0.541 (0.384–0.690) | 0.203 | 2.67 | 0.542 (0.351–0.721) | 24 | 0.077 | 0.118 | 37 | 3 docs with no true candidate |
| invoice_date | weak | 0.844 (0.682–0.931) | 0.633 | 1.33 | 0.744 (0.589–0.854) | 39 | 0.459 | 0.376 | 32 | 4 docs with no true candidate |
| invoice_number | strong | 0.943 (0.814–0.984) | 0.222 | 4.24 | 0.703 (0.542–0.825) | 37 | 0.126 | 0.129 | 35 | 5 docs with no true candidate |
| line_items | weak | 0.972 (0.858–0.995) | 0.718 | 1.35 | 0.972 (0.858–0.995) | 36 | 0.590 | 0.590 | 36 |  |
| subtotal | usable | 0.750 (0.551–0.880) | 0.172 | 4.36 | 0.800 (0.490–0.943) | 10 | 0.032 | 0.077 | 24 | 16 docs with no true candidate |
| tax_amount | strong | 1.000 (0.722–1.000) | 0.125 | 7.97 | 0.000 (0.000–0.390) | 6 | 0.019 | 0.032 | 10 | low_sample (10 docs); 30 docs with no true candidate |
| vendor_name | weak | 0.548 (0.378–0.708) | 0.032 | 17.25 | 0.733 (0.480–0.891) | 15 | 0.011 | 0.023 | 31 | 9 docs with no true candidate |

## POD

| field | verdict | top1 hit (95% CI) | random | lift | prec@floor (CI) | accepted | coverage | prevalence | docs | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| line_items | no_better_than_random | 0.958 (0.798–0.993) | 0.833 | 1.15 | 0.727 (0.558–0.849) | 33 | 0.971 | 0.706 | 24 | accepts almost every candidate; 2 docs with no true candidate |
| pihak_kedua_alamat | no_better_than_random | 0.333 (0.097–0.700) | 0.528 | 0.63 | 0.000 (0.000–0.658) | 2 | 0.037 | 0.111 | 6 | low_sample (6 docs); 18 docs with no true candidate |
| pihak_kedua_nama | weak | 0.214 (0.076–0.476) | 0.033 | 6.59 | - | 0 | 0.000 | 0.013 | 14 | low_sample (14 docs); never reaches the floor; 20 docs with no true candidate |
| pihak_pertama_alamat | no_better_than_random | 0.286 (0.082–0.641) | 0.345 | 0.83 | 0.167 (0.030–0.564) | 6 | 0.111 | 0.130 | 7 | low_sample (7 docs); 17 docs with no true candidate |
| pihak_pertama_nama | strong | 0.929 (0.685–0.987) | 0.032 | 28.75 | - | 0 | 0.000 | 0.013 | 14 | low_sample (14 docs); never reaches the floor; 20 docs with no true candidate |
| pod_date | no_better_than_random | 0.750 (0.531–0.888) | 0.743 | 1.01 | 0.560 (0.371–0.733) | 25 | 0.521 | 0.500 | 20 | 5 docs with no true candidate |
| pod_number | strong | 1.000 (0.722–1.000) | 0.422 | 2.37 | 0.667 (0.208–0.939) | 3 | 0.015 | 0.050 | 10 | low_sample (10 docs); 23 docs with no true candidate |

## RATE_AGREEMENT

| field | verdict | top1 hit (95% CI) | random | lift | prec@floor (CI) | accepted | coverage | prevalence | docs | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| currency | no_better_than_random | 0.333 (0.152–0.583) | 0.297 | 1.12 | 0.250 (0.071–0.591) | 8 | 0.096 | 0.181 | 15 | low_sample (15 docs); 8 docs with no true candidate |
| destination | no_better_than_random | 1.000 (0.676–1.000) | 1.000 | 1.00 | 1.000 (0.906–1.000) | 37 | 0.925 | 0.925 | 8 | low_sample (8 docs); accepts almost every candidate; 1 doc with no true candidate |
| line_items | no_better_than_random | 0.692 (0.500–0.835) | 0.750 | 0.92 | 0.575 (0.422–0.715) | 40 | 0.909 | 0.591 | 26 | accepts almost every candidate |
| load_number | weak | 0.444 (0.246–0.663) | 0.086 | 5.16 | 0.562 (0.332–0.769) | 16 | 0.025 | 0.064 | 18 | low_sample (18 docs); 12 docs with no true candidate |
| rate_card_date | weak | 1.000 (0.610–1.000) | 0.722 | 1.38 | - | 0 | 0.000 | 0.094 | 6 | low_sample (6 docs); never reaches the floor; 16 docs with no true candidate |
| total | usable | 0.833 (0.641–0.933) | 0.521 | 1.60 | 0.773 (0.630–0.872) | 44 | 0.138 | 0.217 | 24 | 6 docs with no true candidate |
| vendor_name | weak | 0.263 (0.118–0.488) | 0.023 | 11.54 | - | 0 | 0.000 | 0.011 | 19 | low_sample (19 docs); never reaches the floor; 11 docs with no true candidate |

## SURAT_JALAN

| field | verdict | top1 hit (95% CI) | random | lift | prec@floor (CI) | accepted | coverage | prevalence | docs | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| line_items | no_better_than_random | 1.000 (0.879–1.000) | 0.940 | 1.06 | 0.824 (0.665–0.917) | 34 | 1.000 | 0.824 | 28 | accepts almost every candidate; 2 docs with no true candidate |
| receiver_address | weak | 0.500 (0.215–0.785) | 0.406 | 1.23 | 0.000 (0.000–0.658) | 2 | 0.020 | 0.079 | 8 | low_sample (8 docs); 24 docs with no true candidate |
| receiver_name | weak | 0.250 (0.102–0.495) | 0.029 | 8.63 | - | 0 | 0.000 | 0.014 | 16 | low_sample (16 docs); never reaches the floor; 16 docs with no true candidate |
| sender_address | weak | 0.538 (0.291–0.768) | 0.372 | 1.45 | 0.467 (0.248–0.699) | 15 | 0.149 | 0.129 | 13 | low_sample (13 docs); 19 docs with no true candidate |
| sender_name | weak | 0.391 (0.222–0.592) | 0.030 | 13.23 | 0.200 (0.036–0.624) | 5 | 0.004 | 0.020 | 23 | 9 docs with no true candidate |
| shipment_date | weak | 0.926 (0.766–0.979) | 0.620 | 1.49 | 0.815 (0.633–0.918) | 27 | 0.415 | 0.477 | 27 | 3 docs with no true candidate |
| surat_jalan_number | usable | 0.733 (0.556–0.858) | 0.249 | 2.95 | 0.545 (0.347–0.731) | 22 | 0.091 | 0.137 | 30 | 2 docs with no true candidate |
