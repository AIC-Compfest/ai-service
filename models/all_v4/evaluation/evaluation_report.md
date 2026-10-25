# Model evaluation (out-of-fold)

Models: `models\all_v4`  |  floor threshold: from bundle (t_unknown) unless overridden

**How to read this.** `top1_hit` is the share of documents where the best-scored candidate is correct; `random_top1` is what picking a candidate at random would give; `top1_lift` = hit / random. Precision at the floor is only meaningful next to `prevalence`, and a `-` means no candidate reached the floor.

Verdicts: no_better_than_random=6, strong=4, usable=7, weak=13

## INVOICE

| field | verdict | top1 hit (95% CI) | random | lift | prec@floor (CI) | accepted | coverage | prevalence | docs | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| currency | usable | 0.686 (0.520–0.814) | 0.301 | 2.28 | 0.613 (0.438–0.763) | 31 | 0.150 | 0.170 | 35 | 1 doc with no true candidate |
| customer_name | weak | 0.581 (0.408–0.736) | 0.035 | 16.53 | 0.700 (0.397–0.892) | 10 | 0.007 | 0.023 | 31 | 9 docs with no true candidate |
| due_date | strong | 0.900 (0.596–0.982) | 0.423 | 2.13 | 0.875 (0.529–0.978) | 8 | 0.094 | 0.118 | 10 | low_sample (10 docs); 26 docs with no true candidate |
| grand_total | usable | 0.649 (0.488–0.782) | 0.238 | 2.73 | 0.667 (0.478–0.814) | 27 | 0.089 | 0.123 | 37 | 3 docs with no true candidate |
| invoice_date | weak | 0.844 (0.682–0.931) | 0.633 | 1.33 | 0.750 (0.598–0.858) | 40 | 0.471 | 0.376 | 32 | 4 docs with no true candidate |
| invoice_number | strong | 0.914 (0.776–0.970) | 0.222 | 4.11 | 0.722 (0.560–0.842) | 36 | 0.122 | 0.129 | 35 | 5 docs with no true candidate |
| line_items | weak | 0.972 (0.858–0.995) | 0.718 | 1.35 | 0.972 (0.858–0.995) | 36 | 0.590 | 0.590 | 36 |  |
| subtotal | usable | 0.708 (0.508–0.851) | 0.226 | 3.14 | 0.700 (0.397–0.892) | 10 | 0.033 | 0.079 | 24 | 16 docs with no true candidate |
| tax_amount | strong | 1.000 (0.722–1.000) | 0.127 | 7.87 | 0.000 (0.000–0.562) | 3 | 0.010 | 0.033 | 10 | low_sample (10 docs); 30 docs with no true candidate |
| vendor_name | weak | 0.581 (0.408–0.736) | 0.032 | 18.27 | 0.875 (0.640–0.965) | 16 | 0.012 | 0.023 | 31 | 9 docs with no true candidate |

## POD

| field | verdict | top1 hit (95% CI) | random | lift | prec@floor (CI) | accepted | coverage | prevalence | docs | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| line_items | no_better_than_random | 0.958 (0.798–0.993) | 0.833 | 1.15 | 0.727 (0.558–0.849) | 33 | 0.971 | 0.706 | 24 | accepts almost every candidate; 2 docs with no true candidate |
| pihak_kedua_alamat | no_better_than_random | 0.333 (0.097–0.700) | 0.528 | 0.63 | 0.000 (0.000–0.434) | 5 | 0.093 | 0.111 | 6 | low_sample (6 docs); 18 docs with no true candidate |
| pihak_kedua_nama | weak | 0.333 (0.152–0.583) | 0.032 | 10.49 | - | 0 | 0.000 | 0.014 | 15 | low_sample (15 docs); never reaches the floor; 19 docs with no true candidate |
| pihak_pertama_alamat | no_better_than_random | 0.286 (0.082–0.641) | 0.345 | 0.83 | 0.167 (0.030–0.564) | 6 | 0.111 | 0.130 | 7 | low_sample (7 docs); 17 docs with no true candidate |
| pihak_pertama_nama | usable | 0.778 (0.548–0.910) | 0.031 | 25.47 | 1.000 (0.772–1.000) | 13 | 0.012 | 0.016 | 18 | low_sample (18 docs); 16 docs with no true candidate |
| pod_date | no_better_than_random | 0.750 (0.531–0.888) | 0.743 | 1.01 | 0.538 (0.355–0.712) | 26 | 0.542 | 0.500 | 20 | 5 docs with no true candidate |
| pod_number | strong | 1.000 (0.722–1.000) | 0.422 | 2.37 | 0.667 (0.208–0.939) | 3 | 0.015 | 0.050 | 10 | low_sample (10 docs); 23 docs with no true candidate |

## RATE_AGREEMENT

| field | verdict | top1 hit (95% CI) | random | lift | prec@floor (CI) | accepted | coverage | prevalence | docs | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| currency | weak | 0.400 (0.198–0.643) | 0.297 | 1.35 | 0.273 (0.097–0.566) | 11 | 0.133 | 0.181 | 15 | low_sample (15 docs); 8 docs with no true candidate |
| line_items | no_better_than_random | 0.692 (0.500–0.835) | 0.750 | 0.92 | 0.575 (0.422–0.715) | 40 | 0.909 | 0.591 | 26 | accepts almost every candidate |
| load_number | weak | 0.444 (0.246–0.663) | 0.086 | 5.16 | 0.588 (0.360–0.784) | 17 | 0.027 | 0.064 | 18 | low_sample (18 docs); 12 docs with no true candidate |
| rate_card_date | weak | 1.000 (0.610–1.000) | 0.722 | 1.38 | - | 0 | 0.000 | 0.094 | 6 | low_sample (6 docs); never reaches the floor; 16 docs with no true candidate |
| total | usable | 0.792 (0.595–0.908) | 0.526 | 1.51 | 0.756 (0.613–0.858) | 45 | 0.147 | 0.225 | 24 | 6 docs with no true candidate |
| vendor_name | weak | 0.211 (0.085–0.433) | 0.023 | 9.23 | - | 0 | 0.000 | 0.011 | 19 | low_sample (19 docs); never reaches the floor; 11 docs with no true candidate |

## SURAT_JALAN

| field | verdict | top1 hit (95% CI) | random | lift | prec@floor (CI) | accepted | coverage | prevalence | docs | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| line_items | no_better_than_random | 1.000 (0.879–1.000) | 0.940 | 1.06 | 0.824 (0.665–0.917) | 34 | 1.000 | 0.824 | 28 | accepts almost every candidate; 2 docs with no true candidate |
| receiver_address | weak | 0.500 (0.215–0.785) | 0.406 | 1.23 | 0.000 (0.000–0.658) | 2 | 0.020 | 0.079 | 8 | low_sample (8 docs); 24 docs with no true candidate |
| receiver_name | weak | 0.188 (0.066–0.430) | 0.029 | 6.47 | - | 0 | 0.000 | 0.014 | 16 | low_sample (16 docs); never reaches the floor; 16 docs with no true candidate |
| sender_address | usable | 0.615 (0.355–0.823) | 0.372 | 1.66 | 0.429 (0.214–0.674) | 14 | 0.139 | 0.129 | 13 | low_sample (13 docs); 19 docs with no true candidate |
| sender_name | weak | 0.391 (0.222–0.592) | 0.030 | 13.23 | 0.200 (0.036–0.624) | 5 | 0.004 | 0.020 | 23 | 9 docs with no true candidate |
| shipment_date | weak | 0.926 (0.766–0.979) | 0.620 | 1.49 | 0.759 (0.579–0.878) | 29 | 0.446 | 0.477 | 27 | 3 docs with no true candidate |
| surat_jalan_number | usable | 0.733 (0.556–0.858) | 0.249 | 2.95 | 0.600 (0.387–0.781) | 20 | 0.083 | 0.137 | 30 | 2 docs with no true candidate |
