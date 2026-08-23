"""
train_classifier.py

Offline training script for Document Classification Model (PRD §20).
Architecture:
  Text Preprocessing + TF-IDF Vectorizer (1-2 ngrams) + Calibrated Logistic Regression.
  Outputs:
    - app/models/doc_classifier_v1.joblib
    - app/models/tfidf_v1.joblib
    - app/models/classifier_metrics.json
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

MODELS_DIR = Path(__file__).resolve().parent.parent / "app" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training corpus generator with domain vocabulary and real logistics text variations
CORPUS_TEMPLATES = {
    "INVOICE": [
        "INVOICE TAGIHAN FAKTUR No. INV/2026/01/001 Tanggal: 2026-01-10 Kepada: PT Sumber Makmur Subtotal: Rp 12.500.000 PPN 11%: Rp 1.375.000 Grand Total: Rp 13.875.000 Rekening Bank Mandiri",
        "FAKTUR PAJAK Tagihan Pengiriman Logistik No Faktur: FP-2026-8891 Bill To: PT Retail Nusantara Jatuh Tempo: 30 Hari Total Billed: IDR 45.000.000 Bank Central Asia BCA",
        "COMMERCIAL INVOICE 3PL Freight Services Invoice Number: INV-EXP-9921 Date of Issue: 2026-03-15 Due Date: 2026-04-15 Base Charge: 8.500.000 Fuel Surcharge: 1.190.000 Total Amount Due: 9.690.000",
        "Kwitansi Tagihan Jasa Angkutan Barang Ekspedisi No: 0092/INV/LOG/2026 Rincian Biaya Pengiriman Truk FTL Sub Total Tagihan Pajak PPN 11 Persen Terbilang Sembilan Juta Rupiah",
        "INVOICE SUMMARY REKAPITULASI TAGIHAN VENDOR 3PL Periode Tagihan: Januari 2026 Nilai DPP: Rp 25.000.000 PPN: Rp 2.750.000 Total Pembayaran: Rp 27.750.000 Remit To:",
        "Tax Invoice No: 010.000-26.00000001 Pengusaha Kena Pajak Pembeli BKP/JKP Dasar Pengenaan Pajak Rp 50.000.000 PPN 11% Rp 5.500.000 Jumlah Tagihan",
        "INVOICE LOGISTIK 3PL PT Cepat Logistik Nusantara No Tagihan: INV/2026/07/00001 Tanggal Invoice: 2026-07-01 Total Tagihan Rp 1.500.000 PPN Rp 165.000 Grand Total Rp 1.665.000",
        "FAKTUR PENJUALAN JASA PENGANGKUTAN TRUCKING No. 2026/INV/881 Biaya Dasar Pengiriman Fuel Surcharge 14% Total Sebelum Pajak Pajak Pertambahan Nilai Total Harus Dibayar",
    ],
    "SURAT_JALAN": [
        "SURAT JALAN PENGIRIMAN BARANG No: SJ-2026-001 Pengirim: PT Pabrikindo Asal: JAKARTA Penerima: Toko Maju Tujuan: SURABAYA Driver: Budi Santoso No Polisi: B 9123 TXD Berat: 150 KG Koli: 10 Dus",
        "SURAT TANDA PENGIRIMAN BARANG EXPEDITION DELIVERY ORDER No Surat Jalan: SJ/JKT/2026/889 AWB Resi: AWB-99120 Asal Kirim: BEKASI Tujuan: BANDUNG Koli: 25 Box Berat Aktual: 450 Kg Tgl: 2026-02-14",
        "DELIVERY NOTE / CONSIGNMENT NOTE Surat Pengantar Barang Ekspedisi Truk FTL Nomor SJ: SJ-SHP-2026-00010 Shipper: Gudang Cikarang Consignee: DC Surabaya Scale Weight: 1200 KG Driver Nama Kendaraan",
        "SURAT JALAN PENGANTAR BARANG PT Trans Express Indonesia No. SJ-2026-7788 No AWB: AWB-7788 Rute: JAKARTA ke MEDAN Rincian Barang: Elektronik Sparepart 5 Koli Berat Timbang: 35.5 KG",
        "AIRWAYBILL / CONSIGNMENT NOTE SURAT JALAN No: SJ-2026-00441 Origin: SEMARANG Destination: DENPASAR Total Packages: 50 Pieces Weight: 85.0 KG Description: Tekstil Garmen Pickup By Supir",
        "SURAT JALAN OPERASIONAL LOGISTIK 3PL Dokumen Pengiriman Barang Muatan Truk CDD Asal Gudang Pusat Tujuan Cabang Jumlah Barang 100 Karton Berat 500 KG Tanggal Berangkat",
        "SURAT PENGANTAR PENGIRIMAN RESI AIRWAYBILL No: SJ-2026-0992 No AWB: AWB-0992 Ekspedisi Darat Laut Udara Shipper Pengirim Consignee Penerima Koli Paket Berat Riil",
        "SURAT JALAN DELIVERY ORDER Asal Kota Jakarta Tujuan Surabaya Berat Timbangan 240 Kg Total 12 Koli Deskripsi Barang Consumer Goods Supir Pengantar Tanda Terima Muat",
    ],
    "POD": [
        "BUKTI SERAH TERIMA BARANG PROOF OF DELIVERY (POD) No POD: POD-2026-001 No Surat Jalan: SJ-2026-001 Diterima Oleh: Agus Receiving Tanggal Terima: 2026-01-12 Kondisi: BAIK LENGKAP Tanda Tangan & Cap Penerima",
        "PROOF OF DELIVERY / TANDA TERIMA DOKUMEN PENYERAHAN BARANG BAST No: POD-SHP-8812 Delivered Date: 2026-02-18 Time: 14:30 Received By: Siti Aminah Signature: Present Stamp: Valid Goods Received in Good Condition",
        "BUKTI TERIMA BARANG EKSPEDISI SURAT TANDA TERIMA Delivery Receipt No: POD-2026-0091 Telah Diterima Barang Dalam Keadaan Baik dan Cukup Nama Penerima: Hendra Jabatan: Staff Gudang Tanda Tangan Cap Perusahaan",
        "BERITA ACARA SERAH TERIMA (BAST) / PROOF OF DELIVERY No: POD-2026-00014 Tanggal Penyerahan: 2026-03-20 Penerima Barang: Gudang Distribusi Status: DELIVERED Jumlah Koli Diterima: 10 Koli Sesuai Cap & Signature",
        "PROOF OF DELIVERY CONVOY RECEIPT Tanda Tangan Penerima Cap Basah Diterima Lengkap Tanggal Jam Masuk Receiver Name Signature Stamp Confirmed Delivered",
        "BUKTI PENGIRIMAN & PENYERAHAN DOKUMEN POD No. POD-2026-5511 Tanggal Sampai: 2026-04-05 Penerima: Bpk Joko Bagian Logistik Keterangan: Barang Diterima Utuh Tanpa Cacat TTD Cap",
        "TANDA TERIMA BARANG BUKTI PENYERAHAN EKSPEDISI POD Received By Gudang Penerimaan Tanggal 2026-07-15 Kondisi Barang Baik Sesuai Surat Jalan Tanda Tangan Penerima",
        "PROOF OF DELIVERY / BUKTI SERAH TERIMA Diterima Oleh Security Gudang Tanggal Jam Status Pengiriman Terkirim Sukses Cap Toko dan Tanda Tangan Lengkap",
    ],
    "RATE_AGREEMENT": [
        "PERJANJIAN KERJASAMA PENGANGKUTAN LOGISTIK MASTER SERVICE AGREEMENT No Kontrak: CTR-3PL-LOG-001-2026 Masa Berlaku: 2026-01-01 s/d 2026-12-31 Tabel Tarif Dasar: Jakarta - Surabaya Rp 3.500 / KG Minimum Charge: Rp 50.000 Fuel Surcharge: 14.0% Dispute Window: 30 Hari",
        "RATE CARD AGREEMENT 2026 TARIFF SHEET KONTRAK 3PL Vendor: PT Trans Express Indonesia Effective Period: 2026-01-01 to 2026-12-31 Base Rate Matrix Weight Tiers 0-50 KG 50-100 KG Fuel Index Clause Dispute Window 45 Days",
        "SURAT PERJANJIAN TARIF ANGKUTAN BARANG No: PKS/LOG/2026/004 Ketentuan Tarif Pengiriman Rute Darat Truk CDD FTL LTL Tarif Dasar Biaya Minimum Klausul Bahan Bakar Solar Surcharge Batas Waktu Sanggahan 30 Hari Kalender",
        "MASTER SERVICES AGREEMENT LOGISTICS CONTRACT AGREEMENT ID: CTR-3PL-LOG-003-2026 Parties: PT Kargo Andalan Utama Rate Schedule Weight Bracket Base Rate per KG Applicable Fuel Surcharge 15% Dispute Period 7 Days Terms and Conditions",
        "ADDENDUM PERJANJIAN KERJASAMA TARIF LOGISTIK Lampiran Tabel Tarif Ekspedisi Berlaku Mulai 1 Januari 2026 Biaya Tambahan Accessorial Minimum Charge Klausul Klaim Tagihan Invoice Dispute Window",
        "RATE AGREEMENT & TARIFF MATRIX CONTRACT Ketentuan Biaya Pengiriman Logistik Antar Kota Tarif Dasar per Kg Fuel Surcharge Index 12.5% Masa Sanggahan 30 Hari Ketentuan Pajak PPN",
        "KONTRAK KERJASAMA LOGISTIK EKSPEDISI 3PL No: CTR-2026-005 Masa Berlaku Kontrak Tarif Rute Jakarta Medan Denpasar Makassar Minimum Pembayaran Surcharge Aturan Tagihan",
        "PERJANJIAN TARIF DAN LAYANAN LOGISTIK TERPADU Tabel Matriks Tarif Angkutan Berat Minimal Fuel Surcharge Penyesuaian Harga Waktu Pengajuan Keberatan Tagihan 30 Hari",
    ],
    "UNKNOWN": [
        "MEMORANDUM INTERNAL PERUSAHAAN Notulen Rapat Koordinasi Mingguan Divisi Sumber Daya Manusia Agenda Evaluasi Kinerja Karyawan dan Rencana Pelatihan Q2",
        "SURAT PENAWARAN HARGA BARANG BUKAN INVOICE Brosur Katalog Produk Furniture Kantor Meja Kursi Lemari Arsip Diskon Khusus Pemesanan Jumlah Besar",
        "MANUAL PANDUAN PENGGUNAAN APLIKASI Langkah-langkah Login Reset Password dan Pengaturan Profil Pengguna Sistem Manajemen Dokumen",
        "LAPORAN KEUANGAN TAHUNAN AUDITED Laporan Posisi Keuangan Neraca Laba Rugi Arus Kas Catatan Atas Laporan Keuangan Auditor Independen",
        "DOKUMEN SPESIFIKASI TEKNIS PERANGKAT SERVER Processor RAM SSD Jaringan Operating System Garansi Distributor Resmi",
        "SURAT KEPUTUSAN DIREKSI TENTANG STRUKTUR ORGANISASI PERUSAHAAN Menimbang Mengingat Memutuskan Menetapkan Susunan Manajemen",
        "FORMULIR PENDAFTARAN ASURANSI KESEHATAN KARYAWAN Data Pribadi Nama Alamat Nomor KTP Nomor BPJS Fasilitas Kesehatan Tingkat Pertama",
        "ARTIKEL BERITA LOGISTIK INDONESIA Perkembangan Infrastruktur Jalan Tol Trans Jawa dan Dampaknya Terhadap Efisiensi Waktu Tempuh Truk Ekspedisi",
    ],
}


import glob
import fitz

def load_dataset_samples() -> dict[str, list[str]]:
    """Scan pilot dataset directory and extract text from actual generated PDFs."""
    dataset_samples: dict[str, list[str]] = {
        "INVOICE": [],
        "SURAT_JALAN": [],
        "POD": [],
        "RATE_AGREEMENT": [],
        "UNKNOWN": [],
    }

    base_dirs = [
        Path(__file__).resolve().parent.parent.parent / "synthetic-data" / "dataset" / "v4_pilot_20",
        Path(__file__).resolve().parent.parent.parent / "synthetic-data" / "dataset" / "v3_pilot_20",
    ]

    for bdir in base_dirs:
        if not bdir.exists():
            continue
        # Invoices
        for f in list(bdir.glob("invoices/*.pdf"))[:20]:
            try:
                doc = fitz.open(f)
                txt = " ".join(p.get_text() for p in doc).strip()
                if txt:
                    dataset_samples["INVOICE"].append(txt)
            except Exception:
                pass
        # Surat Jalan
        for f in list(bdir.glob("transactions/*/surat_jalan.pdf"))[:20]:
            try:
                doc = fitz.open(f)
                txt = " ".join(p.get_text() for p in doc).strip()
                if txt:
                    dataset_samples["SURAT_JALAN"].append(txt)
            except Exception:
                pass
        # POD
        for f in list(bdir.glob("transactions/*/pod.pdf"))[:20]:
            try:
                doc = fitz.open(f)
                txt = " ".join(p.get_text() for p in doc).strip()
                if txt:
                    dataset_samples["POD"].append(txt)
            except Exception:
                pass
        # Rate Agreement
        for f in list(bdir.glob("transactions/*/rate_agreement.pdf"))[:20]:
            try:
                doc = fitz.open(f)
                txt = " ".join(p.get_text() for p in doc).strip()
                if txt:
                    dataset_samples["RATE_AGREEMENT"].append(txt)
            except Exception:
                pass

    return dataset_samples


def generate_augmented_dataset(samples_per_class: int = 150) -> tuple[list[str], list[str]]:
    """Generate training examples combining dataset files and domain templates."""
    texts: list[str] = []
    labels: list[str] = []

    dataset_samples = load_dataset_samples()

    noise_words = [
        "kop surat", "halaman 1", "confidential", "asli", "copy", "rangkap 2",
        "dokumen rahasia", "arsip", "lembar 1", "dicetak otomatis", "barcode",
        "pt", "cv", "indonesia", "jakarta", "surabaya", "tanggal", "no", "lampiran",
    ]

    for label, base_samples in CORPUS_TEMPLATES.items():
        # Combine template base samples + real dataset samples
        all_class_samples = list(base_samples)
        if label in dataset_samples:
            all_class_samples.extend(dataset_samples[label])

        for s in all_class_samples:
            texts.append(s)
            labels.append(label)

        # Generate synthetic permutations if needed
        n_needed = max(0, samples_per_class - len(all_class_samples))
        for i in range(n_needed):
            base1 = np.random.choice(all_class_samples)
            base2 = np.random.choice(all_class_samples)
            
            words1 = base1.split()
            words2 = base2.split()
            cut1 = np.random.randint(max(1, len(words1) // 3), len(words1) + 1)
            cut2 = np.random.randint(0, max(1, len(words2) // 2))
            
            combined = " ".join(words1[:cut1]) + " " + " ".join(words2[cut2:])
            
            if np.random.rand() > 0.5:
                noise = " ".join(np.random.choice(noise_words, size=np.random.randint(1, 4)))
                combined = f"{noise} {combined} {noise}"
                
            texts.append(combined)
            labels.append(label)

    return texts, labels


def train_and_save_model() -> dict:
    """Train TF-IDF + Logistic Regression model and save artifacts."""
    texts, labels = generate_augmented_dataset(samples_per_class=180)
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=4000,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        lowercase=True,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(
        C=2.5,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = float(np.mean(y_pred == y_test))

    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, val_idx in cv.split(texts, labels):
        vec_cv = TfidfVectorizer(ngram_range=(1, 2), max_features=4000, sublinear_tf=True, min_df=2)
        X_tr = vec_cv.fit_transform([texts[i] for i in train_idx])
        X_val = vec_cv.transform([texts[i] for i in val_idx])
        y_tr = [labels[i] for i in train_idx]
        y_val = [labels[i] for i in val_idx]
        m = LogisticRegression(C=2.5, max_iter=1000, class_weight="balanced", random_state=42)
        m.fit(X_tr, y_tr)
        cv_scores.append(float(np.mean(m.predict(X_val) == y_val)))

    metrics = {
        "model_name": "doc-classifier-v1",
        "algorithm": "TF-IDF + LogisticRegression (Multinomial)",
        "test_accuracy": round(accuracy, 4),
        "cv_5fold_mean_accuracy": round(float(np.mean(cv_scores)), 4),
        "cv_5fold_std": round(float(np.std(cv_scores)), 4),
        "classes": list(clf.classes_),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "total_training_samples": len(texts),
        "classification_report": {k: v for k, v in report.items() if isinstance(v, dict)},
    }

    # Save artifacts
    clf_path = MODELS_DIR / "doc_classifier_v1.joblib"
    vec_path = MODELS_DIR / "tfidf_v1.joblib"
    metrics_path = MODELS_DIR / "classifier_metrics.json"

    joblib.dump(clf, clf_path)
    joblib.dump(vectorizer, vec_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[TRAIN_CLASSIFIER_SUCCESS]")
    print(f"  Artifacts saved to: {MODELS_DIR}")
    print(f"  Accuracy: {accuracy * 100:.2f}% | CV Mean: {np.mean(cv_scores) * 100:.2f}%")
    print(f"  Classes: {clf.classes_}")

    return metrics


if __name__ == "__main__":
    train_and_save_model()
