# Integrated AHP-ARIE Decision Support System

Aplikasi Sistem Pendukung Keputusan (*Decision Support System* / DSS) berbasis antarmuka grafis (GUI Desktop) yang mengintegrasikan metode **Analytic Hierarchy Process (AHP)** untuk penentuan bobot kriteria objektif dan metode **Adaptive Ranking with Ideal Evaluation (ARIE)** untuk perankingan alternatif multi-kriteria[cite: 1, 2].

Dikembangkan mengacu pada formulasi matematis artikel ilmiah:
> **Fauzi, N. F. F. M., et al. (2025).** *Adaptive Ranking with Ideal Evaluation: A Novel Similarity-Based Multi-Criteria Decision-Making Method.* European Journal of Pure and Applied Mathematics.

---

## Daftar Isi
- [Latar Belakang & Fitur Utama](#latar-belakang--fitur-utama)
- [Arsitektur Perangkat Lunak](#arsitektur-perangkat-lunak)
- [Tahapan Komputasi Matematis](#tahapan-komputasi-matematis)
- [Struktur Folder](#struktur-folder)
- [Prasyarat & Instalasi](#prasyarat--instalasi)
- [Petunjuk Penggunaan](#petunjuk-penggunaan)
- [Contoh Kasus Uji (Halal Supplier Selection)](#contoh-kasus-uji-halal-supplier-selection)
- [Lisensi](#lisensi)

---

## Latar Belakang & Fitur Utama

Metode ARIE dirancang untuk mengatasi kelemahan metode MCDM klasik (seperti TOPSIS atau ARAS) yang cenderung kaku dalam pembobotan solusi ideal serta rentan menghasilkan lonjakan nilai buatan (*artificial advantages*).

### Fitur Kunci Aplikasi:
1. **Unified Multi-Type Normalization:** Mendukung kriteria bertipe *Benefit* (makin besar makin baik), *Cost* (makin kecil makin baik), serta **Target-Type** (nilai optimal berada pada sasaran numerik spesifik $x^T$)[cite: 2].
2. **Dual-Parameter Tuning:**
   - **Parameter Sensitivitas ($\gamma$):** Mengatur ketajaman penalti deviasi non-linear ($\gamma > 1$ untuk sikap *risk-averse*, $\gamma < 1$ untuk *risk-seeking*).
   - **Koefisien Penyeimbang ($\kappa \in [0, 1]$):** Menyeimbangkan fokus antara mengejar solusi ideal versus menghindari solusi anti-ideal/nadir.
3. **Pembobotan Terintegrasi AHP:** Menghitung vektor bobot prioritas dari matriks perbandingan berpasangan (*pairwise comparison scale 1–9*) lengkap dengan validasi rasio konsistensi Saaty ($CR \le 0{,}10$).
4. **Audit Trail Lengkap:** Menyajikan inspeksi transparan dari matriks ternormalisasi ($R$), matriks terbobot ($V$), nilai patokan ekstrem ($v^{\max}$ dan $v^{\min}$), hingga skor kedekatan relatif ($RC_i$).
5. **Arsitektur Berorientasi Objek (OOP):** Pemisahan tegas antara model domain, mesin komputasi matematis, dan antarmuka Tkinter.

---

## Arsitektur Perangkat Lunak

Program ini dibangun dengan prinsip enkapsulasi OOP:
- **`Criterion`**: Objek entitas yang memvalidasi integritas data kriteria dan target nominal[cite: 2].
- **`AHPCalculator`**: Mesin komputasi matriks berpasangan, *eigenvector* bobot, serta pengujian nilai $CI$ dan $CR$.
- **`ARIECalculator`**: Mesin komputasi murni ARIE (Langkah 1 hingga Langkah 5) yang terbebas dari dependensi GUI.
- **`ARIEMainApp` & UI Tabs**: Lapisan presentasi (*View & Controller*) berbasis Python `tkinter` dan `ttk`.

---

## Tahapan Komputasi Matematis

### 1. Pembobotan AHP
- Matriks perbandingan berpasangan $A = [a_{ij}]_{n \times n}$.
- Normalisasi kolom: $\bar{a}_{ij} = \frac{a_{ij}}{\sum_{k=1}^n a_{kj}}$.
- Bobot kriteria: $w_i = \frac{1}{n} \sum_{j=1}^n \bar{a}_{ij}$.
- Rasio Konsistensi: $CR = \frac{CI}{RI}$, di mana $CI = \frac{\lambda_{\max} - n}{n - 1}$.

### 2. Metode ARIE
- **Normalisasi ($r_{ij}$):**
  - *Benefit:* $r_{ij} = \frac{x_{ij}}{x_j^{\max}}$
  - *Cost:* $r_{ij} = \frac{x_j^{\min}}{x_{ij}}$
  - *Target:* $r_{ij} = 1 - \frac{\vert{}x_{ij} - x_j^T\vert{}}{\max(\vert{}x_j^{\max} - x_j^T\vert{}, \vert{}x_j^{\min} - x_j^T\vert{})}$
- **Matriks Terbobot ($v_{ij}$):** $v_{ij} = w_j \cdot r_{ij}$.
- **Ukuran Kemiripan Solusi Ideal & Nadir:**
  $$Sim_i^{\text{best}} = \sum_{j=1}^n \left(\frac{v_{ij}}{v_j^{\max}}\right)^\gamma, \quad Sim_i^{\text{worst}} = \sum_{j=1}^n \left(\frac{v_j^{\min}}{v_{ij}}\right)^\gamma$$
- **Kedekatan Relatif & Perankingan ($RC_i$):**
  $$RC_i = \frac{\kappa \cdot Sim_i^{\text{best}}}{\kappa \cdot Sim_i^{\text{best}} + (1 - \kappa) \cdot Sim_i^{\text{worst}}}$$
  Alternatif diurutkan secara menurun (*descending*) berdasarkan skor $RC_i$.

---

## Struktur Folder

```text
arie-ahp-dss/
│
├── data/
│   └── halal_case_study.json     # Dataset uji referensi artikel
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── criterion.py          # Entitas domain Criterion
│   │   ├── ahp_engine.py         # Modul matematis AHP
│   │   └── arie_engine.py        # Modul komputasi ARIE (Steps 1–5)
│   │
│   └── views/
│       ├── __init__.py
│       ├── setup_tab.py          # Tab spesifikasi kriteria & parameter
│       ├── ahp_tab.py            # Tab perbandingan berpasangan AHP
│       ├── matrix_tab.py         # Tab input data matriks keputusan X
│       └── result_tab.py         # Tab tabel peringkat & audit trail
│
├── main.py                       # Titik masuk aplikasi (Controller utama)
├── requirements.txt              # Daftar pustaka dependensi
└── README.md                     # Dokumentasi proyek
