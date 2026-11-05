"""
Sistem Pendukung Keputusan (MCDM) Metode ARIE & AHP
Berdasarkan artikel:
Fauzi, N. F. F. M., et al. (2025). "Adaptive Ranking with Ideal Evaluation:
A Novel Similarity-Based Multi-Criteria Decision-Making Method."
European Journal of Pure and Applied Mathematics.

Aplikasi Web Interaktif Berbasis Object-Oriented Programming (OOP) dengan Streamlit.
Jalankan dengan perintah: streamlit run app.py
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import streamlit as st


# ==============================================================================
# 1. DOMAIN MODELS & ENTITIES (OOP)
# ==============================================================================

@dataclass
class Criterion:
    """Representasi entitas Kriteria/Faktor dalam sistem pendukung keputusan."""
    code: str
    name: str
    ctype: str  # 'benefit', 'cost', atau 'target'
    weight: float = 0.2
    target_val: Optional[float] = None

    def validate(self) -> None:
        if self.ctype not in ("benefit", "cost", "target"):
            raise ValueError(f"Tipe kriteria '{self.ctype}' pada {self.code} tidak valid.")
        if self.ctype == "target" and self.target_val is None:
            raise ValueError(f"Kriteria '{self.code}' bertipe 'target' wajib memiliki nilai target (x_T).")


@dataclass
class AHPResult:
    """Menyimpan hasil perhitungan bobot dan evaluasi konsistensi metode AHP."""
    pairwise_matrix: pd.DataFrame
    normalized_matrix: pd.DataFrame
    weights: np.ndarray
    lambda_max: float
    ci: float
    ri: float
    cr: float
    is_consistent: bool


@dataclass
class ARIEResult:
    """Menyimpan seluruh hasil perhitungan bertahap dan perankingan akhir ARIE."""
    raw_matrix: pd.DataFrame
    normalized_matrix: pd.DataFrame
    weighted_matrix: pd.DataFrame
    v_max: pd.Series
    v_min: pd.Series
    sim_best: pd.Series
    sim_worst: pd.Series
    rc_scores: pd.Series
    ranking_df: pd.DataFrame
    gamma: float
    kappa: float


# ==============================================================================
# 2. MATHEMATICAL ENGINES (OOP)
# ==============================================================================

class AHPCalculator:
    """
    Mesin komputasi Analytic Hierarchy Process (AHP) berdasarkan skala Saaty (1-9).
    Menghitung bobot prioritas kriteria, Consistency Index (CI), dan Consistency Ratio (CR).
    """

    # Tabel Random Index (RI) standar Saaty untuk ukuran matriks n = 1 s.d. 20
    RI_TABLE: Dict[int, float] = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
        11: 1.51, 12: 1.54, 13: 1.56, 14: 1.57, 15: 1.59,
        16: 1.60, 17: 1.61, 18: 1.62, 19: 1.63, 20: 1.64
    }

    # Pilihan skala perbandingan Saaty
    SAATY_OPTIONS: List[Tuple[str, float]] = [
        ("1/9 - Mutlak kurang penting (0.111)", 1.0 / 9.0),
        ("1/8 - Sangat-sangat kurang penting (0.125)", 1.0 / 8.0),
        ("1/7 - Sangat kurang penting (0.143)", 1.0 / 7.0),
        ("1/6 - Antara cukup & sangat kurang penting (0.167)", 1.0 / 6.0),
        ("1/5 - Cukup kurang penting (0.200)", 1.0 / 5.0),
        ("1/4 - Antara sedikit & cukup kurang penting (0.250)", 1.0 / 4.0),
        ("1/3 - Sedikit kurang penting (0.333)", 1.0 / 3.0),
        ("1/2 - Antara sama & sedikit kurang penting (0.500)", 1.0 / 2.0),
        ("1 - Sama penting (1.000)", 1.0),
        ("2 - Antara sama & sedikit lebih penting (2.000)", 2.0),
        ("3 - Sedikit lebih penting (3.000)", 3.0),
        ("4 - Antara sedikit & cukup lebih penting (4.000)", 4.0),
        ("5 - Cukup lebih penting (5.000)", 5.0),
        ("6 - Antara cukup & sangat lebih penting (6.000)", 6.0),
        ("7 - Sangat lebih penting (7.000)", 7.0),
        ("8 - Sangat-sangat lebih penting (8.000)", 8.0),
        ("9 - Mutlak lebih penting (9.000)", 9.0),
    ]

    @classmethod
    def compute(cls, matrix: np.ndarray, criteria_codes: List[str]) -> AHPResult:
        """
        Menghitung vektor bobot prioritas dan rasio konsistensi dari matriks berpasangan n x n.
        """
        n = matrix.shape[0]
        if matrix.shape[1] != n:
            raise ValueError("Matriks perbandingan berpasangan AHP harus berukuran persegi (n x n).")
        if np.any(matrix <= 0):
            raise ValueError("Seluruh elemen matriks perbandingan berpasangan harus bernilai positif (> 0).")

        # 1. Normalisasi kolom: a_bar_ij = a_ij / sum_k(a_kj)
        col_sums = matrix.sum(axis=0)
        norm_matrix = matrix / col_sums

        # 2. Vektor bobot prioritas (rata-rata baris dari matriks ternormalisasi)
        weights = norm_matrix.mean(axis=1)
        weights = weights / weights.sum()  # Memastikan total tepat 1.0

        # 3. Menghitung nilai Eigen Maksimum (lambda_max)
        aw = matrix @ weights
        lambda_max = float(np.mean(aw / weights))

        # 4. Menghitung Consistency Index (CI) dan Consistency Ratio (CR)
        if n <= 2:
            ci = 0.0
            ri = 0.0
            cr = 0.0
        else:
            ci = (lambda_max - n) / (n - 1)
            ri = cls.RI_TABLE.get(n, 1.64)
            cr = ci / ri if ri > 0 else 0.0

        is_consistent = bool(cr <= 0.10 + 1e-9)

        df_pairwise = pd.DataFrame(matrix, index=criteria_codes, columns=criteria_codes)
        df_norm = pd.DataFrame(norm_matrix, index=criteria_codes, columns=criteria_codes)

        return AHPResult(
            pairwise_matrix=df_pairwise,
            normalized_matrix=df_norm,
            weights=weights,
            lambda_max=lambda_max,
            ci=float(ci),
            ri=float(ri),
            cr=float(cr),
            is_consistent=is_consistent,
        )


class ARIECalculator:
    """
    Mesin komputasi metode ARIE (Adaptive Ranking with Ideal Evaluation)
    berdasarkan Fauzi et al. (2025).
    """

    def __init__(
        self,
        criteria: List[Criterion],
        gamma: float = 1.0,
        kappa: float = 0.5
    ) -> None:
        if not (0.1 <= gamma <= 3.0):
            raise ValueError("Parameter sensitivitas gamma (γ) harus berada pada rentang [0.1, 3.0].")
        if not (0.0 <= kappa <= 1.0):
            raise ValueError("Koefisien penyeimbang kappa (κ) harus berada pada rentang [0.0, 1.0].")

        for c in criteria:
            c.validate()

        self.criteria = criteria
        self.gamma = float(gamma)
        self.kappa = float(kappa)

    def normalize_matrix(self, df_x: pd.DataFrame) -> pd.DataFrame:
        """
        Tahap 1: Normalisasi Multi-Tipe (r_ij)
        - Benefit : r_ij = x_ij / x_max
        - Cost    : r_ij = x_min / x_ij
        - Target  : r_ij = 1 - (|x_ij - x_T| / max(|x_max - x_T|, |x_min - x_T|))
        """
        r_data: Dict[str, np.ndarray] = {}

        for crit in self.criteria:
            col = df_x[crit.code].astype(float).to_numpy()
            x_max = float(np.max(col))
            x_min = float(np.min(col))

            if crit.ctype == "benefit":
                if x_max == 0:
                    raise ValueError(
                        f"Kriteria Benefit '{crit.code}' memiliki nilai maksimum 0 (pembagian dengan nol)."
                    )
                if np.any(col < 0):
                    raise ValueError(
                        f"Kriteria '{crit.code}' mengandung nilai negatif. Gunakan nilai positif pada matriks keputusan."
                    )
                r_col = col / x_max

            elif crit.ctype == "cost":
                if np.any(col <= 0):
                    raise ValueError(
                        f"Kriteria Cost '{crit.code}' memiliki nilai <= 0 yang menyebabkan pembagian dengan nol."
                    )
                r_col = x_min / col

            elif crit.ctype == "target":
                x_t = float(crit.target_val if crit.target_val is not None else 0.0)
                max_dev = max(abs(x_max - x_t), abs(x_min - x_t))
                if max_dev == 0:
                    # Jika seluruh alternatif tepat bernilai sama dengan x_T
                    r_col = np.ones_like(col, dtype=float)
                else:
                    r_col = 1.0 - (np.abs(col - x_t) / max_dev)
            else:
                raise ValueError(f"Tipe kriteria tidak dikenali: {crit.ctype}")

            r_data[crit.code] = r_col

        return pd.DataFrame(r_data, index=df_x.index)

    def compute_weighted_matrix(self, df_r: pd.DataFrame) -> pd.DataFrame:
        """
        Tahap 2: Pembobotan Matriks (v_ij = w_j * r_ij)
        """
        weights = np.array([c.weight for c in self.criteria], dtype=float)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            raise ValueError("Total bobot kriteria harus lebih besar dari 0.")
        # Pastikan bobot ternormalisasi (sum = 1.0)
        weights = weights / weight_sum

        v_values = df_r.to_numpy() * weights
        return pd.DataFrame(v_values, index=df_r.index, columns=df_r.columns)

    def evaluate(self, df_x: pd.DataFrame, alternative_names: Dict[str, str]) -> ARIEResult:
        """
        Menjalankan keseluruhan tahapan matematis ARIE dari matriks keputusan mentah X
        hingga menghasilkan skor kedekatan relatif (RC_i) dan peringkat akhir.
        """
        # 1. Normalisasi Multi-Tipe (R)
        df_r = self.normalize_matrix(df_x)

        # 2. Matriks Terbobot (V)
        df_v = self.compute_weighted_matrix(df_r)

        # 3. Benchmark Solusi Ideal (v_j^max) & Anti-Ideal (v_j^min)
        v_max = df_v.max(axis=0)
        v_min = df_v.min(axis=0)

        # Penanganan stabilitas numerik apabila sebuah alternatif tepat bernilai 0 pada kriteria target
        # (Sesuai definisi rasio v_j^min / v_ij, jika v_ij = 0 ditambahkan epsilon sangat kecil agar terdefinisi)
        eps = 1e-12
        v_mat = df_v.to_numpy()
        v_max_arr = np.maximum(v_max.to_numpy(), eps)
        v_min_arr = v_min.to_numpy()
        v_mat_safe = np.where(v_mat <= 0, eps, v_mat)

        # Ukuran Kemiripan terhadap solusi Ideal (Sim_i^best) dan Anti-Ideal (Sim_i^worst)
        # Sim_i^best = sum_{j=1}^n (v_ij / v_j^max)^gamma
        sim_best_arr = np.sum(np.power(v_mat / v_max_arr, self.gamma), axis=1)

        # Sim_i^worst = sum_{j=1}^n (v_j^min / v_ij)^gamma
        sim_worst_arr = np.sum(np.power(v_min_arr / v_mat_safe, self.gamma), axis=1)

        sim_best = pd.Series(sim_best_arr, index=df_x.index, name="Sim_best")
        sim_worst = pd.Series(sim_worst_arr, index=df_x.index, name="Sim_worst")

        # 4. Skor Kedekatan Relatif (RC_i)
        # RC_i = (kappa * Sim_i^best) / ((kappa * Sim_i^best) + ((1 - kappa) * Sim_i^worst))
        numerator = self.kappa * sim_best_arr
        denominator = numerator + ((1.0 - self.kappa) * sim_worst_arr)
        rc_arr = np.where(denominator > 0, numerator / denominator, 0.0)
        rc_scores = pd.Series(rc_arr, index=df_x.index, name="RC")

        # 5. Penyusunan Tabel Perankingan Akhir (diurutkan menurun berdasarkan RC_i)
        ranking_df = pd.DataFrame({
            "Kode Alternatif": df_x.index,
            "Nama Alternatif": [alternative_names.get(code, code) for code in df_x.index],
            "Sim_best": sim_best.values,
            "Sim_worst": sim_worst.values,
            "Skor RC": rc_scores.values,
        })

        ranking_df = ranking_df.sort_values(by="Skor RC", ascending=False).reset_index(drop=True)
        ranking_df.insert(0, "Peringkat", np.arange(1, len(ranking_df) + 1))

        return ARIEResult(
            raw_matrix=df_x.copy(),
            normalized_matrix=df_r,
            weighted_matrix=df_v,
            v_max=v_max,
            v_min=v_min,
            sim_best=sim_best,
            sim_worst=sim_worst,
            rc_scores=rc_scores,
            ranking_df=ranking_df,
            gamma=self.gamma,
            kappa=self.kappa,
        )


# ==============================================================================
# 3. BENCHMARK CASE STUDY PROVIDER (OOP)
# ==============================================================================

class BenchmarkDataset:
    """
    Menyediakan studi kasus benchmark pemilihan pemasok halal (Halal Supplier Selection)
    berdasarkan Fauzi et al. (2025).
    """

    @staticmethod
    def get_halal_supplier_case() -> Dict[str, Any]:
        """
        5 Kriteria:
        - C1: Halal Certification (Benefit, w = 0.30)
        - C2: Procurement Cost (Cost, w = 0.20)
        - C3: Delivery Time (Target x_T = 15, w = 0.15)
        - C4: Ingredient Quality (Benefit, w = 0.25)
        - C5: Reputation & Traceability (Benefit, w = 0.10)

        5 Alternatif (S1 - S5):
        Menghasilkan skor benchmark pada gamma=1.0, kappa=0.5:
        - S5: Sim_best = 4.5167, Sim_worst = 2.8778, RC = 0.6108 (#1)
        - S1: Sim_best = 4.3800, Sim_worst = 3.2361, RC = 0.5751 (#2)
        - S4: Sim_best = 3.8333, Sim_worst = 3.3349, RC = 0.5348 (#3)
        - S2: Sim_best = 3.7857, Sim_worst = 3.3778, RC = 0.5285 (#4)
        - S3: Sim_best = 3.1000, Sim_worst = 3.6083, RC = 0.4621 (#5)
        """
        criteria = [
            Criterion("C1", "Halal Certification", "benefit", 0.30, 0.0),
            Criterion("C2", "Procurement Cost", "cost", 0.20, 0.0),
            Criterion("C3", "Delivery Time (Days)", "target", 0.15, 15.0),
            Criterion("C4", "Ingredient Quality", "benefit", 0.25, 0.0),
            Criterion("C5", "Reputation & Traceability", "benefit", 0.10, 0.0),
        ]

        alternatives = [
            {"Kode": "S1", "Nama Alternatif": "Supplier 1 (S1)"},
            {"Kode": "S2", "Nama Alternatif": "Supplier 2 (S2)"},
            {"Kode": "S3", "Nama Alternatif": "Supplier 3 (S3)"},
            {"Kode": "S4", "Nama Alternatif": "Supplier 4 (S4)"},
            {"Kode": "S5", "Nama Alternatif": "Supplier 5 (S5)"},
        ]

        matrix_values = [
            # C1,   C2,   C3,   C4,   C5
            [90.0, 50.0, 14.0, 85.0, 80.0],  # S1
            [80.0, 42.0, 18.0, 75.0, 70.0],  # S2
            [70.0, 60.0, 20.0, 80.0, 60.0],  # S3
            [85.0, 54.0, 12.0, 90.0, 75.0],  # S4
            [95.0, 45.0, 16.0, 90.0, 90.0],  # S5
        ]

        return {
            "n_criteria": 5,
            "n_alternatives": 5,
            "criteria": criteria,
            "alternatives": alternatives,
            "matrix_values": matrix_values,
            "gamma": 1.0,
            "kappa": 0.5,
        }


# ==============================================================================
# 4. STREAMLIT UI & CONTROLLER APPLICATION (OOP)
# ==============================================================================

class ARIEStreamlitApp:
    """
    Kelas Pengendali Utama Antarmuka Streamlit untuk Sistem Pendukung Keputusan AHP-ARIE.
    """

    def __init__(self) -> None:
        st.set_page_config(
            page_title="SPK Metode ARIE & AHP - Fauzi et al. (2025)",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        self._init_session_state()

    def _init_session_state(self) -> None:
        """Inisialisasi state aplikasi pada saat pertama kali dijalankan."""
        if "initialized" not in st.session_state:
            self.load_sample_case_study(notify=False)
            st.session_state["initialized"] = True

    def load_sample_case_study(self, notify: bool = True) -> None:
        """Memuat data studi kasus benchmark Halal Supplier dari Fauzi et al. (2025)."""
        case = BenchmarkDataset.get_halal_supplier_case()
        st.session_state["n_criteria"] = case["n_criteria"]
        st.session_state["n_alternatives"] = case["n_alternatives"]
        st.session_state["gamma"] = case["gamma"]
        st.session_state["kappa"] = case["kappa"]

        # DataFrame Konfigurasi Kriteria
        crit_rows = []
        for c in case["criteria"]:
            crit_rows.append({
                "Kode": c.code,
                "Nama Kriteria": c.name,
                "Tipe": c.ctype,
                "Nilai Target (x_T)": float(c.target_val if c.target_val is not None else 0.0),
                "Bobot": float(c.weight),
            })
        st.session_state["criteria_df"] = pd.DataFrame(crit_rows)

        # DataFrame Konfigurasi Alternatif
        st.session_state["alternatives_df"] = pd.DataFrame(case["alternatives"])

        # DataFrame Matriks Keputusan Mentah (X)
        codes = [c.code for c in case["criteria"]]
        alt_codes = [a["Kode"] for a in case["alternatives"]]
        st.session_state["decision_matrix_df"] = pd.DataFrame(
            case["matrix_values"],
            index=alt_codes,
            columns=codes,
        )

        # Matriks Perbandingan Berpasangan AHP (n x n) default identitas
        n = case["n_criteria"]
        st.session_state["ahp_matrix"] = np.eye(n, dtype=float)
        # Versioning counter untuk mereset data_editor ketika sample dimuat ulang
        st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1

        if notify:
            st.toast("Studi Kasus Benchmark 'Halal Supplier (Fauzi et al., 2025)' berhasil dimuat!", icon="✅")

    def _sync_dimensions(self, new_n_crit: int, new_n_alt: int) -> None:
        """Menyesuaikan ukuran tabel kriteria, alternatif, matriks keputusan, dan matriks AHP secara dinamis."""
        crit_df: pd.DataFrame = st.session_state["criteria_df"].copy()
        alt_df: pd.DataFrame = st.session_state["alternatives_df"].copy()
        mat_df: pd.DataFrame = st.session_state["decision_matrix_df"].copy()
        ahp_mat: np.ndarray = st.session_state["ahp_matrix"].copy()

        old_n_crit = len(crit_df)
        old_n_alt = len(alt_df)

        # 1. Penyesuaian Jumlah Kriteria
        if new_n_crit > old_n_crit:
            new_rows = []
            for j in range(old_n_crit + 1, new_n_crit + 1):
                new_rows.append({
                    "Kode": f"C{j}",
                    "Nama Kriteria": f"Kriteria {j}",
                    "Tipe": "benefit",
                    "Nilai Target (x_T)": 0.0,
                    "Bobot": round(1.0 / new_n_crit, 4),
                })
            crit_df = pd.concat([crit_df, pd.DataFrame(new_rows)], ignore_index=True)
        elif new_n_crit < old_n_crit:
            crit_df = crit_df.iloc[:new_n_crit].reset_index(drop=True)

        # 2. Penyesuaian Jumlah Alternatif
        if new_n_alt > old_n_alt:
            new_alt_rows = []
            for i in range(old_n_alt + 1, new_n_alt + 1):
                new_alt_rows.append({
                    "Kode": f"A{i}",
                    "Nama Alternatif": f"Alternatif {i}",
                })
            alt_df = pd.concat([alt_df, pd.DataFrame(new_alt_rows)], ignore_index=True)
        elif new_n_alt < old_n_alt:
            alt_df = alt_df.iloc[:new_n_alt].reset_index(drop=True)

        # Pastikan kode unik
        crit_codes = [
            str(code).strip() if str(code).strip() else f"C{idx + 1}"
            for idx, code in enumerate(crit_df["Kode"])
        ]
        # Hindari duplikasi kode kriteria
        seen_c = set()
        for idx, c in enumerate(crit_codes):
            if c in seen_c:
                crit_codes[idx] = f"{c}_{idx + 1}"
            seen_c.add(crit_codes[idx])
        crit_df["Kode"] = crit_codes

        alt_codes = [
            str(code).strip() if str(code).strip() else f"A{idx + 1}"
            for idx, code in enumerate(alt_df["Kode"])
        ]
        seen_a = set()
        for idx, a in enumerate(alt_codes):
            if a in seen_a:
                alt_codes[idx] = f"{a}_{idx + 1}"
            seen_a.add(alt_codes[idx])
        alt_df["Kode"] = alt_codes

        # 3. Penyesuaian Matriks Keputusan (X) dengan mempertahankan nilai yang sudah ada
        new_mat = np.full((new_n_alt, new_n_crit), 10.0, dtype=float)
        min_r = min(mat_df.shape[0], new_n_alt)
        min_c = min(mat_df.shape[1], new_n_crit)
        new_mat[:min_r, :min_c] = mat_df.iloc[:min_r, :min_c].to_numpy()
        new_mat_df = pd.DataFrame(new_mat, index=alt_codes, columns=crit_codes)

        # 4. Penyesuaian Matriks Perbandingan Berpasangan AHP
        if ahp_mat.shape[0] != new_n_crit:
            new_ahp = np.eye(new_n_crit, dtype=float)
            min_ahp = min(ahp_mat.shape[0], new_n_crit)
            new_ahp[:min_ahp, :min_ahp] = ahp_mat[:min_ahp, :min_ahp]
            st.session_state["ahp_matrix"] = new_ahp

        st.session_state["criteria_df"] = crit_df
        st.session_state["alternatives_df"] = alt_df
        st.session_state["decision_matrix_df"] = new_mat_df

    def render_sidebar(self) -> None:
        """Merender panel kontrol dimensi, parameter dinamis ARIE (γ dan κ), serta tombol studi kasus."""
        with st.sidebar:
            st.header("⚙️ Konfigurasi & Parameter")

            if st.button("📥 Load Sample Case Study (Halal Supplier)", use_container_width=True, type="primary"):
                self.load_sample_case_study(notify=True)
                st.rerun()

            st.caption("Memuat data benchmark 5 kriteria & 5 alternatif dari artikel Fauzi et al. (2025).")
            st.divider()

            # A. Dimensi Masalah
            st.subheader("1. Dimensi Masalah")
            n_crit = st.number_input(
                "Jumlah Kriteria / Faktor (2 - 20):",
                min_value=2,
                max_value=20,
                value=int(st.session_state["n_criteria"]),
                step=1,
                help="Tentukan banyaknya kriteria/faktor evaluasi (antara 2 hingga 20 kriteria).",
            )
            n_alt = st.number_input(
                "Jumlah Alternatif (2 - 50):",
                min_value=2,
                max_value=50,
                value=int(st.session_state["n_alternatives"]),
                step=1,
                help="Tentukan banyaknya kandidat/alternatif yang dievaluasi (antara 2 hingga 50 alternatif).",
            )

            if n_crit != st.session_state["n_criteria"] or n_alt != st.session_state["n_alternatives"]:
                st.session_state["n_criteria"] = int(n_crit)
                st.session_state["n_alternatives"] = int(n_alt)
                self._sync_dimensions(int(n_crit), int(n_alt))
                st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
                st.rerun()

            col_auto1, col_auto2 = st.columns(2)
            with col_auto1:
                if st.button("🔄 Reset Kode C1..Cn", use_container_width=True):
                    df_c = st.session_state["criteria_df"].copy()
                    df_c["Kode"] = [f"C{j + 1}" for j in range(len(df_c))]
                    df_c["Nama Kriteria"] = [f"Kriteria {j + 1}" for j in range(len(df_c))]
                    st.session_state["criteria_df"] = df_c
                    self._sync_dimensions(len(df_c), len(st.session_state["alternatives_df"]))
                    st.session_state["editor_version"] += 1
                    st.rerun()
            with col_auto2:
                if st.button("🔄 Reset Kode A1..Am", use_container_width=True):
                    df_a = st.session_state["alternatives_df"].copy()
                    df_a["Kode"] = [f"A{i + 1}" for i in range(len(df_a))]
                    df_a["Nama Alternatif"] = [f"Alternatif {i + 1}" for i in range(len(df_a))]
                    st.session_state["alternatives_df"] = df_a
                    self._sync_dimensions(len(st.session_state["criteria_df"]), len(df_a))
                    st.session_state["editor_version"] += 1
                    st.rerun()

            st.divider()

            # C. Parameter Dinamis ARIE
            st.subheader("2. Parameter Dinamis ARIE")

            gamma = st.slider(
                "Parameter Sensitivitas (γ - Gamma):",
                min_value=0.1,
                max_value=3.0,
                value=float(st.session_state["gamma"]),
                step=0.05,
                help="Mengontrol ketajaman penalti non-linear terhadap penyimpangan dari titik benchmark.",
            )
            st.session_state["gamma"] = float(gamma)

            if gamma > 1.0:
                gamma_badge = "🛡️ **Risk-Averse (γ > 1)**: Penalti deviasi diperkuat; mengutamakan performa seimbang."
            elif gamma < 1.0:
                gamma_badge = "🚀 **Risk-Seeking (γ < 1)**: Penalti deviasi diperlembut; mentoleransi kelemahan parsial."
            else:
                gamma_badge = "⚖️ **Risk-Neutral (γ = 1.0)**: Sensitivitas proporsional linier standar."
            st.info(
                f"{gamma_badge}\n\n"
                "• **γ > 1**: *Risk-averse* (menghukum deviasi dari solusi ideal secara ketat).\n"
                "• **γ < 1**: *Risk-seeking* (lebih toleran terhadap deviasi)."
            )

            kappa = st.slider(
                "Koefisien Penyeimbang (κ - Kappa):",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state["kappa"]),
                step=0.05,
                help="Menyeimbangkan bobot kemiripan terhadap solusi ideal (Sim_best) vs anti-ideal (Sim_worst).",
            )
            # Validasi batas ketat [0.0, 1.0]
            kappa = float(np.clip(kappa, 0.0, 1.0))
            st.session_state["kappa"] = kappa

            if kappa > 0.5:
                kappa_badge = "🌟 **Optimis / Ideal-Focused (κ > 0.5)**: Berfokus pada pencapaian solusi ideal terbaik."
            elif kappa < 0.5:
                kappa_badge = "🛑 **Konservatif / Anti-Ideal Avoidance (κ < 0.5)**: Berfokus menghindari kinerja terburuk."
            else:
                kappa_badge = "⚖️ **Seimbang / Balanced (κ = 0.5)**: Bobot setara antara solusi ideal dan anti-ideal."
            st.info(
                f"{kappa_badge}\n\n"
                "• **κ > 0.5**: *Optimis / Ideal-focused*.\n"
                "• **κ < 0.5**: *Konservatif / Anti-ideal avoidance*."
            )

    def render_step1_setup(self) -> None:
        """Merender bagian konfigurasi atribut kriteria dan daftar alternatif."""
        st.subheader("📋 A. Definisi Kriteria & Alternatif")
        ver = st.session_state.get("editor_version", 0)

        col_crit, col_alt = st.columns([3, 2], gap="large")

        with col_crit:
            st.markdown("##### 1. Konfigurasi Kriteria Evaluasi")
            st.caption(
                "Tentukan Kode, Nama Kriteria, Tipe (`benefit`, `cost`, atau `target`), serta **Nilai Target ($x_T$)** "
                "khusus apabila kriteria bertipe `target`."
            )

            edited_crit_df = st.data_editor(
                st.session_state["criteria_df"],
                key=f"crit_editor_{ver}",
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "Kode": st.column_config.TextColumn(
                        "Kode",
                        help="Kode unik kriteria (contoh: C1, C2)",
                        required=True,
                        width="small",
                    ),
                    "Nama Kriteria": st.column_config.TextColumn(
                        "Nama Kriteria",
                        help="Nama deskriptif kriteria/faktor",
                        required=True,
                    ),
                    "Tipe": st.column_config.SelectboxColumn(
                        "Tipe Kriteria",
                        help="'benefit' (makin besar makin baik), 'cost' (makin kecil makin baik), 'target' (optimal pada x_T)",
                        options=["benefit", "cost", "target"],
                        required=True,
                        width="medium",
                    ),
                    "Nilai Target (x_T)": st.column_config.NumberColumn(
                        "Nilai Target (x_T)",
                        help="Digunakan hanya jika Tipe Kriteria = 'target'",
                        format="%.4f",
                        required=True,
                    ),
                    "Bobot": st.column_config.NumberColumn(
                        "Bobot Saat Ini (w_j)",
                        help="Bobot kriteria (dapat diatur melalui bagian B di bawah)",
                        format="%.4f",
                        disabled=True,
                    ),
                },
            )

        with col_alt:
            st.markdown("##### 2. Penamaan Alternatif")
            st.caption("Tentukan Kode (A1, A2, ... atau S1, S2, ...) dan Nama untuk masing-masing alternatif.")

            edited_alt_df = st.data_editor(
                st.session_state["alternatives_df"],
                key=f"alt_editor_{ver}",
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "Kode": st.column_config.TextColumn(
                        "Kode",
                        help="Kode unik alternatif (contoh: A1, A2 atau S1, S2)",
                        required=True,
                        width="small",
                    ),
                    "Nama Alternatif": st.column_config.TextColumn(
                        "Nama Alternatif",
                        help="Nama lengkap kandidat/alternatif",
                        required=True,
                    ),
                },
            )

        # Periksa perubahan kode atau atribut dan sinkronkan ke matriks keputusan
        old_crit_codes = list(st.session_state["criteria_df"]["Kode"])
        old_alt_codes = list(st.session_state["alternatives_df"]["Kode"])

        new_crit_codes = [
            str(c).strip() if str(c).strip() else f"C{i + 1}"
            for i, c in enumerate(edited_crit_df["Kode"])
        ]
        new_alt_codes = [
            str(a).strip() if str(a).strip() else f"A{i + 1}"
            for i, a in enumerate(edited_alt_df["Kode"])
        ]

        edited_crit_df["Kode"] = new_crit_codes
        edited_alt_df["Kode"] = new_alt_codes

        st.session_state["criteria_df"] = edited_crit_df
        st.session_state["alternatives_df"] = edited_alt_df

        if old_crit_codes != new_crit_codes or old_alt_codes != new_alt_codes:
            mat_df = st.session_state["decision_matrix_df"].copy()
            mat_df.columns = new_crit_codes
            mat_df.index = new_alt_codes
            st.session_state["decision_matrix_df"] = mat_df

        # Info ringkas untuk kriteria bertipe target
        target_crits = edited_crit_df[edited_crit_df["Tipe"] == "target"]
        if not target_crits.empty:
            targets_str = ", ".join(
                f"**{row['Kode']} ({row['Nama Kriteria']})** → $x_T = {row['Nilai Target (x_T)']}$"
                for _, row in target_crits.iterrows()
            )
            st.success(f"🎯 **Kriteria Bertipe Target Aktif:** {targets_str}")

    def render_step2_weighting(self) -> None:
        """Merender pilihan metode penentuan bobot: Input Manual Langsung vs Analytic Hierarchy Process (AHP)."""
        st.subheader("⚖️ B. Metode Penentuan Bobot Kriteria")

        weight_method = st.radio(
            "Pilih Metode Penentuan Bobot Kriteria:",
            options=[
                "1. Metode Langsung (Manual Input)",
                "2. Metode Analytic Hierarchy Process (AHP)",
            ],
            horizontal=True,
        )

        crit_df: pd.DataFrame = st.session_state["criteria_df"].copy()
        n = len(crit_df)
        codes = list(crit_df["Kode"])
        names = list(crit_df["Nama Kriteria"])

        if weight_method.startswith("1."):
            st.markdown("##### Input Bobot Kriteria Secara Langsung")
            st.caption(
                "Masukkan nilai bobot untuk setiap kriteria. Apabila total bobot belum bernilai `1.0`, "
                "sistem secara otomatis menormalisasi bobot ($w_j = w_j / \\sum w_k$)."
            )

            cols = st.columns(min(n, 5))
            raw_weights = []
            for idx, row in crit_df.iterrows():
                col_ui = cols[idx % len(cols)]
                with col_ui:
                    val = st.number_input(
                        f"{row['Kode']} - {row['Nama Kriteria']}",
                        min_value=0.0001,
                        max_value=1000.0,
                        value=float(row["Bobot"]) if float(row["Bobot"]) > 0 else 1.0,
                        step=0.05,
                        format="%.4f",
                        key=f"manual_w_{idx}_{st.session_state.get('editor_version', 0)}",
                    )
                    raw_weights.append(val)

            total_raw = float(np.sum(raw_weights))
            norm_weights = [w / total_raw for w in raw_weights]

            # Simpan bobot ternormalisasi ke session_state["criteria_df"]
            st.session_state["criteria_df"]["Bobot"] = norm_weights

            if abs(total_raw - 1.0) > 1e-4:
                st.warning(
                    f"ℹ️ Total bobot input mentah adalah **{total_raw:.4f}** (≠ 1.0). "
                    f"Sistem telah menormalisasi bobot secara otomatis sehingga $\\sum w_j = 1.0000$."
                )
            else:
                st.success("✅ Total bobot kriteria sudah tepat bernilai **1.0000**.")

            w_summary = pd.DataFrame({
                "Kode": codes,
                "Nama Kriteria": names,
                "Tipe": crit_df["Tipe"],
                "Bobot Input": raw_weights,
                "Bobot Ternormalisasi (w_j)": norm_weights,
            })
            st.dataframe(
                w_summary.style.format({"Bobot Input": "{:.4f}", "Bobot Ternormalisasi (w_j)": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.markdown("##### Matriks Perbandingan Berpasangan (Pairwise Comparison Matrix) AHP")
            st.caption(
                "Pilih tingkat kepentingan relatif untuk elemen **segitiga atas** ($a_{ij}$ dengan $i < j$) "
                "menggunakan skala Saaty ($1$ hingga $9$ beserta nilai pecahannya $1/2$ hingga $1/9$). "
                "Nilai diagonal otomatis bernilai `1.0`, dan elemen **segitiga bawah** otomatis terisi nilai resiprokalnya ($a_{ji} = 1 / a_{ij}$)."
            )

            ahp_mat: np.ndarray = st.session_state["ahp_matrix"].copy()
            if ahp_mat.shape != (n, n):
                ahp_mat = np.eye(n, dtype=float)

            option_labels = [opt[0] for opt in AHPCalculator.SAATY_OPTIONS]
            option_values = [opt[1] for opt in AHPCalculator.SAATY_OPTIONS]

            def closest_saaty_index(val: float) -> int:
                diffs = [abs(val - v) for v in option_values]
                return int(np.argmin(diffs))

            col_reset_ahp, _ = st.columns([1, 3])
            with col_reset_ahp:
                if st.button("🔄 Reset Matriks AHP ke Identitas (Semua 1.0)"):
                    st.session_state["ahp_matrix"] = np.eye(n, dtype=float)
                    st.session_state["editor_version"] += 1
                    st.rerun()

            # Input pasangan segitiga atas (i < j)
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            with st.expander("📝 Edit Perbandingan Berpasangan Segitiga Atas (Skala Saaty 1-9 & 1/2-1/9)", expanded=True):
                grid_cols = st.columns(2)
                for p_idx, (i, j) in enumerate(pairs):
                    with grid_cols[p_idx % 2]:
                        current_val = float(ahp_mat[i, j])
                        default_idx = closest_saaty_index(current_val)
                        selected_label = st.selectbox(
                            f"**{codes[i]}** ({names[i]}) terhadap **{codes[j]}** ({names[j]}):",
                            options=option_labels,
                            index=default_idx,
                            key=f"ahp_pair_{i}_{j}_{st.session_state.get('editor_version', 0)}",
                        )
                        chosen_val = option_values[option_labels.index(selected_label)]
                        ahp_mat[i, j] = chosen_val
                        ahp_mat[j, i] = 1.0 / chosen_val

            for d in range(n):
                ahp_mat[d, d] = 1.0

            st.session_state["ahp_matrix"] = ahp_mat

            # Hitung hasil AHP
            ahp_res = AHPCalculator.compute(ahp_mat, codes)

            col_mat_view, col_metrics_view = st.columns([3, 2], gap="large")
            with col_mat_view:
                st.markdown("**Matriks Perbandingan Berpasangan ($A_{n \\times n}$):**")
                st.dataframe(
                    ahp_res.pairwise_matrix.style.format("{:.4f}"),
                    use_container_width=True,
                )

            with col_metrics_view:
                st.markdown("**Hasil Evaluasi Konsistensi & Vektor Prioritas AHP:**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("λ_max", f"{ahp_res.lambda_max:.4f}")
                m2.metric("CI", f"{ahp_res.ci:.4f}")
                m3.metric("RI (Saaty)", f"{ahp_res.ri:.2f}")
                m4.metric("CR", f"{ahp_res.cr:.4f}")

                if ahp_res.is_consistent:
                    st.success(
                        f"✅ **KONSISTEN:** Nilai $CR = {ahp_res.cr:.4f} \\le 0.10$. "
                        "Matriks perbandingan berpasangan memenuhi syarat konsistensi Saaty."
                    )
                else:
                    st.error(
                        f"⚠️ **TIDAK KONSISTEN:** Nilai $CR = {ahp_res.cr:.4f} > 0.10$. "
                        "Disarankan meninjau kembali penilaian perbandingan berpasangan Anda."
                    )

                ahp_weights_df = pd.DataFrame({
                    "Kode": codes,
                    "Nama Kriteria": names,
                    "Bobot Prioritas AHP": ahp_res.weights,
                })
                st.dataframe(
                    ahp_weights_df.style.format({"Bobot Prioritas AHP": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True,
                )

                if st.button("✅ Terapkan Bobot Hasil AHP ke Model ARIE", type="primary", use_container_width=True):
                    st.session_state["criteria_df"]["Bobot"] = ahp_res.weights
                    st.session_state["editor_version"] += 1
                    st.toast("Bobot hasil perhitungan AHP berhasil diterapkan ke model ARIE!", icon="✅")
                    st.rerun()

    def render_step3_decision_matrix(self) -> pd.DataFrame:
        """Merender tabel editable untuk Matriks Keputusan Mentah (X) berukuran alternatif x kriteria."""
        st.subheader("🔢 C. Input Matriks Keputusan ($X$)")
        st.caption(
            "Isi atau ubah nilai kinerja masing-masing alternatif pada setiap kriteria langsung di tabel interaktif berikut."
        )

        crit_df: pd.DataFrame = st.session_state["criteria_df"]
        alt_df: pd.DataFrame = st.session_state["alternatives_df"]
        mat_df: pd.DataFrame = st.session_state["decision_matrix_df"].copy()

        # Pastikan indeks dan kolom sinkron
        mat_df.index = list(alt_df["Kode"])
        mat_df.columns = list(crit_df["Kode"])

        # Tambahkan kolom informasi Nama Alternatif sebagai tampilan index
        display_df = mat_df.copy()
        col_configs = {}
        for _, crow in crit_df.iterrows():
            c_code = crow["Kode"]
            c_name = crow["Nama Kriteria"]
            c_type = crow["Tipe"].upper()
            target_info = f" (x_T={crow['Nilai Target (x_T)']})" if crow["Tipe"] == "target" else ""
            col_configs[c_code] = st.column_config.NumberColumn(
                f"{c_code}: {c_name} [{c_type}{target_info}]",
                help=f"Bobot w = {crow['Bobot']:.4f} | Tipe: {c_type}{target_info}",
                format="%.4f",
                required=True,
            )

        ver = st.session_state.get("editor_version", 0)
        edited_mat_df = st.data_editor(
            display_df,
            key=f"decision_matrix_editor_{ver}",
            use_container_width=True,
            num_rows="fixed",
            column_config=col_configs,
        )

        st.session_state["decision_matrix_df"] = edited_mat_df
        return edited_mat_df

    def render_step4_results_and_audit(self, df_x: pd.DataFrame) -> None:
        """Menjalankan perhitungan ARIE, menampilkan hasil perankingan, grafik bar, tab audit, dan ekspor CSV."""
        st.subheader("🏆 D. Hasil Perankingan, Visualisasi & Audit Matematis ARIE")

        crit_df: pd.DataFrame = st.session_state["criteria_df"]
        alt_df: pd.DataFrame = st.session_state["alternatives_df"]
        gamma: float = float(st.session_state["gamma"])
        kappa: float = float(st.session_state["kappa"])

        # Bangun objek Criterion
        criteria_objs: List[Criterion] = []
        for _, row in crit_df.iterrows():
            criteria_objs.append(
                Criterion(
                    code=str(row["Kode"]),
                    name=str(row["Nama Kriteria"]),
                    ctype=str(row["Tipe"]),
                    weight=float(row["Bobot"]),
                    target_val=float(row["Nilai Target (x_T)"]) if row["Tipe"] == "target" else None,
                )
            )

        alt_name_map = dict(zip(alt_df["Kode"], alt_df["Nama Alternatif"]))

        try:
            calculator = ARIECalculator(criteria=criteria_objs, gamma=gamma, kappa=kappa)
            result = calculator.evaluate(df_x=df_x, alternative_names=alt_name_map)
        except Exception as exc:
            st.error(f"❌ **Terjadi Kesalahan Perhitungan Matematis:** {exc}")
            return

        # Highlight Alternatif Terbaik (#1)
        top_alt = result.ranking_df.iloc[0]
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric(
            "🥇 Peringkat #1 Terbaik",
            f"{top_alt['Kode Alternatif']} - {top_alt['Nama Alternatif']}",
        )
        m_col2.metric("Skor Kedekatan Relatif (RC)", f"{top_alt['Skor RC']:.4f}")
        m_col3.metric("Parameter Sensitivitas (γ)", f"{result.gamma:.2f}")
        m_col4.metric("Koefisien Penyeimbang (κ)", f"{result.kappa:.2f}")

        tab_rank, tab_chart, tab_audit = st.tabs([
            "📊 Tabel Perankingan Akhir & Unduh CSV",
            "📈 Visualisasi Diagram Batang (Bar Chart)",
            "🔍 Audit Perhitungan Matematis Lengkap",
        ])

        # ------------------------------------------------------------------
        # TAB 1: TABEL PERANKINGAN AKHIR & UNDUH CSV
        # ------------------------------------------------------------------
        with tab_rank:
            st.markdown("##### Tabel Perankingan Akhir Metode ARIE")
            st.caption(
                "Alternatif diurutkan secara menurun (*descending*) berdasarkan nilai **Skor Kedekatan Relatif ($RC_i$)**."
            )

            formatted_rank = result.ranking_df.style.format({
                "Sim_best": "{:.4f}",
                "Sim_worst": "{:.4f}",
                "Skor RC": "{:.4f}",
            }).background_gradient(subset=["Skor RC"], cmap="Greens")

            st.dataframe(formatted_rank, use_container_width=True, hide_index=True)

            csv_bytes = result.ranking_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Unduh Hasil Perankingan ke CSV",
                data=csv_bytes,
                file_name=f"hasil_perankingan_ARIE_gamma{result.gamma:.2f}_kappa{result.kappa:.2f}.csv",
                mime="text/csv",
                type="primary",
            )

        # ------------------------------------------------------------------
        # TAB 2: VISUALISASI DIAGRAM BATANG INTERAKTIF
        # ------------------------------------------------------------------
        with tab_chart:
            st.markdown("##### Diagram Batang Urutan Skor Kedekatan Relatif ($RC_i$) Seluruh Alternatif")
            chart_df = result.ranking_df.copy()
            chart_df["Label Alternatif"] = (
                "#" + chart_df["Peringkat"].astype(str) + " "
                + chart_df["Kode Alternatif"] + " (" + chart_df["Nama Alternatif"] + ")"
            )

            # Menggunakan Altair (bawaan Streamlit) agar urutan batang sesuai peringkat & interaktif
            try:
                import altair as alt

                bar_chart = (
                    alt.Chart(chart_df)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X(
                            "Label Alternatif:N",
                            sort=list(chart_df["Label Alternatif"]),
                            title="Alternatif (Urutan Peringkat)",
                            axis=alt.Axis(labelAngle=0),
                        ),
                        y=alt.Y(
                            "Skor RC:Q",
                            title="Skor Kedekatan Relatif (RC_i)",
                            scale=alt.Scale(domain=[0, 1]),
                        ),
                        color=alt.Color(
                            "Skor RC:Q",
                            scale=alt.Scale(scheme="tealblues"),
                            legend=alt.Legend(title="Skor RC"),
                        ),
                        tooltip=[
                            alt.Tooltip("Peringkat:O", title="Peringkat"),
                            alt.Tooltip("Kode Alternatif:N", title="Kode"),
                            alt.Tooltip("Nama Alternatif:N", title="Nama Alternatif"),
                            alt.Tooltip("Skor RC:Q", title="Skor RC", format=".4f"),
                            alt.Tooltip("Sim_best:Q", title="Sim_best", format=".4f"),
                            alt.Tooltip("Sim_worst:Q", title="Sim_worst", format=".4f"),
                        ],
                    )
                    .properties(height=420)
                )

                text_labels = bar_chart.mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-5,
                    fontWeight="bold",
                ).encode(text=alt.Text("Skor RC:Q", format=".4f"))

                st.altair_chart(bar_chart + text_labels, use_container_width=True)
            except Exception:
                st.bar_chart(chart_df.set_index("Label Alternatif")["Skor RC"])

        # ------------------------------------------------------------------
        # TAB 3: AUDIT PERHITUNGAN MATEMATIS LENGKAP
        # ------------------------------------------------------------------
        with tab_audit:
            st.markdown("##### Rincian Tahapan Matematis Metode ARIE (Fauzi et al., 2025)")

            with st.expander("📐 Rumus Matematis ARIE yang Digunakan", expanded=False):
                st.latex(
                    r"r_{ij} = \begin{cases} "
                    r"\dfrac{x_{ij}}{x_j^{\max}} & \text{(Benefit)} \\[1.2ex] "
                    r"\dfrac{x_j^{\min}}{x_{ij}} & \text{(Cost)} \\[1.2ex] "
                    r"1 - \dfrac{|x_{ij} - x_j^T|}{\max(|x_j^{\max} - x_j^T|, |x_j^{\min} - x_j^T|)} & \text{(Target)} "
                    r"\end{cases}"
                )
                st.latex(r"v_{ij} = w_j \cdot r_{ij}")
                st.latex(
                    r"Sim_i^{\text{best}} = \sum_{j=1}^n \left(\frac{v_{ij}}{v_j^{\max}}\right)^\gamma, "
                    r"\quad Sim_i^{\text{worst}} = \sum_{j=1}^n \left(\frac{v_j^{\min}}{v_{ij}}\right)^\gamma"
                )
                st.latex(
                    r"RC_i = \frac{\kappa \cdot Sim_i^{\text{best}}}"
                    r"{\kappa \cdot Sim_i^{\text{best}} + (1 - \kappa) \cdot Sim_i^{\text{worst}}}"
                )

            st.markdown("**1. Matriks Ternormalisasi Multi-Tipe ($R = [r_{ij}]$):**")
            st.dataframe(
                result.normalized_matrix.style.format("{:.4f}"),
                use_container_width=True,
            )

            st.markdown("**2. Matriks Terbobot ($V = [v_{ij}]$ dengan $v_{ij} = w_j \\cdot r_{ij}$):**")
            st.dataframe(
                result.weighted_matrix.style.format("{:.4f}"),
                use_container_width=True,
            )

            st.markdown("**3. Vektor Patokan Solusi Ideal ($v_j^{\\max}$) dan Solusi Anti-Ideal ($v_j^{\\min}$):**")
            benchmark_df = pd.DataFrame(
                [result.v_max.values, result.v_min.values],
                index=["v_max (Solusi Ideal / Best)", "v_min (Solusi Anti-Ideal / Worst)"],
                columns=result.weighted_matrix.columns,
            )
            st.dataframe(
                benchmark_df.style.format("{:.4f}"),
                use_container_width=True,
            )

            st.markdown(
                f"**4. Ukuran Kemiripan ($Sim_i^{{\\text{{best}}}}$ & $Sim_i^{{\\text{{worst}}}}$) dan "
                f"Skor Kedekatan Relatif ($RC_i$) pada $\\gamma = {result.gamma:.2f}, \\kappa = {result.kappa:.2f}$:**"
            )
            sim_audit_df = pd.DataFrame({
                "Kode Alternatif": result.raw_matrix.index,
                "Nama Alternatif": [alt_name_map.get(c, c) for c in result.raw_matrix.index],
                "Sim_best": result.sim_best.values,
                "Sim_worst": result.sim_worst.values,
                "Skor RC": result.rc_scores.values,
            })
            st.dataframe(
                sim_audit_df.style.format({
                    "Sim_best": "{:.4f}",
                    "Sim_worst": "{:.4f}",
                    "Skor RC": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    def run(self) -> None:
        """Titik masuk utama eksekusi antarmuka halaman Streamlit."""
        st.title("🎯 Sistem Pendukung Keputusan Metode ARIE & AHP")
        st.markdown(
            "Implementasi **Adaptive Ranking with Ideal Evaluation (ARIE)** terintegrasi dengan "
            "**Analytic Hierarchy Process (AHP)** berdasarkan artikel ilmiah **Fauzi et al. (2025)**."
        )
        st.divider()

        self.render_sidebar()
        self.render_step1_setup()
        st.divider()
        self.render_step2_weighting()
        st.divider()
        df_x = self.render_step3_decision_matrix()
        st.divider()
        self.render_step4_results_and_audit(df_x)


if __name__ == "__main__":
    app = ARIEStreamlitApp()
    app.run()

