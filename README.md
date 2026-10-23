# Integrated AHP-ARIE Decision Support System

A desktop Decision Support System (DSS) application featuring a graphical user interface (GUI) built in Python. This system integrates the **Analytic Hierarchy Process (AHP)** for objective criteria weighting and the **Adaptive Ranking with Ideal Evaluation (ARIE)** method for multi-criteria alternative evaluation.

Based on the mathematical formulation from:
> **Fauzi, N. F. F. M., et al. (2025).** *Adaptive Ranking with Ideal Evaluation: A Novel Similarity-Based Multi-Criteria Decision-Making Method.* European Journal of Pure and Applied Mathematics.

---

## Table of Contents
- [Overview & Key Features](#overview--key-features)
- [Software Architecture](#software-architecture)
- [Mathematical Formulation](#mathematical-formulation)
- [Project Directory Structure](#project-directory-structure)
- [Installation & Prerequisites](#installation--prerequisites)
- [User Guide](#user-guide)
- [Benchmark Verification Case](#benchmark-verification-case)
- [License](#license)

---

## Overview & Key Features

The ARIE method addresses the structural rigidity of traditional MCDM approaches (such as standard TOPSIS and ARAS) by preventing artificial score inflation, accommodating non-linear risk preferences, and supporting mixed criteria types natively.

### Key Capabilities:
1. **Unified Multi-Type Normalization:** Seamlessly handles *Benefit*, *Cost*, and explicit *Target-Type* criteria ($x^T$).
2. **Behavioral Parameter Tuning:**
   - **Sensitivity Exponent ($\gamma$):** Controls non-linear penalty severity ($\gamma > 1$ models risk-averse behavior; $\gamma < 1$ models risk-seeking tolerance).
   - **Balancing Coefficient ($\kappa \in [0, 1]$):** Adjusts strategic preference between proximity to the ideal solution versus distance from the anti-ideal/nadir benchmark.
3. **Integrated AHP Weighting:** Solves Saaty pairwise comparison matrices ($1$ to $9$ scale) with real-time Consistency Index ($CI$) and Consistency Ratio ($CR \le 0.10$) calculation.
4. **Transparent Audit Trail:** Displays intermediate matrices ($R$ and $V$), benchmark vectors ($v^{\max}$ and $v^{\min}$), and similarity sums ($Sim^{\text{best}}$ and $Sim^{\text{worst}}$).
5. **Object-Oriented Design (OOP):** Modular separation between data domain models, computational engines, and the GUI layer.

---

## Software Architecture

The application is structured into distinct object-oriented components:
- **`Criterion`**: Entity class encapsulating criteria attributes (code, label, weight, optimization direction, and nominal target level).
- **`AHPCalculator`**: Pure mathematical engine for normalizing pairwise matrices, deriving priority weight vectors, and calculating Saaty consistency metrics.
- **`ARIECalculator`**: Standalone computational engine implementing the 5-step ARIE workflow with NumPy and Pandas.
- **`ARIEMainApp` & Custom Frames**: Presentation and controller layer implemented with Python's standard `tkinter` and `ttk`.

---

## Mathematical Formulation

### 1. Analytic Hierarchy Process (AHP) Weighting
Given a pairwise comparison matrix $A = [a_{ij}]_{n \times n}$:
- **Column Normalization:**
  $$\bar{a}_{ij} = \frac{a_{ij}}{\sum_{k=1}^n a_{kj}}$$
- **Priority Weight Vector:**
  $$w_i = \frac{1}{n} \sum_{j=1}^n \bar{a}_{ij}$$
- **Consistency Verification:**
  $$\lambda_{\max} = \frac{1}{n} \sum_{i=1}^n \frac{(A w)_i}{w_i}, \quad CI = \frac{\lambda_{\max} - n}{n - 1}, \quad CR = \frac{CI}{RI}$$
  *(A consistency ratio $CR \le 0.10$ confirms judgment validity).*

### 2. ARIE Decision Framework
- **Step 2: Unified Normalization ($r_{ij}$):**
  - *Benefit criteria:*
    $$r_{ij} = \frac{x_{ij}}{x_j^{\max}}$$
  - *Cost criteria:*
    $$r_{ij} = \frac{x_j^{\min}}{x_{ij}}$$
  - *Target criteria:*
    $$r_{ij} = 1 - \frac{|x_{ij} - x_j^T|}{\max(|x_j^{\max} - x_j^T|, |x_j^{\min} - x_j^T|)}$$
- **Step 3: Weighted Matrix ($v_{ij}$):**
  $$v_{ij} = w_j \cdot r_{ij}, \quad \text{where } \sum_{j=1}^n w_j = 1$$
- **Step 4: Similarity to Ideal & Anti-Ideal Solutions:**
  $$Sim_i^{\text{best}} = \sum_{j=1}^n \left(\frac{v_{ij}}{v_j^{\max}}\right)^\gamma, \quad Sim_i^{\text{worst}} = \sum_{j=1}^n \left(\frac{v_j^{\min}}{v_{ij}}\right)^\gamma$$
- **Step 5: Relative Closeness & Ranking ($RC_i$):**
  $$RC_i = \frac{\kappa \cdot Sim_i^{\text{best}}}{\kappa \cdot Sim_i^{\text{best}} + (1 - \kappa) \cdot Sim_i^{\text{worst}}}$$
  *Alternatives are ranked in descending order of $RC_i \in [0, 1]$.*

---

## Project Directory Structure

```text
arie-ahp-dss/
│
├── data/
│   └── halal_case_study.json     # Benchmark reference dataset
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── criterion.py          # Criterion domain entity
│   │   ├── ahp_engine.py         # AHP pairwise comparison engine
│   │   └── arie_engine.py        # ARIE 5-step computational pipeline
│   │
│   └── views/
│       ├── __init__.py
│       ├── setup_tab.py          # Criteria specification & parameters
│       ├── ahp_tab.py            # Interactive AHP pairwise grid
│       ├── matrix_tab.py         # Decision matrix (X) input grid
│       └── result_tab.py         # Ranking treeview & calculation audit
│
├── main.py                       # Application bootstrap and controller
├── requirements.txt              # Environment dependencies
└── README.md                     # Documentation
```

---

## Installation & Prerequisites

### Prerequisites:
- Python 3.8 or higher.
- `tkinter` (included in standard Python distributions).

### Setup Instructions:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/arie-ahp-dss.git
   cd arie-ahp-dss
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Linux / macOS:
   source venv/bin/activate
   # Windows:
   venv\Scripts\activate
   ```
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   *Required packages:*
   ```text
   numpy>=1.21.0
   pandas>=1.3.0
   ```

---

## User Guide

Launch the desktop interface from your terminal:
```bash
python main.py
```

### Workflow Steps:
1. **Tab 1: Criteria Configuration:**
   - Define criterion dimensions ($n$), alternative counts ($m$), and model parameters ($\gamma$ and $\kappa$).
   - Set criteria properties: Code, Label, Optimization Direction (*benefit*, *cost*, *target*), and nominal target value ($x^T$).
2. **Tab 2: AHP Weighting (Optional):**
   - Click **"Buat Grid Perbandingan AHP"** to generate the pairwise comparison matrix.
   - Enter comparison ratings (supports decimals like `3.0` or fractions like `1/3`).
   - Click **"Hitung Bobot AHP & Terapkan ke Kriteria"**. If $CR \le 0.10$, derived weights are automatically loaded into Tab 1.
3. **Tab 3: Decision Matrix:**
   - Click **"Muat Grid Matriks dari Kriteria"** to construct the evaluation table.
   - Enter performance values $x_{ij}$ for all alternatives.
   - Click **"Hitung Peringkat ARIE"**.
4. **Tab 4: Results & Audit:**
   - Inspect the final ranking order and relative closeness scores ($RC_i$).
   - Review intermediate normalized ($R$) and weighted ($V$) matrices in the audit log.

---

## Benchmark Verification Case

The application includes a built-in benchmark dataset (**"Load Data Kasus Halal (Paper)"**) from Fauzi et al. (2025):

- **Criteria Specification:**
  - $C_1$: Halal Certification ($w = 0.30$, *Benefit*)
  - $C_2$: Procurement Cost ($w = 0.20$, *Cost*)
  - $C_3$: Delivery Time ($w = 0.15$, *Target*, $x^T = 15\text{ days}$)
  - $C_4$: Ingredient Quality ($w = 0.25$, *Benefit*)
  - $C_5$: Reputation & Traceability ($w = 0.10$, *Benefit*)

- **Standard Benchmark Results ($\gamma = 1.0, \kappa = 0.5$):**

| Alternative | $Sim_i^{\text{best}}$ | $Sim_i^{\text{worst}}$ | Relative Closeness ($RC_i$) | Final Rank |
|:---:|:---:|:---:|:---:|:---:|
| **$S_5$** | 4.5167 | 2.8778 | **0.6108** | **#1** |
| **$S_1$** | 4.3800 | 3.2361 | **0.5751** | **#2** |
| **$S_4$** | 3.8333 | 3.3349 | **0.5348** | **#3** |
| **$S_2$** | 3.7857 | 3.3778 | **0.5285** | **#4** |
| **$S_3$** | 3.1000 | 3.6083 | **0.4621** | **#5** |

---

## License

This project is licensed under the [MIT License](LICENSE). You are free to modify, extend, and adapt this implementation for academic research and industrial decision support applications.
