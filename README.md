# Integrated AHP-ARIE Decision Support System (Streamlit Web App)

An interactive web-based Multi-Criteria Decision-Making (MCDM) Decision Support System (DSS) built with **Python** and **Streamlit**. This application integrates the **Analytic Hierarchy Process (AHP)** for objective criteria weighting and the **Adaptive Ranking with Ideal Evaluation (ARIE)** method for multi-criteria alternative evaluation and ranking.

Based on the mathematical formulation from:
> **Fauzi, N. F. F. M., et al. (2025).** *Adaptive Ranking with Ideal Evaluation: A Novel Similarity-Based Multi-Criteria Decision-Making Method.* European Journal of Pure and Applied Mathematics.

---

## Table of Contents
- [Overview & Key Features](#overview--key-features)
- [Object-Oriented Software Architecture](#object-oriented-software-architecture)
- [Mathematical Formulation](#mathematical-formulation)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Application Workflow](#application-workflow)
- [Benchmark Case Study (Halal Supplier Selection)](#benchmark-case-study-halal-supplier-selection)
- [License](#license)

---

## Overview & Key Features

The **ARIE** method overcomes structural limitations of classical MCDM techniques (such as standard TOPSIS and ARAS) by preventing artificial score inflation, supporting non-linear risk preferences, and natively accommodating mixed criteria types.

### Key Features Implemented in `app.py`:
1. **Dynamic Problem Dimensions:**
   - Supports flexible problem sizes from **2 to 20 criteria/factors** ($n$) and **2 to 50 alternatives** ($m$).
   - Automatic or custom naming/coding for criteria ($C_1, C_2, \dots$) and alternatives ($A_1, A_2, \dots$ or $S_1, S_2, \dots$).
2. **Unified Multi-Type Normalization:**
   - Seamlessly evaluates **Benefit** (larger is better), **Cost** (smaller is better), and **Target** (optimal at a specific nominal value $x_T$) criteria within a single decision matrix.
3. **Flexible Criteria Weighting Methods:**
   - **Direct Manual Input:** Enter raw weights directly with automatic normalization ($\sum w_j = 1.0$).
   - **Analytic Hierarchy Process (AHP):** Interactive upper-triangular pairwise comparison inputs using Saaty's scale ($1\text{--}9$ and reciprocals $1/2\text{--}1/9$), automatic diagonal ($1.0$) and lower-triangular reciprocal assignment ($a_{ji} = 1/a_{ij}$), priority vector computation, and real-time consistency verification ($\lambda_{\max}, CI, RI, CR \le 0.10$).
4. **Dynamic Behavioral Parameter Tuning:**
   - **Sensitivity Parameter ($\gamma \in [0.1, 3.0]$, default $1.0$):** Controls non-linear penalty severity ($\gamma > 1$ for *risk-averse*, $\gamma < 1$ for *risk-seeking*, $\gamma = 1$ for *risk-neutral*).
   - **Balancing Coefficient ($\kappa \in [0.0, 1.0]$, default $0.5$):** Adjusts strategic focus between proximity to the ideal solution ($\kappa > 0.5$, *optimistic/ideal-focused*) and avoidance of the anti-ideal solution ($\kappa < 0.5$, *conservative/anti-ideal avoidance*).
5. **Interactive Visualization, Full Mathematical Audit & Export:**
   - Live ranking table with color-gradient highlighting on Relative Closeness ($RC_i$).
   - Interactive **Altair** bar chart ordered by final rank with tooltips.
   - Complete step-by-step mathematical audit tab displaying the Normalized Matrix ($R$), Weighted Matrix ($V$), Ideal/Anti-Ideal benchmarks ($v^{\max}, v^{\min}$), and similarity vectors ($Sim^{\text{best}}, Sim^{\text{worst}}$).
   - One-click **CSV export** of the final ranking table.

---

## Object-Oriented Software Architecture

The entire application is encapsulated cleanly in a single modular file (`app.py`) following Object-Oriented Programming (OOP) principles:

| Class / Component | Responsibility |
| :--- | :--- |
| **`Criterion`** (`@dataclass`) | Domain entity representing a decision criterion (`code`, `name`, `ctype` in `{'benefit', 'cost', 'target'}`, `weight`, and `target_val` $x_T$) with built-in validation. |
| **`AHPResult`** (`@dataclass`) | Data container storing the AHP pairwise matrix, column-normalized matrix, priority weight vector, $\lambda_{\max}$, $CI$, $RI$, $CR$, and consistency status. |
| **`ARIEResult`** (`@dataclass`) | Data container storing all intermediate and final ARIE outputs ($X, R, V, v^{\max}, v^{\min}, Sim^{\text{best}}, Sim^{\text{worst}}, RC_i$, and sorted ranking DataFrame). |
| **`AHPCalculator`** | Pure mathematical engine implementing Saaty's pairwise comparison solver and Random Index ($RI$) lookup table for $n = 1 \dots 20$. |
| **`ARIECalculator`** | Computational engine executing the complete 5-step ARIE mathematical pipeline using vectorized NumPy and Pandas operations. |
| **`BenchmarkDataset`** | Static provider supplying the benchmark *Halal Supplier Selection* case study from Fauzi et al. (2025). |
| **`ARIEStreamlitApp`** | Main UI controller managing Streamlit `session_state`, dynamic dimension synchronization, sidebar controls, interactive `st.data_editor` tables, charts, and audit views. |

---

## Mathematical Formulation

### 1. Analytic Hierarchy Process (AHP) Weighting
Given a pairwise comparison matrix $A = [a_{ij}]_{n \times n}$ where $a_{ii} = 1$ and $a_{ji} = 1 / a_{ij}$:
- **Column Normalization:**
  $$\bar{a}_{ij} = \frac{a_{ij}}{\sum_{k=1}^n a_{kj}}$$
- **Priority Weight Vector:**
  $$w_i = \frac{1}{n} \sum_{j=1}^n \bar{a}_{ij}$$
- **Consistency Evaluation:**
  $$\lambda_{\max} = \frac{1}{n} \sum_{i=1}^n \frac{(A w)_i}{w_i}, \quad CI = \frac{\lambda_{\max} - n}{n - 1}, \quad CR = \frac{CI}{RI_n}$$
  *(The comparison matrix is considered consistent if $CR \le 0.10$).*

### 2. ARIE Method Pipeline
Given a raw decision matrix $X = [x_{ij}]_{m \times n}$ with $m$ alternatives and $n$ criteria:

1. **Multi-Type Normalization ($R = [r_{ij}]$):**
   - **Benefit Criteria:**
     $$r_{ij} = \frac{x_{ij}}{x_j^{\max}}$$
   - **Cost Criteria:**
     $$r_{ij} = \frac{x_j^{\min}}{x_{ij}}$$
   - **Target Criteria (with nominal target $x_j^T$):**
     $$r_{ij} = 1 - \frac{|x_{ij} - x_j^T|}{\max\left(|x_j^{\max} - x_j^T|, |x_j^{\min} - x_j^T|\right)}$$

2. **Weighted Normalized Decision Matrix ($V = [v_{ij}]$):**
   $$v_{ij} = w_j \cdot r_{ij}, \quad \text{where } \sum_{j=1}^n w_j = 1$$

3. **Ideal & Anti-Ideal Benchmarks and Similarity Measures:**
   $$v_j^{\max} = \max_{1 \le i \le m} v_{ij}, \quad v_j^{\min} = \min_{1 \le i \le m} v_{ij}$$
   $$Sim_i^{\text{best}} = \sum_{j=1}^n \left(\frac{v_{ij}}{v_j^{\max}}\right)^\gamma, \quad Sim_i^{\text{worst}} = \sum_{j=1}^n \left(\frac{v_j^{\min}}{v_{ij}}\right)^\gamma$$

4. **Relative Closeness Score ($RC_i$) & Ranking:**
   $$RC_i = \frac{\kappa \cdot Sim_i^{\text{best}}}{\kappa \cdot Sim_i^{\text{best}} + (1 - \kappa) \cdot Sim_i^{\text{worst}}}$$
   Alternatives are ranked in descending order of $RC_i \in [0, 1]$.

---

## Project Structure

```text
MR4053_ARIE/
│
├── app.py             # Complete OOP Streamlit application (Domain Models, Engines, UI)
├── README.md          # Project documentation
└── .gitignore         # Git ignore rules
```

---

## Installation & Usage

### Prerequisites
- Python 3.8 or higher.

### 1. Install Dependencies
Install the required Python packages via `pip`:
```bash
pip install streamlit numpy pandas altair
```

### 2. Run the Application
Launch the Streamlit web server from the project root directory:
```bash
streamlit run app.py
```
The application will automatically open in your default web browser (typically at `http://localhost:8501`).

---

## Application Workflow

1. **Sidebar – Problem Dimensions & Dynamic Parameters:**
   - Click **"📥 Load Sample Case Study (Halal Supplier)"** at any time to load/reset the benchmark dataset from Fauzi et al. (2025).
   - Adjust the number of criteria ($2\text{--}20$) and alternatives ($2\text{--}50$), or reset codes to `C1..Cn` and `A1..Am`.
   - Tune the **Sensitivity Parameter ($\gamma$)** slider ($0.1\text{--}3.0$) and **Balancing Coefficient ($\kappa$)** slider ($0.0\text{--}1.0$) to observe real-time sensitivity effects on the final ranking.
2. **Section A – Criteria & Alternative Definition:**
   - Edit criterion codes, names, optimization types (`benefit`, `cost`, `target`), and target values ($x_T$) directly inside the interactive table.
   - Edit alternative codes and descriptive names.
3. **Section B – Criteria Weighting Method:**
   - Choose **"1. Metode Langsung (Manual Input)"** to input weights directly (auto-normalized if the sum is not $1.0$), **or**
   - Choose **"2. Metode Analytic Hierarchy Process (AHP)"** to perform pairwise comparisons on the upper-triangular elements, inspect $\lambda_{\max}, CI, RI, CR$, and click **"✅ Terapkan Bobot Hasil AHP ke Model ARIE"**.
4. **Section C – Decision Matrix ($X$) Input:**
   - Enter or modify raw performance values $x_{ij}$ for each alternative across all criteria in the editable matrix grid.
5. **Section D – Ranking Results, Visualization & Mathematical Audit:**
   - View the top-ranked alternative summary metrics and full ranking table.
   - Switch to the **Bar Chart** tab for an interactive visual comparison of $RC_i$ scores.
   - Switch to the **Mathematical Audit** tab to inspect $R$, $V$, $v^{\max}$, $v^{\min}$, $Sim^{\text{best}}$, and $Sim^{\text{worst}}$.
   - Click **"📥 Unduh Hasil Perankingan ke CSV"** to download the results.

---

## Benchmark Case Study (Halal Supplier Selection)

The built-in benchmark dataset replicates the **Halal Supplier Selection** problem from Fauzi et al. (2025):

### 1. Criteria Configuration
| Code | Criterion Name | Type | Target ($x_T$) | Weight ($w_j$) |
|:---:|:---|:---:|:---:|:---:|
| **$C_1$** | Halal Certification | `benefit` | — | `0.30` |
| **$C_2$** | Procurement Cost | `cost` | — | `0.20` |
| **$C_3$** | Delivery Time (Days) | `target` | `15.0` | `0.15` |
| **$C_4$** | Ingredient Quality | `benefit` | — | `0.25` |
| **$C_5$** | Reputation & Traceability | `benefit` | — | `0.10` |

### 2. Raw Decision Matrix ($X$)
| Alternative | $C_1$ | $C_2$ | $C_3$ | $C_4$ | $C_5$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$S_1$** (Supplier 1) | 90.0 | 50.0 | 14.0 | 85.0 | 80.0 |
| **$S_2$** (Supplier 2) | 80.0 | 42.0 | 18.0 | 75.0 | 70.0 |
| **$S_3$** (Supplier 3) | 70.0 | 60.0 | 20.0 | 80.0 | 60.0 |
| **$S_4$** (Supplier 4) | 85.0 | 54.0 | 12.0 | 90.0 | 75.0 |
| **$S_5$** (Supplier 5) | 95.0 | 45.0 | 16.0 | 90.0 | 90.0 |

### 3. Benchmark Results ($\gamma = 1.0, \kappa = 0.5$)
| Rank | Alternative | $Sim_i^{\text{best}}$ | $Sim_i^{\text{worst}}$ | Relative Closeness ($RC_i$) |
|:---:|:---:|:---:|:---:|:---:|
| **#1** | **$S_5$ (Supplier 5)** | 4.5167 | 2.8778 | **0.6108** |
| **#2** | **$S_1$ (Supplier 1)** | 4.3800 | 3.2361 | **0.5751** |
| **#3** | **$S_4$ (Supplier 4)** | 3.8333 | 3.3349 | **0.5348** |
| **#4** | **$S_2$ (Supplier 2)** | 3.7857 | 3.3778 | **0.5285** |
| **#5** | **$S_3$ (Supplier 3)** | 3.1000 | 3.6083 | **0.4621** |

---

## License

This project is open-source and available under the MIT License for academic research, teaching, and industrial decision-support applications.
