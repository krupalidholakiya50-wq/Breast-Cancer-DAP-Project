# Breast Cancer Diagnostic Metric Correlation & Tumor Size Regression Study

**Course:** TYBCA Semester 6 — Data Analytics Using Python (DAP)  
**Domain:** Healthcare & Clinical Data Analytics  
**Academic Level:** Undergraduate (TYBCA)  
**Tools & Libraries:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook  

---

## 📋 Standard 16-Chapter Academic Structure & Analysis Flow

The project follows the standard DAP pipeline:
$$\text{Data Loading} \longrightarrow \text{Data Understanding} \longrightarrow \text{Cleaning / Preprocessing} \longrightarrow \text{Univariate} \longrightarrow \text{Bivariate} \longrightarrow \text{Multivariate} \longrightarrow \text{Correlation} \longrightarrow \text{Regression} \longrightarrow \text{Classification} \longrightarrow \text{Evaluation} \longrightarrow \text{Findings}$$

| Chapter | Title | Analysis & Contents |
| :--- | :--- | :--- |
| **01** | **Project Definition** | Title, Clinical Introduction, Problem Statement, Objectives, Scope, Research Questions |
| **02** | **Dataset Details** | 5,000 records, 12 features, Feature Dictionary, Numerical (5) vs Categorical (7) breakdown |
| **03** | **Python Tools & Libraries Used** | Python 3.12, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn with clean academic theme |
| **04** | **Understanding the Data** | Data loading, `head()`, `tail()`, `shape`, `columns`, `info()`, `describe()` |
| **05** | **Understanding Spread of Data** | Numerical spread table (Mean, Median, Min, Max, Std Dev, Q1, Q3, IQR) |
| **06** | **Automating EDA Using Python** | Reusable beginner-friendly function generating automated 12-column summaries |
| **07** | **Handling Missing Data and Outliers** | Quality checks (0 missing, 0 duplicates, 0 anomalous outliers via IQR method) |
| **08** | **Univariate Analysis** | **Charts 1–4:** Histograms (Age, Tumor Size, BMI) + Box Plot (Tumor Size) |
| **09** | **Bivariate Analysis** | **Charts 5–9:** Age vs Tumor Size Scatter Plot, 3 Box Plots, 1 Grouped Bar Chart |
| **10** | **Multivariate Analysis** | **Chart 10:** Single $5 \times 5$ Pair Plot Grid (`hue='Tumor_Type'`) |
| **11** | **Correlation Analysis** | **Chart 11:** Single Pearson Correlation Heatmap ($5 \times 5$) |
| **12** | **Regression Analysis** | Ordinary Least Squares Linear Regression predicting `Tumor_Size` (80/20 train/test split) |
| **13** | **Classification Analysis** | Logistic Regression predicting patient `Survival_Status` (`Alive` / `Deceased`) |
| **14** | **Model Evaluation & Fit Diagnosis** | **Charts 12–13:** Actual vs Predicted Plot, Confusion Matrix, real metrics ($R^2$, MAE, RMSE, Accuracy), Overfitting/Underfitting assessment |
| **15** | **Conclusion / Findings** | Synthesis of clinical takeaways, model interpretations, viva preparation Q&A |
| **16** | **References** | Official Python, Scikit-learn, Pandas, Seaborn, and WHO/ACS oncology references |

---

## 📊 Complete & Exact 13-Chart Visual Catalog

Every chart contains clear titles, labeled axes, centered layout, and a structured 3-point viva explanation (*What it shows, Why it is used, Main observation*):

### A. Univariate Analysis (4 Charts)
1. **Histogram – Age Distribution** (`Age` across 20 bins with KDE curve)
2. **Histogram – Tumor Size Distribution** (`Tumor_Size` in cm across 20 bins with KDE curve)
3. **Histogram – BMI Distribution** (`BMI` in $\text{kg/m}^2$ across 20 bins with KDE curve)
4. **Box Plot – Distribution of Tumor Size** (Median ~5.14 cm, IQR 2.83–7.55 cm, 0 outliers)

### B. Bivariate Analysis (5 Charts)
5. **Scatter Plot – Age vs Tumor Size** (Scatter distribution with linear trend line, N=500 sample)
6. **Box Plot – Tumor Type vs Tumor Size** (Benign vs Malignant tumor size quartile comparison)
7. **Box Plot – Lymph Node Status vs Tumor Size** (Positive vs Negative nodal involvement)
8. **Box Plot – Hormone Receptor Status vs Tumor Size** (Positive, Negative, Unknown receptor status)
9. **Grouped Bar Chart – Tumor Type vs Survival Status** (Survival counts grouped by tumor type)

### C. Multivariate Analysis (1 Chart)
10. **Pair Plot – Multivariate Analysis – Pair Plot** (Single $5 \times 5$ scatter grid with KDE diagonals, `hue='Tumor_Type'`)

### D. Correlation Analysis (1 Chart)
11. **Correlation Heatmap** (Single $5 \times 5$ Pearson correlation matrix with annotated coefficients)

### E. Regression & Classification Model Evaluation (2 Charts)
12. **Actual vs Predicted Tumor Size** (Scatter plot with ideal $y = x$ reference line)
13. **Confusion Matrix – Survival Status Classification** ($2 \times 2$ annotated heatmap: TP, TN, FP, FN)

---

## 📈 Real Calculated Model Performance Metrics

### A. Linear Regression (`Tumor_Size` Target)
- **Model:** Ordinary Least Squares Linear Regression (`sklearn.linear_model.LinearRegression`)
- **Train/Test Partition:** 80% Training ($N=4,000$), 20% Testing ($N=1,000$) (`random_state=42`)
- **$R^2$ Score:** `0.0024` (Train) / `-0.0004` (Test)
- **Mean Absolute Error (MAE):** `2.3537 cm` (Train) / `2.3825 cm` (Test)
- **Root Mean Squared Error (RMSE):** `2.7241 cm` (Train) / `2.7529 cm` (Test)

### B. Logistic Regression (`Survival_Status` Target)
- **Model:** Logistic Regression (`max_iter=1000, random_state=42`)
- **Classification Accuracy:** `52.20%` (522 / 1,000 test cases correctly classified)
- **Confusion Matrix:** 375 True Alive, 147 True Deceased, 146 False Deceased, 332 False Alive

### C. Overfitting / Underfitting Assessment
- **Regression MAE Gap:** $|2.3537 - 2.3825| = 0.0288\text{ cm}$
- **Classification Accuracy Gap:** $|52.00\% - 52.20\%| = 0.20\%$
- **Diagnosis:** Both models generalize consistently between training and test sets with no evidence of overfitting.

---

## 📁 Project Directory Structure

```
Breast_Cancer_DAP_Project
│
├── data/
│   └── breast_cancer_5000.csv              # Clinical dataset (5,000 rows, 12 columns)
│
├── notebook/
│   └── Breast_Cancer_Diagnostic_Metric_Correlation_Tumor_Size_Regression.ipynb  # Executed Master Notebook (13 Charts)
│
├── build_master_notebook.py               # Autonomous notebook generator & executor
├── verify_project.py                      # Audit script verifying 13 charts & 16 chapters
├── README.md                              # Academic documentation & viva preparation guide
└── requirements.txt                       # Python dependencies
```

---

## 🚀 Execution & Verification Commands

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Master Notebook (Builds & runs all 13 charts):**
   ```bash
   python build_master_notebook.py
   ```

3. **Run Audit Verification (Validates 0 errors & exact 13 charts):**
   ```bash
   python verify_project.py
   ```

4. **Launch Interactive Jupyter Notebook:**
   ```bash
   jupyter notebook notebook/Breast_Cancer_Diagnostic_Metric_Correlation_Tumor_Size_Regression.ipynb
   ```
