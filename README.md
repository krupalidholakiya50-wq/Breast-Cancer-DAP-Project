# Breast Cancer Diagnostic Metric Correlation & Tumor Size Regression Study

**Course:** TYBCA Data Analytics Project  
**Domain:** Healthcare & Clinical Data Analytics  
**Academic Level:** Undergraduate (TYBCA)  
**Tools & Libraries:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook  

---

## 1. Project Overview

The project **Breast Cancer Diagnostic Metric Correlation & Tumor Size Regression Study** performs exploratory data analytics, regression modeling, and classification on real-world clinical breast cancer patient records.

The primary objectives are:
1. Understanding demographic and diagnostic metrics of breast cancer patients (Age, BMI, Blood Pressure, Follow-Up Duration, Tumor Type, Lymph Node Status, Hormone Receptor Status, Genetic Mutation, Treatment, and Survival Status).
2. Investigating the spread and distribution of **Tumor Size** and clinical diagnostic features using univariate and bivariate statistical visualizations.
3. Conducting varied bivariate analysis (Tumor Type vs Tumor Size Box Plot, Lymph Node Status vs Tumor Size Box Plot, Hormone Receptor Status vs Tumor Size Box Plot, Tumor Type vs Survival Status Grouped Bar Chart, and Age vs Tumor Size Scatter Plot).
4. Performing multivariate pair plot analysis and computing Pearson correlation among numerical diagnostic metrics.
5. Building and evaluating a **Linear Regression** model to predict tumor size based on patient diagnostic attributes.
6. Building and evaluating a **Logistic Regression** classification model to predict patient **Survival Status** with confusion matrix analysis.

---

## 2. Dataset Information

- **Dataset Source:** Kaggle Breast Cancer Analysis (`data/breast_cancer_5000.csv`)
- **Total Records:** 5,000 patient records
- **Total Columns:** 12 features
- **Data Quality:** 0 missing values, 0 duplicate records

### Feature Description:
1. `Patient_ID` – Unique patient identifier
2. `Age` – Patient age (in years)
3. `Tumor_Size` – Measured tumor dimension (in cm) [Target Variable for Regression]
4. `Tumor_Type` – Tumor classification (`Malignant`, `Benign`)
5. `Lymph_Node_Status` – Nodal involvement (`Positive`, `Negative`)
6. `Hormone_Receptor_Status` – Receptor expression (`Positive`, `Negative`, `Unknown`)
7. `Genetic_Mutation` – Genetic testing result (`BRCA1`, `BRCA2`, `Other`)
8. `Treatment` – Primary therapy administered (`Surgery`, `Radiation`, `Chemotherapy`, `Hormone Therapy`)
9. `Survival_Status` – Clinical outcome (`Alive`, `Deceased`) [Target Variable for Classification]
10. `Follow_Up_Duration` – Post-diagnosis monitoring duration (in months)
11. `Blood_Pressure` – Baseline blood pressure measurement (in mmHg)
12. `BMI` – Body Mass Index (in kg/m²)

---

## 3. Project Structure

```
Breast_Cancer_DAP_Project
│
├── data
│   └── breast_cancer_5000.csv
│
├── notebook
│   └── Breast_Cancer_Diagnostic_Metric_Correlation_Tumor_Size_Regression.ipynb
│
├── Breast_Cancer_DAP_Project_Report.docx
├── Breast_Cancer_DAP_Project_Report.pdf
├── README.md
└── requirements.txt
```

---

## 4. Installation & Setup

1. **Clone or Open the Project Folder:**
   ```bash
   cd D:\Breast_Cancer_DAP_Project
   ```

2. **Install Required Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Open `notebook/Breast_Cancer_Diagnostic_Metric_Correlation_Tumor_Size_Regression.ipynb` and run all cells sequentially.

---

## 5. Summary of Analysis & Visualizations

| Section | Analysis Description | Chart Type / Details |
| :--- | :--- | :--- |
| **5. Spread of Data** | Numerical spread analysis | 3 Separate Histograms (Tumor Size, Age, BMI) |
| **6. Univariate Analysis** | Single-variable distributions | 16 Separate Figures (Histograms, Box Plots, Bar Charts) |
| **7. Tumor Size Distribution** | Deep-dive into target variable | Separate Histogram (KDE) + Box Plot |
| **8. Diagnostic Metric Distribution** | Clinical measurements spread | 4 Separate Figures (Age, BMI, BP, Follow-up) |
| **9. Bivariate Analysis** | Varied diagnostic comparisons | 5 Distinct Charts (3 Box Plots + 1 Grouped Bar Chart + 1 Scatter Plot) |
| **10. Outcome Analysis** | Overall patient survival rate | Single Bar Chart (`Alive` vs `Deceased`) |
| **11. Multivariate Analysis** | Cross-feature interactions | Clean Corner Pair Plot of 5 numerical metrics (N=500 Sample) |
| **12. Correlation Analysis** | Linear association strength | Pearson Correlation Heatmap ($5 \times 5$) |
| **13. Tumor Size Regression** | Predictive modeling | Scikit-learn Linear Regression (80/20 split) |
| **14. Classification** | Survival outcome prediction | Scikit-learn Logistic Regression + Confusion Matrix |

---

## 6. Model Results

### A. Tumor Size Regression (Linear Regression)
- **Model:** Ordinary Least Squares Linear Regression
- **Train/Test Split:** 80% Training (4,000 samples), 20% Testing (1,000 samples)
- **Target Variable ($y$):** `Tumor_Size`
- **$R^2$ Score:** `-0.0006`
- **Mean Absolute Error (MAE):** `2.3822 cm`
- **Root Mean Squared Error (RMSE):** `2.7532 cm`

### B. Survival Status Classification (Logistic Regression)
- **Model:** Logistic Regression (`max_iter=1000, random_state=42`)
- **Train/Test Split:** 80% Training (4,000 samples), 20% Testing (1,000 samples)
- **Target Variable ($y$):** `Survival_Status`
- **Accuracy:** `52.20%` (522 / 1,000 test cases correctly classified)
- **Confusion Matrix:** 375 True Alive, 147 True Deceased

---

## 7. Project Deliverables

- **Jupyter Notebook:** Fully executed and documented in `notebook/Breast_Cancer_Diagnostic_Metric_Correlation_Tumor_Size_Regression.ipynb`
- **DOCX Academic Report:** 29-section TYBCA academic report at `Breast_Cancer_DAP_Project_Report.docx`
- **PDF Academic Report:** High-resolution formatted PDF at `Breast_Cancer_DAP_Project_Report.pdf`
