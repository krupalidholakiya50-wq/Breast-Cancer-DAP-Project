import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('docs/diagrams', exist_ok=True)

def setup_academic_canvas(figsize=(10, 6.5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    return fig, ax

def draw_box(ax, x, y, w, h, text, font_size=9.5, bold=False, fill_color='white', align='center', title=""):
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=1.2, edgecolor='black', facecolor=fill_color, zorder=2
    )
    ax.add_patch(rect)
    
    font_weight = 'bold' if bold else 'normal'
    if align == 'center':
        tx, ty = x + w / 2, y + h / 2
        ha, va = 'center', 'center'
    elif align == 'left':
        tx, ty = x + 2, y + h / 2
        ha, va = 'left', 'center'
        
    full_text = f"{title}\n{text}" if title else text
    ax.text(tx, ty, full_text, ha=ha, va=va, fontsize=font_size,
            fontfamily='Arial', color='black', fontweight=font_weight, zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, text="", text_pos=None, font_size=8.5):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.2, shrinkA=0, shrinkB=0),
        zorder=2
    )
    if text:
        tx = (x1 + x2) / 2 if text_pos is None else text_pos[0]
        ty = (y1 + y2) / 2 if text_pos is None else text_pos[1]
        ax.text(tx, ty, text, ha='center', va='center', fontsize=font_size,
                fontfamily='Arial', color='black', bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='none'), zorder=4)

# =========================================================================
# 1. Project Workflow Diagram
# =========================================================================
def create_diagram_1():
    fig, ax = setup_academic_canvas(figsize=(10.5, 7.0))
    ax.text(50, 95, "PROJECT WORKFLOW DIAGRAM", ha='center', va='center', fontsize=13, fontweight='bold', fontfamily='Arial')
    ax.text(50, 91.5, "Breast Cancer Diagnostic Metric Correlation & Tumor Size Regression Study", ha='center', va='center', fontsize=10, fontfamily='Arial')

    steps = [
        ("Step 1: Data Collection & Loading", "Import Libraries (Pandas, NumPy, Scikit-learn)\nLoad 'breast_cancer_5000.csv' (5,000 Rows, 12 Columns)"),
        ("Step 2: Data Understanding & Cleaning", "Inspect Dimensions, Column Data Types, and Info\nVerify Missing Values (0) & Outliers via IQR Method"),
        ("Step 3: Exploratory Data Analysis (EDA)", "Univariate Analysis: Histograms & Box Plots\nBivariate & Multivariate: Scatter Plot & Pair Plot"),
        ("Step 4: Correlation & Feature Preprocessing", "Compute Pearson Correlation Matrix ($5 \\times 5$ Heatmap)\nOne-Hot Encode Categorical Predictor Features"),
        ("Step 5: Supervised Machine Learning", "Linear Regression: Predict Primary Tumor Size ($y$)\nLogistic Regression: Classify Survival Status ($y$)"),
        ("Step 6: Model Evaluation & Diagnosis", "Evaluate Real Metrics ($R^2$, MAE, RMSE, Accuracy, CM)\nAssess Overfitting/Underfitting & Final Conclusion")
    ]
    
    y_starts = [78, 64, 50, 36, 22, 8]
    for i, (title, desc) in enumerate(steps):
        y = y_starts[i]
        draw_box(ax, 15, y, 70, 9.5, desc, font_size=9, bold=False, title=f"[{title}]")
        if i < len(steps) - 1:
            draw_arrow(ax, 50, y, 50, y_starts[i+1] + 9.5)
            
    plt.tight_layout()
    plt.savefig('docs/diagrams/01_project_workflow_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

# =========================================================================
# 2. Data Flow Diagram – Level 0 (Context Diagram)
# =========================================================================
def create_diagram_2():
    fig, ax = setup_academic_canvas(figsize=(10.5, 6.0))
    ax.text(50, 94, "DATA FLOW DIAGRAM (DFD) — LEVEL 0 (CONTEXT DIAGRAM)", ha='center', va='center', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.text(50, 90, "Overall System Boundary & External Entities", ha='center', va='center', fontsize=9.5, fontfamily='Arial')

    # External Entity 1
    draw_box(ax, 5, 40, 20, 20, "Data Analyst /\nStudent Researcher", font_size=10, bold=True)
    
    # Process 0.0
    draw_box(ax, 38, 35, 26, 30, "0.0\n\nBreast Cancer\nDiagnostic & Regression\nAnalytics System", font_size=10, bold=True)
    
    # External Entity 2
    draw_box(ax, 75, 40, 20, 20, "Academic Evaluator /\nClinical Faculty", font_size=10, bold=True)

    # Data flows
    draw_arrow(ax, 25, 53, 38, 53, "Clinical Dataset (CSV)\n& Parameters", (31.5, 58))
    draw_arrow(ax, 38, 43, 25, 43, "System Execution\nLogs & Errors", (31.5, 38))
    
    draw_arrow(ax, 64, 53, 75, 53, "Regression Metrics\n& Evaluation Reports", (69.5, 58))
    draw_arrow(ax, 64, 43, 75, 43, "Visual Charts &\nClassification Results", (69.5, 38))

    plt.tight_layout()
    plt.savefig('docs/diagrams/02_dfd_level_0.png', dpi=300, bbox_inches='tight')
    plt.close()

# =========================================================================
# 3. Data Flow Diagram – Level 1
# =========================================================================
def create_diagram_3():
    fig, ax = setup_academic_canvas(figsize=(11.0, 7.5))
    ax.text(50, 95, "DATA FLOW DIAGRAM (DFD) — LEVEL 1", ha='center', va='center', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.text(50, 91.5, "Major Subsystems & Inter-Process Data Stores", ha='center', va='center', fontsize=9.5, fontfamily='Arial')

    # Processes
    draw_box(ax, 6, 68, 22, 14, "1.0\nLoad & Inspect\nDataset", font_size=9, bold=True)
    draw_box(ax, 39, 68, 22, 14, "2.0\nPerform Exploratory\nData Analysis", font_size=9, bold=True)
    draw_box(ax, 72, 68, 22, 14, "3.0\nFeature Encoding &\nTrain-Test Split", font_size=9, bold=True)
    draw_box(ax, 39, 28, 22, 14, "4.0\nTrain Supervised\nML Models", font_size=9, bold=True)
    draw_box(ax, 72, 28, 22, 14, "5.0\nModel Evaluation &\nMetric Calculation", font_size=9, bold=True)

    # Data stores
    draw_box(ax, 6, 45, 22, 9, "D1: breast_cancer_5000.csv", font_size=8.5, fill_color='#f8f9fa')
    draw_box(ax, 39, 48, 22, 9, "D2: Visual Plots & Matrices", font_size=8.5, fill_color='#f8f9fa')
    draw_box(ax, 72, 48, 22, 9, "D3: Train / Test Matrices", font_size=8.5, fill_color='#f8f9fa')

    # Arrows
    draw_arrow(ax, 17, 45, 17, 68, "Raw Records", (17, 56))
    draw_arrow(ax, 28, 75, 39, 75, "Clean Data", (33.5, 78))
    draw_arrow(ax, 50, 68, 50, 57, "Visual Summaries", (50, 63))
    draw_arrow(ax, 61, 75, 72, 75, "Feature Table", (66.5, 78))
    draw_arrow(ax, 83, 68, 83, 57, "X, y Arrays", (83, 63))
    draw_arrow(ax, 83, 48, 50, 42, "Train Data", (66, 46))
    draw_arrow(ax, 61, 35, 72, 35, "Trained Models", (66.5, 38))
    draw_arrow(ax, 83, 28, 83, 15, "Final Metrics &\nReports", (83, 20))
    
    # User box
    draw_box(ax, 39, 8, 22, 11, "External User /\nFaculty Evaluator", font_size=9, bold=True)
    draw_arrow(ax, 72, 13, 61, 13, "Outputs", (66.5, 16))

    plt.tight_layout()
    plt.savefig('docs/diagrams/03_dfd_level_1.png', dpi=300, bbox_inches='tight')
    plt.close()

# =========================================================================
# 4. Data Flow Diagram – Level 2 (Supervised ML Modeling Subsystem)
# =========================================================================
def create_diagram_4():
    fig, ax = setup_academic_canvas(figsize=(11.0, 7.5))
    ax.text(50, 95, "DATA FLOW DIAGRAM (DFD) — LEVEL 2", ha='center', va='center', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.text(50, 91.5, "Detailed Decomposition of Supervised Machine Learning Subsystem (Process 4.0)", ha='center', va='center', fontsize=9.5, fontfamily='Arial')

    draw_box(ax, 5, 68, 24, 13, "4.1\nFeature Selection &\nTarget Assignment", font_size=8.5, bold=True)
    draw_box(ax, 38, 68, 24, 13, "4.2\nOne-Hot Categorical\nEncoding", font_size=8.5, bold=True)
    draw_box(ax, 71, 68, 24, 13, "4.3\n80/20 Train-Test\nSplit Partitioning", font_size=8.5, bold=True)
    
    draw_box(ax, 5, 28, 24, 13, "4.4\nLinear Regression\nModel Training", font_size=8.5, bold=True)
    draw_box(ax, 38, 28, 24, 13, "4.5\nTest Prediction\nGeneration (y_pred)", font_size=8.5, bold=True)
    draw_box(ax, 71, 28, 24, 13, "4.6\nModel Evaluation\nMetric Computation", font_size=8.5, bold=True)

    draw_arrow(ax, 29, 74.5, 38, 74.5, "Selected Cols", (33.5, 78))
    draw_arrow(ax, 62, 74.5, 71, 74.5, "Encoded Matrix", (66.5, 78))
    draw_arrow(ax, 83, 68, 17, 41, "X_train, y_train", (50, 56))
    draw_arrow(ax, 29, 34.5, 38, 34.5, "Fitted Model", (33.5, 38))
    draw_arrow(ax, 62, 34.5, 71, 34.5, "y_pred, y_test", (66.5, 38))
    
    # Store at bottom
    draw_box(ax, 38, 8, 24, 9, "D4: Model Evaluation Store\n(R², MAE, RMSE, Acc)", font_size=8, fill_color='#f8f9fa')
    draw_arrow(ax, 83, 28, 62, 12.5, "Calculated Results", (74, 19))

    plt.tight_layout()
    plt.savefig('docs/diagrams/04_dfd_level_2.png', dpi=300, bbox_inches='tight')
    plt.close()

# =========================================================================
# 5. System Architecture Diagram
# =========================================================================
def create_diagram_5():
    fig, ax = setup_academic_canvas(figsize=(10.5, 7.5))
    ax.text(50, 96, "SYSTEM ARCHITECTURE DIAGRAM", ha='center', va='center', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.text(50, 92.5, "Layered Architecture of Breast Cancer Data Analytics System", ha='center', va='center', fontsize=9.5, fontfamily='Arial')

    layers = [
        ("1. PRESENTATION & INTERACTION LAYER", "Jupyter Notebook Interface (.ipynb) | Academic Report (.docx / .pdf) | Markdown Summaries"),
        ("2. MACHINE LEARNING & EVALUATION LAYER", "Linear Regression (Tumor Size) | Logistic Regression (Survival) | Metrics (R², MAE, RMSE, Acc, CM)"),
        ("3. EXPLORATORY DATA ANALYSIS (EDA) LAYER", "Univariate Analysis (Histograms, Box Plot) | Bivariate (Scatter, Box, Bar) | Multivariate (Pair Plot, Heatmap)"),
        ("4. DATA PROCESSING & TRANSFORMATION LAYER", "Data Cleaning | Missing Value Verification | IQR Outlier Check | One-Hot Feature Encoding (Pandas, NumPy)"),
        ("5. DATA SOURCE & STORAGE LAYER", "Clinical Dataset File: 'data/breast_cancer_5000.csv' (5,000 Patient Records, 12 Diagnostic Variables)")
    ]

    y_pos = [76, 59, 42, 25, 8]
    for i, (layer_title, layer_desc) in enumerate(layers):
        y = y_pos[i]
        draw_box(ax, 10, y, 80, 11, layer_desc, font_size=8.5, bold=False, title=f"[{layer_title}]")
        if i < len(layers) - 1:
            draw_arrow(ax, 50, y, 50, y_pos[i+1] + 11)

    plt.tight_layout()
    plt.savefig('docs/diagrams/05_system_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

# =========================================================================
# 6. Data Processing Flow Diagram
# =========================================================================
def create_diagram_6():
    fig, ax = setup_academic_canvas(figsize=(10.5, 7.5))
    ax.text(50, 96, "DATA PROCESSING FLOW DIAGRAM", ha='center', va='center', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.text(50, 92.5, "Sequential Execution Steps for Data Preparation and Verification", ha='center', va='center', fontsize=9.5, fontfamily='Arial')

    nodes = [
        ("Start: Load Raw Clinical CSV", "Read 'breast_cancer_5000.csv' into Pandas DataFrame"),
        ("Data Inspection", "Execute head(), tail(), shape (5000, 12), info(), and describe()"),
        ("Data Cleaning & Validation", "Check Missing Values (0 found) & Duplicate Rows (0 found)"),
        ("Outlier Detection (IQR Method)", "Compute Q1, Q3, IQR for Numerical Features (0 extreme outliers)"),
        ("Automated Summary Pipeline", "Run automated_eda_summary() loop across all 12 columns"),
        ("Feature Transformation", "Apply One-Hot Encoding on Categorical Columns (pd.get_dummies)"),
        ("End: Processed Matrix Ready", "Generate structured Feature Matrix (X) and Target Vector (y)")
    ]

    y_pos = [81, 68, 55, 42, 29, 16, 3]
    for i, (title, desc) in enumerate(nodes):
        y = y_pos[i]
        draw_box(ax, 15, y, 70, 8.5, desc, font_size=8.5, bold=False, title=f"{title}")
        if i < len(nodes) - 1:
            draw_arrow(ax, 50, y, 50, y_pos[i+1] + 8.5)

    plt.tight_layout()
    plt.savefig('docs/diagrams/06_data_processing_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

# =========================================================================
# 7. Machine Learning / Regression Workflow Diagram
# =========================================================================
def create_diagram_7():
    fig, ax = setup_academic_canvas(figsize=(10.5, 7.5))
    ax.text(50, 96, "MACHINE LEARNING / REGRESSION WORKFLOW DIAGRAM", ha='center', va='center', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.text(50, 92.5, "Predictive Modeling Lifecycle for Primary Tumor Size Prediction", ha='center', va='center', fontsize=9.5, fontfamily='Arial')

    ml_steps = [
        ("1. Feature Selection & Setup", "Predictors (X): Age, BMI, BP, Follow-up, Tumor_Type, Lymph_Node, Hormone_Receptor, Genetics, Treatment\nTarget (y): Primary Tumor_Size (Continuous, in cm)"),
        ("2. Data Partitioning (Train/Test Split)", "80% Training Set (4,000 samples) | 20% Testing Set (1,000 samples)\nApplied Scikit-learn train_test_split with random_state=42"),
        ("3. Model Initialization & Fitting", "Algorithm: Ordinary Least Squares Linear Regression (LinearRegression)\nFit model weights on training data (X_train, y_train)"),
        ("4. Prediction Generation", "Generate predictions (y_pred) on unseen test set (X_test)\nBuild Actual vs. Predicted comparison table for sample records"),
        ("5. Performance Metric Evaluation", "Compute Real Calculated Metrics: R² Score, MAE, MSE, and RMSE\nPlot Actual vs. Predicted Scatter Plot with 45° Ideal Fit Line (y = x)"),
        ("6. Generalization Diagnosis", "Compare Training vs. Testing MAE and R² (No Overfitting detected)\nSynthesize findings and limitations for clinical oncology")
    ]

    y_pos = [77, 62, 47, 32, 17, 2]
    for i, (title, desc) in enumerate(ml_steps):
        y = y_pos[i]
        draw_box(ax, 12, y, 76, 10.5, desc, font_size=8.5, bold=False, title=f"[{title}]")
        if i < len(ml_steps) - 1:
            draw_arrow(ax, 50, y, 50, y_pos[i+1] + 10.5)

    plt.tight_layout()
    plt.savefig('docs/diagrams/07_ml_regression_workflow_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

# =========================================================================
# 8. Use Case Diagram
# =========================================================================
def create_diagram_8():
    fig, ax = setup_academic_canvas(figsize=(10.5, 7.5))
    ax.text(50, 96, "USE CASE DIAGRAM", ha='center', va='center', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.text(50, 92.5, "System Actors & Core Functional Use Cases", ha='center', va='center', fontsize=9.5, fontfamily='Arial')

    # Boundary
    rect = patches.Rectangle((28, 4), 68, 84, linewidth=1.2, edgecolor='black', facecolor='none', linestyle='-')
    ax.add_patch(rect)
    ax.text(62, 85, "Breast Cancer Data Analytics System Boundary", ha='center', va='center', fontsize=9.5, fontweight='bold')

    # Actor
    draw_box(ax, 4, 38, 18, 22, "Actor:\n\nTYBCA Student /\nData Analyst", font_size=9, bold=True)

    use_cases = [
        "UC-1: Load & Validate Dataset (5,000 Records)",
        "UC-2: Check Missing Values & Outliers (IQR Method)",
        "UC-3: Generate Univariate & Bivariate Visualizations",
        "UC-4: Compute Pearson Correlation Heatmap",
        "UC-5: Train Linear Regression Model (Tumor Size)",
        "UC-6: Train Logistic Regression Model (Survival)",
        "UC-7: Calculate Model Evaluation Metrics (R², MAE, RMSE)",
        "UC-8: Evaluate Overfitting / Underfitting Generalization"
    ]

    y_pos = [75, 65, 55, 45, 35, 25, 15, 5]
    for i, uc in enumerate(use_cases):
        y = y_pos[i]
        draw_box(ax, 34, y, 56, 7.5, uc, font_size=8.5, bold=False, align='center')
        draw_arrow(ax, 22, 49, 34, y + 3.75)

    plt.tight_layout()
    plt.savefig('docs/diagrams/08_use_case_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

# =========================================================================
# 9. Entity-Relationship (ER) Diagram
# =========================================================================
def create_diagram_9():
    fig, ax = setup_academic_canvas(figsize=(10.5, 7.0))
    ax.text(50, 95, "ENTITY-RELATIONSHIP (ER) DIAGRAM", ha='center', va='center', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.text(50, 91.5, "Logical Data Entities and Attributes in Clinical Dataset", ha='center', va='center', fontsize=9.5, fontfamily='Arial')

    # Entity 1: PATIENT
    p_text = "PK: Patient_ID (VARCHAR)\nAge (INT)\nBMI (FLOAT)\nBlood_Pressure (INT)"
    draw_box(ax, 5, 42, 25, 25, p_text, font_size=8.5, title="[PATIENT_MASTER]")

    # Relationship 1
    draw_box(ax, 35, 50, 10, 9, "Has", font_size=9, bold=True)
    draw_arrow(ax, 30, 54.5, 35, 54.5)
    draw_arrow(ax, 45, 54.5, 50, 54.5)
    ax.text(32, 57, "1", fontsize=8.5, fontweight='bold')
    ax.text(48, 57, "1", fontsize=8.5, fontweight='bold')

    # Entity 2: TUMOR_DIAGNOSIS
    t_text = "PK/FK: Patient_ID (VARCHAR)\nTumor_Size (FLOAT) [Target]\nTumor_Type (VARCHAR)\nLymph_Node_Status (VARCHAR)\nHormone_Receptor_Status (VARCHAR)\nGenetic_Mutation (VARCHAR)"
    draw_box(ax, 50, 38, 30, 32, t_text, font_size=8, title="[TUMOR_DIAGNOSIS]")

    # Relationship 2
    draw_box(ax, 60, 23, 10, 9, "Receives", font_size=9, bold=True)
    draw_arrow(ax, 65, 38, 65, 32)
    draw_arrow(ax, 65, 23, 65, 17)
    ax.text(67, 35, "1", fontsize=8.5, fontweight='bold')
    ax.text(67, 20, "1", fontsize=8.5, fontweight='bold')

    # Entity 3: TREATMENT_OUTCOME
    o_text = "PK/FK: Patient_ID (VARCHAR)\nTreatment (VARCHAR)\nFollow_Up_Duration (INT)\nSurvival_Status (VARCHAR) [Target]"
    draw_box(ax, 48, 2, 34, 15, o_text, font_size=8, title="[CLINICAL_OUTCOME]")

    plt.tight_layout()
    plt.savefig('docs/diagrams/09_er_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

# =========================================================================
# 10. Project Input → Process → Output (IPO) Diagram
# =========================================================================
def create_diagram_10():
    fig, ax = setup_academic_canvas(figsize=(11.0, 7.0))
    ax.text(50, 95, "INPUT — PROCESS — OUTPUT (IPO) DIAGRAM", ha='center', va='center', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.text(50, 91.5, "Functional Transformation Pipeline for Data Analytics Project", ha='center', va='center', fontsize=9.5, fontfamily='Arial')

    input_text = "• 'breast_cancer_5000.csv'\n• 5,000 Patient Observations\n• 12 Clinical Variables\n  - Age, BMI, Blood Pressure\n  - Tumor_Size, Tumor_Type\n  - Lymph Node Status\n  - Hormone Receptor Status\n  - Genetic Mutation\n  - Treatment Modality\n  - Follow-Up Duration\n  - Survival Status\n• Train-Test Split Ratio (80/20)\n• Random State (42)"
    draw_box(ax, 5, 8, 27, 76, input_text, font_size=8, align='left', title="[INPUT]")

    process_text = "1. Data Validation & Inspection:\n   - Check nulls (0) & duplicates (0)\n   - IQR Outlier verification\n\n2. Exploratory Data Analysis:\n   - 4 Univariate Charts\n   - 5 Bivariate Charts\n   - 1 Pair Plot ($5 \\times 5$ Grid)\n   - 1 Pearson Heatmap\n\n3. Feature Transformation:\n   - One-hot encoding of categories\n\n4. Supervised Model Training:\n   - OLS Linear Regression\n   - Logistic Regression\n\n5. Evaluation & Diagnosis:\n   - Metric computation\n   - Overfitting vs Underfitting"
    draw_box(ax, 36.5, 8, 27, 76, process_text, font_size=8, align='left', title="[PROCESS]")

    output_text = "• 13 Verified Academic Charts:\n  - 3 Spread Histograms\n  - 1 Tumor Size Box Plot\n  - 1 Scatter Plot (Age vs Size)\n  - 3 Bivariate Box Plots\n  - 1 Grouped Bar Chart\n  - 1 Multivariate Pair Plot\n  - 1 Correlation Heatmap\n  - 1 Actual vs Predicted Plot\n  - 1 Confusion Matrix Heatmap\n\n• Regression Performance:\n  - Test $R^2 = -0.0004$\n  - Test MAE = 2.3825 cm\n  - Test RMSE = 2.7529 cm\n\n• Classification Performance:\n  - Test Accuracy = 52.20%\n\n• Viva-Ready Academic Reports"
    draw_box(ax, 68, 8, 27, 76, output_text, font_size=8, align='left', title="[OUTPUT]")

    draw_arrow(ax, 32, 46, 36.5, 46)
    draw_arrow(ax, 63.5, 46, 68, 46)

    plt.tight_layout()
    plt.savefig('docs/diagrams/10_ipo_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Generating 10 MS-Word-styled Academic Diagrams...")
    create_diagram_1()
    create_diagram_2()
    create_diagram_3()
    create_diagram_4()
    create_diagram_5()
    create_diagram_6()
    create_diagram_7()
    create_diagram_8()
    create_diagram_9()
    create_diagram_10()
    print("All 10 diagrams successfully generated in 'docs/diagrams/'!")

if __name__ == '__main__':
    main()
