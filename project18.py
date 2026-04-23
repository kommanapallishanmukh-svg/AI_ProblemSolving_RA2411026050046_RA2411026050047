#   STUDENT EXAM SCORE PREDICTOR — ML Regression Model
#   Features: Hours Studied, Attendance, Prior Score,
#             Sleep Hours → Predicts Exam Score
# ============================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1.  DATA GENERATION  (synthetic if no CSV)
# ─────────────────────────────────────────────
def generate_synthetic_data(n=500):
    """Create realistic synthetic student data."""
    np.random.seed(42)
    hours_studied    = np.random.uniform(1, 10, n)
    attendance       = np.random.uniform(50, 100, n)
    prior_scores     = np.random.uniform(40, 100, n)
    sleep_hours      = np.random.uniform(4, 10, n)

    # Target: weighted combination + noise
    exam_score = (
        5.0  * hours_studied  +
        0.3  * attendance     +
        0.4  * prior_scores   +
        1.5  * sleep_hours    +
        np.random.normal(0, 5, n)
    ).clip(0, 100)

    df = pd.DataFrame({
        "Hours_Studied"  : np.round(hours_studied, 2),
        "Attendance_%"   : np.round(attendance, 2),
        "Prior_Score"    : np.round(prior_scores, 2),
        "Sleep_Hours"    : np.round(sleep_hours, 2),
        "Exam_Score"     : np.round(exam_score, 2),
    })
    return df


# ─────────────────────────────────────────────
# 2.  DATA LOADING
# ─────────────────────────────────────────────
def load_data():
    print("\n" + "="*55)
    print("   STUDENT EXAM SCORE PREDICTOR")
    print("="*55)
    print("\nHow would you like to provide the dataset?")
    print("  [1] Load from CSV file")
    print("  [2] Use synthetic data (auto-generated)")
    print("  [3] Enter data manually (console input)")

    choice = input("\nEnter your choice (1 / 2 / 3): ").strip()

    if choice == "1":
        path = input("Enter the full path to your CSV file: ").strip().strip('"')
        if not os.path.exists(path):
            print(f"  ✗ File not found: {path}")
            print("  → Falling back to synthetic data.\n")
            return generate_synthetic_data()
        df = pd.read_csv(path)
        print(f"\n  ✓ Loaded {len(df)} rows from '{os.path.basename(path)}'")
        return df

    elif choice == "3":
        print("\nEnter student records (type 'done' when finished).")
        print("Format: hours_studied, attendance_%, prior_score, sleep_hours, exam_score")
        records = []
        while True:
            row = input("  Row: ").strip()
            if row.lower() == "done":
                break
            try:
                vals = [float(v) for v in row.split(",")]
                if len(vals) == 5:
                    records.append(vals)
                else:
                    print("  ✗ Need exactly 5 values.")
            except ValueError:
                print("  ✗ Invalid input — use numbers separated by commas.")
        if len(records) < 10:
            print("  ✗ Too few records. Need at least 10. Using synthetic data.")
            return generate_synthetic_data()
        df = pd.DataFrame(records,
                          columns=["Hours_Studied","Attendance_%",
                                   "Prior_Score","Sleep_Hours","Exam_Score"])
        return df

    else:   # default / choice == "2"
        print("\n  → Generating 500-row synthetic dataset …")
        df = generate_synthetic_data()
        df.to_csv("synthetic_student_data.csv", index=False)
        print("  ✓ Saved to 'synthetic_student_data.csv'")
        return df


# ─────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    print("\n─── Preprocessing ───────────────────────────────")
    print(f"  Shape before : {df.shape}")

    # Auto-detect feature & target columns
    target_candidates = [c for c in df.columns
                         if any(k in c.lower() for k in
                                ["score","grade","mark","result","exam","gpa"])]
    if not target_candidates:
        target_candidates = [df.columns[-1]]

    target_col = target_candidates[0]
    feature_cols = [c for c in df.columns if c != target_col]

    print(f"  Target column  : {target_col}")
    print(f"  Feature columns: {feature_cols}")

    # Keep only numeric columns
    df_num = df[feature_cols + [target_col]].select_dtypes(include=[np.number])
    missing = df_num.isnull().sum().sum()
    if missing:
        df_num = df_num.fillna(df_num.median())
        print(f"  Filled {missing} missing values with column medians.")

    print(f"  Shape after  : {df_num.shape}")

    X = df_num[feature_cols].values
    y = df_num[target_col].values
    return X, y, feature_cols, target_col


# ─────────────────────────────────────────────
# 4.  MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(X, y, feature_cols):
    print("\n─── Training ────────────────────────────────────")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Linear Regression"  : LinearRegression(),
        "Random Forest"      : RandomForestRegressor(n_estimators=100,
                                                     random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        r2    = r2_score(y_test, preds)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = {"model": model, "preds": preds,
                         "r2": r2, "mae": mae, "rmse": rmse}
        print(f"\n  ▸ {name}")
        print(f"      R²   : {r2:.4f}")
        print(f"      MAE  : {mae:.2f}")
        print(f"      RMSE : {rmse:.2f}")

    # Pick best model by R²
    best_name = max(results, key=lambda k: results[k]["r2"])
    best      = results[best_name]
    print(f"\n  ★  Best model: {best_name}  (R² = {best['r2']:.4f})")

    return best["model"], scaler, X_test, y_test, best["preds"], best_name, results


# ─────────────────────────────────────────────
# 5.  VISUALIZATIONS
# ─────────────────────────────────────────────
def visualize(y_test, preds, feature_cols, X_train, best_name, results):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Student Exam Score Predictor — Results", fontsize=16, fontweight="bold")

    # (a) Actual vs Predicted
    ax = axes[0, 0]
    ax.scatter(y_test, preds, alpha=0.6, color="#4C72B0", edgecolors="white", s=60)
    lims = [min(y_test.min(), preds.min()) - 2, max(y_test.max(), preds.max()) + 2]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title(f"Actual vs Predicted  ({best_name})")
    ax.legend()

    # (b) Residuals
    ax = axes[0, 1]
    residuals = y_test - preds
    ax.hist(residuals, bins=30, color="#55A868", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")

    # (c) Model Comparison bar chart
    ax = axes[1, 0]
    names = list(results.keys())
    r2s   = [results[n]["r2"] for n in names]
    bars  = ax.bar(names, r2s, color=["#4C72B0","#DD8452"], edgecolor="white")
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("R² Score")
    ax.set_title("Model Comparison (R²)")

    # (d) Feature importance (correlation heatmap on training data)
    ax = axes[1, 1]
    df_heat = pd.DataFrame(X_train, columns=feature_cols)
    corr = df_heat.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix")

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    plt.show()
    print("\n  ✓ Chart saved as 'results.png'")


# ─────────────────────────────────────────────
# 6.  PREDICT NEW STUDENTS (INTERACTIVE)
# ─────────────────────────────────────────────
def predict_new(model, scaler, feature_cols):
    print("\n─── Predict New Student Score ───────────────────")
    while True:
        print(f"\n  Enter values for: {feature_cols}")
        print("  (or type 'exit' to quit)")
        raw = input("  Input: ").strip()
        if raw.lower() == "exit":
            print("\n  Goodbye! 👋\n")
            break
        try:
            vals = [float(v) for v in raw.split(",")]
            if len(vals) != len(feature_cols):
                print(f"  ✗ Expected {len(feature_cols)} values.")
                continue
            X_new = scaler.transform([vals])
            score = model.predict(X_new)[0]
            score = np.clip(score, 0, 100)
            print(f"\n  🎯 Predicted Exam Score : {score:.1f} / 100")
            grade = ("A" if score >= 85 else "B" if score >= 70 else
                     "C" if score >= 55 else "D" if score >= 40 else "F")
            print(f"     Grade                : {grade}")
        except ValueError:
            print("  ✗ Please enter numeric values separated by commas.")


# ─────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────
def main():
    # Step A: Load data
    df = load_data()
    print("\n  First 5 rows of your dataset:")
    print(df.head().to_string(index=False))

    # Step B: Preprocess
    X, y, feature_cols, target_col = preprocess(df)

    if len(X) < 20:
        print("  ✗ Dataset too small (need ≥ 20 rows). Exiting.")
        return

    # Step C: Train & Evaluate
    model, scaler, X_test, y_test, preds, best_name, results = \
        train_and_evaluate(X, y, feature_cols)

    # Step D: Visualize
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    visualize(y_test, preds, feature_cols, X_train, best_name, results)

    # Step E: Predict new inputs
    predict_new(model, scaler, feature_cols)


if __name__ == "__main__":
    main()
