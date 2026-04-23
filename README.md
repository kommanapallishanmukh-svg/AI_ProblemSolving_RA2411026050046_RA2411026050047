# AI_ProblemSolving_RA2411026050046_RA2411026050047
# AI Problem Solving — Student Exam Score Predictor

> Predicting student academic performance using Linear Regression (scikit-learn)

---

## Problem Description

Given measurable study-related features for a student, this model estimates their likely exam score on a **0–100 scale**.

The problem is a classic **supervised regression** task — the model learns from historical student data (features + known scores) and then predicts scores for new students it has never seen.

### Input Features

| Feature | Description | Typical Range |
|---|---|---|
| Hours Studied | Weekly study hours before exam | 1 – 12 hrs |
| Attendance % | Percentage of classes attended | 40 – 100 % |
| Previous Score | Score from the most recent prior exam | 30 – 95 pts |
| Assignments Done | Number of assignments submitted (out of 10) | 0 – 10 |

### Target Variable

| Variable | Description | Range |
|---|---|---|
| Exam Score | Final exam result to predict | 0 – 100 |

---

## Algorithm Used

**Linear Regression** — `sklearn.linear_model.LinearRegression`

Linear Regression models the target as a weighted linear combination of the input features:

```
Exam Score = w₁·(Hours) + w₂·(Attendance) + w₃·(Prev Score) + w₄·(Assignments) + b
```

Where `w₁…w₄` are learned weights (coefficients) and `b` is the bias (intercept), determined by minimising the Mean Squared Error using the Ordinary Least Squares solution:

```
w = (XᵀX)⁻¹ Xᵀy
```

### Full ML Pipeline

```
Raw Data
   │
   ▼
Handle Missing Values  ← fill NaN with column median; drop residual NaN rows
   │
   ▼
Train / Test Split     ← default 80% train, 20% test (configurable)
   │
   ▼
StandardScaler         ← zero mean, unit variance (fit on train only)
   │
   ▼
LinearRegression.fit() ← learns weights on scaled training data
   │
   ▼
Predict on Test Set    ← clip predictions to [0, 100]
   │
   ▼
Evaluate Metrics       ← R², MAE, RMSE
   │
   ▼
Predict New Students   ← scale input → model.predict() → display
```

### Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| R² Score | 1 − SS_res/SS_tot | 1.0 = perfect fit; 0 = predicts mean |
| MAE | mean(\|y − ŷ\|) | Average error in exam points |
| RMSE | √mean((y − ŷ)²) | Penalises large errors more than MAE |

---

## Folder Structure

```
AI_ProblemSolving_<RegisterNumber>/
│
├── StudentScorePredictor/
│   ├── student_score_predictor.py   ← Main application
│   └── README.md                    ← This file
```

---

## Execution Steps

### Prerequisites

Install the required Python libraries:

```bash
pip install scikit-learn pandas numpy
```

> If you get a system error, use:
> ```bash
> pip install scikit-learn pandas numpy --break-system-packages
> ```

### Running the Application

```bash
python student_score_predictor.py
```

### Step-by-Step Usage

The application has **three tabs**:

#### Tab ① — Dataset

Choose one of three data sources:

**Option A — Upload CSV**
- Click Browse and select a `.csv` file
- Required columns: `Hours Studied`, `Attendance %`, `Previous Score`, `Assignments Done`, `Exam Score`
- Missing values are auto-filled with column medians

**Option B — Synthetic Dataset** *(recommended for quick demo)*
- Set the number of rows (default: 120, minimum: 10)
- Click **Generate** — a realistic dataset is created instantly

**Option C — Manual Entry**
- Fill in all five fields for one student row
- Click **+ Add Row** to accumulate rows (minimum 5 required)
- Click **Use Manual Data** to load

After loading, the Data Preview table on the right populates with up to 200 rows.

#### Tab ② — Train Model

- Set the test split % (5–50, default 20%)
- Click **Train Model**
- The Metrics panel updates with R², MAE, RMSE, and individual feature coefficients

#### Tab ③ — Predict

- Enter values for all four features
- Click **Predict Score**
- The app displays the predicted score, a grade label, and appends to the session history log

### CSV Format Reference

```csv
Hours Studied,Attendance %,Previous Score,Assignments Done,Exam Score
7.5,88.0,74.0,9,81.3
3.2,61.0,55.0,4,52.7
11.0,97.0,91.0,10,95.1
```

---

## Sample Output

### Model Metrics (trained on 120-row synthetic dataset)

```
┌─────────────┬────────────┬──────────────┐
│  R² Score   │    MAE     │     RMSE     │
│   0.9293    │   2.93 pts │   3.94 pts   │
└─────────────┴────────────┴──────────────┘

Training rows: 96    Test rows: 24

Feature Coefficients (scaled):
  Hours Studied      +14.2341
  Attendance %       + 2.8812
  Previous Score     +13.0457
  Assignments Done   + 7.6123
```

### Single Student Prediction

**Input:**
```
Hours Studied    : 8.0
Attendance %     : 90.0
Previous Score   : 76.0
Assignments Done : 8
```

**Output:**
```
Predicted Exam Score
        83.6
  Grade: Good
```

### Grade Bands

```
90 – 100  →  Excellent
75 –  89  →  Good
60 –  74  →  Average
40 –  59  →  Below Average
 0 –  39  →  At Risk
```

### App Screenshot

![Student Score Predictor UI](screenshot.png)

---

## Error Handling

| Error Message | Cause | Fix |
|---|---|---|
| `⚠ Invalid value for "Attendance %": "abc"` | Non-numeric input | Enter a valid number |
| `⚠ Not Enough Data — Enter at least 5 rows` | Too few manual rows | Add more rows or use Synthetic |
| `⚠ Train the model first (Tab ②)` | Predict before training | Complete Tab ① and Tab ② first |
| `⚠ Missing columns: Assignments Done` | Wrong CSV format | Check column names match exactly |

---

## Dependencies

| Package | Purpose | Install |
|---|---|---|
| `tkinter` | GUI (built into Python) | Built-in |
| `scikit-learn` | Linear Regression, StandardScaler | `pip install scikit-learn` |
| `pandas` | Data loading and processing | `pip install pandas` |
| `numpy` | Numerical operations | `pip install numpy` |
