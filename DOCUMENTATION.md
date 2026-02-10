# Employee Performance & Productivity Dashboard — Documentation

## 1. Project Overview

**Employee Performance & Productivity** is a data analytics and predictive modeling project that helps organizations understand and predict employee performance and productivity using historical HR and work data. The project delivers an interactive **Streamlit dashboard** for exploration, visualization, and ML-based predictions.

---

## 2. What This Project Is About

- **Purpose:** Analyze employee performance and productivity, visualize patterns by department/job/education, and predict future performance and productivity scores from input features.
- **Audience:** HR teams, managers, and analysts who need to:
  - Monitor workforce metrics and distributions
  - Compare performance and productivity across departments, job titles, and education levels
  - Predict performance or productivity scores for hypothetical or new employees
  - Identify top performers and top producers and search individual employee records
- **Output:** A single web application (dashboard) that combines:
  - Filtered overview metrics and charts
  - Performance and productivity distributions and breakdowns
  - Two prediction tools (Performance Score and Productivity Score)
  - Employee insights: top lists, scatter plots, and employee search

---

## 3. Project Structure

```
Employee Performance and Productivity/
├── dashboard.py              # Main Streamlit app (nav, filters, tabs, prediction forms, models)
├── styles.css                # Custom UI (navbar, footer, form section headers, chart layout)
├── Extended_Employee_Performance_and_Productivity_Data.csv   # Dataset
├── Performance Evaluation.ipynb   # Notebook for performance analysis/experiments
├── Productivity Evaluation.ipynb  # Notebook for productivity analysis/experiments
├── requirements.txt          # Python dependencies
├── run_dashboard.ps1         # PowerShell script to run the dashboard
├── .streamlit/
│   └── config.toml           # Streamlit theme (light mode, colors)
├── DOCUMENTATION.md          # This file
└── README.md                 # Project overview and quick start
```

---

## 4. Dataset

- **File:** `Extended_Employee_Performance_and_Productivity_Data.csv`
- **Size:** 100,000 rows × 20 columns

### Columns

| Column | Description |
|--------|-------------|
| `Employee_ID` | Unique employee identifier |
| `Department` | Department (e.g. IT, Finance, HR, Sales) |
| `Gender` | Employee gender |
| `Age` | Age in years |
| `Job_Title` | Role (e.g. Developer, Manager, Analyst) |
| `Hire_Date` | Date of hire |
| `Years_At_Company` | Tenure in years |
| `Education_Level` | Education level |
| `Performance_Score` | Performance score (existing target for performance model) |
| `Monthly_Salary` | Monthly salary |
| `Work_Hours_Per_Week` | Weekly work hours |
| `Projects_Handled` | Number of projects handled |
| `Overtime_Hours` | Overtime hours |
| `Sick_Days` | Sick days taken |
| `Remote_Work_Frequency` | Remote work frequency (e.g. percentage or scale) |
| `Team_Size` | Team size |
| `Training_Hours` | Training hours |
| `Promotions` | Number of promotions |
| `Employee_Satisfaction_Score` | Satisfaction score |
| `Resigned` | Whether the employee resigned |

The app derives **`Hire_Year`** from `Hire_Date` and **`Productivity_Score`** from a weighted formula (see below).

---

## 5. Productivity Score (Derived Metric)

Productivity is not a raw column; it is computed in the app:

- **Formula (conceptual):**  
  `Productivity_Raw = 0.45×Projects_Handled + 0.25×Work_Hours_Per_Week + 0.20×Overtime_Hours − 0.15×Sick_Days + 0.05×Training_Hours`
- **Scaling:** Values are scaled to a **1–100** range using `MinMaxScaler`, then rounded to two decimals as **Productivity_Score**.

This score is used in visualizations, filters, and the productivity prediction model.

---

## 6. Dashboard Features (Navigation & Tabs)

The dashboard uses a **top navigation bar** for switching content and a **sidebar** for filters.

### Top Navigation

- **Overview** — Default tab; KPIs and main distributions.
- **Performance** — Performance charts and prediction form.
- **Productivity** — Productivity charts and prediction form.
- **Insights** — Top performers/producers, scatter plots, employee search.

Navigation stays in the same browser tab (query parameter `?nav=...`). The navbar is fixed, full-width, and styled via `styles.css` (`.premium-nav`).

### Sidebar Filters

- **Department** — Filter by department or “All”
- **Job Title** — Filter by job title or “All”
- **Education Level** — Filter by education level or “All”

All charts and metrics respect these filters (where applicable).

### Footer

A custom footer appears at the bottom: “© 2025 Employee Performance & Productivity. All rights reserved.” Styled in `styles.css` (`.footer-pro`); full content width, no nav links.

---

### Tab 1: Overview

- **Metrics (KPI cards):** Total employees, average performance score, average productivity score, average monthly salary, average satisfaction.
- **Charts:**
  - Performance score distribution (histogram)
  - Productivity score distribution (histogram)
  - Average performance score by department (bar chart)
  - Average productivity score by department (bar chart)
  - Additional visualizations (e.g. performance/productivity by job title or education, scatter plots).
- All use **filtered** data based on sidebar selections.

---

### Tab 2: Performance Evaluation

- **Charts:** Performance by job title, performance by education level, and other performance-related plots (filtered). Chart layout is consistent via a shared `_chart_layout()` and `styles.css`.
- **Performance Evaluation Model:**
  - Trained **Gradient Boosting Regressor** to predict **Performance_Score** from:
    - Categorical: Department, Gender, Job_Title, Education_Level (one-hot encoded)
    - Numerical: Age, Years_At_Company, Monthly_Salary, Work_Hours_Per_Week, Projects_Handled, Overtime_Hours, Sick_Days, Remote_Work_Frequency, Team_Size, Training_Hours, Promotions, Employee_Satisfaction_Score, Hire_Year
  - Preprocessing: numerical features standardized, categorical one-hot encoded via `ColumnTransformer`.
  - Model is cached (`@st.cache_resource`) so it is trained once per run.
- **Predict Performance Score:** Input form grouped into three sections (styled with `.form-section-header` in CSS):
  - **Profile & role** — Department, Job Title, Education Level, Gender; then Age, Years at Company, Hire Year
  - **Work & output** — Work Hours/Week, Projects Handled, Overtime, Sick Days; Remote Work %, Team Size
  - **Growth & satisfaction** — Monthly Salary, Training Hours, Promotions, Satisfaction (1–5)
  - Full-width “Predict Performance Score” button; app shows predicted score and level (e.g. Excellent / Good / Below average).

---

### Tab 3: Productivity Evaluation

- **Charts:** Productivity by job title and other productivity-related visualizations (filtered).
- **Productivity Evaluation Model:**
  - Same approach as performance: **Gradient Boosting Regressor** predicting **Productivity_Score** (the derived 1–100 score).
  - Same preprocessing and caching strategy; same 17 input features.
- **Predict Productivity Score:** Input form uses the same grouped layout as Performance (Profile & role, Work & output, Growth & satisfaction). Caption: “Productivity is driven by work output, hours, training, and engagement.” App shows predicted productivity score and a brief level/description.

---

### Tab 4: Employee Insights

- **Top Performers (Performance):** Bar chart and table of top 10 employees by **Performance_Score** (filtered).
- **Top Producers (Productivity):** Bar chart and table of top 10 employees by **Productivity_Score** (filtered).
- **Detailed analysis:** Scatter plots (e.g. Performance vs Salary, Productivity vs Work Hours), possibly by department.
- **Search Employee:** Search by **Employee ID**; display that employee’s key metrics and details.

---

## 7. Technology Stack

| Component | Technology |
|----------|------------|
| App framework | **Streamlit** |
| Data handling | **pandas**, **NumPy** |
| Visualizations | **Plotly** (Plotly Express and Graph Objects) |
| ML / preprocessing | **scikit-learn** (GradientBoostingRegressor, StandardScaler, MinMaxScaler, OneHotEncoder, ColumnTransformer, Pipeline, train_test_split, metrics) |

### Dependencies (requirements.txt)

- `streamlit>=1.28.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`
- `plotly>=5.17.0`

---

## 8. How to Run the Project

### Prerequisites

- Python 3.8+ (3.9+ recommended)
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the dashboard

**Option A — Command line**

```bash
streamlit run dashboard.py
```

**Option B — PowerShell (Windows)**

```powershell
.\run_dashboard.ps1
```

Or from any folder:

```powershell
cd "path\to\Employee Performance and Productivity"
streamlit run dashboard.py
```

- Default URL: **http://localhost:8501**
- Use the sidebar to change filters; use the tabs to switch between Overview, Performance Evaluation, Productivity Evaluation, and Employee Insights.

---

## 9. Configuration & Custom UI

- **Streamlit theme:** `.streamlit/config.toml` forces **light** theme and sets primary color to `#1E88E5`, with white/light backgrounds so tables and inputs are readable and consistent.
- **Custom CSS (`styles.css`):** Loaded by the dashboard for:
  - **Navbar** — Fixed top bar (`.premium-nav`), active-tab styling, same-tab navigation.
  - **Footer** — Full-width footer (`.footer-pro`), copyright line, no extra space below.
  - **Layout** — Content spacing, block-container padding, no horizontal scroll; chart layout via shared `_chart_layout()` in `dashboard.py`.
  - **Form sections** — Prediction form section headers (`.form-section-header`): uppercase, secondary color, spacing.
  - **Variables** — CSS custom properties for colors, radii, shadows, and typography (e.g. `--primary`, `--card`, `--radius`).

---

## 10. Summary

This project provides a single, self-contained dashboard to:

1. **Explore** employee performance and productivity with sidebar filters and charts.
2. **Navigate** via a top navbar (Overview, Performance, Productivity, Insights) with a custom footer.
3. **Understand** how performance and productivity vary by department, job title, and education.
4. **Predict** performance and productivity scores for new or hypothetical employees using two Gradient Boosting models; prediction forms are grouped into Profile & role, Work & output, and Growth & satisfaction.
5. **Identify** top performers and top producers and **look up** individual employees by ID.

All of this is backed by the **Extended_Employee_Performance_and_Productivity_Data.csv** dataset and a derived **Productivity_Score** metric, with a consistent UI built in Streamlit, Plotly, and custom CSS (`styles.css`).
