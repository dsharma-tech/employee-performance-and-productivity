import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Employee Performance & Productivity Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load professional SaaS dashboard theme + Inter font
import os
st.markdown("""<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">""", unsafe_allow_html=True)
_css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.css")
if os.path.exists(_css_path):
    with open(_css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>/* styles.css not found */</style>", unsafe_allow_html=True)

# Force navbar to touch top + page ends at footer (no blank space below)
st.markdown("""
<style>
html, body { padding-top: 0 !important; margin-top: 0 !important; padding-bottom: 0 !important; margin-bottom: 0 !important; height: auto !important; min-height: auto !important; }
.stApp, .stApp > div, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > div,
section.main, div[class*="block-container"], div[class*="blockContainer"] {
  padding-top: 0 !important; margin-top: 0 !important; padding-bottom: 0 !important; margin-bottom: 0 !important;
  height: auto !important; min-height: auto !important;
}
header, [data-testid="stHeader"], [data-testid="stToolbar"] {
  display: none !important; height: 0 !important; min-height: 0 !important;
  padding: 0 !important; margin: 0 !important; overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

# Modern Plotly chart theme (professional, matches dashboard CSS)
def _chart_layout(height=400):
    return dict(
        height=height,
        template="plotly_white",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#0f172a", size=13, family="Inter, -apple-system, sans-serif"),
        title=dict(font=dict(size=17, color="#0f172a", family="Inter, sans-serif"), x=0.02, xanchor="left"),
        margin=dict(t=56, b=48, l=56, r=32),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", font=dict(size=12)),
        xaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            title_font=dict(size=12, color="#64748b"),
            tickfont=dict(size=11, color="#475569"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            title_font=dict(size=12, color="#64748b"),
            tickfont=dict(size=11, color="#475569"),
        ),
        legend=dict(
            font=dict(size=11, color="#475569"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
    )


@st.cache_data
def load_data():
    """Load and preprocess the employee data"""
    df = pd.read_csv("Extended_Employee_Performance_and_Productivity_Data.csv")
    df["Hire_Date"] = pd.to_datetime(df["Hire_Date"])
    df["Hire_Year"] = df["Hire_Date"].dt.year
    return df

@st.cache_data
def calculate_productivity_score(df):
    """Calculate productivity score for the dataframe"""
    df_copy = df.copy()
    df_copy["Productivity_Raw"] = (
        (0.45 * df_copy["Projects_Handled"]) +
        (0.25 * df_copy["Work_Hours_Per_Week"]) +
        (0.20 * df_copy["Overtime_Hours"]) -
        (0.15 * df_copy["Sick_Days"]) +
        (0.05 * df_copy["Training_Hours"])
    )
    scaler = MinMaxScaler(feature_range=(1, 100))
    df_copy["Productivity_Score"] = scaler.fit_transform(df_copy[["Productivity_Raw"]])
    df_copy["Productivity_Score"] = df_copy["Productivity_Score"].round(2)
    return df_copy

@st.cache_resource
def train_performance_model(df):
    """Train performance evaluation model"""
    df_model = df.copy()
    df_model["Performance_Score_noisy"] = df_model["Performance_Score"].copy()
    
    # Define features
    categorical_features = ["Department", "Gender", "Job_Title", "Education_Level"]
    numerical_features = [
        "Age", "Years_At_Company", "Monthly_Salary", "Work_Hours_Per_Week",
        "Projects_Handled", "Overtime_Hours", "Sick_Days", "Remote_Work_Frequency",
        "Team_Size", "Training_Hours", "Promotions", "Employee_Satisfaction_Score",
        "Hire_Year"
    ]
    
    X = df_model.drop(columns=["Performance_Score", "Performance_Score_noisy", 
                               "Employee_ID", "Hire_Date", "Resigned"])
    y = df_model["Performance_Score_noisy"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(random_state=42)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return pipeline, rmse, r2

@st.cache_resource
def train_productivity_model(df):
    """Train productivity evaluation model"""
    df_model = df.copy()
    df_model = calculate_productivity_score(df_model)
    
    categorical_features = ["Department", "Gender", "Job_Title", "Education_Level"]
    numerical_features = [
        "Age", "Years_At_Company", "Monthly_Salary", "Work_Hours_Per_Week",
        "Projects_Handled", "Overtime_Hours", "Sick_Days",
        "Remote_Work_Frequency", "Team_Size", "Training_Hours",
        "Promotions", "Employee_Satisfaction_Score", "Hire_Year"
    ]
    
    X = df_model.drop(columns=["Productivity_Score", "Productivity_Raw",
                               "Performance_Score", "Employee_ID", "Hire_Date", "Resigned"])
    y = df_model["Productivity_Score"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(random_state=42)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return pipeline, rmse, r2

def main():
    # Sync nav from URL so navbar links work (no Home/Settings)
    valid_tabs = ["Overview", "Performance", "Productivity", "Insights"]
    nav_from_url = st.query_params.get("nav")
    if nav_from_url in valid_tabs:
        st.session_state["nav_tab"] = nav_from_url
    if "nav_tab" not in st.session_state or st.session_state["nav_tab"] == "Home":
        st.session_state["nav_tab"] = "Overview"
    
    current = st.session_state["nav_tab"]
    
    # ----- Premium navbar: JS-based nav so it always opens in same tab (no new tab) -----
    nav_html = f'''
    <nav class="premium-nav">
        <div class="premium-nav-inner">
            <a href="javascript:void(0)" class="premium-nav-brand" data-nav="Overview" role="button">
                <span class="premium-nav-logo">📊</span>
                <span class="premium-nav-title">Employee Performance & Productivity</span>
            </a>
            <div class="premium-nav-links">
                <a href="javascript:void(0)" class="premium-nav-link{" premium-nav-link-active" if current == "Overview" else ""}" data-nav="Overview" role="button">Overview</a>
                <a href="javascript:void(0)" class="premium-nav-link{" premium-nav-link-active" if current == "Performance" else ""}" data-nav="Performance" role="button">Performance</a>
                <a href="javascript:void(0)" class="premium-nav-link{" premium-nav-link-active" if current == "Productivity" else ""}" data-nav="Productivity" role="button">Productivity</a>
                <a href="javascript:void(0)" class="premium-nav-link{" premium-nav-link-active" if current == "Insights" else ""}" data-nav="Insights" role="button">Insights</a>
            </div>
        </div>
    </nav>
    <script>
    (function() {{
        function goNav(tab) {{
            window.location.href = (window.location.pathname || '/') + '?nav=' + encodeURIComponent(tab);
        }}
        document.querySelectorAll('.premium-nav a[data-nav]').forEach(function(el) {{
            el.addEventListener('click', function(e) {{
                e.preventDefault();
                e.stopPropagation();
                goNav(el.getAttribute('data-nav'));
            }});
        }});
    }})();
    </script>
    <div class="premium-nav-spacer"></div>
    '''
    try:
        st.html(nav_html, unsafe_allow_javascript=True)
    except (TypeError, AttributeError):
        # Older Streamlit: st.html may not exist or may lack unsafe_allow_javascript
        st.markdown(nav_html, unsafe_allow_html=True)
    
    # ----- Dashboard: hide sidebar so filters live below navbar only -----
    st.markdown("""
        <style>
        [data-testid="stSidebar"], section[data-testid="stSidebar"] { display: none !important; }
        .main .block-container { max-width: 100%%; padding-left: 2rem; padding-right: 2rem; }
        </style>
    """, unsafe_allow_html=True)
    
    # ----- Dashboard: load data and filters -----
    df = load_data()
    df_with_productivity = calculate_productivity_score(df)
    
    # Filter row below navbar (no grey bar wrapper)
    filter_col1, filter_col2, filter_col3, _ = st.columns([1.2, 1.2, 1.2, 2])
    departments = ["All"] + sorted(df["Department"].unique().tolist())
    job_titles = ["All"] + sorted(df["Job_Title"].unique().tolist())
    education_levels = ["All"] + sorted(df["Education_Level"].unique().tolist())
    
    with filter_col1:
        selected_department = st.selectbox(
            "**Department**",
            departments,
            help="Filter by department or show All.",
            key="filter_department"
        )
    with filter_col2:
        selected_job_title = st.selectbox(
            "**Job Title**",
            job_titles,
            help="Filter by job title or show All.",
            key="filter_job_title"
        )
    with filter_col3:
        selected_education = st.selectbox(
            "**Education Level**",
            education_levels,
            help="Filter by education level or show All.",
            key="filter_education"
        )
    
    filtered_df = df_with_productivity.copy()
    if selected_department != "All":
        filtered_df = filtered_df[filtered_df["Department"] == selected_department]
    if selected_job_title != "All":
        filtered_df = filtered_df[filtered_df["Job_Title"] == selected_job_title]
    if selected_education != "All":
        filtered_df = filtered_df[filtered_df["Education_Level"] == selected_education]
    
    # Main content by nav selection
    if st.session_state["nav_tab"] == "Overview":
        st.header("📊 Overview Dashboard")
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Employees", f"{len(filtered_df):,}")
        
        with col2:
            avg_performance = filtered_df["Performance_Score"].mean()
            st.metric("Avg Performance Score", f"{avg_performance:.2f}")
        
        with col3:
            avg_productivity = filtered_df["Productivity_Score"].mean()
            st.metric("Avg Productivity Score", f"{avg_productivity:.2f}")
        
        with col4:
            avg_salary = filtered_df["Monthly_Salary"].mean()
            st.metric("Avg Monthly Salary", f"${avg_salary:,.0f}")
        
        with col5:
            avg_satisfaction = filtered_df["Employee_Satisfaction_Score"].mean()
            st.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}")
        
        st.markdown("---")
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance Score Distribution
            fig = px.histogram(
                filtered_df, 
                x="Performance_Score",
                nbins=5,
                title="Performance Score Distribution",
                labels={"Performance_Score": "Performance Score", "count": "Number of Employees"},
                color_discrete_sequence=['#2563EB']
            )
            fig.update_layout(**_chart_layout(400))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Productivity Score Distribution
            fig = px.histogram(
                filtered_df,
                x="Productivity_Score",
                nbins=50,
                title="Productivity Score Distribution",
                labels={"Productivity_Score": "Productivity Score", "count": "Number of Employees"},
                color_discrete_sequence=['#e85a70']
            )
            fig.update_layout(**_chart_layout(400))
            st.plotly_chart(fig, use_container_width=True)
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance by Department
            dept_perf = filtered_df.groupby("Department")["Performance_Score"].mean().sort_values(ascending=False)
            fig = px.bar(
                x=dept_perf.index,
                y=dept_perf.values,
                title="Average Performance Score by Department",
                labels={"x": "Department", "y": "Average Performance Score"},
                color_discrete_sequence=['#2563EB']
            )
            fig.update_layout(**_chart_layout(400))
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Productivity by Department
            dept_prod = filtered_df.groupby("Department")["Productivity_Score"].mean().sort_values(ascending=False)
            fig = px.bar(
                x=dept_prod.index,
                y=dept_prod.values,
                title="Average Productivity Score by Department",
                labels={"x": "Department", "y": "Average Productivity Score"},
                color_discrete_sequence=['#e85a70']
            )
            fig.update_layout(**_chart_layout(400))
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Charts Row 3
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance vs Productivity Scatter
            fig = px.scatter(
                filtered_df.sample(min(5000, len(filtered_df))),
                x="Performance_Score",
                y="Productivity_Score",
                color="Department",
                size="Monthly_Salary",
                hover_data=["Job_Title", "Years_At_Company"],
                title="Performance vs Productivity",
                labels={"Performance_Score": "Performance Score", 
                       "Productivity_Score": "Productivity Score"},
                color_discrete_sequence=['#2563EB', '#e85a70', '#e88a9a', '#93C5FD', '#f5c6ce', '#E2E8F0']
            )
            fig.update_layout(**_chart_layout(400))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation Heatmap
            numeric_cols = ["Performance_Score", "Productivity_Score", "Monthly_Salary",
                          "Work_Hours_Per_Week", "Projects_Handled", "Employee_Satisfaction_Score"]
            corr_matrix = filtered_df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap",
                color_continuous_scale=[[0, '#E2E8F0'], [0.5, '#93C5FD'], [1, '#2563EB']]
            )
            fig.update_layout(**_chart_layout(400))
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Performance Evaluation
    elif st.session_state["nav_tab"] == "Performance":
        st.header("⭐ Performance Evaluation Model")
        
        # Train model and show metrics
        with st.spinner("Training performance model..."):
            perf_model, perf_rmse, perf_r2 = train_performance_model(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model R² Score", f"{perf_r2:.4f}")
        with col2:
            st.metric("RMSE", f"{perf_rmse:.4f}")
        with col3:
            st.metric("Model Type", "Gradient Boosting")
        
        st.markdown("---")
        
        # Performance Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance by Job Title
            job_perf = filtered_df.groupby("Job_Title")["Performance_Score"].mean().sort_values(ascending=False)
            fig = px.bar(
                x=job_perf.index,
                y=job_perf.values,
                title="Average Performance by Job Title",
                labels={"x": "Job Title", "y": "Average Performance Score"},
                color_discrete_sequence=['#2563EB']
            )
            fig.update_layout(**_chart_layout(400))
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance by Education Level
            edu_perf = filtered_df.groupby("Education_Level")["Performance_Score"].mean().sort_values(ascending=False)
            fig = px.bar(
                x=edu_perf.index,
                y=edu_perf.values,
                title="Average Performance by Education Level",
                labels={"x": "Education Level", "y": "Average Performance Score"},
                color_discrete_sequence=['#e85a70']
            )
            fig.update_layout(**_chart_layout(400))
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance Prediction
        st.subheader("🔮 Predict Performance Score")
        with st.expander("ℹ️ What does Performance Score mean?", expanded=False):
            st.markdown("""
            **Performance Score Scale (1-5):** 5 = Excellent · 4 = Very Good · 3 = Good · 2 = Needs Improvement · 1 = Poor  
            The model uses profile, work output, and engagement factors to predict score.
            """)
        
        st.markdown('<p class="form-section-header">Profile & role</p>', unsafe_allow_html=True)
        r1a, r1b, r1c, r1d = st.columns(4)
        with r1a:
            pred_dept = st.selectbox("Department", df["Department"].unique(), key="perf_dept")
        with r1b:
            pred_job = st.selectbox("Job Title", df["Job_Title"].unique(), key="perf_job")
        with r1c:
            pred_edu = st.selectbox("Education Level", df["Education_Level"].unique(), key="perf_edu")
        with r1d:
            pred_gender = st.selectbox("Gender", df["Gender"].unique(), key="perf_gender")
        
        r2a, r2b, r2c = st.columns(3)
        with r2a:
            pred_age = st.number_input("Age", min_value=18, max_value=70, value=35, key="perf_age")
        with r2b:
            pred_years = st.number_input("Years at Company", min_value=0, max_value=50, value=5, key="perf_years")
        with r2c:
            pred_hire_year = st.number_input("Hire Year", min_value=2000, max_value=2024, value=2020, key="perf_hire_year")
        
        st.markdown('<p class="form-section-header">Work & output</p>', unsafe_allow_html=True)
        r3a, r3b, r3c, r3d = st.columns(4)
        with r3a:
            pred_hours = st.number_input("Work Hours/Week", min_value=20, max_value=80, value=40, key="perf_hours")
        with r3b:
            pred_projects = st.number_input("Projects Handled", min_value=0, max_value=100, value=10, key="perf_projects")
        with r3c:
            pred_overtime = st.number_input("Overtime Hours", min_value=0, max_value=100, value=5, key="perf_overtime")
        with r3d:
            pred_sick = st.number_input("Sick Days", min_value=0, max_value=50, value=3, key="perf_sick")
        
        r4a, r4b = st.columns(2)
        with r4a:
            pred_remote = st.number_input("Remote Work %", min_value=0, max_value=100, value=50, key="perf_remote")
        with r4b:
            pred_team = st.number_input("Team Size", min_value=1, max_value=50, value=10, key="perf_team")
        
        st.markdown('<p class="form-section-header">Growth & satisfaction</p>', unsafe_allow_html=True)
        r5a, r5b, r5c, r5d = st.columns(4)
        with r5a:
            pred_salary = st.number_input("Monthly Salary", min_value=1000, max_value=50000, value=5000, key="perf_salary")
        with r5b:
            pred_training = st.number_input("Training Hours", min_value=0, max_value=200, value=20, key="perf_training")
        with r5c:
            pred_promotions = st.number_input("Promotions", min_value=0, max_value=10, value=1, key="perf_promotions")
        with r5d:
            pred_satisfaction = st.number_input("Satisfaction (1–5)", min_value=1.0, max_value=5.0, value=3.5, step=0.1, key="perf_satisfaction")
        
        if st.button("Predict Performance Score", type="primary", use_container_width=True):
            input_data = pd.DataFrame({
                "Department": [pred_dept],
                "Gender": [pred_gender],
                "Job_Title": [pred_job],
                "Education_Level": [pred_edu],
                "Age": [pred_age],
                "Years_At_Company": [pred_years],
                "Monthly_Salary": [pred_salary],
                "Work_Hours_Per_Week": [pred_hours],
                "Projects_Handled": [pred_projects],
                "Overtime_Hours": [pred_overtime],
                "Sick_Days": [pred_sick],
                "Remote_Work_Frequency": [pred_remote],
                "Team_Size": [pred_team],
                "Training_Hours": [pred_training],
                "Promotions": [pred_promotions],
                "Employee_Satisfaction_Score": [pred_satisfaction],
                "Hire_Year": [pred_hire_year]
            })
            
            prediction = perf_model.predict(input_data)[0]
            predicted_score_raw = prediction
            predicted_score = max(1, min(5, round(prediction)))
            
            # Determine performance level
            if predicted_score >= 4.5:
                level = "Excellent"
                emoji = "⭐"
                description = "Outstanding performance, exceeds all expectations"
            elif predicted_score >= 3.5:
                level = "Very Good"
                emoji = "👍"
                description = "Strong performance, consistently meets and exceeds expectations"
            elif predicted_score >= 2.5:
                level = "Good"
                emoji = "✓"
                description = "Meets expectations, satisfactory performance"
            elif predicted_score >= 1.5:
                level = "Needs Improvement"
                emoji = "⚠️"
                description = "Below expectations, requires development"
            else:
                level = "Poor"
                emoji = "❌"
                description = "Significantly below expectations, requires immediate attention"
            
            st.success(f"""
            **{emoji} Predicted Performance Score: {predicted_score_raw:.2f}** (Rounded: **{predicted_score}**)
            
            **Performance Level: {level}**
            
            *{description}*
            """)
            
            st.info(f"""
            **Understanding Performance Scores:**
            - **Score 5**: Excellent - Top performer, exceeds all expectations
            - **Score 4**: Very Good - Strong performance, exceeds expectations  
            - **Score 3**: Good - Meets expectations, satisfactory performance
            - **Score 2**: Needs Improvement - Below expectations, needs development
            - **Score 1**: Poor - Significantly below expectations, requires attention
            """)
    
    # Tab 3: Productivity Evaluation
    elif st.session_state["nav_tab"] == "Productivity":
        st.header("🚀 Productivity Evaluation Model")
        
        # Train model and show metrics
        with st.spinner("Training productivity model..."):
            prod_model, prod_rmse, prod_r2 = train_productivity_model(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model R² Score", f"{prod_r2:.4f}")
        with col2:
            st.metric("RMSE", f"{prod_rmse:.4f}")
        with col3:
            st.metric("Model Type", "Gradient Boosting")
        
        st.markdown("---")
        
        # Productivity Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Productivity by Job Title
            job_prod = filtered_df.groupby("Job_Title")["Productivity_Score"].mean().sort_values(ascending=False)
            fig = px.bar(
                x=job_prod.index,
                y=job_prod.values,
                title="Average Productivity by Job Title",
                labels={"x": "Job Title", "y": "Average Productivity Score"},
                color_discrete_sequence=['#e85a70']
            )
            fig.update_layout(**_chart_layout(400))
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Productivity Trends
            year_prod = filtered_df.groupby("Hire_Year")["Productivity_Score"].mean()
            fig = px.line(
                x=year_prod.index,
                y=year_prod.values,
                title="Productivity Trends by Hire Year",
                labels={"x": "Hire Year", "y": "Average Productivity Score"},
                markers=True,
                color_discrete_sequence=['#2563EB']
            )
            fig.update_layout(**_chart_layout(400))
            st.plotly_chart(fig, use_container_width=True)
        
        # Productivity Prediction
        st.subheader("🔮 Predict Productivity Score")
        st.caption("Productivity is driven by work output, hours, training, and engagement.")
        
        st.markdown("**Profile & role**")
        p1a, p1b, p1c, p1d = st.columns(4)
        with p1a:
            pred_dept_prod = st.selectbox("Department", df["Department"].unique(), key="prod_dept")
        with p1b:
            pred_job_prod = st.selectbox("Job Title", df["Job_Title"].unique(), key="prod_job")
        with p1c:
            pred_edu_prod = st.selectbox("Education Level", df["Education_Level"].unique(), key="prod_edu")
        with p1d:
            pred_gender_prod = st.selectbox("Gender", df["Gender"].unique(), key="prod_gender")
        
        p2a, p2b, p2c = st.columns(3)
        with p2a:
            pred_age_prod = st.number_input("Age", min_value=18, max_value=70, value=35, key="prod_age")
        with p2b:
            pred_years_prod = st.number_input("Years at Company", min_value=0, max_value=50, value=5, key="prod_years")
        with p2c:
            pred_hire_year_prod = st.number_input("Hire Year", min_value=2000, max_value=2024, value=2020, key="prod_hire_year")
        
        st.markdown('<p class="form-section-header">Work & output</p>', unsafe_allow_html=True)
        p3a, p3b, p3c, p3d = st.columns(4)
        with p3a:
            pred_hours_prod = st.number_input("Work Hours/Week", min_value=20, max_value=80, value=40, key="prod_hours")
        with p3b:
            pred_projects_prod = st.number_input("Projects Handled", min_value=0, max_value=100, value=10, key="prod_projects")
        with p3c:
            pred_overtime_prod = st.number_input("Overtime Hours", min_value=0, max_value=100, value=5, key="prod_overtime")
        with p3d:
            pred_sick_prod = st.number_input("Sick Days", min_value=0, max_value=50, value=3, key="prod_sick")
        
        p4a, p4b = st.columns(2)
        with p4a:
            pred_remote_prod = st.number_input("Remote Work %", min_value=0, max_value=100, value=50, key="prod_remote")
        with p4b:
            pred_team_prod = st.number_input("Team Size", min_value=1, max_value=50, value=10, key="prod_team")
        
        st.markdown('<p class="form-section-header">Growth & satisfaction</p>', unsafe_allow_html=True)
        p5a, p5b, p5c, p5d = st.columns(4)
        with p5a:
            pred_salary_prod = st.number_input("Monthly Salary", min_value=1000, max_value=50000, value=5000, key="prod_salary")
        with p5b:
            pred_training_prod = st.number_input("Training Hours", min_value=0, max_value=200, value=20, key="prod_training")
        with p5c:
            pred_promotions_prod = st.number_input("Promotions", min_value=0, max_value=10, value=1, key="prod_promotions")
        with p5d:
            pred_satisfaction_prod = st.number_input("Satisfaction (1–5)", min_value=1.0, max_value=5.0, value=3.5, step=0.1, key="prod_satisfaction")
        
        if st.button("Predict Productivity Score", type="primary", use_container_width=True):
            input_data = pd.DataFrame({
                "Department": [pred_dept_prod],
                "Gender": [pred_gender_prod],
                "Job_Title": [pred_job_prod],
                "Education_Level": [pred_edu_prod],
                "Age": [pred_age_prod],
                "Years_At_Company": [pred_years_prod],
                "Monthly_Salary": [pred_salary_prod],
                "Work_Hours_Per_Week": [pred_hours_prod],
                "Projects_Handled": [pred_projects_prod],
                "Overtime_Hours": [pred_overtime_prod],
                "Sick_Days": [pred_sick_prod],
                "Remote_Work_Frequency": [pred_remote_prod],
                "Team_Size": [pred_team_prod],
                "Training_Hours": [pred_training_prod],
                "Promotions": [pred_promotions_prod],
                "Employee_Satisfaction_Score": [pred_satisfaction_prod],
                "Hire_Year": [pred_hire_year_prod]
            })
            
            prediction = prod_model.predict(input_data)[0]
            predicted_score = max(1, min(100, round(prediction, 2)))
            
            # Determine productivity level
            if predicted_score >= 80:
                level = "Excellent"
                emoji = "🚀"
                description = "Exceptional productivity, high output, minimal sick days"
            elif predicted_score >= 60:
                level = "Very Good"
                emoji = "⭐"
                description = "Strong productivity, good work output"
            elif predicted_score >= 40:
                level = "Good"
                emoji = "👍"
                description = "Average productivity, meets standard expectations"
            elif predicted_score >= 20:
                level = "Below Average"
                emoji = "⚠️"
                description = "Low productivity, needs improvement"
            else:
                level = "Poor"
                emoji = "❌"
                description = "Very low productivity, requires immediate attention"
            
            st.success(f"""
            **{emoji} Predicted Productivity Score: {predicted_score:.2f}**
            
            **Productivity Level: {level}**
            
            *{description}*
            """)
            
            st.info(f"""
            **Understanding Productivity Scores:**
            - **80-100**: Excellent - Exceptional productivity, high output
            - **60-79**: Very Good - Strong productivity, good work output
            - **40-59**: Good - Average productivity, meets expectations
            - **20-39**: Below Average - Low productivity, needs improvement
            - **1-19**: Poor - Very low productivity, requires attention
            
            The score considers projects handled, work hours, overtime, sick days, and training.
            """)
    
    # Tab 4: Employee Insights
    elif st.session_state["nav_tab"] == "Insights":
        st.header("👤 Employee Insights & Analysis")
        
        # Top Performers & Top Producers - with bar charts and tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Performers (Performance)")
            top_performers = filtered_df.nlargest(10, "Performance_Score")[
                ["Employee_ID", "Department", "Job_Title", "Performance_Score", 
                 "Productivity_Score", "Monthly_Salary"]
            ]
            if len(top_performers) > 0:
                # Bar chart: Performance Score by Employee
                top_performers_display = top_performers.copy()
                top_performers_display["Label"] = "ID " + top_performers_display["Employee_ID"].astype(str) + " · " + top_performers_display["Job_Title"]
                fig_perf = px.bar(
                    top_performers_display,
                    x="Label",
                    y="Performance_Score",
                    color="Performance_Score",
                    color_continuous_scale=["#93C5FD", "#2563EB", "#1D4ED8"],
                    labels={"Performance_Score": "Performance Score", "Label": "Employee"},
                    title="Top 10 by Performance Score"
                )
                _lay = {**_chart_layout(350), "showlegend": False, "xaxis_tickangle": -45, "margin": dict(b=120), "coloraxis_showscale": False}
                fig_perf.update_layout(**_lay)
                fig_perf.update_traces(marker_line_color="white", marker_line_width=1)
                st.plotly_chart(fig_perf, use_container_width=True, key="chart_top_performers")
            st.caption("Data table below")
            if len(top_performers) > 0:
                st.table(top_performers.reset_index(drop=True))
            else:
                st.info("No data for current filters. Select **All** in sidebar to see top performers.")
        
        with col2:
            st.subheader("🚀 Top Producers (Productivity)")
            top_producers = filtered_df.nlargest(10, "Productivity_Score")[
                ["Employee_ID", "Department", "Job_Title", "Performance_Score",
                 "Productivity_Score", "Monthly_Salary"]
            ]
            if len(top_producers) > 0:
                # Bar chart: Productivity Score by Employee
                top_producers_display = top_producers.copy()
                top_producers_display["Label"] = "ID " + top_producers_display["Employee_ID"].astype(str) + " · " + top_producers_display["Job_Title"]
                fig_prod = px.bar(
                    top_producers_display,
                    x="Label",
                    y="Productivity_Score",
                    color="Productivity_Score",
                    color_continuous_scale=["#93C5FD", "#2563EB", "#1D4ED8"],
                    labels={"Productivity_Score": "Productivity Score", "Label": "Employee"},
                    title="Top 10 by Productivity Score"
                )
                _lay = {**_chart_layout(350), "showlegend": False, "xaxis_tickangle": -45, "margin": dict(b=120), "coloraxis_showscale": False}
                fig_prod.update_layout(**_lay)
                fig_prod.update_traces(marker_line_color="white", marker_line_width=1)
                st.plotly_chart(fig_prod, use_container_width=True, key="chart_top_producers")
            st.caption("Data table below")
            if len(top_producers) > 0:
                st.table(top_producers.reset_index(drop=True))
            else:
                st.info("No data for current filters. Select **All** in sidebar to see top producers.")
        
        st.markdown("---")
        
        # Detailed Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance vs Salary
            fig = px.scatter(
                filtered_df.sample(min(5000, len(filtered_df))),
                x="Monthly_Salary",
                y="Performance_Score",
                color="Department",
                size="Years_At_Company",
                hover_data=["Job_Title", "Productivity_Score"],
                title="Performance vs Salary",
                labels={"Monthly_Salary": "Monthly Salary ($)", 
                       "Performance_Score": "Performance Score"},
                color_discrete_sequence=['#2563EB', '#e85a70', '#e88a9a', '#93C5FD', '#f5c6ce', '#E2E8F0']
            )
            fig.update_layout(**_chart_layout(400))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Productivity vs Work Hours
            fig = px.scatter(
                filtered_df.sample(min(5000, len(filtered_df))),
                x="Work_Hours_Per_Week",
                y="Productivity_Score",
                color="Department",
                size="Projects_Handled",
                hover_data=["Job_Title", "Overtime_Hours"],
                title="Productivity vs Work Hours",
                labels={"Work_Hours_Per_Week": "Work Hours Per Week", 
                       "Productivity_Score": "Productivity Score"},
                color_discrete_sequence=['#2563EB', '#e85a70', '#e88a9a', '#93C5FD', '#f5c6ce', '#E2E8F0']
            )
            fig.update_layout(**_chart_layout(400))
            st.plotly_chart(fig, use_container_width=True)
        
        # Employee Search
        st.subheader("🔍 Search Employee")
        employee_id = st.number_input("Enter Employee ID", min_value=1, max_value=len(df), value=1)
        
        if st.button("View Employee Details"):
            employee = df[df["Employee_ID"] == employee_id]
            if len(employee) > 0:
                emp = employee.iloc[0]
                emp_prod = filtered_df[filtered_df["Employee_ID"] == employee_id]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Performance Score", f"{emp['Performance_Score']}")
                with col2:
                    if len(emp_prod) > 0:
                        st.metric("Productivity Score", f"{emp_prod.iloc[0]['Productivity_Score']:.2f}")
                with col3:
                    st.metric("Department", emp["Department"])
                with col4:
                    st.metric("Job Title", emp["Job_Title"])
                
                st.dataframe(employee.T, use_container_width=True)
            else:
                st.error("Employee not found!")

    # Traditional professional footer (no divider — footer has border-top)
    st.markdown(
        '<footer class="footer-pro">© 2025 Employee Performance & Productivity. All rights reserved.</footer>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
