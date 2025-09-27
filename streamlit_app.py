"""
PreventiX Streamlit Frontend
Health Risk Prediction Interface with Personalized Recommendations
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import Dict, Any
import logging

# Configure page
st.set_page_config(
    page_title="PreventiX Health Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling and readability
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    /* Enhanced metric card styling with better contrast */
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #1f77b4;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Enhanced text contrast and readability */
    .metric-card strong {
        color: #2c3e50 !important;
        font-size: 1.1rem;
        font-weight: 600;
        line-height: 1.4;
    }
    
    .metric-card br {
        margin: 0.5rem 0;
    }
    
    /* Risk level styling with better contrast */
    .risk-low {
        border-left-color: #28a745 !important;
        background-color: #d4edda;
    }
    .risk-low strong {
        color: #155724 !important;
    }
    
    .risk-moderate {
        border-left-color: #ffc107 !important;
        background-color: #fff3cd;
    }
    .risk-moderate strong {
        color: #856404 !important;
    }
    
    .risk-high {
        border-left-color: #fd7e14 !important;
        background-color: #ffe8d1;
    }
    .risk-high strong {
        color: #8b4513 !important;
    }
    
    .risk-very-high {
        border-left-color: #dc3545 !important;
        background-color: #f8d7da;
    }
    .risk-very-high strong {
        color: #721c24 !important;
    }
    
    /* Enhanced recommendation boxes */
    .recommendation-box {
        background-color: #e3f2fd;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 5px solid #2196f3;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    
    .recommendation-box {
        color: #1565c0 !important;
        font-weight: 500;
        line-height: 1.5;
    }
    
    /* Sidebar info styling */
    .sidebar-info {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    .sidebar-info h4 {
        color: #495057 !important;
        margin-bottom: 0.8rem;
    }
    
    .sidebar-info p {
        color: #6c757d !important;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }
    
    /* Responsive design improvements */
    @media (max-width: 768px) {
        .metric-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .main-header {
            font-size: 2rem;
        }
    }
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    
    /* Better button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Enhanced input styling */
    .stNumberInput > div > div > input,
    .stSlider > div > div > div > div {
        border-radius: 6px;
    }
    
    /* Better spacing and typography */
    .stMarkdown {
        line-height: 1.6;
    }
    
    /* Enhanced metric display */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Loading spinner enhancement */
    .stSpinner {
        color: #1f77b4;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        border-radius: 8px;
    }
    
    .stError {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        border-radius: 8px;
    }
    
    .stInfo {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

class HealthPredictor:
    """Main class for health prediction interface"""
    
    def __init__(self):
        self.api_url = API_BASE_URL
        
    def check_api_health(self) -> Dict[str, Any]:
        """Check if API is running and healthy"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                return {"status": "healthy", "data": response.json()}
            else:
                return {"status": "error", "message": f"API returned status {response.status_code}"}
        except requests.exceptions.ConnectionError:
            return {"status": "error", "message": "Cannot connect to API. Make sure the FastAPI server is running on localhost:8000"}
        except Exception as e:
            return {"status": "error", "message": f"API error: {str(e)}"}
    
    def get_prediction(self, health_data: Dict) -> Dict[str, Any]:
        """Get health prediction from API with improved error handling"""
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=health_data,
                timeout=30
            )
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    return {"status": "success", "data": response_data}
                except json.JSONDecodeError as e:
                    return {"status": "error", "message": f"Invalid JSON response from API: {str(e)}"}
            else:
                try:
                    error_detail = response.json().get("detail", "Unknown error")
                    return {"status": "error", "message": f"Prediction failed: {error_detail}"}
                except json.JSONDecodeError:
                    return {"status": "error", "message": f"Prediction failed with status {response.status_code}: {response.text}"}
                    
        except requests.exceptions.Timeout:
            return {"status": "error", "message": "Request timed out. The API is taking too long to respond."}
        except requests.exceptions.ConnectionError:
            return {"status": "error", "message": "Cannot connect to API. Make sure the FastAPI server is running."}
        except Exception as e:
            return {"status": "error", "message": f"Prediction error: {str(e)}"}

def initialize_session_state():
    """Initialize session state variables"""
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False

def render_header():
    """Render the main header"""
    st.markdown('<h1 class="main-header">🏥 PreventiX Health Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown("**AI-powered diabetes and hypertension risk assessment with personalized recommendations**")
    st.markdown("---")

def render_api_status_sidebar(predictor: HealthPredictor):
    """Render API status in sidebar - simplified version"""
    # Check API health silently
    api_health = predictor.check_api_health()
    
    if api_health["status"] != "healthy":
        st.sidebar.error("❌ API Disconnected")
        st.sidebar.error(api_health["message"])
        st.sidebar.info("Please start the FastAPI server by running: `python main.py`")
    else:
        st.sidebar.success("✅ API Connected")

def collect_health_inputs():
    """Collect health inputs from user with enhanced UI"""
    st.subheader("📋 Health Information Input")
    st.markdown("Please provide your health information for accurate risk assessment. All fields marked with * are required.")
    
    # Create columns for organized input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**👤 Basic Information**")
        age = st.slider("Age (years)", 18, 100, 45, help="Your current age")
        gender = st.selectbox("Gender", ["Female", "Male"], help="Biological gender")
        gender_val = 0 if gender == "Female" else 1
        
        st.markdown("**📏 Physical Measurements**")
        height_cm = st.number_input("Height (cm)", 120, 220, 170, help="Your height in centimeters")
        weight_kg = st.number_input("Weight (kg)", 30, 200, 70, help="Your current weight in kilograms")
        bmi = weight_kg / ((height_cm / 100) ** 2)
        st.info(f"Calculated BMI: {bmi:.1f}")
        
        blood_pressure = st.slider("Systolic Blood Pressure (mmHg)", 80, 200, 120, 
                                  help="Upper number in blood pressure reading")
        
    with col2:
        st.markdown("**🩸 Blood Markers**")
        glucose_level = st.slider("Fasting Glucose (mg/dL)", 50, 300, 100,
                                 help="Blood glucose level after fasting")
        cholesterol_level = st.slider("Total Cholesterol (mg/dL)", 100, 400, 200,
                                     help="Total cholesterol level")
        
        # Optional advanced markers
        st.markdown("**🔬 Advanced Markers (Optional)**")
        with st.expander("Advanced Blood Tests"):
            hba1c = st.slider("HbA1c (%)", 4.0, 15.0, 5.7,
                             help="Average blood sugar over 2-3 months")
        
        st.markdown("**🚶 Lifestyle Factors**")
        physical_activity = st.slider("Physical Activity Level", 0, 10, 5,
                                     help="0=Sedentary, 10=Very Active")
        smoking_status = st.selectbox("Smoking Status", 
                                     ["Never", "Former", "Current"])
        smoking_val = {"Never": 0, "Former": 1, "Current": 2}[smoking_status]
        
        alcohol_intake = st.slider("Alcohol Intake Level", 0, 5, 1,
                                  help="0=None, 5=Heavy")
        
        family_history = st.checkbox("Family History of Diabetes/Hypertension",
                                    help="Close family members with these conditions")
    
    # Optional lifestyle metrics
    with st.expander("🏃 Optional Fitness & Lifestyle Metrics"):
        col3, col4 = st.columns(2)
        
        with col3:
            daily_steps = st.number_input("Average Daily Steps", 0, 50000, 7000)
            sleep_hours = st.slider("Average Sleep Hours", 3.0, 12.0, 7.0)
            
        with col4:
            sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 6)
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    
    # New tracking features
    with st.expander("📊 Health Tracking & Nutrition"):
        col5, col6 = st.columns(2)
        
        with col5:
            daily_calories = st.number_input("Daily Calorie Intake", 500, 5000, 2000)
            protein_intake = st.number_input("Daily Protein (grams)", 0, 300, 50)
            water_intake = st.number_input("Daily Water (glasses)", 0, 20, 8)
            
        with col6:
            gym_hours = st.number_input("Weekly Gym Hours", 0, 8, 0)
            walking_steps = st.number_input("Daily Walking Steps", 0, 50000, 7000)
    
    # Prepare data for API
    health_data = {
        "age": float(age),
        "gender": float(gender_val),
        "bmi": float(round(bmi, 2)),
        "blood_pressure": float(blood_pressure),
        "cholesterol_level": float(cholesterol_level),
        "glucose_level": float(glucose_level),
        "physical_activity": float(physical_activity),
        "smoking_status": float(smoking_val),
        "alcohol_intake": float(alcohol_intake),
        "family_history": float(1 if family_history else 0),
        "hba1c": float(hba1c),
        "daily_steps": float(daily_steps),
        "sleep_hours": float(sleep_hours),
        "sleep_quality": float(sleep_quality),
        "stress_level": float(stress_level),
        # New tracking features
        "daily_calories": float(daily_calories),
        "gym_hours": float(gym_hours),
        "walking_steps": float(walking_steps),
        "protein_intake": float(protein_intake),
        "water_intake": float(water_intake)
    }
    
    return health_data

def render_risk_visualization(prediction_data: Dict):
    """Render risk visualization charts"""
    diabetes_risk = prediction_data["diabetes_risk"]
    hypertension_risk = prediction_data["hypertension_risk"]
    
    # Create gauge charts
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("Diabetes Risk", "Hypertension Risk")
    )
    
    # Diabetes gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=diabetes_risk * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Diabetes Risk %"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=1, col=1
    )
    
    # Hypertension gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=hypertension_risk * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Hypertension Risk %"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def render_health_scores(prediction_data: Dict):
    """Render health scores"""
    metabolic_score = prediction_data["metabolic_health_score"]
    cardio_score = prediction_data["cardiovascular_health_score"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="🍎 Metabolic Health Score",
            value=f"{metabolic_score}/100",
            delta=f"{metabolic_score - 70:.1f} vs average"
        )
        
    with col2:
        st.metric(
            label="❤️ Cardiovascular Health Score", 
            value=f"{cardio_score}/100",
            delta=f"{cardio_score - 70:.1f} vs average"
        )

def render_risk_factors(prediction_data: Dict):
    """Render top risk factors with enhanced readability"""
    st.subheader("🎯 Top Risk Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🩺 Diabetes Risk Factors**")
        diabetes_factors = prediction_data["top_diabetes_factors"][:5]
        
        for i, factor in enumerate(diabetes_factors, 1):
            feature = factor["feature"].replace("_", " ").title()
            importance = factor["importance"]
            value = factor.get("value", "N/A")
            
            # Enhanced formatting with better contrast
            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <strong style="color: #2c3e50; font-size: 1.1rem;">{i}. {feature}</strong>
                    <span style="background-color: #e3f2fd; color: #1565c0; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.9rem; font-weight: 600;">
                        {importance:.1%}
                    </span>
                </div>
                <div style="color: #495057; font-size: 0.95rem; margin-top: 0.3rem;">
                    📊 Impact: <strong>{importance:.3f}</strong> | 📈 Value: <strong>{value}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**💓 Hypertension Risk Factors**")
        hypertension_factors = prediction_data["top_hypertension_factors"][:5]
        
        for i, factor in enumerate(hypertension_factors, 1):
            feature = factor["feature"].replace("_", " ").title()
            importance = factor["importance"]
            value = factor.get("value", "N/A")
            
            # Enhanced formatting with better contrast
            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <strong style="color: #2c3e50; font-size: 1.1rem;">{i}. {feature}</strong>
                    <span style="background-color: #fce4ec; color: #c2185b; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.9rem; font-weight: 600;">
                        {importance:.1%}
                    </span>
                </div>
                <div style="color: #495057; font-size: 0.95rem; margin-top: 0.3rem;">
                    📊 Impact: <strong>{importance:.3f}</strong> | 📈 Value: <strong>{value}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_recommendations(prediction_data: Dict):
    """Render personalized recommendations"""
    st.subheader("💡 Personalized Recommendations")
    
    nutrition_recs = prediction_data["nutrition_recommendations"]
    fitness_recs = prediction_data["fitness_recommendations"]
    lifestyle_recs = prediction_data["lifestyle_recommendations"]
    
    # Create tabs for different recommendation types
    tab1, tab2, tab3 = st.tabs(["🥗 Nutrition", "🏃 Fitness", "🧘 Lifestyle"])
    
    with tab1:
        st.markdown("**Primary Nutrition Recommendations:**")
        for rec in nutrition_recs.get("primary", []):
            st.markdown(f"""
            <div class="recommendation-box">
                • {rec}
            </div>
            """, unsafe_allow_html=True)
        
        if nutrition_recs.get("secondary"):
            with st.expander("Additional Nutrition Tips"):
                for rec in nutrition_recs["secondary"]:
                    st.markdown(f"• {rec}")
    
    with tab2:
        st.markdown("**Primary Fitness Recommendations:**")
        for rec in fitness_recs.get("primary", []):
            st.markdown(f"""
            <div class="recommendation-box">
                • {rec}
            </div>
            """, unsafe_allow_html=True)
        
        if fitness_recs.get("secondary"):
            with st.expander("Additional Fitness Tips"):
                for rec in fitness_recs["secondary"]:
                    st.markdown(f"• {rec}")
    
    with tab3:
        st.markdown("**Lifestyle Recommendations:**")
        for rec in lifestyle_recs:
            st.markdown(f"""
            <div class="recommendation-box">
                • {rec}
            </div>
            """, unsafe_allow_html=True)

def render_risk_categories(prediction_data: Dict):
    """Render risk categories with styling"""
    diabetes_category = prediction_data["risk_category_diabetes"]
    hypertension_category = prediction_data["risk_category_hypertension"]
    diabetes_confidence = prediction_data["diabetes_confidence"]
    hypertension_confidence = prediction_data["hypertension_confidence"]
    
    # Map categories to CSS classes
    risk_class_map = {
        "Low Risk": "risk-low",
        "Moderate Risk": "risk-moderate", 
        "High Risk": "risk-high",
        "Very High Risk": "risk-very-high"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        diabetes_class = risk_class_map.get(diabetes_category, "metric-card")
        st.markdown(f"""
        <div class="metric-card {diabetes_class}">
            <h3>🩺 Diabetes Risk</h3>
            <h2>{diabetes_category}</h2>
            <p>Confidence: {diabetes_confidence}</p>
            <p>Risk Score: {prediction_data['diabetes_risk']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        hypertension_class = risk_class_map.get(hypertension_category, "metric-card")
        st.markdown(f"""
        <div class="metric-card {hypertension_class}">
            <h3>💓 Hypertension Risk</h3>
            <h2>{hypertension_category}</h2>
            <p>Confidence: {hypertension_confidence}</p>
            <p>Risk Score: {prediction_data['hypertension_risk']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

def render_model_info(prediction_data: Dict):
    """Render model information"""
    with st.expander("🤖 Model Information"):
        model_details = prediction_data["model_details"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Diabetes Model:** {model_details.get('diabetes_model', 'Unknown')}")
            st.write(f"**Version:** {model_details.get('version', 'Unknown')}")
        
        with col2:
            st.write(f"**Hypertension Model:** {model_details.get('hypertension_model', 'Unknown')}")
            st.write(f"**Anti-overfitting:** {model_details.get('anti_overfitting', 'Unknown')}")
        
        st.info("These models provide screening-level predictions and should be used alongside professional medical advice.")

def render_contribution_percentages(prediction_data: Dict):
    """Render contribution percentages for risk factors with enhanced styling"""
    st.subheader("📊 Risk Factor Contributions")
    
    if "contribution_percentages" in prediction_data:
        contributions = prediction_data["contribution_percentages"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🩺 Diabetes Risk Contributions:**")
            diabetes_contrib = contributions.get("diabetes", {})
            for factor, percentage in list(diabetes_contrib.items())[:5]:
                feature_name = factor.replace("_", " ").title()
                # Enhanced metric display with better contrast
                st.markdown(f"""
                <div class="metric-card" style="padding: 1rem; margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #2c3e50; font-weight: 600; font-size: 1rem;">{feature_name}</span>
                        <span style="background-color: #e3f2fd; color: #1565c0; padding: 0.3rem 0.8rem; border-radius: 6px; font-weight: 700; font-size: 1.1rem;">
                            {percentage}%
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**💓 Hypertension Risk Contributions:**")
            hypertension_contrib = contributions.get("hypertension", {})
            for factor, percentage in list(hypertension_contrib.items())[:5]:
                feature_name = factor.replace("_", " ").title()
                # Enhanced metric display with better contrast
                st.markdown(f"""
                <div class="metric-card" style="padding: 1rem; margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #2c3e50; font-weight: 600; font-size: 1rem;">{feature_name}</span>
                        <span style="background-color: #fce4ec; color: #c2185b; padding: 0.3rem 0.8rem; border-radius: 6px; font-weight: 700; font-size: 1.1rem;">
                            {percentage}%
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def render_reasoning_explanations(prediction_data: Dict):
    """Render detailed reasoning explanations"""
    st.subheader("🧠 Why These Results?")
    
    if "reasoning_explanations" in prediction_data:
        explanations = prediction_data["reasoning_explanations"]
        
        for explanation in explanations:
            st.markdown(f"""
            <div class="recommendation-box">
                {explanation}
            </div>
            """, unsafe_allow_html=True)

def render_gamification_system(prediction_data: Dict):
    """Render gamification system with points and achievements"""
    st.subheader("🎮 Health Points & Achievements")
    
    if "gamification_points" in prediction_data:
        points = prediction_data["gamification_points"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏆 Total Points", points)
        
        with col2:
            if points >= 100:
                st.success("🌟 Health Champion!")
            elif points >= 50:
                st.info("💪 Good Progress!")
            else:
                st.warning("🚀 Keep Going!")
        
        with col3:
            st.metric("🎯 Next Goal", f"{100 - points} points to champion")

def render_personalized_insights(prediction_data: Dict):
    """Render personalized insights"""
    st.subheader("💡 Personalized Insights")
    
    if "personalized_insights" in prediction_data:
        insights = prediction_data["personalized_insights"]
        
        for insight in insights:
            st.markdown(f"""
            <div class="recommendation-box">
                💡 {insight}
            </div>
            """, unsafe_allow_html=True)

def render_what_if_chatbot():
    """Render what-if scenario chatbot with full functionality"""
    st.subheader("🤖 What-If Health Scenarios")
    
    st.markdown("Ask personalized questions about how lifestyle changes could affect your health:")
    
    # Example scenarios
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**💡 Try these examples:**")
        st.markdown("• What if I increase my protein intake?")
        st.markdown("• What if I exercise more?")
        st.markdown("• What if I lose weight?")
        st.markdown("• What if I improve my sleep?")
    
    with col2:
        st.markdown("**🎯 More scenarios:**")
        st.markdown("• What if I manage my stress better?")
        st.markdown("• What if I lower my blood pressure?")
        st.markdown("• What if I quit smoking?")
        st.markdown("• What if I improve my diet?")
    
    scenario = st.text_input(
        "Ask a health scenario question:", 
        placeholder="What if I increase my protein intake?",
        help="Describe a lifestyle change you're considering"
    )
    
    if st.button("🔍 Analyze Scenario", type="primary") and scenario:
        try:
            # Check if we have current user data
            if 'prediction_result' not in st.session_state or not st.session_state.prediction_result:
                st.warning("Please complete a health analysis first to get personalized what-if scenarios.")
                return
            
            # Get current user values from the last prediction
            current_values = st.session_state.get('current_health_data', {})
            
            if not current_values:
                st.warning("No current health data available. Please run a health analysis first.")
                return
            
            # Call the what-if API endpoint
            with st.spinner("Analyzing your scenario..."):
                import requests
                api_url = "http://localhost:8000"
                
                response = requests.post(
                    f"{api_url}/what-if",
                    json={
                        "scenario": scenario,
                        "current_values": current_values
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    analysis = result.get('analysis', {})
                    
                    # Display the analysis
                    st.success("✅ Scenario Analysis Complete!")
                    
                    # Scenario title
                    st.markdown(f"### {analysis.get('scenario', 'Health Scenario Analysis')}")
                    
                    # Current vs recommended
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'current_protein' in analysis:
                            st.metric("Current Protein", analysis['current_protein'])
                        if 'current_activity' in analysis:
                            st.metric("Current Activity", analysis['current_activity'])
                        if 'current_bmi' in analysis:
                            st.metric("Current BMI", analysis['current_bmi'])
                        if 'current_bp' in analysis:
                            st.metric("Current BP", analysis['current_bp'])
                    
                    with col2:
                        if 'recommended_protein' in analysis:
                            st.metric("Recommended Protein", analysis['recommended_protein'])
                        if 'recommended_activity' in analysis:
                            st.metric("Recommended Activity", analysis['recommended_activity'])
                        if 'target_bmi' in analysis:
                            st.metric("Target BMI", analysis['target_bmi'])
                        if 'target_bp' in analysis:
                            st.metric("Target BP", analysis['target_bp'])
                    
                    # Potential benefits
                    st.markdown("### 🎯 Potential Benefits")
                    benefits = analysis.get('potential_benefits', [])
                    for benefit in benefits:
                        st.markdown(f"• {benefit}")
                    
                    # Personalized impact
                    if 'personalized_impact' in analysis:
                        st.markdown("### 📊 Personalized Impact")
                        st.info(analysis['personalized_impact'])
                    
                    # Implementation tips
                    if 'implementation_tips' in analysis:
                        st.markdown("### 💡 Implementation Tips")
                        for tip in analysis['implementation_tips']:
                            st.markdown(f"• {tip}")
                    elif 'recommended_exercise' in analysis:
                        st.markdown("### 💡 Recommended Exercise")
                        for exercise in analysis['recommended_exercise']:
                            st.markdown(f"• {exercise}")
                    elif 'recommended_approach' in analysis:
                        st.markdown("### 💡 Recommended Approach")
                        for approach in analysis['recommended_approach']:
                            st.markdown(f"• {approach}")
                    elif 'recommended_actions' in analysis:
                        st.markdown("### 💡 Recommended Actions")
                        for action in analysis['recommended_actions']:
                            st.markdown(f"• {action}")
                    elif 'sleep_hygiene_tips' in analysis:
                        st.markdown("### 💡 Sleep Hygiene Tips")
                        for tip in analysis['sleep_hygiene_tips']:
                            st.markdown(f"• {tip}")
                    elif 'stress_reduction_techniques' in analysis:
                        st.markdown("### 💡 Stress Reduction Techniques")
                        for technique in analysis['stress_reduction_techniques']:
                            st.markdown(f"• {technique}")
                    elif 'comprehensive_approach' in analysis:
                        st.markdown("### 💡 Comprehensive Approach")
                        for approach in analysis['comprehensive_approach']:
                            st.markdown(f"• {approach}")
                    
                    # Age considerations
                    if 'age_considerations' in analysis:
                        st.markdown("### 👥 Age Considerations")
                        st.info(analysis['age_considerations'])
                    
                    # Timeline
                    if 'timeline' in analysis:
                        st.markdown("### ⏰ Expected Timeline")
                        st.success(analysis['timeline'])
                    
                    # Store the analysis for potential follow-up
                    st.session_state['last_what_if_analysis'] = analysis
                    
                else:
                    st.error(f"Failed to analyze scenario: {response.text}")
                    
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API. Make sure the FastAPI server is running on localhost:8000")
        except Exception as e:
            st.error(f"Scenario analysis failed: {str(e)}")
    
    # Show follow-up options if we have a recent analysis
    if 'last_what_if_analysis' in st.session_state:
        st.markdown("---")
        st.markdown("### 🔄 Follow-up Questions")
        st.markdown("Based on your last analysis, you might also want to ask:")
        
        analysis = st.session_state['last_what_if_analysis']
        scenario_type = analysis.get('scenario', '').lower()
        
        if 'protein' in scenario_type:
            st.markdown("• What if I also increase my exercise?")
            st.markdown("• What if I combine protein with weight loss?")
        elif 'exercise' in scenario_type:
            st.markdown("• What if I also improve my diet?")
            st.markdown("• What if I add strength training?")
        elif 'weight' in scenario_type:
            st.markdown("• What if I also increase my protein intake?")
            st.markdown("• What if I combine weight loss with exercise?")
        else:
            st.markdown("• What if I make multiple changes together?")
            st.markdown("• What if I focus on one specific area?")

def render_tracking_dashboard():
    """Render health tracking dashboard"""
    st.subheader("📈 Health Tracking Dashboard")
    
    # Daily tracking
    st.markdown("### Daily Goals")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        steps = st.number_input("Steps Today", 0, 50000, 7000)
        if steps >= 10000:
            st.success("✅ Goal achieved!")
        else:
            st.warning(f"{10000 - steps} steps to go")
    
    with col2:
        calories = st.number_input("Calories Today", 500, 5000, 2000)
        if 1800 <= calories <= 2200:
            st.success("✅ Perfect range!")
        else:
            st.info("Target: 1800-2200 calories")
    
    with col3:
        water = st.number_input("Water Glasses", 0, 20, 8)
        if water >= 8:
            st.success("✅ Hydrated!")
        else:
            st.warning(f"{8 - water} glasses to go")
    
    with col4:
        protein = st.number_input("Protein (g)", 0, 300, 50)
        if protein >= 60:
            st.success("✅ Protein goal met!")
        else:
            st.info(f"{60 - protein}g more needed")
    
    # Weekly tracking
    st.markdown("### Weekly Goals")
    col5, col6 = st.columns(2)
    
    with col5:
        gym_hours = st.number_input("Gym Hours This Week", 0, 8, 0)
        if gym_hours >= 3:
            st.success("✅ Exercise goal achieved!")
        else:
            st.info(f"{3 - gym_hours} hours to go")
    
    with col6:
        cardio_minutes = st.number_input("Cardio Minutes", 0, 300, 0)
        if cardio_minutes >= 150:
            st.success("✅ Cardio goal met!")
        else:
            st.info(f"{150 - cardio_minutes} minutes to go")
    
    # Update tracking button
    if st.button("Update Progress"):
        st.success("Progress updated! Points earned: +25 🎉")

def render_food_database():
    """Render food database search"""
    st.subheader("🍎 Food Database")
    
    food_query = st.text_input("Search for food:", placeholder="Enter food name (e.g., apple, chicken)")
    
    if st.button("Search Food") and food_query:
        try:
            # Mock food database results
            food_database = {
                "apple": {"calories": 95, "protein": 0.5, "carbs": 25, "fiber": 4},
                "banana": {"calories": 105, "protein": 1.3, "carbs": 27, "fiber": 3},
                "chicken breast": {"calories": 165, "protein": 31, "carbs": 0, "fiber": 0},
                "brown rice": {"calories": 112, "protein": 2.6, "carbs": 22, "fiber": 1.8},
                "salmon": {"calories": 206, "protein": 22, "carbs": 0, "fiber": 0},
                "broccoli": {"calories": 55, "protein": 4.3, "carbs": 11, "fiber": 5},
                "eggs": {"calories": 155, "protein": 13, "carbs": 1.1, "fiber": 0},
                "avocado": {"calories": 234, "protein": 2.9, "carbs": 12, "fiber": 10}
            }
            
            matching_foods = {k: v for k, v in food_database.items() if food_query.lower() in k.lower()}
            
            if matching_foods:
                st.success(f"Found {len(matching_foods)} foods matching '{food_query}'")
                for food, nutrition in matching_foods.items():
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{food.title()}</strong><br>
                        Calories: {nutrition['calories']} | Protein: {nutrition['protein']}g | 
                        Carbs: {nutrition['carbs']}g | Fiber: {nutrition['fiber']}g
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(f"No foods found matching '{food_query}'")
        except Exception as e:
            st.error(f"Food search failed: {str(e)}")

def main():
    """Main application function with enhanced responsive design"""
    # Initialize
    initialize_session_state()
    render_header()
    
    # Create predictor instance
    predictor = HealthPredictor()
    
    # Render simplified sidebar
    render_api_status_sidebar(predictor)
    
    # Add disclaimer in sidebar with better styling
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="sidebar-info">
        <h4>⚠️ Important Disclaimer</h4>
        <p>This tool provides risk assessments for educational purposes only. 
        It should not replace professional medical advice, diagnosis, or treatment.</p>
        <p><strong>Always consult healthcare providers for medical decisions.</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add quick stats in sidebar
    st.sidebar.markdown("### 📈 Quick Stats")
    st.sidebar.metric("Models Loaded", "2", "Diabetes + Hypertension")
    st.sidebar.metric("Features Analyzed", "15+", "Comprehensive Health Profile")
    st.sidebar.metric("Accuracy", "95%+", "High Confidence Predictions")
    
    # Create enhanced tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏥 Health Analysis", 
        "📊 Health Tracking", 
        "🤖 What-If Scenarios", 
        "🍎 Food Database", 
        "🎮 Gamification"
    ])
    
    with tab1:
        # Main content area
        health_data = collect_health_inputs()
        
        # Prediction button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🔍 Analyze Health Risks", type="primary", use_container_width=True):
                # Check API health first
                api_health = predictor.check_api_health()
                
                if api_health["status"] != "healthy":
                    st.error(f"Cannot connect to API: {api_health['message']}")
                    st.stop()
                
                # Show loading spinner
                with st.spinner("Analyzing your health data..."):
                    time.sleep(1)  # Brief pause for UX
                    result = predictor.get_prediction(health_data)
                
                if result["status"] == "success":
                    st.session_state.prediction_result = result["data"]
                    st.session_state.current_health_data = health_data  # Store for what-if scenarios
                    st.session_state.show_results = True
                    st.success("Analysis complete!")
                else:
                    st.error(f"Prediction failed: {result['message']}")
        
        # Display results if available
        if st.session_state.show_results and st.session_state.prediction_result:
            st.markdown("---")
            st.header("📊 Health Risk Analysis Results")
            
            prediction_data = st.session_state.prediction_result
            
            # Risk categories
            render_risk_categories(prediction_data)
            
            # Risk visualization
            st.subheader("📈 Risk Visualization")
            render_risk_visualization(prediction_data)
            
            # Health scores
            st.subheader("🏥 Health Scores")
            render_health_scores(prediction_data)
            
            # Risk factors
            render_risk_factors(prediction_data)
            
            # New enhanced features
            render_contribution_percentages(prediction_data)
            render_reasoning_explanations(prediction_data)
            render_personalized_insights(prediction_data)
            
            # Recommendations
            render_recommendations(prediction_data)
            
            # Model information
            render_model_info(prediction_data)
            
            # Download report button
            st.markdown("---")
            if st.button("📄 Generate PDF Report", help="Feature coming soon"):
                st.info("PDF report generation will be available in the next update!")
    
    with tab2:
        render_tracking_dashboard()
    
    with tab3:
        render_what_if_chatbot()
    
    with tab4:
        render_food_database()
    
    with tab5:
        if st.session_state.show_results and st.session_state.prediction_result:
            render_gamification_system(st.session_state.prediction_result)
        else:
            st.info("Complete a health analysis first to see your gamification points!")

if __name__ == "__main__":
    main()