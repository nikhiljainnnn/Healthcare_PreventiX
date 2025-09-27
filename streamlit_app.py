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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .risk-low {
        border-left-color: #28a745 !important;
        background-color: #d4edda;
    }
    .risk-moderate {
        border-left-color: #ffc107 !important;
        background-color: #fff3cd;
    }
    .risk-high {
        border-left-color: #fd7e14 !important;
        background-color: #ffe8d1;
    }
    .risk-very-high {
        border-left-color: #dc3545 !important;
        background-color: #f8d7da;
    }
    .recommendation-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .sidebar-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
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
    """Render API status in sidebar"""
    st.sidebar.markdown("### 🔧 System Status")
    
    with st.sidebar.container():
        api_health = predictor.check_api_health()
        
        if api_health["status"] == "healthy":
            st.sidebar.success("✅ API Connected")
            api_data = api_health["data"]
            
            with st.sidebar.expander("API Details"):
                st.write(f"**Status:** {api_data.get('status', 'Unknown')}")
                st.write(f"**Models Loaded:** {'Yes' if api_data.get('models_loaded') else 'No'}")
                st.write(f"**Features:** {api_data.get('features_count', 0)}")
                
                model_types = api_data.get('model_types', {})
                if model_types:
                    st.write("**Model Types:**")
                    st.write(f"- Diabetes: {model_types.get('diabetes', 'Unknown')}")
                    st.write(f"- Hypertension: {model_types.get('hypertension', 'Unknown')}")
        else:
            st.sidebar.error("❌ API Disconnected")
            st.sidebar.error(api_health["message"])
            st.sidebar.info("Please start the FastAPI server by running: `python main.py`")

def collect_health_inputs():
    """Collect health inputs from user"""
    st.subheader("📋 Health Information Input")
    
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
        "stress_level": float(stress_level)
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
    """Render top risk factors"""
    st.subheader("🎯 Top Risk Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Diabetes Risk Factors**")
        diabetes_factors = prediction_data["top_diabetes_factors"][:5]
        
        for i, factor in enumerate(diabetes_factors, 1):
            feature = factor["feature"].replace("_", " ").title()
            importance = factor["importance"]
            value = factor.get("value", "N/A")
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>{i}. {feature}</strong><br>
                Impact: {importance:.3f} | Value: {value}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Hypertension Risk Factors**")
        hypertension_factors = prediction_data["top_hypertension_factors"][:5]
        
        for i, factor in enumerate(hypertension_factors, 1):
            feature = factor["feature"].replace("_", " ").title()
            importance = factor["importance"]
            value = factor.get("value", "N/A")
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>{i}. {feature}</strong><br>
                Impact: {importance:.3f} | Value: {value}
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

def main():
    """Main application function"""
    # Initialize
    initialize_session_state()
    render_header()
    
    # Create predictor instance
    predictor = HealthPredictor()
    
    # Render sidebar
    render_api_status_sidebar(predictor)
    
    # Add disclaimer in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="sidebar-info">
        <h4>⚠️ Important Disclaimer</h4>
        <p>This tool provides risk assessments for educational purposes only. 
        It should not replace professional medical advice, diagnosis, or treatment.</p>
        <p><strong>Always consult healthcare providers for medical decisions.</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
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
        
        # Recommendations
        render_recommendations(prediction_data)
        
        # Model information
        render_model_info(prediction_data)
        
        # Download report button
        st.markdown("---")
        if st.button("📄 Generate PDF Report", help="Feature coming soon"):
            st.info("PDF report generation will be available in the next update!")

if __name__ == "__main__":
    main()