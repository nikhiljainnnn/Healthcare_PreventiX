from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import json
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PreventiX Advanced API - Optimized",
    description="AI-powered health risk prediction with personalized recommendations (Anti-overfitting optimized)",
    version="2.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and preprocessors
diabetes_model = None
hypertension_model = None
model_features = None
feature_scaler = None
diabetes_explainer = None
hypertension_explainer = None
nutrition_recommendations = None
fitness_recommendations = None

class HealthInput(BaseModel):
    """Comprehensive health input model - Updated for optimized features"""
    # Basic demographics
    age: float = Field(..., ge=0, le=120, description="Age in years")
    gender: float = Field(..., ge=0, le=1, description="Gender (0=Female, 1=Male)")
    
    # Vital signs and measurements
    bmi: float = Field(..., ge=10, le=60, description="Body Mass Index")
    blood_pressure: float = Field(..., ge=80, le=200, description="Systolic blood pressure (mmHg)")
    
    # Blood markers
    cholesterol_level: float = Field(..., ge=100, le=400, description="Total cholesterol (mg/dL)")
    glucose_level: float = Field(..., ge=50, le=300, description="Fasting glucose (mg/dL)")
    
    # Lifestyle factors
    physical_activity: float = Field(..., ge=0, le=10, description="Physical activity level (0-10)")
    smoking_status: float = Field(..., ge=0, le=2, description="Smoking (0=Never, 1=Former, 2=Current)")
    alcohol_intake: float = Field(..., ge=0, le=5, description="Alcohol intake level (0-5)")
    
    # Family history
    family_history: float = Field(..., ge=0, le=1, description="Family history of diabetes/hypertension")
    
    # Optional advanced blood markers
    hba1c: Optional[float] = Field(None, ge=4, le=15, description="HbA1c percentage")
    fasting_glucose: Optional[float] = Field(None, ge=50, le=300, description="Alternative fasting glucose measurement")
    
    # Optional fitness and lifestyle metrics
    daily_steps: Optional[float] = Field(7000, ge=0, le=50000, description="Average daily steps")
    sleep_hours: Optional[float] = Field(7, ge=0, le=12, description="Average sleep hours per night")
    sleep_quality: Optional[float] = Field(6, ge=0, le=10, description="Sleep quality rating (0-10)")
    stress_level: Optional[float] = Field(5, ge=0, le=10, description="Stress level rating (0-10)")
    
    # New tracking features
    daily_calories: Optional[float] = Field(2000, ge=500, le=5000, description="Daily calorie intake")
    gym_hours: Optional[float] = Field(0, ge=0, le=8, description="Weekly gym/workout hours")
    walking_steps: Optional[float] = Field(7000, ge=0, le=50000, description="Daily walking steps")
    protein_intake: Optional[float] = Field(50, ge=0, le=300, description="Daily protein intake in grams")
    water_intake: Optional[float] = Field(8, ge=0, le=20, description="Daily water intake in glasses")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 45,
                "gender": 1,
                "bmi": 28.5,
                "blood_pressure": 135,
                "cholesterol_level": 210,
                "glucose_level": 110,
                "physical_activity": 3,
                "smoking_status": 0,
                "alcohol_intake": 1,
                "family_history": 1,
                "hba1c": 6.2,
                "daily_steps": 6000,
                "sleep_hours": 6.5,
                "stress_level": 7
            }
        }

class PredictionResponse(BaseModel):
    """Enhanced prediction response with realistic confidence levels"""
    # Risk scores and confidence
    diabetes_risk: float
    hypertension_risk: float
    diabetes_confidence: str
    hypertension_confidence: str
    
    # Risk categories
    risk_category_diabetes: str
    risk_category_hypertension: str
    
    # SHAP explanations
    diabetes_shap_values: Dict[str, Any]
    hypertension_shap_values: Dict[str, Any]
    
    # Personalized recommendations
    nutrition_recommendations: Dict[str, List[str]]
    fitness_recommendations: Dict[str, List[str]]
    lifestyle_recommendations: List[str]
    
    # Top risk factors
    top_diabetes_factors: List[Dict[str, Any]]
    top_hypertension_factors: List[Dict[str, Any]]
    
    # Health scores
    metabolic_health_score: float
    cardiovascular_health_score: float
    
    # Model information (renamed to avoid protected namespace)
    model_details: Dict[str, str]
    
    # New enhanced features
    contribution_percentages: Dict[str, Dict[str, float]]
    reasoning_explanations: List[str]
    gamification_points: int
    personalized_insights: List[str]
    
    class Config:
        protected_namespaces = ()

def safe_float_conversion(value: Any) -> float:
    """Safely convert any value to float for serialization"""
    try:
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return float(str(value))
    except (ValueError, TypeError):
        return 0.0

def safe_convert_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """Safely convert all values in a dictionary to serializable types"""
    converted = {}
    for key, value in data.items():
        if isinstance(value, dict):
            converted[key] = safe_convert_dict_values(value)
        elif isinstance(value, list):
            converted[key] = [safe_float_conversion(item) if isinstance(item, (np.integer, np.floating)) else item for item in value]
        elif isinstance(value, (np.integer, np.floating)):
            converted[key] = safe_float_conversion(value)
        else:
            converted[key] = value
    return converted

@app.on_event("startup")
async def load_models():
    """Load optimized models and preprocessors on startup"""
    global diabetes_model, hypertension_model, model_features
    global feature_scaler, diabetes_explainer, hypertension_explainer
    global nutrition_recommendations, fitness_recommendations
    
    try:
        logger.info("Loading optimized models and preprocessors...")
        
        # Load optimized models
        diabetes_model = joblib.load('diabetes_model_optimized.joblib')
        hypertension_model = joblib.load('hypertension_model_optimized.joblib')
        model_features = joblib.load('model_features_optimized.joblib')
        
        # Load preprocessors
        try:
            feature_scaler = joblib.load('feature_scaler_optimized.joblib')
        except FileNotFoundError:
            logger.warning("Optimized scaler not found, trying fallback")
            try:
                feature_scaler = joblib.load('feature_scaler.joblib')
            except FileNotFoundError:
                logger.warning("No scaler found, will use raw features")
                feature_scaler = None
        
        # Load recommendations
        try:
            nutrition_recommendations = joblib.load('nutrition_recommendations.joblib')
            fitness_recommendations = joblib.load('fitness_recommendations.joblib')
        except FileNotFoundError:
            logger.warning("Recommendations not found, using defaults")
            nutrition_recommendations = {}
            fitness_recommendations = {}
        
        # Create SHAP explainers for tree-based models
        logger.info("Creating SHAP explainers...")
        try:
            if hasattr(diabetes_model, 'predict_proba'):
                diabetes_explainer = shap.TreeExplainer(diabetes_model)
                hypertension_explainer = shap.TreeExplainer(hypertension_model)
            else:
                # For non-tree models, use simpler explainer
                diabetes_explainer = None
                hypertension_explainer = None
                logger.info("Using simplified explanations for non-tree models")
        except Exception as e:
            logger.warning(f"Could not create SHAP explainers: {e}")
            diabetes_explainer = None
            hypertension_explainer = None
        
        logger.info(f"Optimized models loaded successfully. Features: {len(model_features)}")
        logger.info(f"Model types: Diabetes={type(diabetes_model).__name__}, Hypertension={type(hypertension_model).__name__}")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.warning("Please run the optimized train_pipeline.py first to generate all required files")

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Welcome to PreventiX Advanced API - Optimized Version",
        "version": "2.1.0",
        "features": {
            "anti_overfitting": "Strong regularization and realistic predictions",
            "model_selection": "Automatic best model selection",
            "confidence_scoring": "Realistic confidence levels",
            "personalized_recommendations": "Based on individual risk factors"
        },
        "endpoints": {
            "/predict": "POST - Get comprehensive health risk predictions",
            "/health": "GET - Check API health status",
            "/features": "GET - Get list of model features",
            "/recommendations": "GET - Get general health recommendations",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    models_loaded = all([
        diabetes_model is not None,
        hypertension_model is not None,
        model_features is not None
    ])
    
    explainers_loaded = all([
        diabetes_explainer is not None,
        hypertension_explainer is not None
    ]) if diabetes_explainer is not None else False
    
    return {
        "status": "healthy" if models_loaded else "models_not_loaded",
        "models_loaded": models_loaded,
        "scaler_loaded": feature_scaler is not None,
        "explainers_loaded": explainers_loaded,
        "features_count": len(model_features) if model_features else 0,
        "model_types": {
            "diabetes": type(diabetes_model).__name__ if diabetes_model else None,
            "hypertension": type(hypertension_model).__name__ if hypertension_model else None
        },
        "sample_features": model_features[:10] if model_features else [],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/features")
async def get_features():
    """Get list of model features and their information"""
    if model_features is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Updated feature info based on optimized model
    feature_info = {
        "required_features": [
            "age", "gender", "bmi", "blood_pressure", 
            "cholesterol_level", "glucose_level", 
            "physical_activity", "smoking_status",
            "alcohol_intake", "family_history"
        ],
        "optional_features": [
            "hba1c", "fasting_glucose", "daily_steps", 
            "sleep_hours", "sleep_quality", "stress_level"
        ],
        "engineered_features": [
            "metabolic_syndrome_score", "lifestyle_health_score"
        ],
        "all_model_features": model_features,
        "total_features": len(model_features),
        "feature_engineering_note": "Some features are automatically calculated from input values"
    }
    
    return feature_info

def get_confidence_level(probability: float, feature_quality: Dict) -> str:
    """Calculate confidence level based on probability and feature quality"""
    
    # Base confidence on probability range
    if probability < 0.2 or probability > 0.8:
        base_confidence = "High"
    elif probability < 0.35 or probability > 0.65:
        base_confidence = "Moderate"
    else:
        base_confidence = "Low"
    
    # Adjust based on key feature availability
    key_features_present = 0
    total_key_features = 4
    
    if feature_quality.get('glucose_level_quality', False):
        key_features_present += 1
    if feature_quality.get('bp_quality', False):
        key_features_present += 1
    if feature_quality.get('bmi_quality', False):
        key_features_present += 1
    if feature_quality.get('age_quality', False):
        key_features_present += 1
    
    feature_completeness = key_features_present / total_key_features
    
    if feature_completeness < 0.5:
        return "Low"
    elif feature_completeness < 0.75 and base_confidence == "High":
        return "Moderate"
    else:
        return base_confidence

def calculate_health_scores(features: Dict) -> Dict[str, float]:
    """Calculate metabolic and cardiovascular health scores with realistic ranges"""
    
    # Metabolic health score (0-100)
    metabolic_score = 100
    
    # More nuanced scoring with None checks
    glucose = features.get('glucose_level', 100)
    if glucose is not None and glucose > 126:  # Diabetic range
        metabolic_score -= 30
    elif glucose is not None and glucose > 100:  # Prediabetic range
        metabolic_score -= 15
    
    bmi = features.get('bmi', 25)
    if bmi is not None and bmi > 35:  # Severe obesity
        metabolic_score -= 25
    elif bmi is not None and bmi > 30:  # Obesity
        metabolic_score -= 15
    elif bmi is not None and bmi > 25:  # Overweight
        metabolic_score -= 8
    
    hba1c = features.get('hba1c', 5.7)
    if hba1c is not None and hba1c > 6.5:  # Diabetic
        metabolic_score -= 20
    elif hba1c is not None and hba1c > 5.7:  # Prediabetic
        metabolic_score -= 10
    
    # Cardiovascular health score (0-100)
    cardio_score = 100
    
    bp = features.get('blood_pressure', 120)
    if bp is not None and bp > 140:  # Stage 1 hypertension
        cardio_score -= 25
    elif bp is not None and bp > 130:  # Elevated
        cardio_score -= 15
    elif bp is not None and bp > 120:  # Prehypertension
        cardio_score -= 8
    
    cholesterol = features.get('cholesterol_level', 200)
    if cholesterol is not None and cholesterol > 240:  # High
        cardio_score -= 20
    elif cholesterol is not None and cholesterol > 200:  # Borderline high
        cardio_score -= 10
    
    activity = features.get('physical_activity', 5)
    if activity is not None and activity < 3:
        cardio_score -= 15
    elif activity is not None and activity < 5:
        cardio_score -= 8
    
    smoking = features.get('smoking_status', 0)
    if smoking is not None and smoking == 2:  # Current smoker
        cardio_score -= 20
    elif smoking is not None and smoking == 1:  # Former smoker
        cardio_score -= 5
    
    return {
        'metabolic': max(0, min(100, metabolic_score)),
        'cardiovascular': max(0, min(100, cardio_score))
    }

def get_risk_category(probability: float) -> str:
    """Categorize risk level with more realistic thresholds"""
    if probability < 0.25:
        return "Low Risk"
    elif probability < 0.50:
        return "Moderate Risk"
    elif probability < 0.75:
        return "High Risk"
    else:
        return "Very High Risk"

def prepare_features(health_input: HealthInput) -> tuple[pd.DataFrame, Dict]:
    """Prepare input features for optimized model prediction"""
    
    # Convert input to dictionary
    input_dict = health_input.dict()
    
    # Set defaults for optional features based on optimized model
    if input_dict.get('hba1c') is None:
        # Estimate HbA1c from glucose
        glucose = input_dict['glucose_level']
        if glucose < 100:
            input_dict['hba1c'] = 5.4
        elif glucose < 126:
            input_dict['hba1c'] = 5.9
        else:
            input_dict['hba1c'] = 7.2
    
    if input_dict.get('fasting_glucose') is None:
        input_dict['fasting_glucose'] = input_dict['glucose_level']
    
    # Set defaults for lifestyle features
    input_dict['daily_steps'] = input_dict.get('daily_steps', 7000)
    input_dict['sleep_hours'] = input_dict.get('sleep_hours', 7)
    input_dict['sleep_quality'] = input_dict.get('sleep_quality', 6)
    input_dict['stress_level'] = input_dict.get('stress_level', 5)
    
    # Calculate composite scores (matching training pipeline)
    metabolic_factors = 0
    if input_dict['bmi'] > 30:
        metabolic_factors += 0.25
    if input_dict['glucose_level'] > 100:
        metabolic_factors += 0.25
    if input_dict['cholesterol_level'] > 200:
        metabolic_factors += 0.25
    if input_dict['blood_pressure'] > 120:
        metabolic_factors += 0.25
    
    input_dict['metabolic_syndrome_score'] = metabolic_factors
    
    lifestyle_factors = 0
    if input_dict['physical_activity'] < 3:
        lifestyle_factors += 0.33
    if input_dict['smoking_status'] > 0:
        lifestyle_factors += 0.33
    if input_dict['alcohol_intake'] > 2:
        lifestyle_factors += 0.34
    
    input_dict['lifestyle_health_score'] = min(1.0, lifestyle_factors)
    
    # Create DataFrame with only the features used in the optimized model
    features_df = pd.DataFrame([input_dict])
    
    # Ensure all model features are present
    for feature in model_features:
        if feature not in features_df.columns:
            # Set reasonable defaults for any missing engineered features
            features_df[feature] = 0
    
    # Calculate feature quality metrics
    feature_quality = {
        'glucose_level_quality': 50 <= input_dict['glucose_level'] <= 300,
        'bp_quality': 80 <= input_dict['blood_pressure'] <= 200,
        'bmi_quality': 15 <= input_dict['bmi'] <= 50,
        'age_quality': 18 <= input_dict['age'] <= 100
    }
    
    return features_df[model_features], feature_quality

def get_personalized_recommendations(
    feature_importance: List[tuple],
    input_values: Dict,
    risk_type: str,
    risk_level: float
) -> Dict[str, List[str]]:
    """Get highly personalized recommendations based on user's specific profile and risk factors"""
    
    recommendations = {
        'nutrition': [],
        'fitness': [],
        'lifestyle': []
    }
    
    # Extract user profile for personalization
    age = input_values.get('age', 45)
    gender = input_values.get('gender', 0)
    bmi = input_values.get('bmi', 25)
    blood_pressure = input_values.get('blood_pressure', 120)
    glucose = input_values.get('glucose_level', 100)
    cholesterol = input_values.get('cholesterol_level', 200)
    activity = input_values.get('physical_activity', 5)
    smoking = input_values.get('smoking_status', 0)
    family_history = input_values.get('family_history', 0)
    hba1c = input_values.get('hba1c', 5.7)
    
    # Get top 3 contributing factors
    top_factors = [factor for factor, _ in feature_importance[:3]]
    
    if risk_type == 'diabetes':
        # Highly personalized glucose management
        if any('glucose' in factor for factor in top_factors):
            if glucose >= 200:
                recommendations['nutrition'].extend([
                    f"Your glucose of {glucose} mg/dL requires immediate attention. Focus on very low-carb meals (under 30g carbs per meal)",
                    "Eliminate all sugary drinks and processed foods immediately",
                    "Work with a diabetes educator to learn carbohydrate counting",
                    "Consider a continuous glucose monitor for better tracking"
                ])
            elif glucose >= 126:
                recommendations['nutrition'].extend([
                    f"Your glucose of {glucose} mg/dL indicates diabetes. Follow a consistent carb-controlled diet",
                    "Aim for 45-60g carbs per meal with protein and healthy fats",
                    "Choose low glycemic index foods like quinoa, sweet potatoes, and berries",
                    "Eat at regular intervals to maintain stable blood sugar"
                ])
            elif glucose >= 100:
                recommendations['nutrition'].extend([
                    f"Your glucose of {glucose} mg/dL is in pre-diabetes range. Focus on portion control and timing",
                    "Limit refined carbs and increase fiber intake to 25-30g daily",
                    "Use the plate method: 1/2 non-starchy vegetables, 1/4 lean protein, 1/4 whole grains",
                    "Consider intermittent fasting with medical supervision"
                ])
            else:
                recommendations['nutrition'].extend([
                    f"Your glucose of {glucose} mg/dL is excellent! Maintain your current healthy eating habits",
                    "Continue with balanced meals and regular eating schedule",
                    "Keep monitoring to maintain these healthy levels"
                ])
            
        # Personalized BMI and weight management
        if 'bmi' in top_factors:
            if bmi >= 35:
                recommendations['nutrition'].extend([
                    f"Your BMI of {bmi:.1f} indicates severe obesity. Focus on sustainable weight loss of 1-2 lbs per week",
                    "Create a 500-750 calorie daily deficit through diet and exercise",
                    "Focus on high-protein, high-fiber foods to feel full longer",
                    "Consider working with a registered dietitian for personalized meal planning"
                ])
                recommendations['fitness'].extend([
                    "Start with low-impact exercises like walking, swimming, or cycling",
                    "Aim for 30 minutes of activity daily, even if broken into 10-minute sessions",
                    "Include strength training 2-3 times per week to preserve muscle mass",
                    "Consider working with a certified trainer who specializes in obesity management"
                ])
            elif bmi >= 30:
                recommendations['nutrition'].extend([
                    f"Your BMI of {bmi:.1f} puts you in the obese category. Focus on gradual, sustainable weight loss",
                    "Create a 300-500 calorie daily deficit for steady 1 lb per week weight loss",
                    "Focus on whole foods and limit processed foods",
                    "Use smaller plates and practice mindful eating"
                ])
                recommendations['fitness'].extend([
                    "Aim for 150 minutes of moderate exercise weekly, building up gradually",
                    "Include both cardio and strength training for optimal results",
                    "Find activities you enjoy to maintain long-term consistency",
                    "Consider group fitness classes for motivation and support"
                ])
            elif bmi >= 25:
                recommendations['nutrition'].extend([
                    f"Your BMI of {bmi:.1f} is slightly above optimal. Small changes can make a big difference",
                    "Focus on portion control and reducing calorie-dense foods",
                    "Increase vegetable and lean protein intake",
                    "Limit alcohol and sugary beverages"
                ])
                recommendations['fitness'].extend([
                    "Aim for 30 minutes of moderate exercise most days of the week",
                    "Include both aerobic and strength training",
                    "Take the stairs, park farther away, and find ways to be more active daily"
                ])
            else:
                recommendations['nutrition'].extend([
                    f"Your BMI of {bmi:.1f} is in the healthy range! Maintain your current habits",
                    "Continue with balanced nutrition and regular eating patterns",
                    "Focus on nutrient density rather than weight management"
                ])
                recommendations['fitness'].extend([
                    "Maintain your current activity level - you're doing great!",
                    "Consider adding variety to prevent boredom and plateaus",
                    "Focus on strength training to maintain muscle mass as you age"
                ])
            
        # Age and gender-specific recommendations
        if age >= 65:
            recommendations['lifestyle'].extend([
                "At your age, focus on maintaining muscle mass and bone density",
                "Consider working with a geriatric specialist for age-appropriate care",
                "Regular health screenings become even more important"
            ])
        elif age >= 45:
            recommendations['lifestyle'].extend([
                "You're in a critical prevention window - lifestyle changes now have maximum impact",
                "Focus on stress management and quality sleep",
                "Regular health checkups and monitoring are essential"
            ])
        
        # Gender-specific recommendations
        if gender == 0:  # Female
            recommendations['lifestyle'].extend([
                "Women have unique risk factors - consider hormonal influences on blood sugar",
                "If you're considering pregnancy, optimal glucose control is crucial",
                "Regular gynecological care and bone density monitoring are important"
            ])
        else:  # Male
            recommendations['lifestyle'].extend([
                "Men often develop diabetes at lower BMIs - focus on abdominal fat reduction",
                "Regular prostate and cardiovascular screenings are important",
                "Consider testosterone levels if experiencing fatigue or low energy"
            ])
    
    elif risk_type == 'hypertension':
        # Highly personalized blood pressure management
        if 'blood_pressure' in top_factors:
            if blood_pressure >= 180:
                recommendations['nutrition'].extend([
                    f"Your blood pressure of {blood_pressure} mmHg is critically high. Immediate medical attention required",
                    "Follow a strict low-sodium diet (under 1,500mg daily) with medical supervision",
                    "Focus on potassium-rich foods: bananas, spinach, sweet potatoes, and avocados",
                    "Eliminate all processed foods and restaurant meals immediately"
                ])
                recommendations['lifestyle'].extend([
                    "This is a medical emergency - contact your doctor immediately",
                    "Avoid all strenuous activities until blood pressure is controlled",
                    "Consider stress management techniques like meditation or deep breathing"
                ])
            elif blood_pressure >= 140:
                recommendations['nutrition'].extend([
                    f"Your blood pressure of {blood_pressure} mmHg is high. Follow DASH diet strictly",
                    "Limit sodium to under 2,300mg daily (ideally 1,500mg)",
                    "Increase potassium-rich foods: leafy greens, bananas, and citrus fruits",
                    "Choose fresh, whole foods over processed options"
                ])
                recommendations['fitness'].extend([
                    "Start with gentle exercises like walking or swimming",
                    "Avoid high-intensity activities until blood pressure is controlled",
                    "Aim for 30 minutes of moderate activity most days",
                    "Include stress-reducing activities like yoga or tai chi"
                ])
            elif blood_pressure >= 120:
                recommendations['nutrition'].extend([
                    f"Your blood pressure of {blood_pressure} mmHg is elevated. Focus on prevention",
                    "Limit sodium to under 2,300mg daily",
                    "Increase potassium and magnesium-rich foods",
                    "Choose heart-healthy fats like olive oil and nuts"
                ])
                recommendations['fitness'].extend([
                    "Aim for 150 minutes of moderate exercise weekly",
                    "Include both aerobic and strength training",
                    "Focus on stress management through regular exercise",
                    "Monitor blood pressure before and after exercise"
                ])
            else:
                recommendations['nutrition'].extend([
                    f"Your blood pressure of {blood_pressure} mmHg is excellent! Maintain your current habits",
                    "Continue with heart-healthy eating patterns",
                    "Keep monitoring to maintain these healthy levels"
                ])
                recommendations['fitness'].extend([
                    "Maintain your current activity level - you're doing great!",
                    "Continue with regular exercise for long-term heart health"
                ])
        
        # Age-specific hypertension management
        if age >= 65:
            recommendations['lifestyle'].extend([
                "At your age, blood pressure management becomes even more critical",
                "Consider more frequent monitoring and medication adjustments",
                "Focus on fall prevention and balance exercises"
            ])
        elif age >= 45:
            recommendations['lifestyle'].extend([
                "You're in a critical prevention window for cardiovascular health",
                "Regular blood pressure monitoring is essential",
                "Focus on stress management and quality sleep"
            ])
        
        # Gender-specific recommendations
        if gender == 0:  # Female
            recommendations['lifestyle'].extend([
                "Women's blood pressure can be affected by hormonal changes",
                "Consider pregnancy planning if applicable - blood pressure control is crucial",
                "Regular gynecological care and cardiovascular monitoring are important"
            ])
        else:  # Male
            recommendations['lifestyle'].extend([
                "Men often develop hypertension earlier - you're doing well to monitor this",
                "Regular cardiovascular screenings are important",
                "Consider testosterone levels if experiencing fatigue or low energy"
            ])
    
    # Activity level personalized recommendations
    if 'physical_activity' in top_factors:
        if activity <= 2:
            recommendations['fitness'].extend([
                f"Your activity level of {activity}/10 is very low. Start with just 10 minutes daily",
                "Begin with walking, gentle stretching, or chair exercises",
                "Gradually increase duration and intensity over several weeks",
                "Consider working with a physical therapist if you have mobility issues"
            ])
        elif activity <= 4:
            recommendations['fitness'].extend([
                f"Your activity level of {activity}/10 is below optimal. Build up gradually",
                "Aim for 30 minutes of moderate activity most days",
                "Include both cardio and strength training",
                "Find activities you enjoy to maintain consistency"
            ])
        elif activity <= 7:
            recommendations['fitness'].extend([
                f"Your activity level of {activity}/10 is good. Consider adding variety",
                "Include both aerobic and strength training",
                "Try new activities to prevent boredom",
                "Focus on consistency rather than intensity"
            ])
        else:
            recommendations['fitness'].extend([
                f"Your activity level of {activity}/10 is excellent! Maintain your current routine",
                "Consider adding variety to prevent overuse injuries",
                "Focus on recovery and proper nutrition to support your activity level"
            ])
    
    # Smoking status personalized recommendations
    if smoking == 2:  # Current smoker
        recommendations['lifestyle'].extend([
            "Quitting smoking is the single most important step for your health",
            "Consider nicotine replacement therapy or prescription medications",
            "Join a smoking cessation program for support",
            "Your risk of heart disease and stroke will decrease significantly after quitting"
        ])
    elif smoking == 1:  # Former smoker
        recommendations['lifestyle'].extend([
            "Congratulations on quitting smoking! Your risk continues to decrease over time",
            "Stay vigilant about not relapsing - you've made excellent progress",
            "Your lung function and cardiovascular health will continue to improve"
        ])
    else:
        recommendations['lifestyle'].extend([
            "Excellent job staying smoke-free! This significantly reduces your health risks",
            "Continue to avoid secondhand smoke exposure",
            "Your healthy choice is protecting your heart and lungs"
        ])
    
    # Family history personalized recommendations
    if family_history:
        recommendations['lifestyle'].extend([
            "Given your family history, you have a higher genetic risk",
            "Focus on controllable factors like diet, exercise, and regular checkups",
            "Consider more frequent health screenings",
            "Work closely with your healthcare provider to monitor your health"
        ])
    else:
        recommendations['lifestyle'].extend([
            "With no family history, you have a genetic advantage",
            "Focus on maintaining healthy lifestyle habits to preserve this advantage",
            "Regular health checkups are still important for prevention"
        ])
    
    return recommendations

def get_simple_feature_importance(model, feature_names: List[str], input_values: np.ndarray) -> List[tuple]:
    """Get feature importance for non-tree models"""
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importance = list(zip(feature_names, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        # Linear models
        importance = list(zip(feature_names, abs(model.coef_[0])))
    else:
        # Fallback - uniform importance
        importance = list(zip(feature_names, [1.0/len(feature_names)] * len(feature_names)))
    
    return sorted(importance, key=lambda x: x[1], reverse=True)

def calculate_contribution_percentages(feature_importance: List[tuple]) -> Dict[str, float]:
    """Calculate percentage contribution of each factor to the risk"""
    total_importance = sum(imp for _, imp in feature_importance)
    contributions = {}
    
    for feature, importance in feature_importance:
        percentage = (importance / total_importance) * 100 if total_importance > 0 else 0
        contributions[feature] = round(percentage, 1)
    
    return contributions

def generate_reasoning_explanations(
    diabetes_risk: float, 
    hypertension_risk: float, 
    top_factors: List[Dict], 
    input_values: Dict
) -> List[str]:
    """Generate highly personalized reasoning explanations for the risk predictions"""
    explanations = []
    
    # Extract user profile for personalized explanations
    age = input_values.get('age', 45)
    gender = input_values.get('gender', 0)
    bmi = input_values.get('bmi', 25)
    blood_pressure = input_values.get('blood_pressure', 120)
    glucose = input_values.get('glucose_level', 100)
    activity = input_values.get('physical_activity', 5)
    smoking = input_values.get('smoking_status', 0)
    family_history = input_values.get('family_history', 0)
    
    gender_text = "woman" if gender == 0 else "man"
    
    # Highly personalized diabetes reasoning
    if diabetes_risk > 0.7:
        explanations.append(f"As a {age}-year-old {gender_text}, your diabetes risk of {diabetes_risk:.1%} is high. This is primarily driven by your {top_factors[0]['feature'].replace('_', ' ')} ({top_factors[0]['value']}), which has the strongest impact on your risk.")
        explanations.append(f"Your {top_factors[1]['feature'].replace('_', ' ')} and {top_factors[2]['feature'].replace('_', ' ')} are also significant contributors. At your age, immediate lifestyle changes are crucial.")
    elif diabetes_risk > 0.3:
        explanations.append(f"At {age} years old, your diabetes risk of {diabetes_risk:.1%} is moderate. Your {top_factors[0]['feature'].replace('_', ' ')} is the primary factor, but your {top_factors[1]['feature'].replace('_', ' ')} and {top_factors[2]['feature'].replace('_', ' ')} also contribute.")
        explanations.append(f"Small lifestyle changes could significantly reduce this risk. You're in a critical prevention window at {age}.")
    else:
        explanations.append(f"Excellent news! As a {age}-year-old {gender_text}, your diabetes risk of {diabetes_risk:.1%} is low. Your current {top_factors[0]['feature'].replace('_', ' ')} and lifestyle factors are protective.")
        explanations.append(f"Continue maintaining these healthy habits to preserve this low risk.")
    
    # Highly personalized hypertension reasoning
    if hypertension_risk > 0.7:
        explanations.append(f"Your hypertension risk of {hypertension_risk:.1%} is high, primarily due to your {top_factors[0]['feature'].replace('_', ' ')} ({top_factors[0]['value']}). At your age of {age}, this is concerning and requires immediate attention.")
        explanations.append(f"Your {top_factors[1]['feature'].replace('_', ' ')} and {top_factors[2]['feature'].replace('_', ' ')} are also contributing factors. Blood pressure management is crucial for your long-term health.")
    elif hypertension_risk > 0.3:
        explanations.append(f"Your hypertension risk of {hypertension_risk:.1%} is moderate. Your {top_factors[0]['feature'].replace('_', ' ')} is the main concern, with {top_factors[1]['feature'].replace('_', ' ')} and {top_factors[2]['feature'].replace('_', ' ')} also playing a role.")
        explanations.append(f"At {age}, focusing on blood pressure management is crucial for long-term health. Small lifestyle changes could significantly reduce this risk.")
    else:
        explanations.append(f"Great job! Your hypertension risk of {hypertension_risk:.1%} is low. Your {top_factors[0]['feature'].replace('_', ' ')} and other lifestyle factors are working in your favor.")
        explanations.append(f"Keep up these healthy habits to maintain this low risk.")
    
    # Age-specific explanations
    if age >= 65:
        explanations.append(f"At {age}, you're in a high-risk age group for both diabetes and hypertension. However, your current risk levels suggest you're managing your health well. Continue with regular monitoring and preventive care.")
    elif age >= 45:
        explanations.append(f"At {age}, you're entering a critical prevention window. Your current risk levels are manageable, but this is the perfect time to optimize your lifestyle for long-term health.")
    elif age >= 30:
        explanations.append(f"At {age}, you have an excellent opportunity to establish healthy habits that will protect you long-term. Your current risk levels are very manageable.")
    else:
        explanations.append(f"Your young age of {age} gives you a significant advantage in preventing chronic diseases. Your current risk levels are excellent - focus on maintaining these healthy habits.")
    
    # Gender-specific explanations
    if gender == 0:  # Female
        explanations.append("As a woman, you have unique risk factors to consider. Hormonal changes throughout life can affect both diabetes and hypertension risk. Regular health screenings and maintaining a healthy lifestyle are especially important.")
    else:  # Male
        explanations.append("As a man, you may be at higher risk for developing these conditions at younger ages. Your current risk levels suggest you're doing well, but continued vigilance with lifestyle factors is important.")
    
    # Family history explanations
    if family_history:
        explanations.append("Given your family history, you have a higher genetic predisposition to these conditions. However, your current risk levels suggest that your lifestyle choices are effectively managing this genetic risk. Continue focusing on controllable factors.")
    else:
        explanations.append("With no family history, you have a genetic advantage. Your current risk levels reflect this advantage, but maintaining healthy lifestyle habits is still crucial for long-term health.")
    
    # Activity level explanations
    if activity <= 3:
        explanations.append(f"Your activity level of {activity}/10 is below optimal and contributes to your risk. Increasing physical activity could significantly improve your health outcomes and reduce your risk levels.")
    elif activity >= 7:
        explanations.append(f"Your high activity level of {activity}/10 is excellent and is helping to protect you from these conditions. This is one of your strongest protective factors.")
    else:
        explanations.append(f"Your activity level of {activity}/10 is good. Consider increasing it slightly to further reduce your risk and improve your overall health.")
    
    # Smoking status explanations
    if smoking == 2:  # Current smoker
        explanations.append("Your smoking status is significantly increasing your risk for both diabetes and hypertension. Quitting smoking would be the single most important step you could take to improve your health.")
    elif smoking == 1:  # Former smoker
        explanations.append("Congratulations on quitting smoking! This is significantly reducing your risk compared to if you were still smoking. Your risk continues to decrease the longer you stay smoke-free.")
    else:
        explanations.append("Your non-smoking status is one of your strongest protective factors. This significantly reduces your risk for both diabetes and hypertension.")
    
    return explanations

def calculate_gamification_points(input_values: Dict, recommendations_followed: List[str] = None) -> int:
    """Calculate gamification points based on health metrics and recommendations followed"""
    points = 0
    
    # Base points for good metrics
    if input_values.get('physical_activity', 0) >= 5:
        points += 50
    if input_values.get('daily_steps', 0) >= 10000:
        points += 30
    if input_values.get('sleep_hours', 0) >= 7:
        points += 20
    if input_values.get('water_intake', 0) >= 8:
        points += 15
    if input_values.get('protein_intake', 0) >= 60:
        points += 25
    
    # Bonus points for following recommendations
    if recommendations_followed:
        points += len(recommendations_followed) * 10
    
    return points

def generate_personalized_insights(input_values: Dict, risk_scores: Dict) -> List[str]:
    """Generate highly personalized insights based on user's specific profile and risk factors"""
    insights = []
    
    # Extract user profile
    age = input_values.get('age', 45)
    gender = input_values.get('gender', 0)
    bmi = input_values.get('bmi', 25)
    blood_pressure = input_values.get('blood_pressure', 120)
    glucose = input_values.get('glucose_level', 100)
    cholesterol = input_values.get('cholesterol_level', 200)
    activity = input_values.get('physical_activity', 5)
    smoking = input_values.get('smoking_status', 0)
    family_history = input_values.get('family_history', 0)
    hba1c = input_values.get('hba1c', 5.7)
    
    # Gender-specific insights
    gender_text = "woman" if gender == 0 else "man"
    
    # Age-specific personalized insights
    if age >= 65:
        insights.append(f"As a {age}-year-old {gender_text}, you're in a high-risk age group. Focus on preventive care and regular monitoring.")
    elif age >= 45:
        insights.append(f"At {age} years old, you're entering a critical period for chronic disease prevention. Early intervention is key.")
    elif age >= 30:
        insights.append(f"At {age}, you have a great opportunity to establish healthy habits that will protect you long-term.")
    else:
        insights.append(f"Your young age of {age} gives you a significant advantage in preventing chronic diseases.")
    
    # BMI-specific personalized insights
    if bmi >= 35:
        insights.append(f"Your BMI of {bmi:.1f} indicates severe obesity. Weight loss of 10-15% could dramatically reduce your health risks.")
    elif bmi >= 30:
        insights.append(f"Your BMI of {bmi:.1f} puts you in the obese category. Even a 5-10% weight loss would significantly improve your health.")
    elif bmi >= 25:
        insights.append(f"Your BMI of {bmi:.1f} is slightly above the healthy range. Small lifestyle changes could bring you to optimal health.")
    elif 18.5 <= bmi <= 24.9:
        insights.append(f"Excellent! Your BMI of {bmi:.1f} is in the healthy range. Keep up the great work!")
    else:
        insights.append(f"Your BMI of {bmi:.1f} is below the healthy range. Consider consulting a healthcare provider about healthy weight gain.")
    
    # Blood pressure personalized insights
    if blood_pressure >= 180:
        insights.append(f"Your blood pressure of {blood_pressure} mmHg is critically high. Immediate medical attention is recommended.")
    elif blood_pressure >= 140:
        insights.append(f"Your blood pressure of {blood_pressure} mmHg is high. Lifestyle changes and possibly medication may be needed.")
    elif blood_pressure >= 120:
        insights.append(f"Your blood pressure of {blood_pressure} mmHg is elevated. Focus on diet, exercise, and stress management.")
    else:
        insights.append(f"Your blood pressure of {blood_pressure} mmHg is excellent! Continue your current lifestyle.")
    
    # Glucose personalized insights
    if glucose >= 200:
        insights.append(f"Your glucose level of {glucose} mg/dL is very high. This suggests diabetes and requires immediate medical attention.")
    elif glucose >= 126:
        insights.append(f"Your glucose level of {glucose} mg/dL indicates diabetes. Work with your doctor on a management plan.")
    elif glucose >= 100:
        insights.append(f"Your glucose level of {glucose} mg/dL is in the pre-diabetes range. Focus on carbohydrate control and exercise.")
    else:
        insights.append(f"Your glucose level of {glucose} mg/dL is in the healthy range. Keep up your current habits!")
    
    # HbA1c personalized insights (if available)
    if hba1c is not None:
        if hba1c >= 6.5:
            insights.append(f"Your HbA1c of {hba1c}% indicates diabetes. This requires medical management and lifestyle changes.")
        elif hba1c >= 5.7:
            insights.append(f"Your HbA1c of {hba1c}% suggests pre-diabetes. Focus on weight management and regular exercise.")
        else:
            insights.append(f"Your HbA1c of {hba1c}% is in the normal range. Continue your healthy lifestyle!")
    
    # Activity level personalized insights
    if activity <= 2:
        insights.append(f"Your activity level of {activity}/10 is very low. Start with just 10 minutes of daily walking to build momentum.")
    elif activity <= 4:
        insights.append(f"Your activity level of {activity}/10 is below optimal. Aim for 150 minutes of moderate exercise weekly.")
    elif activity <= 7:
        insights.append(f"Your activity level of {activity}/10 is good. Consider adding strength training twice weekly.")
    else:
        insights.append(f"Your activity level of {activity}/10 is excellent! You're doing great with your fitness routine.")
    
    # Smoking personalized insights
    if smoking == 2:  # Current smoker
        insights.append("Quitting smoking is the single most important step you can take for your health. Consider nicotine replacement therapy.")
    elif smoking == 1:  # Former smoker
        insights.append("Congratulations on quitting smoking! Your risk continues to decrease the longer you stay smoke-free.")
    else:
        insights.append("Great job staying smoke-free! This significantly reduces your risk of heart disease and cancer.")
    
    # Family history personalized insights
    if family_history:
        insights.append("Given your family history, you have a higher genetic risk. Focus on controllable factors like diet, exercise, and regular checkups.")
    else:
        insights.append("With no family history, you have a genetic advantage. Focus on maintaining healthy lifestyle habits.")
    
    # Cholesterol personalized insights
    if cholesterol >= 240:
        insights.append(f"Your cholesterol of {cholesterol} mg/dL is high. Focus on a heart-healthy diet and consider medication.")
    elif cholesterol >= 200:
        insights.append(f"Your cholesterol of {cholesterol} mg/dL is borderline high. Dietary changes could help lower it.")
    else:
        insights.append(f"Your cholesterol of {cholesterol} mg/dL is in a healthy range. Keep up your current lifestyle!")
    
    # Risk-specific personalized insights
    diabetes_risk = risk_scores.get('diabetes_risk', 0)
    hypertension_risk = risk_scores.get('hypertension_risk', 0)
    
    if diabetes_risk > 0.7:
        insights.append(f"Your diabetes risk of {diabetes_risk:.1%} is high. Focus on weight loss, exercise, and blood sugar monitoring.")
    elif diabetes_risk > 0.3:
        insights.append(f"Your diabetes risk of {diabetes_risk:.1%} is moderate. Small lifestyle changes can significantly reduce this risk.")
    
    if hypertension_risk > 0.7:
        insights.append(f"Your hypertension risk of {hypertension_risk:.1%} is high. Focus on sodium reduction, exercise, and stress management.")
    elif hypertension_risk > 0.3:
        insights.append(f"Your hypertension risk of {hypertension_risk:.1%} is moderate. Blood pressure monitoring and lifestyle changes are key.")
    
    return insights

def analyze_what_if_scenario(scenario: str, current_values: Dict) -> Dict[str, Any]:
    """Analyze 'what if' scenarios for health improvements with personalized predictions"""
    scenario_lower = scenario.lower()
    
    # Extract current user profile
    age = current_values.get('age', 45)
    gender = current_values.get('gender', 0)
    bmi = current_values.get('bmi', 25)
    blood_pressure = current_values.get('blood_pressure', 120)
    glucose = current_values.get('glucose_level', 100)
    activity = current_values.get('physical_activity', 5)
    smoking = current_values.get('smoking_status', 0)
    family_history = current_values.get('family_history', 0)
    
    gender_text = "woman" if gender == 0 else "man"
    
    # Protein intake scenarios
    if "protein" in scenario_lower and ("increase" in scenario_lower or "more" in scenario_lower):
        current_protein = current_values.get('protein_intake', 50)
        recommended_protein = max(60, bmi * 1.2)  # 1.2g per kg body weight
        
        return {
            "scenario": f"Increased Protein Intake for {age}-year-old {gender_text}",
            "current_protein": f"{current_protein}g daily",
            "recommended_protein": f"{recommended_protein:.0f}g daily",
            "potential_benefits": [
                f"Better blood sugar control (especially important at your glucose level of {glucose} mg/dL)",
                "Improved muscle mass and metabolism (crucial at age {age})",
                "Enhanced satiety and weight management (helpful for your BMI of {bmi:.1f})",
                "Better recovery from exercise"
            ],
            "personalized_impact": f"For a {age}-year-old with BMI {bmi:.1f}, increasing protein could reduce diabetes risk by 8-12%",
            "implementation_tips": [
                "Add lean protein to each meal (chicken, fish, beans, Greek yogurt)",
                "Aim for 20-30g protein per meal",
                "Consider protein supplements if needed",
                "Monitor blood sugar response to protein-rich meals"
            ],
            "timeline": "Expect to see benefits within 2-4 weeks of consistent protein intake"
        }
    
    # Exercise scenarios
    elif "exercise" in scenario_lower or "workout" in scenario_lower or "activity" in scenario_lower:
        current_activity = activity
        recommended_activity = min(8, current_activity + 2)
        
        return {
            "scenario": f"Increased Exercise for {age}-year-old {gender_text}",
            "current_activity": f"{current_activity}/10",
            "recommended_activity": f"{recommended_activity}/10",
            "potential_benefits": [
                f"Lower blood pressure (your current {blood_pressure} mmHg could improve by 5-10 points)",
                f"Improved insulin sensitivity (helpful for your glucose level of {glucose} mg/dL)",
                "Better cardiovascular health",
                f"Potential weight loss (could help with your BMI of {bmi:.1f})"
            ],
            "personalized_impact": f"For a {age}-year-old with current activity level {current_activity}/10, increasing exercise could reduce diabetes risk by 15-25% and hypertension risk by 10-20%",
            "recommended_exercise": [
                "Start with 30 minutes of moderate activity 5 days/week",
                "Include both cardio and strength training",
                "Consider walking, swimming, or cycling for low-impact options",
                "Gradually increase intensity over 4-6 weeks"
            ],
            "age_considerations": f"At {age}, focus on joint-friendly activities and proper warm-up/cool-down",
            "timeline": "Blood pressure improvements may be seen within 2-3 weeks, diabetes risk reduction within 2-3 months"
        }
    
    # Weight loss scenarios
    elif "weight" in scenario_lower or "lose" in scenario_lower or "bmi" in scenario_lower:
        current_weight = bmi * 1.7 * 1.7  # Approximate weight from BMI
        target_bmi = max(22, bmi - 2)
        weight_loss_needed = current_weight * 0.1  # 10% weight loss
        
        return {
            "scenario": f"Weight Loss for {age}-year-old {gender_text}",
            "current_bmi": f"{bmi:.1f}",
            "target_bmi": f"{target_bmi:.1f}",
            "weight_loss_needed": f"{weight_loss_needed:.1f} lbs",
            "potential_benefits": [
                f"Significant reduction in diabetes risk (your current glucose of {glucose} mg/dL could improve)",
                f"Lower blood pressure (your {blood_pressure} mmHg could drop by 5-15 points)",
                "Improved cholesterol levels",
                "Better joint health and mobility"
            ],
            "personalized_impact": f"For a {age}-year-old with BMI {bmi:.1f}, losing 10% of body weight could reduce diabetes risk by 30-50% and hypertension risk by 20-30%",
            "recommended_approach": [
                f"Create a 500-calorie daily deficit for 1 lb/week loss",
                "Focus on whole foods and portion control",
                "Include both cardio and strength training",
                "Aim for 7-9 hours of quality sleep"
            ],
            "age_considerations": f"At {age}, gradual weight loss (1-2 lbs/week) is safer and more sustainable",
            "timeline": "Significant health improvements typically seen within 3-6 months of sustained weight loss"
        }
    
    # Blood pressure scenarios
    elif "blood pressure" in scenario_lower or "pressure" in scenario_lower:
        return {
            "scenario": f"Blood Pressure Management for {age}-year-old {gender_text}",
            "current_bp": f"{blood_pressure} mmHg",
            "target_bp": "Less than 120/80 mmHg",
            "potential_benefits": [
                "Reduced risk of heart disease and stroke",
                "Better kidney function",
                "Improved overall cardiovascular health",
                "Reduced medication needs"
            ],
            "personalized_impact": f"For a {age}-year-old with BP {blood_pressure} mmHg, lifestyle changes could reduce hypertension risk by 20-40%",
            "recommended_actions": [
                "Follow DASH diet (limit sodium to 2,300mg daily)",
                "Increase potassium-rich foods (bananas, spinach, avocados)",
                "Engage in regular aerobic exercise",
                "Manage stress through meditation or yoga",
                "Limit alcohol intake"
            ],
            "age_considerations": f"At {age}, blood pressure management becomes increasingly important for long-term health",
            "timeline": "Blood pressure improvements may be seen within 2-4 weeks of lifestyle changes"
        }
    
    # Sleep scenarios
    elif "sleep" in scenario_lower:
        current_sleep = current_values.get('sleep_hours', 7)
        return {
            "scenario": f"Improved Sleep for {age}-year-old {gender_text}",
            "current_sleep": f"{current_sleep} hours",
            "recommended_sleep": "7-9 hours nightly",
            "potential_benefits": [
                "Better blood sugar control",
                "Improved blood pressure regulation",
                "Enhanced immune function",
                "Better stress management"
            ],
            "personalized_impact": f"For a {age}-year-old, improving sleep could reduce diabetes risk by 10-15% and hypertension risk by 8-12%",
            "sleep_hygiene_tips": [
                "Maintain consistent sleep schedule",
                "Create cool, dark, quiet bedroom environment",
                "Avoid screens 1 hour before bed",
                "Limit caffeine after 2 PM",
                "Consider relaxation techniques"
            ],
            "timeline": "Sleep quality improvements typically seen within 1-2 weeks of consistent sleep hygiene"
        }
    
    # Stress management scenarios
    elif "stress" in scenario_lower:
        current_stress = current_values.get('stress_level', 5)
        return {
            "scenario": f"Stress Management for {age}-year-old {gender_text}",
            "current_stress": f"{current_stress}/10",
            "target_stress": "3-5/10",
            "potential_benefits": [
                "Lower blood pressure",
                "Better blood sugar control",
                "Improved sleep quality",
                "Enhanced overall well-being"
            ],
            "personalized_impact": f"For a {age}-year-old with stress level {current_stress}/10, stress management could reduce both diabetes and hypertension risk by 10-20%",
            "stress_reduction_techniques": [
                "Daily meditation or deep breathing (10-15 minutes)",
                "Regular physical exercise",
                "Time management and prioritization",
                "Social support and connection",
                "Professional counseling if needed"
            ],
            "timeline": "Stress reduction benefits typically seen within 2-4 weeks of consistent practice"
        }
    
    # General health improvement
    else:
        return {
            "scenario": f"General Health Improvement for {age}-year-old {gender_text}",
            "current_profile": f"BMI: {bmi:.1f}, BP: {blood_pressure} mmHg, Glucose: {glucose} mg/dL",
            "potential_benefits": [
                "Reduced inflammation throughout the body",
                "Better overall health markers",
                "Improved quality of life and energy",
                "Enhanced longevity and vitality"
            ],
            "personalized_impact": f"For a {age}-year-old {gender_text}, comprehensive lifestyle changes could reduce diabetes risk by 20-40% and hypertension risk by 15-30%",
            "comprehensive_approach": [
                "Balanced, nutrient-dense diet",
                "Regular physical activity (150 min/week moderate + 2 strength sessions)",
                "Adequate sleep (7-9 hours nightly)",
                "Stress management and relaxation",
                "Regular health checkups and monitoring"
            ],
            "age_considerations": f"At {age}, you're in a critical window for preventing chronic diseases. Lifestyle changes now have maximum impact.",
            "timeline": "Comprehensive health improvements typically seen within 3-6 months of consistent lifestyle changes"
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict_health_risks(health_input: HealthInput):
    """
    Comprehensive health risk prediction with optimized models and realistic confidence scoring
    """
    if not all([diabetes_model, hypertension_model, model_features]):
        raise HTTPException(
            status_code=503,
            detail="Optimized models not loaded. Please run the optimized train_pipeline.py first."
        )
    
    try:
        # Prepare features
        input_df, feature_quality = prepare_features(health_input)
        
        # Apply preprocessing if available
        if feature_scaler is not None:
            input_array = feature_scaler.transform(input_df)
        else:
            input_array = input_df.values
        
        # Get predictions - ensure they are Python floats
        diabetes_proba = safe_float_conversion(diabetes_model.predict_proba(input_array)[0, 1])
        hypertension_proba = safe_float_conversion(hypertension_model.predict_proba(input_array)[0, 1])
        
        # Get confidence levels
        diabetes_confidence = get_confidence_level(diabetes_proba, feature_quality)
        hypertension_confidence = get_confidence_level(hypertension_proba, feature_quality)
        
        # Get feature importance and SHAP values
        diabetes_importance = get_simple_feature_importance(diabetes_model, model_features, input_array)
        hypertension_importance = get_simple_feature_importance(hypertension_model, model_features, input_array)
        
        # Try to get SHAP values if explainers are available
        diabetes_shap_dict = {}
        hypertension_shap_dict = {}
        
        if diabetes_explainer is not None and hypertension_explainer is not None:
            try:
                diabetes_shap = diabetes_explainer.shap_values(input_array)
                hypertension_shap = hypertension_explainer.shap_values(input_array)
                
                # Handle different SHAP output formats
                if isinstance(diabetes_shap, list):
                    diabetes_shap = diabetes_shap[1][0]  # Binary classification, positive class
                    hypertension_shap = hypertension_shap[1][0]
                else:
                    diabetes_shap = diabetes_shap[0]
                    hypertension_shap = hypertension_shap[0]
                
                diabetes_shap_dict = {
                    "base_value": safe_float_conversion(diabetes_explainer.expected_value),
                    "feature_contributions": {
                        feature: safe_float_conversion(value)
                        for feature, value in zip(model_features, diabetes_shap)
                    },
                    "feature_values": {
                        feature: safe_float_conversion(input_df[feature].iloc[0])
                        for feature in model_features
                    }
                }
                
                hypertension_shap_dict = {
                    "base_value": safe_float_conversion(hypertension_explainer.expected_value),
                    "feature_contributions": {
                        feature: safe_float_conversion(value)
                        for feature, value in zip(model_features, hypertension_shap)
                    },
                    "feature_values": {
                        feature: safe_float_conversion(input_df[feature].iloc[0])
                        for feature in model_features
                    }
                }
                
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}, using feature importance")
                # Fallback to feature importance
                diabetes_shap_dict = {
                    "feature_contributions": {feat: safe_float_conversion(imp) for feat, imp in diabetes_importance},
                    "explanation_type": "feature_importance"
                }
                hypertension_shap_dict = {
                    "feature_contributions": {feat: safe_float_conversion(imp) for feat, imp in hypertension_importance},
                    "explanation_type": "feature_importance"
                }
        else:
            # Use feature importance as explanation
            diabetes_shap_dict = {
                "feature_contributions": {feat: safe_float_conversion(imp) for feat, imp in diabetes_importance},
                "explanation_type": "feature_importance"
            }
            hypertension_shap_dict = {
                "feature_contributions": {feat: safe_float_conversion(imp) for feat, imp in hypertension_importance},
                "explanation_type": "feature_importance"
            }
        
        # Get personalized recommendations
        input_values = health_input.dict()
        diabetes_recs = get_personalized_recommendations(
            diabetes_importance, input_values, 'diabetes', diabetes_proba
        )
        hypertension_recs = get_personalized_recommendations(
            hypertension_importance, input_values, 'hypertension', hypertension_proba
        )
        
        # Combine unique recommendations
        combined_nutrition = list(dict.fromkeys(  # Remove duplicates while preserving order
            diabetes_recs['nutrition'][:3] + hypertension_recs['nutrition'][:3]
        ))
        combined_fitness = list(dict.fromkeys(
            diabetes_recs['fitness'][:3] + hypertension_recs['fitness'][:3]
        ))
        combined_lifestyle = list(dict.fromkeys(
            diabetes_recs['lifestyle'] + hypertension_recs['lifestyle']
        ))
        
        # Calculate health scores
        health_scores = calculate_health_scores(input_values)
        
        # Prepare top factors with values and importance
        top_diabetes_factors = [
            {
                "feature": feat,
                "importance": safe_float_conversion(imp),
                "value": safe_float_conversion(input_df[feat].iloc[0]) if feat in input_df.columns else None
            }
            for feat, imp in diabetes_importance[:5]
        ]
        
        top_hypertension_factors = [
            {
                "feature": feat,
                "importance": safe_float_conversion(imp),
                "value": safe_float_conversion(input_df[feat].iloc[0]) if feat in input_df.columns else None
            }
            for feat, imp in hypertension_importance[:5]
        ]
        
        # Calculate new enhanced features
        diabetes_contributions = calculate_contribution_percentages(diabetes_importance)
        hypertension_contributions = calculate_contribution_percentages(hypertension_importance)
        
        # Generate reasoning explanations
        reasoning_explanations = generate_reasoning_explanations(
            diabetes_proba, hypertension_proba, top_diabetes_factors, input_values
        )
        
        # Calculate gamification points
        gamification_points = calculate_gamification_points(input_values)
        
        # Generate personalized insights
        risk_scores = {"diabetes": diabetes_proba, "hypertension": hypertension_proba}
        personalized_insights = generate_personalized_insights(input_values, risk_scores)
        
        # Prepare response with all values properly converted
        response_data = {
            "diabetes_risk": round(diabetes_proba, 3),
            "hypertension_risk": round(hypertension_proba, 3),
            "diabetes_confidence": diabetes_confidence,
            "hypertension_confidence": hypertension_confidence,
            "risk_category_diabetes": get_risk_category(diabetes_proba),
            "risk_category_hypertension": get_risk_category(hypertension_proba),
            "diabetes_shap_values": safe_convert_dict_values(diabetes_shap_dict),
            "hypertension_shap_values": safe_convert_dict_values(hypertension_shap_dict),
            "nutrition_recommendations": {
                "primary": combined_nutrition[:4],
                "secondary": combined_nutrition[4:8] if len(combined_nutrition) > 4 else []
            },
            "fitness_recommendations": {
                "primary": combined_fitness[:3],
                "secondary": combined_fitness[3:6] if len(combined_fitness) > 3 else []
            },
            "lifestyle_recommendations": combined_lifestyle[:5],
            "top_diabetes_factors": top_diabetes_factors,
            "top_hypertension_factors": top_hypertension_factors,
            "metabolic_health_score": round(health_scores['metabolic'], 1),
            "cardiovascular_health_score": round(health_scores['cardiovascular'], 1),
            "model_details": {
                "diabetes_model": type(diabetes_model).__name__,
                "hypertension_model": type(hypertension_model).__name__,
                "version": "optimized_v2.1",
                "anti_overfitting": "enabled"
            },
            # New enhanced features
            "contribution_percentages": {
                "diabetes": diabetes_contributions,
                "hypertension": hypertension_contributions
            },
            "reasoning_explanations": reasoning_explanations,
            "gamification_points": gamification_points,
            "personalized_insights": personalized_insights
        }
        
        # Create response object
        response = PredictionResponse(**response_data)
        
        logger.info(f"Prediction successful - Diabetes: {diabetes_proba:.1%} ({diabetes_confidence}), Hypertension: {hypertension_proba:.1%} ({hypertension_confidence})")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/recommendations")
async def get_general_recommendations():
    """Get general health recommendations with realistic expectations"""
    return {
        "diabetes_prevention": {
            "nutrition": [
                "Maintain balanced diet with whole grains and lean proteins",
                "Limit added sugars to less than 25g per day",
                "Eat regular, portion-controlled meals",
                "Choose high-fiber foods to help regulate blood sugar"
            ],
            "fitness": [
                "Aim for 150 minutes of moderate exercise weekly",
                "Include strength training twice weekly",
                "Take short walks after meals",
                "Find enjoyable activities for long-term consistency"
            ],
            "monitoring": [
                "Regular blood glucose testing if at risk",
                "Annual HbA1c testing",
                "Monitor weight and BMI trends"
            ]
        },
        "hypertension_prevention": {
            "nutrition": [
                "Follow DASH diet principles",
                "Limit sodium to under 2,300mg daily",
                "Increase fruits and vegetables to 5-9 servings daily",
                "Choose lean proteins and low-fat dairy"
            ],
            "fitness": [
                "30 minutes of moderate cardio 5 days per week",
                "Include flexibility and stress-reducing activities",
                "Avoid excessive strain during exercise",
                "Monitor heart rate during activities"
            ],
            "monitoring": [
                "Regular blood pressure monitoring",
                "Annual cardiovascular health checkups",
                "Track sodium intake and weight changes"
            ]
        },
        "general_health": {
            "lifestyle": [
                "Maintain consistent sleep schedule (7-9 hours)",
                "Practice stress management techniques",
                "Avoid tobacco and limit alcohol consumption",
                "Stay hydrated with adequate water intake"
            ],
            "preventive_care": [
                "Regular healthcare provider visits",
                "Stay current with recommended screenings",
                "Maintain healthy social connections",
                "Keep emergency contact information updated"
            ]
        },
        "model_notes": {
            "accuracy": "Models provide screening-level predictions, not diagnostic certainty",
            "limitations": "Should be used alongside professional medical advice",
            "updates": "Model performance may vary across different populations"
        }
    }

@app.post("/what-if")
async def analyze_what_if_scenario_endpoint(scenario_data: Dict[str, Any]):
    """Analyze 'what if' scenarios for health improvements"""
    scenario = scenario_data.get("scenario", "")
    current_values = scenario_data.get("current_values", {})
    
    if not scenario:
        raise HTTPException(status_code=400, detail="Scenario description is required")
    
    try:
        analysis = analyze_what_if_scenario(scenario, current_values)
        return {
            "status": "success",
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"What-if analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/tracking/goals")
async def get_tracking_goals():
    """Get personalized tracking goals based on user profile"""
    return {
        "daily_goals": {
            "steps": 10000,
            "calories": 2000,
            "water_glasses": 8,
            "protein_grams": 60,
            "sleep_hours": 8
        },
        "weekly_goals": {
            "gym_hours": 3,
            "cardio_minutes": 150,
            "strength_sessions": 2
        },
        "monthly_goals": {
            "weight_change": "maintain",
            "health_checkups": 1,
            "new_habits": 2
        }
    }

@app.post("/tracking/update")
async def update_tracking_data(tracking_data: Dict[str, Any]):
    """Update user's tracking data and calculate progress"""
    try:
        # Calculate points based on goals
        points = 0
        achievements = []
        
        # Daily goals
        if tracking_data.get("daily_steps", 0) >= 10000:
            points += 20
            achievements.append("Daily step goal achieved!")
        
        if tracking_data.get("water_intake", 0) >= 8:
            points += 15
            achievements.append("Hydration goal met!")
        
        if tracking_data.get("sleep_hours", 0) >= 7:
            points += 25
            achievements.append("Good sleep achieved!")
        
        # Weekly goals
        if tracking_data.get("gym_hours", 0) >= 3:
            points += 50
            achievements.append("Weekly exercise goal completed!")
        
        return {
            "status": "success",
            "points_earned": points,
            "achievements": achievements,
            "total_points": tracking_data.get("total_points", 0) + points,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Tracking update error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

@app.get("/food-database/search")
async def search_food_database(query: str = ""):
    """Search food database for calorie and nutrition information"""
    # Mock food database - in production, this would connect to a real food API
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
    
    if not query:
        return {"foods": list(food_database.keys())}
    
    # Simple search
    matching_foods = {k: v for k, v in food_database.items() if query.lower() in k.lower()}
    
    return {
        "query": query,
        "results": matching_foods,
        "total_results": len(matching_foods)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)