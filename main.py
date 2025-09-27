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
    
    # More nuanced scoring
    glucose = features.get('glucose_level', 100)
    if glucose > 126:  # Diabetic range
        metabolic_score -= 30
    elif glucose > 100:  # Prediabetic range
        metabolic_score -= 15
    
    bmi = features.get('bmi', 25)
    if bmi > 35:  # Severe obesity
        metabolic_score -= 25
    elif bmi > 30:  # Obesity
        metabolic_score -= 15
    elif bmi > 25:  # Overweight
        metabolic_score -= 8
    
    hba1c = features.get('hba1c', 5.7)
    if hba1c > 6.5:  # Diabetic
        metabolic_score -= 20
    elif hba1c > 5.7:  # Prediabetic
        metabolic_score -= 10
    
    # Cardiovascular health score (0-100)
    cardio_score = 100
    
    bp = features.get('blood_pressure', 120)
    if bp > 140:  # Stage 1 hypertension
        cardio_score -= 25
    elif bp > 130:  # Elevated
        cardio_score -= 15
    elif bp > 120:  # Prehypertension
        cardio_score -= 8
    
    cholesterol = features.get('cholesterol_level', 200)
    if cholesterol > 240:  # High
        cardio_score -= 20
    elif cholesterol > 200:  # Borderline high
        cardio_score -= 10
    
    activity = features.get('physical_activity', 5)
    if activity < 3:
        cardio_score -= 15
    elif activity < 5:
        cardio_score -= 8
    
    smoking = features.get('smoking_status', 0)
    if smoking == 2:  # Current smoker
        cardio_score -= 20
    elif smoking == 1:  # Former smoker
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
    """Get personalized recommendations based on top risk factors"""
    
    recommendations = {
        'nutrition': [],
        'fitness': [],
        'lifestyle': []
    }
    
    # Get top 3 contributing factors
    top_factors = [factor for factor, _ in feature_importance[:3]]
    
    if risk_type == 'diabetes':
        # Glucose-specific recommendations
        if any('glucose' in factor for factor in top_factors):
            recommendations['nutrition'].extend([
                "Choose complex carbohydrates over simple sugars",
                "Aim for 25-30g fiber daily from whole foods",
                "Use the plate method: 1/2 vegetables, 1/4 lean protein, 1/4 whole grains",
                "Monitor portion sizes and eat at regular intervals"
            ])
            
        # BMI-specific recommendations
        if 'bmi' in top_factors and input_values.get('bmi', 25) > 25:
            recommendations['nutrition'].extend([
                "Create a moderate 300-500 calorie deficit for gradual weight loss",
                "Focus on nutrient-dense, low-calorie foods",
                "Stay hydrated with water before meals"
            ])
            recommendations['fitness'].extend([
                "Aim for 150 minutes of moderate aerobic activity weekly",
                "Include resistance training 2-3 times per week",
                "Take short walks after meals to help control blood sugar"
            ])
            
        # Activity-specific recommendations
        if 'physical_activity' in top_factors:
            recommendations['fitness'].extend([
                "Start with 10-15 minute walks and gradually increase",
                "Find activities you enjoy to maintain consistency",
                "Monitor blood glucose before and after exercise if diabetic"
            ])
    
    elif risk_type == 'hypertension':
        # Blood pressure specific recommendations
        if 'blood_pressure' in top_factors:
            recommendations['nutrition'].extend([
                "Follow DASH diet principles",
                "Limit sodium to less than 2,300mg daily",
                "Increase potassium-rich foods (bananas, spinach, avocados)",
                "Choose fresh foods over processed options"
            ])
            recommendations['fitness'].extend([
                "Engage in 30 minutes of moderate cardio 5 days per week",
                "Include activities like brisk walking, swimming, or cycling",
                "Practice relaxation techniques to manage stress"
            ])
            
        # Age and metabolic factors
        if any(factor in top_factors for factor in ['age', 'metabolic_syndrome_score']):
            recommendations['lifestyle'].extend([
                "Maintain regular sleep schedule (7-9 hours)",
                "Practice stress management techniques",
                "Consider meditation or yoga for relaxation"
            ])
    
    # General lifestyle recommendations based on risk factors
    if input_values.get('smoking_status', 0) > 0:
        recommendations['lifestyle'].append("Consider smoking cessation programs and support")
        
    if input_values.get('stress_level', 5) > 7:
        recommendations['lifestyle'].extend([
            "Practice daily stress reduction techniques",
            "Consider professional stress management counseling"
        ])
        
    if input_values.get('sleep_quality', 6) < 5:
        recommendations['lifestyle'].extend([
            "Establish consistent bedtime routine",
            "Create optimal sleep environment (cool, dark, quiet)"
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
            }
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)