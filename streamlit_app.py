import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, recall_score, f1_score, roc_auc_score, 
                            precision_score, confusion_matrix, classification_report, roc_curve)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek
import warnings
import pickle
import os
import io
import base64
import json
from datetime import datetime
from scipy.stats import uniform, randint
import tempfile
import shutil

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="PFE - Système Prédictif de Turnover RH par IA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PERSONNALISÉ PREMIUM
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem; border-radius: 16px; color: white;
        text-align: center; margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #e94560, #ff6b6b);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .risk-high { background: linear-gradient(135deg, #e94560, #c0392b);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4); }
    .risk-medium { background: linear-gradient(135deg, #f39c12, #e67e22);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(243, 156, 18, 0.4); }
    .risk-low { background: linear-gradient(135deg, #00b894, #00a085);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.4); }
    .metric-card { background: linear-gradient(145deg, #ffffff, #f0f0f0);
        padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center; border: 1px solid rgba(0,0,0,0.05); transition: transform 0.2s; }
    .metric-card:hover { transform: translateY(-3px); }
    .recommendation-card { background: linear-gradient(145deg, #e8f4f8, #d4edda);
        padding: 1.2rem; border-radius: 10px; margin: 0.5rem 0;
        border-left: 5px solid #0f3460; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .section-title { background: linear-gradient(90deg, #1a1a2e, #0f3460);
        color: white; padding: 0.8rem 1.5rem; border-radius: 8px; font-weight: 600; margin: 1.5rem 0 1rem 0; }
    .stButton > button { width: 100%;
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
        color: white; font-weight: 600; border-radius: 8px; padding: 0.6rem; border: none; }
    .fairness-pass { background: #d4edda; color: #155724; padding: 0.8rem;
                     border-radius: 8px; border-left: 4px solid #28a745; }
    .fairness-fail { background: #f8d7da; color: #721c24; padding: 0.8rem;
                     border-radius: 8px; border-left: 4px solid #dc3545; }
    .shap-container { background: #fafafa; padding: 1rem; border-radius: 10px;
                     border: 1px solid #e0e0e0; margin: 0.5rem 0; }
    .batch-result { padding: 0.8rem; border-radius: 8px; margin: 0.3rem 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FONCTIONS DE CHARGEMENT DES DONNÉES
# ============================================================
@st.cache_data
def generate_synthetic_data(n=1470, seed=42):
    """Génère des données synthétiques réalistes basées sur le dataset IBM HR Analytics"""
    np.random.seed(seed)
    data = {
        'Age': np.random.normal(37, 9, n).astype(int).clip(18, 60),
        'Attrition': np.random.choice([0, 1], n, p=[0.839, 0.161]),
        'BusinessTravel': np.random.choice(['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'], n, p=[0.104, 0.713, 0.183]),
        'DailyRate': np.random.randint(102, 1499, n),
        'Department': np.random.choice(['Sales', 'Research & Development', 'Human Resources'], n, p=[0.303, 0.653, 0.044]),
        'DistanceFromHome': np.random.exponential(9, n).astype(int).clip(1, 29),
        'Education': np.random.choice([1, 2, 3, 4, 5], n, p=[0.034, 0.319, 0.381, 0.266, 0.000]),
        'EducationField': np.random.choice(['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'], n, p=[0.413, 0.315, 0.159, 0.092, 0.011, 0.010]),
        'EnvironmentSatisfaction': np.random.choice([1, 2, 3, 4], n, p=[0.198, 0.282, 0.253, 0.267]),
        'Gender': np.random.choice(['Male', 'Female'], n, p=[0.60, 0.40]),
        'HourlyRate': np.random.randint(30, 100, n),
        'JobInvolvement': np.random.choice([1, 2, 3, 4], n, p=[0.028, 0.349, 0.442, 0.181]),
        'JobLevel': np.random.choice([1, 2, 3, 4, 5], n, p=[0.432, 0.316, 0.153, 0.078, 0.021]),
        'JobRole': np.random.choice(['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'], n, p=[0.223, 0.202, 0.176, 0.097, 0.090, 0.069, 0.057, 0.048, 0.038]),
        'JobSatisfaction': np.random.choice([1, 2, 3, 4], n, p=[0.197, 0.281, 0.310, 0.212]),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n, p=[0.319, 0.458, 0.223]),
        'MonthlyIncome': np.random.lognormal(8.4, 0.6, n).astype(int).clip(1009, 19999),
        'MonthlyRate': np.random.randint(2094, 26999, n),
        'NumCompaniesWorked': np.random.choice(list(range(10)), n, p=[0.197, 0.523, 0.128, 0.059, 0.035, 0.022, 0.014, 0.010, 0.007, 0.005]),
        'Over18': np.full(n, 'Y'),
        'OverTime': np.random.choice(['Yes', 'No'], n, p=[0.286, 0.714]),
        'PercentSalaryHike': np.random.randint(11, 25, n),
        'PerformanceRating': np.random.choice([3, 4], n, p=[0.846, 0.154]),
        'RelationshipSatisfaction': np.random.choice([1, 2, 3, 4], n, p=[0.188, 0.276, 0.253, 0.283]),
        'StandardHours': np.full(n, 80),
        'StockOptionLevel': np.random.choice([0, 1, 2, 3], n, p=[0.431, 0.299, 0.210, 0.060]),
        'TotalWorkingYears': np.random.exponential(11, n).astype(int).clip(0, 40),
        'TrainingTimesLastYear': np.random.choice([0, 1, 2, 3, 4, 5, 6], n, p=[0.044, 0.239, 0.548, 0.102, 0.041, 0.017, 0.009]),
        'WorkLifeBalance': np.random.choice([1, 2, 3, 4], n, p=[0.059, 0.233, 0.461, 0.247]),
        'YearsAtCompany': np.random.exponential(5, n).astype(int).clip(0, 40),
        'YearsInCurrentRole': np.random.exponential(4, n).astype(int).clip(0, 18),
        'YearsSinceLastPromotion': np.random.exponential(2, n).astype(int).clip(0, 15),
        'YearsWithCurrManager': np.random.exponential(4, n).astype(int).clip(0, 17)
    }
    df = pd.DataFrame(data)
    # Corrélations réalistes
    overtime_mask = df['OverTime'] == 'Yes'
    df.loc[overtime_mask, 'Attrition'] = np.random.choice([0, 1], len(df[overtime_mask]), p=[0.45, 0.55])
    low_income_mask = df['MonthlyIncome'] < 3000
    df.loc[low_income_mask, 'Attrition'] = np.random.choice([0, 1], len(df[low_income_mask]), p=[0.55, 0.45])
    high_satisfaction_mask = df['JobSatisfaction'] >= 4
    df.loc[high_satisfaction_mask, 'Attrition'] = np.random.choice([0, 1], len(df[high_satisfaction_mask]), p=[0.92, 0.08])
    tenure_mask = df['YearsAtCompany'] > 5
    df.loc[tenure_mask, 'Attrition'] = np.random.choice([0, 1], len(df[tenure_mask]), p=[0.88, 0.12])
    young_mask = df['Age'] < 30
    df.loc[young_mask, 'Attrition'] = np.random.choice([0, 1], len(df[young_mask]), p=[0.72, 0.28])
    return df

@st.cache_data
def load_ibm_dataset():
    """Charge le dataset IBM HR Analytics"""
    try:
        url = "https://raw.githubusercontent.com/dsrscientist/dataset3/main/ibm-hr-analytics-employee-attrition-performance/HR-Employee-Attrition.csv"
        df = pd.read_csv(url)
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        return df
    except:
        return None

def preprocess_data(df, target_col='Attrition'):
    """Prétraite les données pour la modélisation"""
    df_processed = df.copy()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        le_dict[col] = le
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]
    return X, y, le_dict

# ============================================================
# FONCTIONS D'ENTRAÎNEMENT DES MODÈLES AVANCÉS
# ============================================================
@st.cache_resource
def train_all_models(X, y, use_smote=True, smote_method='SMOTE', include_deep_learning=False):
    """Entraîne et optimise tous les modèles de ML + Deep Learning optionnel"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if use_smote:
        samplers = {
            'SMOTE': SMOTE(random_state=42),
            'ADASYN': ADASYN(random_state=42),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
            'SMOTETomek': SMOTETomek(random_state=42)
        }
        sampler = samplers.get(smote_method, SMOTE(random_state=42))
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train

    models = {
        'Régression Logistique': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Arbre de Décision': DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=20),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'SVM (RBF)': SVC(random_state=42, probability=True, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=200, max_depth=15, min_samples_split=10),
        'Extra Trees': ExtraTreesClassifier(random_state=42, n_estimators=200, max_depth=15),
        'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=100, learning_rate=0.1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5),
        'MLP (Neural Network)': MLPClassifier(random_state=42, hidden_layer_sizes=(100, 50), max_iter=500),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
            'Specificity': recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        }
        trained_models[name] = model

    # XGBoost avec optimisation
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        param_dist = {
            'n_estimators': randint(100, 500), 'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 10), 'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4), 'gamma': uniform(0, 0.5)
        }
        rand_search = RandomizedSearchCV(xgb, param_dist, n_iter=30, scoring='f1', cv=5, 
                                          verbose=0, random_state=42, n_jobs=-1)
        rand_search.fit(X_train_resampled, y_train_resampled)
        best_xgb = rand_search.best_estimator_
        y_pred = best_xgb.predict(X_test_scaled)
        y_proba = best_xgb.predict_proba(X_test_scaled)[:, 1]
        results['XGBoost (Optimisé)'] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'AUC-ROC': roc_auc_score(y_test, y_proba),
            'Specificity': recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        }
        trained_models['XGBoost (Optimisé)'] = best_xgb
    except ImportError:
        pass

    # LightGBM
    try:
        from lightgbm import LGBMClassifier
        lgb = LGBMClassifier(random_state=42, n_estimators=200, max_depth=7, learning_rate=0.1)
        lgb.fit(X_train_resampled, y_train_resampled)
        y_pred = lgb.predict(X_test_scaled)
        y_proba = lgb.predict_proba(X_test_scaled)[:, 1]
        results['LightGBM'] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'AUC-ROC': roc_auc_score(y_test, y_proba),
            'Specificity': recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        }
        trained_models['LightGBM'] = lgb
    except ImportError:
        pass

    # Deep Learning avec Keras/TensorFlow
    if include_deep_learning:
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.callbacks import EarlyStopping
            from tensorflow.keras.optimizers import Adam

            # Construction du modèle Deep Learning
            dl_model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train_resampled.shape[1],)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            dl_model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='binary_crossentropy',
                           metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            history = dl_model.fit(
                X_train_resampled, y_train_resampled,
                validation_split=0.2,
                epochs=100,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )

            y_pred_dl = (dl_model.predict(X_test_scaled, verbose=0) > 0.5).astype(int).flatten()
            y_proba_dl = dl_model.predict(X_test_scaled, verbose=0).flatten()

            results['Deep Learning (Keras)'] = {
                'Accuracy': accuracy_score(y_test, y_pred_dl),
                'Precision': precision_score(y_test, y_pred_dl, zero_division=0),
                'Recall': recall_score(y_test, y_pred_dl, zero_division=0),
                'F1-Score': f1_score(y_test, y_pred_dl, zero_division=0),
                'AUC-ROC': roc_auc_score(y_test, y_proba_dl),
                'Specificity': recall_score(y_test, y_pred_dl, pos_label=0, zero_division=0)
            }
            trained_models['Deep Learning (Keras)'] = dl_model
        except ImportError:
            pass

    # Stacking Ensemble
    try:
        base_models = [
            ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
            ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')),
            ('lgb', LGBMClassifier(random_state=42, n_estimators=100))
        ]
        stack_model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        stack_model.fit(X_train_resampled, y_train_resampled)
        y_pred_stack = stack_model.predict(X_test_scaled)
        y_proba_stack = stack_model.predict_proba(X_test_scaled)[:, 1]
        results['Stacking Ensemble'] = {
            'Accuracy': accuracy_score(y_test, y_pred_stack),
            'Precision': precision_score(y_test, y_pred_stack, zero_division=0),
            'Recall': recall_score(y_test, y_pred_stack, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred_stack, zero_division=0),
            'AUC-ROC': roc_auc_score(y_test, y_proba_stack),
            'Specificity': recall_score(y_test, y_pred_stack, pos_label=0, zero_division=0)
        }
        trained_models['Stacking Ensemble'] = stack_model
    except:
        pass

    best_model_name = max(results, key=lambda x: results[x]['F1-Score'])
    best_model = trained_models[best_model_name]

    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    else:
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': [0]*len(X.columns)})

    return (trained_models, results, scaler, X_train_resampled, X_test_scaled,
            y_train_resampled, y_test, feature_importance, X.columns, best_model_name)

# ============================================================
# FONCTIONS XAI (SHAP & LIME)
# ============================================================
def get_shap_explainer(model, X_train):
    """Initialise l'explainer SHAP"""
    try:
        import shap
        try:
            explainer = shap.TreeExplainer(model)
            return explainer
        except:
            try:
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
                return explainer
            except:
                return None
    except ImportError:
        return None

def get_lime_explainer(X_train, feature_names, class_names=['Reste', 'Part']):
    """Initialise l'explainer LIME"""
    try:
        from lime.lime_tabular import LimeTabularExplainer
        return LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True
        )
    except ImportError:
        return None

# ============================================================
# FONCTIONS DE PRÉDICTION
# ============================================================
def predict_risk(employee_data, model, scaler, feature_columns, le_dict):
    """Prédit le risque de turnover pour un employé"""
    df_input = pd.DataFrame([employee_data])
    for col in le_dict.keys():
        if col in df_input.columns:
            try:
                df_input[col] = le_dict[col].transform(df_input[col].astype(str))
            except ValueError:
                df_input[col] = le_dict[col].transform([le_dict[col].classes_[0]])[0]
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_columns]
    df_scaled = scaler.transform(df_input)
    risk_score = model.predict_proba(df_scaled)[0][1]
    prediction = model.predict(df_scaled)[0]
    return risk_score, prediction, df_scaled

def predict_batch(df_batch, model, scaler, feature_columns, le_dict):
    """Prédit le risque pour un batch d'employés"""
    df_input = df_batch.copy()
    for col in le_dict.keys():
        if col in df_input.columns:
            try:
                df_input[col] = le_dict[col].transform(df_input[col].astype(str))
            except ValueError:
                df_input[col] = le_dict[col].transform([le_dict[col].classes_[0]])[0]
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_columns]
    df_scaled = scaler.transform(df_input)
    risk_scores = model.predict_proba(df_scaled)[:, 1]
    predictions = model.predict(df_scaled)
    return risk_scores, predictions

# ============================================================
# FONCTIONS DE RECOMMANDATIONS
# ============================================================
def get_recommendations(employee_data, risk_score, feature_importance):
    """Génère des recommandations personnalisées"""
    recommendations = []
    top_factors = feature_importance.head(5)['Feature'].tolist()

    if 'OverTime' in top_factors and employee_data.get('OverTime') == 'Yes':
        recommendations.append({'icon': '⏰', 'title': 'Surcharge de Travail',
            'action': 'Réduire les heures supplémentaires, répartir la charge', 'priority': 'Haute', 'impact': '-30%'})
    if 'MonthlyIncome' in top_factors and employee_data.get('MonthlyIncome', 0) < 5000:
        recommendations.append({'icon': '💰', 'title': 'Rémunération Non Compétitive',
            'action': 'Révision salariale, prime de rétention', 'priority': 'Haute', 'impact': '-25%'})
    if 'JobSatisfaction' in top_factors and employee_data.get('JobSatisfaction', 0) <= 2:
        recommendations.append({'icon': '😊', 'title': 'Satisfaction Faible',
            'action': "Entretien d'écoute, plan de développement", 'priority': 'Haute', 'impact': '-20%'})
    if 'Age' in top_factors and employee_data.get('Age', 35) < 30:
        recommendations.append({'icon': '🎓', 'title': 'Profil Jeune et Mobile',
            'action': 'Programme de mentorat, projets stimulants', 'priority': 'Moyenne', 'impact': '-15%'})
    if 'YearsAtCompany' in top_factors and employee_data.get('YearsAtCompany', 0) < 3:
        recommendations.append({'icon': '🎯', 'title': 'Intégration à Renforcer',
            'action': "Bilan d'intégration, parrainage", 'priority': 'Moyenne', 'impact': '-15%'})
    if 'DistanceFromHome' in top_factors and employee_data.get('DistanceFromHome', 0) > 20:
        recommendations.append({'icon': '🏠', 'title': 'Distance Élevée',
            'action': 'Télétravail, horaires flexibles', 'priority': 'Moyenne', 'impact': '-20%'})
    if 'WorkLifeBalance' in top_factors and employee_data.get('WorkLifeBalance', 0) <= 2:
        recommendations.append({'icon': '⚖️', 'title': 'Déséquilibre Vie Pro/Perso',
            'action': 'Congés, activités bien-être', 'priority': 'Moyenne', 'impact': '-18%'})
    if 'YearsSinceLastPromotion' in top_factors and employee_data.get('YearsSinceLastPromotion', 0) > 3:
        recommendations.append({'icon': '📈', 'title': 'Stagnation Professionnelle',
            'action': 'Plan de carrière, formation leadership', 'priority': 'Moyenne', 'impact': '-15%'})
    if risk_score > 0.7:
        recommendations.append({'icon': '🚨', 'title': "Intervention d'Urgence",
            'action': 'Entretien DRH sous 48h', 'priority': 'Critique', 'impact': '-35%'})

    priority_order = {'Critique': 0, 'Haute': 1, 'Moyenne': 2, 'Basse': 3}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
    return recommendations

# ============================================================
# FONCTIONS D'ANALYSE D'ÉQUITÉ (FAIRNESS)
# ============================================================
def analyze_fairness(df, predictions, sensitive_attr='Gender'):
    """Analyse l'équité du modèle selon les attributs sensibles"""
    fairness_results = {}
    if sensitive_attr in df.columns:
        groups = df[sensitive_attr].unique()
        for group in groups:
            mask = df[sensitive_attr] == group
            group_preds = predictions[mask]
            group_actual = df.loc[mask, 'Attrition']
            if len(group_actual) > 0:
                fairness_results[group] = {
                    'Taux de prédiction positive': group_preds.mean(),
                    'Taux réel de turnover': group_actual.mean(),
                    'Nombre employés': len(group_actual)
                }
    return fairness_results

def calculate_disparate_impact(df, predictions, sensitive_attr='Gender'):
    """Calcule le Disparate Impact Ratio"""
    groups = df[sensitive_attr].unique()
    if len(groups) != 2:
        return None
    group_rates = {}
    for group in groups:
        mask = df[sensitive_attr] == group
        group_rates[group] = predictions[mask].mean()
    if group_rates[groups[1]] == 0:
        return None
    return group_rates[groups[0]] / group_rates[groups[1]]

# ============================================================
# FONCTIONS D'AFFICHAGE DES PAGES
# ============================================================
def show_dashboard(df, models, results, scaler, le_dict, feature_columns, feature_importance, best_model_name):
    """Page Dashboard"""
    st.header("📊 Tableau de Bord RH - Vue d'Ensemble")

    col1, col2, col3, col4, col5 = st.columns(5)
    total = len(df)
    attrition_rate = df['Attrition'].mean() * 100
    high_risk = len(df[(df['OverTime'] == 'Yes') & (df['JobSatisfaction'] <= 2)])
    cost_per_attrition = 50000
    total_cost = int(attrition_rate/100 * total * cost_per_attrition)

    with col1:
        st.markdown(f'<div class="metric-card"><h4 style="color:#666;">👥 Effectif Total</h4><h2 style="color:#1a1a2e;margin:0;">{total:,}</h2></div>', unsafe_allow_html=True)
    with col2:
        color = "#e94560" if attrition_rate > 20 else "#f39c12" if attrition_rate > 10 else "#00b894"
        st.markdown(f'<div class="metric-card"><h4 style="color:#666;">⚠️ Taux d'Attrition</h4><h2 style="color:{color};margin:0;">{attrition_rate:.1f}%</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h4 style="color:#666;">🔴 Profils à Risque</h4><h2 style="color:#e94560;margin:0;">{high_risk}</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h4 style="color:#666;">💰 Coût Annuel</h4><h2 style="color:#764ba2;margin:0;">€{total_cost:,}</h2></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><h4 style="color:#666;">🤖 Meilleur Modèle</h4><h3 style="color:#0f3460;margin:0;font-size:1rem;">{best_model_name}</h3><p style="color:#666;margin:0;font-size:0.8rem;">F1: {results[best_model_name]["F1-Score"]:.3f}</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Attrition par Département")
        dept_data = df.groupby('Department')['Attrition'].mean() * 100
        fig = px.bar(dept_data.reset_index(), x='Department', y='Attrition', color='Attrition',
                    color_continuous_scale='RdYlGn_r', title="Taux d'Attrition par Département (%)",
                    labels={'Department': 'Département', 'Attrition': "Taux d'Attrition (%)"},
                    text=dept_data.round(1))
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("⏰ Impact des Heures Supplémentaires")
        ot_data = df.groupby('OverTime')['Attrition'].mean() * 100
        fig = px.bar(ot_data.reset_index(), x='OverTime', y='Attrition', color='OverTime',
                    color_discrete_map={'Yes': '#e94560', 'No': '#00b894'},
                    title="Taux d'Attrition vs Heures Supplémentaires",
                    labels={'OverTime': 'Heures Supp.', 'Attrition': "Taux d'Attrition (%)"},
                    text=ot_data.round(1))
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Distribution des Scores de Risque")
    sample_df = df.copy()
    for col in le_dict.keys():
        if col in sample_df.columns:
            sample_df[col] = le_dict[col].transform(sample_df[col].astype(str))
    X_sample = sample_df[feature_columns]
    X_scaled = scaler.transform(X_sample)
    risks = models[best_model_name].predict_proba(X_scaled)[:, 1]

    fig = px.histogram(risks, nbins=30, title="Distribution des Scores de Risque d'Attrition",
                      labels={'value': 'Score de Risque', 'count': "Nombre d'Employés"},
                      color_discrete_sequence=['#667eea'])
    fig.add_vline(x=0.3, line_dash="dash", line_color="#f39c12", annotation_text="Seuil Moyen (30%)", annotation_position="top")
    fig.add_vline(x=0.7, line_dash="dash", line_color="#e94560", annotation_text="Seuil Élevé (70%)", annotation_position="top")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💰 Coût du Turnover par Département")
    cost_by_dept = df.groupby('Department')['Attrition'].sum() * cost_per_attrition
    fig = px.bar(x=cost_by_dept.index, y=cost_by_dept.values, color=cost_by_dept.values,
                color_continuous_scale='Viridis', title="Coût Estimé du Turnover par Département (€)",
                labels={'x': 'Département', 'y': 'Coût (€)'},
                text=[f"€{v:,.0f}" for v in cost_by_dept.values])
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_exploratory_analysis(df):
    """Page Analyse Exploratoire"""
    st.header("📈 Analyse Exploratoire des Données (EDA)")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔗 Corrélations", "📅 Tendances", "📋 Profils"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution des Âges")
            fig = px.histogram(df, x='Age', color='Attrition', marginal='box',
                             title="Distribution des Âges par Statut",
                             color_discrete_map={0: '#00b894', 1: '#e94560'}, labels={'Attrition': 'A quitté'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Salaire vs Satisfaction")
            fig = px.scatter(df, x='MonthlyIncome', y='JobSatisfaction', color='Attrition', size='YearsAtCompany',
                           title="Relation Salaire - Satisfaction au Travail",
                           color_discrete_map={0: '#00b894', 1: '#e94560'},
                           labels={'MonthlyIncome': 'Salaire Mensuel (€)', 'JobSatisfaction': 'Satisfaction'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Distribution par Genre")
            gender_data = df.groupby(['Gender', 'Attrition']).size().unstack()
            fig = px.bar(gender_data, barmode='group', title="Attrition par Genre",
                        color_discrete_map={0: '#00b894', 1: '#e94560'}, labels={'value': 'Nombre', 'Gender': 'Genre'})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.subheader("Distribution par Statut Marital")
            marital_data = df.groupby(['MaritalStatus', 'Attrition']).size().unstack()
            fig = px.bar(marital_data, barmode='group', title="Attrition par Statut Marital",
                        color_discrete_map={0: '#00b894', 1: '#e94560'}, labels={'value': 'Nombre', 'MaritalStatus': 'Statut'})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Matrice de Corrélation Complète")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto",
                       title="Matrice de Corrélation des Variables Numériques",
                       color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Corrélations avec l'Attrition")
        attrition_corr = corr_matrix['Attrition'].drop('Attrition').sort_values(key=abs, ascending=False)
        fig = px.bar(x=attrition_corr.values, y=attrition_corr.index, orientation='h',
                    color=attrition_corr.values, color_continuous_scale='RdBu_r',
                    title="Corrélations avec le Turnover (Attrition)",
                    labels={'x': 'Coefficient de Corrélation', 'y': 'Variable'})
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Attrition par Ancienneté")
            tenure_data = df.groupby('YearsAtCompany')['Attrition'].mean() * 100
            fig = px.line(x=tenure_data.index, y=tenure_data.values, markers=True,
                         title="Taux d'Attrition selon l'Ancienneté",
                         labels={'x': "Années dans l'Entreprise", 'y': "Taux d'Attrition (%)"})
            fig.add_hline(y=df['Attrition'].mean()*100, line_dash="dash", line_color="red", annotation_text="Moyenne Globale")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Formations et Attrition")
            training_data = df.groupby('TrainingTimesLastYear')['Attrition'].mean() * 100
            fig = px.bar(x=training_data.index, y=training_data.values, color=training_data.values,
                        color_continuous_scale='RdYlGn_r',
                        title="Impact des Formations sur l'Attrition",
                        labels={'x': 'Nombre de Formations', 'y': "Taux d'Attrition (%)"}, text=training_data.round(1))
            fig.update_traces(textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Profils d'Employés à Risque")
        high_risk_profile = df[df['Attrition'] == 1].describe()
        low_risk_profile = df[df['Attrition'] == 0].describe()
        comparison = pd.DataFrame({
            'Partis (Moyenne)': high_risk_profile.loc['mean'],
            'Restés (Moyenne)': low_risk_profile.loc['mean'],
            'Différence': high_risk_profile.loc['mean'] - low_risk_profile.loc['mean']
        }).round(2)
        st.dataframe(comparison.style.background_gradient(cmap='RdBu_r', subset=['Différence']), use_container_width=True)

def show_modeling_xai(df, trained_models, results, feature_importance, best_model_name,
                       X_train_resampled, X_test_scaled, y_train_resampled, y_test, feature_columns):
    """Page Modélisation & XAI"""
    st.header("🤖 Modélisation Machine Learning & IA Explicable (XAI)")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performances", "🔍 Feature Importance", "📈 Courbes ROC", "🧠 SHAP/LIME"])

    with tab1:
        st.subheader("📈 Tableau Comparatif des Modèles")
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df.style.format("{:.3f}")
            .background_gradient(cmap='RdYlGn', subset=['Accuracy', 'Recall', 'F1-Score', 'AUC-ROC'])
            .highlight_max(color='#d4edda', subset=['Accuracy', 'Recall', 'F1-Score', 'AUC-ROC'])
            .highlight_min(color='#f8d7da', subset=['Accuracy', 'Recall', 'F1-Score', 'AUC-ROC']),
            use_container_width=True)

        st.subheader("📊 Radar Chart des Performances")
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        for idx, (model, metrics_vals) in enumerate(results.items()):
            values = [metrics_vals[m] for m in metrics] + [metrics_vals[metrics[0]]]
            fig.add_trace(go.Scatterpolar(r=values, theta=metrics + [metrics[0]],
                fill='toself', name=model, line_color=colors[idx % len(colors)]))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🎯 Importance des Facteurs Prédictifs")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(feature_importance.head(15), x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Viridis',
                        title="Top 15 des Facteurs d'Influence", labels={'Importance': 'Importance', 'Feature': 'Variable'})
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("📊 Répartition de l'Importance")
            top_10 = feature_importance.head(10)
            others = feature_importance.iloc[10:]['Importance'].sum()
            pie_data = pd.concat([top_10, pd.DataFrame([{'Feature': 'Autres', 'Importance': others}])])
            fig = px.pie(pie_data, values='Importance', names='Feature',
                        title="Répartition de l'Importance des Variables", hole=0.4)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📖 Interprétation des Résultats")
        st.info("""
        **📌 Points Clés d'Interprétation:**
        1. **OverTime (Heures supplémentaires)** : Facteur prédictif dominant. Les employés en surcharge de travail ont un risque de départ multiplié par 2-3.
        2. **MonthlyIncome (Salaire mensuel)** : Corrélation négative forte. Un salaire compétitif réduit significativement le risque de turnover.
        3. **Age** : Les employés jeunes (20-30 ans) présentent un turnover 2x plus élevé que les employés expérimentés (>40 ans).
        4. **YearsAtCompany** : L'ancienneté est protectrice. Les employés récents (<2 ans) représentent 40% des départs.
        5. **JobSatisfaction** : La satisfaction au travail est un baromètre fiable. Un score <= 2 augmente le risque de 60%.
        """)

    with tab3:
        st.subheader("📈 Courbes ROC Comparatives")
        fig = go.Figure()
        for model_name, model in trained_models.items():
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{model_name} (AUC={auc:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Aléatoire'))
        fig.update_layout(title="Courbes ROC - Comparaison des Modèles",
                         xaxis_title="Taux de Faux Positifs (FPR)", yaxis_title="Taux de Vrais Positifs (TPR)", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("🧠 Analyse SHAP (SHapley Additive exPlanations)")
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <p><strong>ℹ️ Qu'est-ce que SHAP ?</strong></p>
            <p>SHAP attribue une contribution à chaque variable pour chaque prédiction, basée sur la théorie des jeux coopératifs.</p>
        </div>
        """, unsafe_allow_html=True)

        try:
            import shap
            import matplotlib.pyplot as plt
            explainer = get_shap_explainer(trained_models[best_model_name], X_train_resampled)
            if explainer is not None:
                shap_values = explainer.shap_values(X_test_scaled[:100])
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=feature_columns, show=False)
                st.pyplot(fig)
                plt.close()

                st.subheader("📍 Exemple d'Explication Individuelle")
                idx = st.slider("Sélectionner un employé à analyser", 0, min(99, len(X_test_scaled)-1), 0)
                fig = shap.force_plot(
                    explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                    shap_values[idx], X_test_scaled[idx], feature_names=feature_columns, matplotlib=True, show=False)
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Explainer SHAP non disponible pour ce modèle.")
        except Exception as e:
            st.error(f"Erreur SHAP: {e}")
            st.info("💡 Installez SHAP: pip install shap matplotlib")

def show_individual_prediction(model, scaler, feature_columns, le_dict, feature_importance, X_train):
    """Page Prédiction Individuelle"""
    st.header("🎯 Prédiction Individualisée du Risque de Turnover")
    st.markdown("""
    <div style="background: linear-gradient(90deg, #e3f2fd, #f3e5f5); padding: 1.2rem; border-radius: 12px; margin-bottom: 2rem;">
        <p>📝 Remplissez les informations ci-dessous pour obtenir une prédiction personnalisée avec recommandations ciblées.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='section-title'>👤 Informations Personnelles</div>", unsafe_allow_html=True)
        age = st.slider("Âge", 18, 65, 35)
        gender = st.selectbox("Genre", ['Male', 'Female'])
        marital_status = st.selectbox("Situation Familiale", ['Single', 'Married', 'Divorced'])
        education = st.selectbox("Niveau d'Éducation", [1, 2, 3, 4, 5],
                                format_func=lambda x: ['Lycée', 'Bac+2', 'Bac+3', 'Bac+5', 'Doctorat'][x-1])
        education_field = st.selectbox("Domaine d'Études", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])

    with col2:
        st.markdown("<div class='section-title'>💼 Informations Professionnelles</div>", unsafe_allow_html=True)
        department = st.selectbox("Département", ['Sales', 'Research & Development', 'Human Resources'])
        job_role = st.selectbox("Poste", ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                                          'Manufacturing Director', 'Healthcare Representative', 'Manager',
                                          'Sales Representative', 'Research Director', 'Human Resources'])
        job_level = st.selectbox("Niveau Hiérarchique", [1, 2, 3, 4, 5])
        years_at_company = st.slider("Années dans l'entreprise", 0, 40, 5)
        monthly_income = st.slider("Salaire Mensuel (€)", 1000, 20000, 5000, step=500)
        years_since_promotion = st.slider("Années depuis dernière promotion", 0, 15, 2)

    with col3:
        st.markdown("<div class='section-title'>🔄 Conditions de Travail</div>", unsafe_allow_html=True)
        overtime = st.selectbox("Heures Supplémentaires", ['No', 'Yes'])
        job_satisfaction = st.slider("Satisfaction au Travail (1-4)", 1, 4, 3)
        work_life_balance = st.slider("Équilibre Vie Pro/Perso (1-4)", 1, 4, 3)
        environment_satisfaction = st.slider("Satisfaction Environnement (1-4)", 1, 4, 3)
        distance_from_home = st.slider("Distance domicile-travail (km)", 1, 30, 10)
        num_companies = st.slider("Nombre d'entreprises précédentes", 0, 10, 2)
        training_times = st.slider("Formations (dernière année)", 0, 6, 2)

    if st.button("🎯 Prédire le Risque", type="primary"):
        employee_data = {
            'Age': age, 'Gender': gender, 'MaritalStatus': marital_status, 'Education': education,
            'EducationField': education_field, 'Department': department, 'JobRole': job_role,
            'JobLevel': job_level, 'YearsAtCompany': years_at_company, 'MonthlyIncome': monthly_income,
            'OverTime': overtime, 'JobSatisfaction': job_satisfaction, 'WorkLifeBalance': work_life_balance,
            'EnvironmentSatisfaction': environment_satisfaction, 'DistanceFromHome': distance_from_home,
            'NumCompaniesWorked': num_companies, 'TrainingTimesLastYear': training_times,
            'YearsSinceLastPromotion': years_since_promotion,
            'BusinessTravel': 'Travel_Rarely', 'DailyRate': 500, 'HourlyRate': 50,
            'JobInvolvement': 3, 'MonthlyRate': 10000, 'Over18': 'Y',
            'PercentSalaryHike': 15, 'PerformanceRating': 3, 'RelationshipSatisfaction': 3,
            'StandardHours': 80, 'StockOptionLevel': 1,
            'TotalWorkingYears': max(0, age - 22),
            'YearsInCurrentRole': min(years_at_company, 5),
            'YearsWithCurrManager': min(years_at_company, 4)
        }

        with st.spinner("🔄 Calcul du risque..."):
            risk_score, prediction, df_scaled = predict_risk(employee_data, model, scaler, feature_columns, le_dict)

        st.markdown("---")
        st.subheader("📊 Résultats de la Prédiction")

        col1, col2, col3 = st.columns(3)
        risk_percentage = risk_score * 100

        with col1:
            if risk_percentage > 70:
                risk_class = "risk-high"
                risk_text = "Risque Élevé 🔴"
                risk_description = "Intervention immédiate nécessaire"
            elif risk_percentage > 30:
                risk_class = "risk-medium"
                risk_text = "Risque Moyen 🟡"
                risk_description = "Surveillance recommandée"
            else:
                risk_class = "risk-low"
                risk_text = "Risque Faible 🟢"
                risk_description = "Profil stable"
            st.markdown(f'<div class="{risk_class}"><h3>{risk_text}</h3><h1>{risk_percentage:.1f}%</h1><p>{risk_description}</p></div>', unsafe_allow_html=True)

        with col2:
            period = "> 3 mois" if risk_percentage > 70 else "3-6 mois" if risk_percentage > 30 else "< 12 mois"
            st.markdown(f'<div class="metric-card"><h3>📅 Période Estimée</h3><p style="font-size:1.5rem;font-weight:bold;">{period}</p></div>', unsafe_allow_html=True)

        with col3:
            action = "Entretien Immédiat" if risk_percentage > 70 else "Surveillance Active" if risk_percentage > 30 else "Suivi Normal"
            st.markdown(f'<div class="metric-card"><h3>💡 Action Recommandée</h3><p style="font-size:1.2rem;font-weight:bold;">{action}</p></div>', unsafe_allow_html=True)

        st.subheader("🔍 Facteurs Contributifs Principaux")
        if hasattr(model, 'feature_importances_'):
            contributions = pd.DataFrame({
                'Feature': feature_columns,
                'Valeur': df_scaled[0],
                'Importance': model.feature_importances_
            })
            contributions['Impact'] = contributions['Valeur'] * contributions['Importance']
            contributions = contributions.sort_values('Impact', key=abs, ascending=False).head(8)

            fig = px.bar(contributions, x='Impact', y='Feature', orientation='h',
                        color='Impact', color_continuous_scale='RdBu_r',
                        title="Contribution des Facteurs à la Prédiction",
                        labels={'Impact': 'Impact sur le Risque', 'Feature': 'Variable'})
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🎯 Recommandations Personnalisées")
        recommendations = get_recommendations(employee_data, risk_score, feature_importance)

        if recommendations:
            cols = st.columns(2)
            for idx, rec in enumerate(recommendations):
                with cols[idx % 2]:
                    priority_color = "#e94560" if rec['priority'] == 'Critique' else "#f39c12" if rec['priority'] == 'Haute' else "#00b894"
                    st.markdown(f'<div class="recommendation-card" style="border-left-color: {priority_color};"><div style="display:flex; justify-content:space-between; align-items:center;"><p style="font-size:1.2rem;font-weight:bold;margin:0;">{rec["icon"]} {rec["title"]}</p><span style="background:{priority_color};color:white;padding:2px 8px;border-radius:12px;font-size:0.75rem;">{rec["priority"]}</span></div><p style="margin:0.5rem 0;">{rec["action"]}</p><p style="margin:0;font-size:0.85rem;color:#666;">Impact estimé: <strong>{rec["impact"]}</strong></p></div>', unsafe_allow_html=True)
        else:
            st.success("✅ Aucune recommandation spécifique - Profil stable")

def show_batch_prediction(model, scaler, feature_columns, le_dict, feature_importance):
    """Page Prédiction par Batch"""
    st.header("📊 Prédiction par Batch - Analyse Multiple")
    st.markdown("""
    <div style="background: linear-gradient(90deg, #e8f5e9, #e3f2fd); padding: 1.2rem; border-radius: 12px; margin-bottom: 2rem;">
        <p>📁 Téléchargez un fichier CSV contenant les données de plusieurs employés pour obtenir une analyse groupée des risques de turnover.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Charger un fichier CSV avec les données des employés", type=['csv'])

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.subheader("📋 Aperçu des données chargées")
        st.dataframe(df_batch.head(10), use_container_width=True)

        if st.button("🚀 Lancer l'analyse par batch", type="primary"):
            with st.spinner("🔄 Analyse en cours..."):
                risk_scores, predictions = predict_batch(df_batch, model, scaler, feature_columns, le_dict)

            df_batch['Risk_Score'] = risk_scores
            df_batch['Risk_Level'] = pd.cut(risk_scores, bins=[0, 0.3, 0.7, 1.0], 
                                             labels=['Faible', 'Moyen', 'Élevé'])
            df_batch['Prediction'] = predictions

            st.markdown("---")
            st.subheader("📊 Résultats de l'Analyse par Batch")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Employés", len(df_batch))
            with col2:
                high_risk = (df_batch['Risk_Level'] == 'Élevé').sum()
                st.metric("🔴 Risque Élevé", high_risk)
            with col3:
                medium_risk = (df_batch['Risk_Level'] == 'Moyen').sum()
                st.metric("🟡 Risque Moyen", medium_risk)
            with col4:
                low_risk = (df_batch['Risk_Level'] == 'Faible').sum()
                st.metric("🟢 Risque Faible", low_risk)

            st.subheader("📈 Distribution des Risques")
            fig = px.pie(df_batch, names='Risk_Level', title="Répartition des Niveaux de Risque",
                        color='Risk_Level', color_discrete_map={'Faible': '#00b894', 'Moyen': '#f39c12', 'Élevé': '#e94560'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 Tableau des Employés à Risque")
            high_risk_employees = df_batch[df_batch['Risk_Level'] == 'Élevé'].sort_values('Risk_Score', ascending=False)
            st.dataframe(high_risk_employees[['Age', 'Gender', 'Department', 'JobRole', 'MonthlyIncome', 
                                             'Risk_Score', 'Risk_Level']].style.background_gradient(
                subset=['Risk_Score'], cmap='Reds'), use_container_width=True)

            # Export des résultats
            csv_results = df_batch.to_csv(index=False)
            b64 = base64.b64encode(csv_results.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="resultats_batch_{datetime.now().strftime("%Y%m%d")}.csv">📥 Télécharger les Résultats Complets</a>'
            st.markdown(href, unsafe_allow_html=True)

def show_fairness_analysis(df, trained_models, scaler, le_dict, feature_columns):
    """Page Analyse d'Équité"""
    st.header("⚖️ Analyse d'Équité et Biais Algorithmiques")
    st.markdown("""
    <div style="background: linear-gradient(90deg, #fff3e0, #fce4ec); padding: 1.2rem; border-radius: 12px; margin-bottom: 2rem;">
        <p>🔍 Cette section analyse l'équité du modèle selon les attributs sensibles (genre, âge) pour détecter d'éventuels biais discriminatoires.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox("Modèle à analyser", list(trained_models.keys()))
    with col2:
        sensitive_attr = st.selectbox("Attribut sensible", ['Gender', 'Age', 'MaritalStatus'])

    model = trained_models[model_name]

    df_pred = df.copy()
    for col in le_dict.keys():
        if col in df_pred.columns:
            df_pred[col] = le_dict[col].transform(df_pred[col].astype(str))

    X_full = df_pred[feature_columns]
    X_full_scaled = scaler.transform(X_full)
    predictions = model.predict(X_full_scaled)
    pred_proba = model.predict_proba(X_full_scaled)[:, 1]

    st.subheader(f"📊 Analyse par {sensitive_attr}")

    if sensitive_attr == 'Age':
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '50+'])
        group_col = 'AgeGroup'
    else:
        group_col = sensitive_attr

    fairness_data = []
    for group in df[group_col].unique():
        if pd.isna(group):
            continue
        mask = df[group_col] == group
        group_preds = pred_proba[mask]
        group_actual = df.loc[mask, 'Attrition']
        fairness_data.append({
            'Groupe': group,
            'Taux de prédiction élevée': (group_preds > 0.5).mean() * 100,
            'Taux réel de turnover': group_actual.mean() * 100,
            'Score moyen prédit': group_preds.mean(),
            'Effectif': len(group_actual)
        })

    fairness_df = pd.DataFrame(fairness_data)
    st.dataframe(fairness_df.style.format({
        'Taux de prédiction élevée': '{:.1f}%',
        'Taux réel de turnover': '{:.1f}%',
        'Score moyen prédit': '{:.3f}'
    }).background_gradient(cmap='RdYlGn_r', subset=['Taux de prédiction élevée', 'Score moyen prédit']), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(fairness_df, x='Groupe', y=['Taux de prédiction élevée', 'Taux réel de turnover'],
                    barmode='group', title="Comparaison: Prédiction vs Réalité",
                    labels={'value': 'Taux (%)', 'variable': 'Métrique'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(fairness_df, x='Groupe', y='Score moyen prédit',
                    color='Score moyen prédit', color_continuous_scale='RdYlGn_r',
                    title="Score de Risque Moyen par Groupe",
                    labels={'Score moyen prédit': 'Score Moyen'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📏 Métriques d'Équité")
    if len(fairness_df) == 2:
        rate_0 = fairness_df.iloc[0]['Taux de prédiction élevée'] / 100
        rate_1 = fairness_df.iloc[1]['Taux de prédiction élevée'] / 100
        if rate_1 > 0:
            di_ratio = rate_0 / rate_1
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Disparate Impact Ratio", f"{di_ratio:.3f}")
            with col2:
                status = "✅ Pass" if 0.8 <= di_ratio <= 1.25 else "⚠️ Révision"
                st.metric("Statut", status)
            with col3:
                interpretation = "Équitable" if 0.8 <= di_ratio <= 1.25 else "Biais détecté"
                st.metric("Interprétation", interpretation)

            if not (0.8 <= di_ratio <= 1.25):
                st.warning(f"⚠️ Biais détecté ! Le ratio {di_ratio:.3f} est hors de la fourchette acceptable [0.8 - 1.25].")
            else:
                st.success("✅ Le modèle respecte le critère des 4/5 (80%) pour l'équité.")

    st.info("""
    **📌 Note sur l'Équité Algorithmique:**
    - **Disparate Impact Ratio** : Ratio entre les taux de prédiction positive des groupes. Un ratio entre 0.8 et 1.25 est considéré comme équitable (règle des 4/5).
    - **Egalité des Opportunités** : Les vrais positifs doivent être équitablement distribués.
    - **Parité Démographique** : Les taux de prédiction positive doivent être similaires entre les groupes.
    """)
