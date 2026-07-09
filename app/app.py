import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------
# App Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Artifact Loading & Validation
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models/final_model_pipeline.pkl"
SCHEMA_PATH = PROJECT_ROOT / "models/feature_schema.pkl"
PLOTS_DIR = PROJECT_ROOT / "assets/plots"

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        st.error(f"❌ Critical Error: Model artifact not found at {MODEL_PATH}")
        st.stop()
    if not SCHEMA_PATH.exists():
        st.error(f"❌ Critical Error: Schema artifact not found at {SCHEMA_PATH}")
        st.stop()
        
    model = joblib.load(MODEL_PATH)
    schema = joblib.load(SCHEMA_PATH)
    
    # Validation
    if not isinstance(schema, list):
        st.error("❌ Critical Error: feature_schema.pkl is not a list.")
        st.stop()
    if len(schema) != 27:
        st.error(f"❌ Critical Error: Expected exactly 27 features in schema, found {len(schema)}.")
        st.stop()
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        st.error("❌ Critical Error: Loaded model does not support predict/predict_proba.")
        st.stop()
    if not hasattr(model, "named_steps") or list(model.named_steps) != ["imputer", "scaler", "model"]:
        st.error("Critical Error: Loaded model pipeline contract is incompatible.")
        st.stop()
    if not hasattr(model, "feature_names_in_") or list(model.feature_names_in_) != schema:
        st.error("Critical Error: Model feature contract does not match feature_schema.pkl.")
        st.stop()
        
    return model, schema

try:
    model, feature_schema = load_artifacts()
except Exception as e:
    st.error(f"❌ Critical Error during artifact loading: {e}")
    st.stop()

# ---------------------------------------------------------
# Helper Functions for Inference
# ---------------------------------------------------------
def prepare_inference_data(inputs, schema):
    """
    Transforms human-readable UI inputs into the exact 27-column numeric DataFrame 
    required by the frozen Phase 4 final model schema.
    """
    validation_errors = validate_business_rules(inputs)
    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    # 1. Base Mapping
    data = {}
    
    # Gender
    data["Gender"] = 0 if inputs["Gender"] == "Female" else 1
    
    # Simple Yes/No Binary
    for col in ["Senior Citizen", "Partner", "Dependents", "Phone Service", "Paperless Billing"]:
        data[col] = 1 if inputs[col] == "Yes" else 0
        
    # Multiple Lines
    if inputs["Multiple Lines"] == "Yes":
        data["Multiple Lines"] = 1
    else:
        data["Multiple Lines"] = 0
        
    # Internet Service Base
    is_fiber = 1 if inputs["Internet Service"] == "Fiber optic" else 0
    is_no_internet = 1 if inputs["Internet Service"] == "No" else 0
    data["Internet Service_Fiber optic"] = is_fiber
    data["Internet Service_No"] = is_no_internet
    
    # Internet Add-ons (Force 0 if no internet)
    internet_addons = [
        "Online Security", "Online Backup", "Device Protection", 
        "Tech Support", "Streaming TV", "Streaming Movies"
    ]
    for col in internet_addons:
        if inputs["Internet Service"] == "No":
            data[col] = 0
        else:
            data[col] = 1 if inputs[col] == "Yes" else 0
            
    # Contract
    if inputs["Contract"] == "Month-to-month":
        data["Contract"] = 0
    elif inputs["Contract"] == "One year":
        data["Contract"] = 1
    else:
        data["Contract"] = 2
        
    # Payment Method (Base is Bank transfer)
    data["Payment Method_Credit card (automatic)"] = 1 if inputs["Payment Method"] == "Credit card (automatic)" else 0
    data["Payment Method_Electronic check"] = 1 if inputs["Payment Method"] == "Electronic check" else 0
    data["Payment Method_Mailed check"] = 1 if inputs["Payment Method"] == "Mailed check" else 0

    # Numeric Basics
    tenure = inputs["Tenure Months"]
    monthly_charges = inputs["Monthly Charges"]
    total_charges = inputs["Total Charges"]
    
    data["Tenure Months"] = tenure
    data["Monthly Charges"] = monthly_charges
    data["Total Charges"] = total_charges
    
    # 2. Engineered Features Reconstruction (Phase 2 alignment)
    # Tenure_Group
    if tenure <= 12:
        data["Tenure_Group"] = 0
    elif tenure <= 24:
        data["Tenure_Group"] = 1
    elif tenure <= 48:
        data["Tenure_Group"] = 2
    else:
        data["Tenure_Group"] = 3
        
    # Num_Add_On_Services
    num_addons = sum(data[col] for col in internet_addons)
    data["Num_Add_On_Services"] = num_addons
    
    # Has_Online_Services
    data["Has_Online_Services"] = 1 if (data["Online Security"] == 1 or data["Online Backup"] == 1) else 0
    
    # Avg_Monthly_Spend
    if tenure > 0:
        data["Avg_Monthly_Spend"] = float(np.round(total_charges / tenure, 2))
    else:
        data["Avg_Monthly_Spend"] = monthly_charges
        
    # Is_Long_Term_Contract
    data["Is_Long_Term_Contract"] = 1 if inputs["Contract"] in ["One year", "Two year"] else 0

    # 3. Construct exact schema DataFrame
    input_df = pd.DataFrame([data])
    
    # Validation step
    missing_cols = set(schema) - set(input_df.columns)
    if missing_cols:
        st.error(f"Schema mismatch! Missing calculated columns: {missing_cols}")
        st.stop()
        
    return input_df[schema] # Reorder strictly to schema


def validate_business_rules(inputs, strict_financial_consistency=False):
    """
    Lightweight guardrails for impossible UI states before model inference.
    These checks protect the frozen model contract without changing the model.
    """
    errors = []

    tenure = inputs["Tenure Months"]
    monthly_charges = inputs["Monthly Charges"]
    total_charges = inputs["Total Charges"]

    if tenure < 0 or monthly_charges < 0 or total_charges < 0:
        errors.append("Tenure, Monthly Charges, and Total Charges must be non-negative.")

    if inputs["Phone Service"] == "No" and inputs["Multiple Lines"] == "Yes":
        errors.append("Multiple Lines cannot be Yes when Phone Service is No.")

    internet_addons = [
        "Online Security", "Online Backup", "Device Protection",
        "Tech Support", "Streaming TV", "Streaming Movies"
    ]
    if inputs["Internet Service"] == "No":
        invalid_addons = [
            addon for addon in internet_addons
            if inputs[addon] not in ["No", "No internet service"]
        ]
        if invalid_addons:
            errors.append("Internet add-ons cannot be Yes when Internet Service is No.")

    if strict_financial_consistency and tenure == 0 and total_charges != 0:
        errors.append("Total Charges must be 0 when Tenure Months is 0.")

    if tenure > 0 and total_charges == 0:
        errors.append("Total Charges must be positive when Tenure Months is greater than 0.")

    return errors

# ---------------------------------------------------------
# App Layout
# ---------------------------------------------------------
st.title("📡 Telco Customer Churn Predictor")
st.markdown("Predict the likelihood of a customer churning based on our finalized Logistic Regression pipeline.")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction App", "📊 Dashboard & Insights", "ℹ️ Project & Model Summary"])

# ---------------------------------------------------------
# TAB 1: PREDICTION APP
# ---------------------------------------------------------
with tab1:
    st.header("Customer Profile")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            
            st.subheader("Account Status")
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=70.0, step=0.5)
            total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)

        with col2:
            st.subheader("Phone Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            
            st.subheader("Internet Services")
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            
        with col3:
            st.subheader("Internet Add-ons")
            st.caption("Requires Fiber optic or DSL")
            sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            protect = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            
        submit = st.form_submit_button("Predict Churn Risk", type="primary")

    if submit:
        # Collect
        raw_inputs = {
            "Gender": gender, "Senior Citizen": senior, "Partner": partner, "Dependents": dependents,
            "Tenure Months": tenure, "Contract": contract, "Paperless Billing": paperless,
            "Payment Method": payment, "Monthly Charges": monthly, "Total Charges": total,
            "Phone Service": phone, "Multiple Lines": multiple, "Internet Service": internet,
            "Online Security": sec, "Online Backup": backup, "Device Protection": protect,
            "Tech Support": tech, "Streaming TV": tv, "Streaming Movies": movies
        }

        validation_errors = validate_business_rules(raw_inputs, strict_financial_consistency=True)
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            st.stop()
        
        # Safe logic override for UI clarity
        if internet == "No":
            for addon in ["Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"]:
                raw_inputs[addon] = "No"

        # Prepare
        try:
            input_df = prepare_inference_data(raw_inputs, feature_schema)
            
            # Predict
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            churn_prob = probabilities[1]
            stay_prob = probabilities[0]
        except Exception as e:
            st.error(f"❌ Inference Error: Could not process prediction. Details: {e}")
            st.stop()
        
        # Determine Display
        if churn_prob >= 0.70:
            risk_cat, risk_color = "High Risk", "red"
        elif churn_prob >= 0.40:
            risk_cat, risk_color = "Medium Risk", "orange"
        else:
            risk_cat, risk_color = "Low Risk", "green"
            
        result_text = "🚨 Likely to Churn" if prediction == 1 else "✅ Likely to Stay"
        
        st.markdown("---")
        st.header("Prediction Results")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Model Prediction", result_text)
        with res_col2:
            st.metric("Churn Probability", f"{churn_prob:.1%}")
        with res_col3:
            st.markdown(f"### Risk Category: <span style='color:{risk_color}'>{risk_cat}</span>", unsafe_allow_html=True)
            
        st.subheader("Business Explanation")
        factors = []
        if contract == "Month-to-month": factors.append("Month-to-month contracts lack long-term commitment.")
        if tenure <= 6: factors.append("Low tenure indicates the customer is very new and less established.")
        if internet == "Fiber optic": factors.append("Fiber optic users historically show higher churn volatility.")
        if payment == "Electronic check": factors.append("Electronic check payments correlate with higher attrition.")
        if sec == "No" and tech == "No" and internet != "No": 
            factors.append("Lack of online security and tech support services decreases stickiness.")
            
        if prediction == 1:
            st.warning("This customer profile flags as a churn risk based on historical patterns.")
            if factors:
                for f in factors: st.markdown(f"- {f}")
        else:
            st.success("This customer profile aligns with historical retention patterns.")
            st.markdown("- Steady indicators suggest low immediate flight risk.")
            
        with st.expander("🛠️ View Internal Model Input Data (27-Column Schema)"):
            st.dataframe(input_df)

# ---------------------------------------------------------
# TAB 2: DASHBOARD & INSIGHTS
# ---------------------------------------------------------
with tab2:
    st.header("Project Visualizations")
    st.markdown("These plots were generated during the EDA and Modeling phases.")
    
    def load_image(subpath):
        img_path = PLOTS_DIR / subpath
        if img_path.exists():
            st.image(Image.open(img_path), use_column_width=True)
        else:
            st.info(f"Image not found: {subpath}")
            
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Churn Distribution")
        load_image("eda/01_churn_distribution.png")
        st.subheader("Contract vs Churn")
        load_image("eda/04_contract_vs_churn.png")
    with col_b:
        st.subheader("Tenure Distribution")
        load_image("eda/02_tenure_distribution.png")
        st.subheader("ROC Curve Comparison")
        load_image("models/roc_curve_xgb_vs_lr.png")

# ---------------------------------------------------------
# TAB 3: PROJECT & MODEL SUMMARY
# ---------------------------------------------------------
with tab3:
    st.header("DEPI Graduation Project Context")
    st.markdown("""
    **Customer Churn Prediction and Analysis**
    
    This application represents the final Phase 5 deployment of the ML pipeline.
    
    ### Model Selection
    - **Champion Model:** Logistic Regression
    - **Why not XGBoost?:** In Phase 4, both XGBoost and Logistic Regression were tuned and tested. Our primary business objective was to maximize **Recall** (catching as many actual churners as possible). Logistic Regression cleanly outperformed XGBoost on Recall. Therefore, the simpler baseline won the comparison, which is a highly professional and robust outcome.
    
    ### Architecture Integrity
    - **No Retraining:** This app strictly performs inference using `models/final_model_pipeline.pkl`.
    - **Schema Bound:** Inputs are rigorously adapted to the exact 27-column numeric schema generated during Phase 2 (`models/feature_schema.pkl`), without calling dynamic dummy encoders.
    - **Leakage Free:** Future-dependent columns like *Churn Score* or *Churn Reason* are completely absent from the input structure.
    """)
