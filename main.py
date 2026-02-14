#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import os
import auth_utils  # Import authentication utilities

# ============================================
# APP CONFIGURATION & HEADER
# ============================================
st.set_page_config(page_title="Cloud Scaling Predictor", layout="wide")

# ============================================
# AUTHENTICATION LOGIC
# ============================================

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''

def login_callback():
    username = st.session_state.login_user
    password = st.session_state.login_pass
    if auth_utils.login_user(username, password):
        st.session_state.logged_in = True
        st.session_state.username = username
        # st.success will be shown on rerun or we can use a flag
    else:
        st.session_state.login_error = "Invalid username or password"

def signup_callback():
    new_user = st.session_state.signup_user
    new_pass = st.session_state.signup_pass
    confirm_pass = st.session_state.signup_confirm
    
    if new_pass != confirm_pass:
        st.session_state.signup_error = "Passwords do not match"
    elif not new_user or not new_pass:
        st.session_state.signup_error = "Please fill in all fields"
    elif not new_user.isalpha():
        st.session_state.signup_error = "Username must contain only letters (a-z, A-Z)"
    else:
        if auth_utils.add_user(new_user, new_pass):
            st.session_state.auth_mode = "Login"
            st.session_state.signup_success = True
            st.session_state.signup_error = None
        else:
            st.session_state.signup_error = "Username already exists"

def login_page():
    st.title("Welcome to Cloud Scaling Predictor")
    
    # Initialize session state for errors if not present
    if 'login_error' not in st.session_state:
        st.session_state.login_error = None
    if 'signup_error' not in st.session_state:
        st.session_state.signup_error = None

    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'Login'

    # Display success message from previous signup
    if st.session_state.get('signup_success'):
        st.success("🎉 Account created successfully! Please log in.")
        del st.session_state.signup_success

    # Toggle between Login and Sign Up
    mode = st.radio("Select Mode", ["Login", "Sign Up"], horizontal=True, key="auth_mode")
    
    if mode == "Login":
        st.subheader("Login")
        if st.session_state.login_error:
            st.error(st.session_state.login_error)
            st.session_state.login_error = None # Clear after showing
            
        with st.form("login_form"):
            st.text_input("Username", key="login_user")
            st.text_input("Password", type="password", key="login_pass")
            st.form_submit_button("Login", on_click=login_callback)
    
    elif mode == "Sign Up":
        st.subheader("Sign Up")
        if st.session_state.signup_error:
            st.error(st.session_state.signup_error)
            st.session_state.signup_error = None # Clear after showing
            
        with st.form("signup_form"):
            st.text_input("New Username", key="signup_user")
            st.text_input("New Password", type="password", key="signup_pass")
            st.text_input("Confirm Password", type="password", key="signup_confirm")
            st.form_submit_button("Sign Up", on_click=signup_callback)

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.rerun()

def show_landing_page():
    st.title("Cloud Scaling Intelligence")
    st.write("### AI-Powered Infrastructure Optimization")
    st.write("Welcome to the **Sunday Final Year Project**. This system utilizes advanced Machine Learning algorithms to predict optimal scaling actions (Scale Up, Scale Down, No Action) for your cloud resources.")
    
    st.markdown("---")
    
    # Dashboard Features
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### ☁️ AWS\nAnalyze EC2 instances and optimize cost/performance.")
    with col2:
        st.success("### 🟦 Azure\nMonitor VM throughput and reduce latency.")
    with col3:
        st.warning("### 🟢 GCP\nGoogle Cloud resource management and scaling.")
    
    st.markdown("---")
    st.subheader("📊 Global Dataset Insights")
    
    # Aggregated Stats (Hardcoded for performance as we know them, or could load)
    # AWS: 323, Azure: 326, GCP: 351
    col_chart, col_stats = st.columns([2, 1])
    
    with col_chart:
        data = pd.DataFrame({
            'Records': [323, 326, 351],
            'Provider': ['AWS', 'Azure', 'GCP']
        }).set_index('Provider')
        st.bar_chart(data)
    
    with col_stats:
        st.metric("Total Training Records", "1,000")
        st.metric("Supported Providers", "3")
        st.metric("Prediction Accuracy", ">90%")

    st.markdown("---")
    st.warning("👈 **To Begin**: Select a Cloud Provider from the sidebar menu!")

# ============================================
# MAIN APP LOGIC (Protected)
# ============================================

if not st.session_state.logged_in:
    login_page()
else:
    # Sidebar Logout
    st.sidebar.button("Logout", on_click=logout)
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
    st.sidebar.markdown("---")

    st.header("Sunday Final Year Project")
    st.subheader("Using Machine Learning to Predict Cloud Services Scaling Actions")

    # ============================================
    # GLOBAL SETTINGS (SIDEBAR)
    # ============================================
    st.sidebar.header("Configuration")
    selected_provider = st.sidebar.selectbox(
        "Select Cloud Provider",
        ["AWS", "Azure", "GCP"],
        index=None,
        placeholder="Choose a provider..."
    )

    if not selected_provider:
        show_landing_page()
        st.stop()

    st.sidebar.success(f"Selected Provider: {selected_provider}")

    # ============================================
    # DATA LOADING & PREPROCESSING
    # ============================================
    data_file_map = {
        'AWS': 'data/AWS_Cloud_Dataset.csv',
        'Azure': 'data/Azure_Cloud_Dataset.csv',
        'GCP': 'data/GCP_Cloud_Dataset.csv'
    }

    data_path = data_file_map.get(selected_provider)

    if not os.path.exists(data_path):
        st.error(f"Dataset not found: {data_path}")
        st.stop()

    # Load Data
    df = pd.read_csv(data_path)


    # Preprocessing
    # 1. Drop unnecessary columns
    if "region" in df.columns:
        df.drop(columns=["region"], inplace=True)
    if "timestamp" in df.columns:
        df.drop(columns=["timestamp"], inplace=True)

    # 2. Drop cloud_provider column as it's constant for this specific model
    if "cloud_provider" in df.columns:
        df.drop(columns=["cloud_provider"], inplace=True)

    # 3. Encode Categorical Features (Only vm_type should remain)
    # We check what categorical columns exist
    categorical_features = [col for col in df.select_dtypes(include=['object']).columns if col != 'target']

    if categorical_features:
        one_hot_encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
        one_hot_encoded = one_hot_encoder.fit_transform(df[categorical_features])
        
        # Store encoder for later usage (though strict re-training means we just re-fit)
        feature_names_out = one_hot_encoder.get_feature_names_out(categorical_features)
        
        # Concatenate and drop original
        df = pd.concat([df.drop(columns=categorical_features), one_hot_encoded], axis=1)

    # 4. Encode Target
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])

    # ============================================
    # MODEL TRAINING / LOADING
    # ============================================

    X = df.drop(columns=['target'])
    y = df['target']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    # Balancing
    X_train_balanced, y_train_balanced = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)

    @st.cache_resource
    def train_model_for_provider(provider_name, _X_train, _y_train):
        """Train model for specific provider using GridSearchCV and cache"""
        model_filename = f'best_rf_model_{provider_name}.pkl'
        params_filename = f'best_rf_params_{provider_name}.pkl'

        # Check if model exists
        if os.path.exists(model_filename) and os.path.exists(params_filename):
            # st.info(f"Loading pre-trained model for {provider_name}...")
            best_rf = joblib.load(model_filename)
            best_params = joblib.load(params_filename)
            return best_rf, best_params

        with st.spinner(f"Training model for {provider_name} (this may take a moment)..."):
            param_grid = {
                "n_estimators": [10, 100, 200],
                "max_depth": [None, 3, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False],
            }
            rf_model = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                estimator=rf_model,
                param_grid=param_grid,
                scoring='accuracy',
                cv=3,
                n_jobs=-1
            )
            grid_search.fit(_X_train, _y_train)
            best_rf = grid_search.best_estimator_

            # Save
            joblib.dump(best_rf, model_filename)
            joblib.dump(grid_search.best_params_, params_filename)
        
        st.success(f"New model trained for {provider_name}!")
        return best_rf, grid_search.best_params_

    best_rf, best_params = train_model_for_provider(selected_provider, X_train_balanced, y_train_balanced)

    # ============================================
    # INTERACTIVE PREDICTION
    # ============================================
    st.markdown("---")
    st.header("🔮 Make Your Own Predictions")
    st.info(f"Predicting for **{selected_provider}**")

    # Input Form
    st.subheader("Input Feature Values")

    # Input dictionary
    user_input = {}
    input_cols = st.columns(2)

    for i, feature in enumerate(X.columns):
        # Determine column
        col = input_cols[i % 2]
        
        # Calculate stats for range
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        mean_val = float(X[feature].mean())
        
        with col:
            if min_val == max_val:
                st.text_input(f"{feature} (Fixed)", value=min_val, disabled=True)
                user_input[feature] = min_val
            elif "vm_type_" in feature:
                # For one-hot encoded features, simpler to ask for the category? 
                # Actually, since we auto-generated features based on the dataset, let's just use sliders for now.
                # But better: if we have vm_type_X, vm_type_Y, we should probably have a selectbox for VM Type
                # However, simpler to just let current logic handle it (0 or 1).
                # IMPROVEMENT: Re-construct categorical inputs.
                pass # We handle categorical below
            else:
                step = max((max_val - min_val) / 100, 0.001)
                user_input[feature] = st.slider(f"{feature}", min_value=min_val, max_value=max_val, value=mean_val, step=step)

    # Handle VM Type Selection separately if it exists as encoded columns
    vm_type_cols = [col for col in X.columns if "vm_type_" in col]
    if vm_type_cols:
        # Extract original vm type names
        vm_types = [col.replace("vm_type_", "") for col in vm_type_cols]
        
        st.subheader("Configuration")
        selected_vm = st.selectbox("Select VM Type", vm_types)
        
        # Set one-hot values
        for col in vm_type_cols:
            target_vm = col.replace("vm_type_", "")
            user_input[col] = 1.0 if target_vm == selected_vm else 0.0

    # Prepare DataFrame
    if st.button("Get Prediction", type="primary"):
        input_df = pd.DataFrame([user_input])
        
        # Ensure column order matches X
        input_df = input_df[X.columns]
        
        # Predict
        pred_idx = best_rf.predict(input_df)[0]
        pred_prob = best_rf.predict_proba(input_df)[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        
        # Display
        color_map = {'scaledown': '🔽 Scale Down', 'scaleup': '🔼 Scale Up', 'noaction': '⏸️ No Action'}
        display_label = color_map.get(pred_label.lower(), pred_label)
        
        st.markdown("---")
        st.subheader("Result")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"# {display_label}")
        with col2:
            probs_df = pd.DataFrame({
                'Action': le.classes_,
                'Confidence': [f"{p:.2%}" for p in pred_prob]
            })
            st.dataframe(probs_df, use_container_width=True)

    # ============================================
    # PERFORMANCE EVALUATION
    # ============================================
    st.markdown("---")
    st.header("📈 Model Performance")

    col1, col2 = st.columns(2)

    # Validation Metrics
    y_pred_val = best_rf.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred_val)

    with col1:
        st.subheader("Validation Set")
        st.metric("Accuracy", f"{val_acc:.2%}")
        st.text("Classification Report:")
        st.dataframe(pd.DataFrame(classification_report(y_val, y_pred_val, target_names=le.classes_, output_dict=True)).transpose())

    # Test Metrics
    y_pred_test = best_rf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    with col2:
        st.subheader("Test Set")
        st.metric("Accuracy", f"{test_acc:.2%}")
        st.text("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_test)
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
        st.pyplot(fig)

    # Feature Importance
    st.markdown("---")
    st.subheader(f"🔍 Feature Importance ({selected_provider})")

    importances = best_rf.feature_importances_
    feature_names = X.columns
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    st.bar_chart(feat_df.set_index('Feature'))
