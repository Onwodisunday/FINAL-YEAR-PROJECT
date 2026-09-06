#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
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

    # ============================================
    # GREY WOLF OPTIMIZER (GWO) FOR HYPERPARAMETER TUNING
    # ============================================
    def decode_wolf(position):
        """Convert a continuous wolf position vector into RF hyperparameters."""
        n_estimators    = int(np.clip(round(position[0]), 10, 300))
        max_depth_raw   = position[1]
        # Treat values below 1.0 as None (no depth limit), otherwise use int
        max_depth       = None if max_depth_raw < 1.0 else int(np.clip(round(max_depth_raw), 1, 30))
        min_samples_leaf = int(np.clip(round(position[2]), 1, 8))
        bootstrap       = bool(round(np.clip(position[3], 0, 1)))
        return {
            "n_estimators":     n_estimators,
            "max_depth":        max_depth,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap":        bootstrap,
        }

    def gwo_fitness(position, X_tr, y_tr, X_val, y_val):
        """Evaluate one wolf: train RF with decoded params, return validation accuracy."""
        params = decode_wolf(position)
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        model.fit(X_tr, y_tr)
        return accuracy_score(y_val, model.predict(X_val))

    def grey_wolf_optimizer(X_tr, y_tr, X_val, y_val,
                            n_wolves=10, max_iter=20):
        """
        GWO minimises a cost; we maximise accuracy so cost = 1 - accuracy.

        Search space (4 dimensions):
          [0] n_estimators    : 10 – 300  (continuous, rounded to int)
          [1] max_depth       :  0 – 30   (< 1.0 → None, else rounded to int)
          [2] min_samples_leaf:  1 – 8    (continuous, rounded to int)
          [3] bootstrap       :  0 – 1    (rounded to 0/1 → False/True)
        """
        import numpy as np

        lb = np.array([10,  0.0, 1, 0])
        ub = np.array([300, 30.0, 8, 1])
        dim = len(lb)

        # Initialise pack randomly within bounds
        wolves = lb + np.random.rand(n_wolves, dim) * (ub - lb)
        fitness = np.array([gwo_fitness(w, X_tr, y_tr, X_val, y_val) for w in wolves])

        # Alpha = best, Beta = 2nd, Delta = 3rd
        sorted_idx = np.argsort(fitness)[::-1]
        alpha_pos, beta_pos, delta_pos = wolves[sorted_idx[0]].copy(), wolves[sorted_idx[1]].copy(), wolves[sorted_idx[2]].copy()
        alpha_score = fitness[sorted_idx[0]]

        for t in range(max_iter):
            a = 2 - 2 * (t / max_iter)          # linearly decreases 2 → 0

            for i in range(n_wolves):
                new_pos = np.zeros(dim)
                for leader in [alpha_pos, beta_pos, delta_pos]:
                    r1, r2 = np.random.rand(dim), np.random.rand(dim)
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = np.abs(C * leader - wolves[i])
                    new_pos += leader - A * D
                wolves[i] = np.clip(new_pos / 3, lb, ub)

            # Re-evaluate
            fitness = np.array([gwo_fitness(w, X_tr, y_tr, X_val, y_val) for w in wolves])
            sorted_idx = np.argsort(fitness)[::-1]
            if fitness[sorted_idx[0]] > alpha_score:
                alpha_score = fitness[sorted_idx[0]]
                alpha_pos   = wolves[sorted_idx[0]].copy()
                beta_pos    = wolves[sorted_idx[1]].copy()
                delta_pos   = wolves[sorted_idx[2]].copy()

        best_params = decode_wolf(alpha_pos)
        return best_params, alpha_score

    @st.cache_resource
    def train_model_for_provider(provider_name, _X_train, _y_train, _X_val, _y_val):
        """Train model for specific provider using GWO and cache the result."""
        import numpy as np
        model_filename  = f'best_rf_model_{provider_name}.pkl'
        params_filename = f'best_rf_params_{provider_name}.pkl'

        # Load cached model if it exists
        if os.path.exists(model_filename) and os.path.exists(params_filename):
            best_rf     = joblib.load(model_filename)
            best_params = joblib.load(params_filename)
            return best_rf, best_params

        with st.spinner(f"Running Grey Wolf Optimizer for {provider_name}…"):
            np.random.seed(42)
            best_params, best_val_acc = grey_wolf_optimizer(
                _X_train, _y_train, _X_val, _y_val,
                n_wolves=10, max_iter=20
            )

            best_rf = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
            best_rf.fit(_X_train, _y_train)

            joblib.dump(best_rf,     model_filename)
            joblib.dump(best_params, params_filename)

        st.success(f"GWO tuning complete for {provider_name}! Best val accuracy: {best_val_acc:.2%}")
        return best_rf, best_params

    best_rf, best_params = train_model_for_provider(selected_provider, X_train_balanced, y_train_balanced, X_val, y_val)

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
