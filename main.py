#!/usr/bin/env python
# coding: utf-8

# ## Sunday Final Year Project
# Using Machine Learning to Predict Cloud Services, if  it will be required to scale up, down or no action taken based on the resource usage. using different algorithms to find the best model for the prediction.

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import os


# In[6]:


st.header("Sunday Final Year Project")
st.subheader("Using Machine Learning to Predict Cloud Services Scaling Actions")


# In[ ]:


# In[ ]:


df = pd.read_csv('data/Cloud_Dataset.csv')


# In[68]:


df.head()


# In[69]:


df["vm_type"].unique()


# In[70]:


df.drop(columns=["region"], inplace=True)


# In[71]:


one_hot_encoder = OneHotEncoder(
    sparse_output=False).set_output(transform="pandas")
categorical_features = ['cloud_provider', "vm_type"]
one_hot_encoded = one_hot_encoder.fit_transform(df[categorical_features])


# In[72]:


one_hot_encoded.head()


# In[73]:


df = pd.concat(
    [df.drop(columns=categorical_features), one_hot_encoded], axis=1)
df.head()


# In[74]:


df.info()


# In[86]:


df["timestamp"] = pd.to_datetime(df["timestamp"])


# In[87]:


df.info()


# In[76]:


df.head()


# In[96]:


df.drop(columns=['timestamp'], inplace=True)


# In[77]:


df["target"].unique()


# In[78]:


df["target"].value_counts(normalize=True)


# In[79]:


df["target"].value_counts(normalize=True).plot(kind='bar')
plt.xlabel('Target Classes')
plt.ylabel('Proportion [%]')
plt.title('Distribution of Target Classes')


# In[80]:


le = LabelEncoder()
df['target'] = le.fit_transform(df['target'])


# In[81]:


df["target"].info()


# In[98]:


X = df.drop(columns=['target'])
y = df['target']


# In[99]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)


# In[100]:


X_train_balanced, y_train_balanced = RandomOverSampler(
    random_state=42).fit_resample(X_train, y_train)


# In[101]:


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)


# In[103]:


y_pred = rf_model.predict(X_val)

validation_accuracy = accuracy_score(y_val, y_pred)
st.header("Performance on validation data")
print(f"Validation Accuracy: {validation_accuracy:.4f}")
st.write(f"Validation Accuracy: {validation_accuracy:.4f}")

# In[107]:


print("Classification report", classification_report(y_val, y_pred))


# In[111]:


cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Validation Set")
plt.show()


# In[114]:


@st.cache_resource
def train_model_with_gridsearch(X_train, y_train):
    """Train model using GridSearchCV and cache the result"""
    model_path = 'best_rf_model.pkl'
    params_path = 'best_rf_params.pkl'

    # Check if model already exists
    if os.path.exists(model_path) and os.path.exists(params_path):
        st.info("Loading pre-trained model...")
        best_rf = joblib.load(model_path)
        best_params = joblib.load(params_path)
        return best_rf, best_params

    st.info("Training model with GridSearchCV (this may take a few minutes)...")

    param_grid = {
        "n_estimators": [10, 100, 200, 300],
        "max_depth": [None, 3, 5, 10, 20, 30],
        "max_features": ['auto', 'sqrt', 'log2', None],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }
    rf_model = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_

    # Save the model and parameters
    joblib.dump(best_rf, model_path)
    joblib.dump(grid_search.best_params_, params_path)
    st.success("Model trained and saved!")

    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best score", grid_search.best_score_)

    return best_rf, grid_search.best_params_


# Displaying the best parameters
best_rf, best_params = train_model_with_gridsearch(
    X_train_balanced, y_train_balanced)

st.subheader("🎯 Best Hyperparameters Found")
st.markdown("---")

# Display hyperparameters in a nice table format
params_df = pd.DataFrame(list(best_params.items()), columns=[
                         'Hyperparameter', 'Value'])
st.dataframe(params_df, use_container_width=True)

# Also display as metrics for key parameters
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("N Estimators", best_params.get('n_estimators', 'N/A'))
with col2:
    st.metric("Max Depth", best_params.get('max_depth', 'N/A'))
with col3:
    st.metric("Min Samples Leaf", best_params.get('min_samples_leaf', 'N/A'))
with col4:
    st.metric("Bootstrap", best_params.get('bootstrap', 'N/A'))

# In[115]:

st.markdown("---")
st.header("📈 Model Performance on Test Data")

y_pred = best_rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.4f}")

# Display test accuracy prominently
st.metric("🎯 Test Accuracy", f"{test_accuracy:.2%}")

# Display classification report
st.subheader("Classification Report")
report = classification_report(
    y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df, use_container_width=True)

# Also display as text for better readability
st.text(classification_report(y_test, y_pred, target_names=le.classes_))


# In[116]:


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Test Set')
st.pyplot(plt.gcf())


# In[117]:

st.markdown("---")
st.header("🔍 Feature Importance Analysis")

# Feature importance
importances = best_rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)


# In[118]:

# Display top features
st.subheader("Top 10 Most Important Features")
fig, ax = plt.subplots(figsize=(10, 6))
top_10 = feature_importance_df.head(10)
ax.barh(top_10['Feature'], top_10['Importance'], color='steelblue')
ax.set_xlabel('Importance Score')
ax.set_ylabel('Feature')
ax.set_title('Top 10 Feature Importance')
plt.tight_layout()
st.pyplot(fig)

# Display all features in a table
st.subheader("All Features Importance Scores")
st.dataframe(feature_importance_df, use_container_width=True)


# In[119]:


feature_importance_df.head(10)


# In[121]:


pred_df = pd.DataFrame({
    'Actual': le.inverse_transform(y_test),
    "predicted": le.inverse_transform(y_pred)
})
pred_df.tail()


# In[ ]:

# ============================================
# INTERACTIVE PREDICTION SECTION
# ============================================

st.markdown("---")
st.header("🔮 Make Your Own Predictions")
st.write("Enter feature values below to predict whether the cloud service should scale up, scale down, or take no action.")

# Create a sidebar for user inputs
st.subheader("Input Feature Values")

# Get feature names
feature_names = X.columns.tolist()

# Create input columns for user to enter values
col1, col2 = st.columns(2)

# Dictionary to store user inputs
user_input = {}

# Create input fields for each feature
for i, feature in enumerate(feature_names):
    # Get min and max from training data for slider bounds
    feature_min = float(X[feature].min())
    feature_max = float(X[feature].max())
    feature_mean = float(X[feature].mean())

    # Alternate between columns
    if i % 2 == 0:
        col = col1
    else:
        col = col2

    with col:
        # Handle constant features (where min == max)
        if feature_min == feature_max:
            st.number_input(
                f"{feature} (constant)",
                value=feature_mean,
                disabled=True
            )
            user_input[feature] = feature_mean
        else:
            # Calculate step - ensure it's not too small
            feature_range = feature_max - feature_min
            step = max(feature_range / 100, 0.001)

            user_input[feature] = st.slider(
                f"{feature}",
                min_value=feature_min,
                max_value=feature_max,
                value=feature_mean,
                step=step
            )

# Create prediction button
if st.button("🚀 Get Prediction", key="predict_btn"):
    # Prepare the input data
    user_input_df = pd.DataFrame([user_input])

    # Make prediction
    prediction = best_rf.predict(user_input_df)[0]
    prediction_proba = best_rf.predict_proba(user_input_df)[0]

    # Convert prediction to class label
    prediction_label = le.inverse_transform([prediction])[0]

    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    # Color mapping for predictions
    color_map = {
        'scaledown': '🔽 Scale Down',
        'scaleup': '🔼 Scale Up',
        'noaction': '⏸️ No Action'
    }

    result_display = color_map.get(prediction_label.lower(), prediction_label)

    # Display in a nice box
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info(f"# {result_display}")

    # Show confidence scores
    st.subheader("Confidence Scores")
    confidence_df = pd.DataFrame({
        'Action': le.classes_,
        'Confidence (%)': [f"{prob*100:.2f}%" for prob in prediction_proba]
    })
    st.dataframe(confidence_df, use_container_width=True)

    # Show input values used
    with st.expander("View Input Features"):
        input_display_df = pd.DataFrame([user_input]).T
        input_display_df.columns = ['Value']
        st.dataframe(input_display_df)
