import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from xgboost import XGBClassifier
from assigCode import fff_injection_df, rpm_injection_df, no_injection_df

# Combine the datasets
combined_data = pd.concat([fff_injection_df, rpm_injection_df, no_injection_df], ignore_index=True)

# Before training, convert data types to float
combined_data['Speed'] = pd.to_numeric(combined_data['Speed'], errors='coerce')
combined_data['RPM'] = pd.to_numeric(combined_data['RPM'], errors='coerce')

# Check for any missing values introduced due to conversion
print(combined_data.isnull().sum())

# Assuming you've handled NaN values appropriately (if any):
# Splitting the dataset
X = combined_data[['Speed', 'RPM']]
y = combined_data['attack'].apply(lambda x: 1 if x > 0 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize and train the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Continue with predictions and evaluation as before
predictions = xgb_model.predict(X_test)

# Proceed with evaluation metrics and further analysis
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

classif_report = classification_report(y_test, predictions)
print("\nClassification Report:")
print(classif_report)

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

importances = xgb_model.feature_importances_
feature_names = X.columns
print(f"Feature Importances: {dict(zip(feature_names, importances))}")
