import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from assigCode import fff_injection_df, rpm_injection_df, no_injection_df

# Combine the datasets
combined_data = pd.concat([fff_injection_df, rpm_injection_df, no_injection_df], ignore_index=True)

# Feature and target variable definition
X = combined_data[['Speed', 'RPM']]
y = combined_data['attack'].apply(lambda x: 1 if x > 0 else 0)

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Cross-validation for performance measure
cv_scores = cross_val_score(decision_tree_model, X, y, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean CV score: {cv_scores.mean()}')

# Train the Decision Tree model
decision_tree_model.fit(X_train, y_train)

# Predict and evaluate
predictions = decision_tree_model.predict(X_test)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Create a classification report
classif_report = classification_report(y_test, predictions)
print("\nClassification Report:")
print(classif_report)

# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Feature importance
importances = decision_tree_model.feature_importances_
feature_names = X.columns
print(f"Feature Importances: {dict(zip(feature_names, importances))}")
