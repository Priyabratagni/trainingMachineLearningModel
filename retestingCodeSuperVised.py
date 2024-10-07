import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from assigCode import fff_injection_df, rpm_injection_df, no_injection_df

# Combine the datasets
combined_data = pd.concat([fff_injection_df, rpm_injection_df, no_injection_df], ignore_index=True)

# Print the merged DataFrame
print('Combined DataFrame:\n', combined_data)

# Convert 'Speed' and 'RPM' to numeric if needed
combined_data['Speed'] = pd.to_numeric(combined_data['Speed'])
combined_data['RPM'] = pd.to_numeric(combined_data['RPM'])

# Define features and target variable
X = combined_data[['Speed', 'RPM']]
y = combined_data['attack']

# Convert target variable to integer
y = y.astype(int)

# Check for NaN values in target variable
if y.isnull().any():
    print("NaN values found in target variable.")
    y.dropna(inplace=True)
    # Align X with the updated y
    X = X.loc[y.index]

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the SVM classifier
svm_model = SVC(random_state=42)

# Train the SVM model
svm_model.fit(X_train, y_train)

# Predict and evaluate
predictions = svm_model.predict(X_test)

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

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='icefire', xticklabels=['No Injection', 'Attack'], yticklabels=['No Injection', 'Attack'])
# plt.title(f'SVM Model Confusion Matrix (Accuracy: {accuracy:.2f})')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()
