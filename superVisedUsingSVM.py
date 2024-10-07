import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from assigCode import fff_injection_df, rpm_injection_df, no_injection_df

# Combine data from Assignment 1
combined_data = pd.concat([fff_injection_df, rpm_injection_df, no_injection_df], ignore_index=True)

# Define the Variables
X = combined_data[['Speed', 'RPM']]
y = combined_data['attack']

# Split the data (3/4) for training and (1/4) for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Convert the data obtain to float numberic as the data is not in proper format
X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)

# Intializing the SVM model
svm_model = SVC(random_state=42)

# Tarain the Model
svm_model.fit(X_train, y_train)

# Predic testing data
predictions = svm_model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
print('\n\n')
print("Confusion Matrix:")
print(conf_matrix)
print('\n\n')

# Classification Report
classif_report = classification_report(y_test, predictions)
print('\n\n')
print("\nClassification Report:")
print(classif_report)
print('\n\n')

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
