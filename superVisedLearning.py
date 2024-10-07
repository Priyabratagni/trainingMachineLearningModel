import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from assigCode import fff_injection_df, rpm_injection_df,no_injection_df

#combine the data for applying Random Forest
combine_file = pd.concat([fff_injection_df, rpm_injection_df, no_injection_df], ignore_index = True)

print(combine_file['attack'].value_counts())

#Define Speed and RPM
X = combine_file[['Speed', 'RPM']]
y = combine_file['attack'].apply(lambda x: 1 if x > 0 else 0)

#Split the data for training (3/4) and testing (1/4) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Random Forest
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

#Train the Model
random_forest_model.fit(X_train, y_train)

#Testing the model
Prediction = random_forest_model.predict(X_test)

# Create a Confusion matrix
conff_matrix = confusion_matrix(y_test, Prediction)
print('\n\n')
print("Confusion Matrix:")
print(conff_matrix)
print('\n\n')
#Create a Classification report
classif_report = classification_report(y_test, Prediction)
print('\n\n')
print("\nClassification Report:")
print(classif_report)

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, Prediction)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
