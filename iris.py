


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import pickle

# # Load the dataset
# loan_dataset = pd.read_csv('./demo1_data.csv')

# # Feature Engineering and Data Preprocessing


# loan_dataset = loan_dataset.dropna()

# loan_dataset = loan_dataset.replace(to_replace="N", value="0")
# loan_dataset = loan_dataset.replace(to_replace="Y", value="1")
# loan_dataset = loan_dataset.replace(to_replace="Yes", value="1")
# loan_dataset = loan_dataset.replace(to_replace="No", value="0")
# loan_dataset = loan_dataset.replace(to_replace="Male", value="1")
# loan_dataset = loan_dataset.replace(to_replace="Female", value="0")
# loan_dataset = loan_dataset.replace(to_replace="Graduate", value="1")
# loan_dataset = loan_dataset.replace(to_replace="Not Graduate", value="0")
# loan_dataset = loan_dataset.replace(to_replace="Rural", value="0")
# loan_dataset = loan_dataset.replace(to_replace="Urban", value="2")
# loan_dataset = loan_dataset.replace(to_replace="Semiurban", value="1")
# loan_dataset = loan_dataset.replace(to_replace="3+", value="4")


# # ... (Your preprocessing code here)

# # Separate features and target variable
# X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
# Y = loan_dataset['Loan_Status']

# # Split the data into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

# # Model Selection and Hyperparameter Tuning (Random Forest)
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [5, 10, 15],
#     'min_samples_split': [2, 5, 10]
# }

# rf_classifier = RandomForestClassifier(random_state=2)
# grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, n_jobs=-1)
# grid_search.fit(X_train, Y_train)

# best_rf_classifier = grid_search.best_estimator_

# # Evaluate the model
# X_train_prediction = best_rf_classifier.predict(X_train)
# training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# X_test_prediction = best_rf_classifier.predict(X_test)
# test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
# percentage_approval = (X_test_prediction == 'Y').mean() * 100

# # print('Percentage of loan approvals in the test data:', percentage_approval)

# # Save the trained model
# filename = 'trained_model1_rf.sav'
# pickle.dump(best_rf_classifier, open(filename, 'wb'))






import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
loan_dataset = pd.read_csv('./demo1_data.csv')

# Feature Engineering and Data Preprocessing


loan_dataset = loan_dataset.dropna()

loan_dataset = loan_dataset.replace(to_replace="N", value="0")
loan_dataset = loan_dataset.replace(to_replace="Y", value="1")
loan_dataset = loan_dataset.replace(to_replace="Yes", value="1")
loan_dataset = loan_dataset.replace(to_replace="No", value="0")
loan_dataset = loan_dataset.replace(to_replace="Male", value="1")
loan_dataset = loan_dataset.replace(to_replace="Female", value="0")
loan_dataset = loan_dataset.replace(to_replace="Graduate", value="1")
loan_dataset = loan_dataset.replace(to_replace="Not Graduate", value="0")
loan_dataset = loan_dataset.replace(to_replace="Rural", value="0")
loan_dataset = loan_dataset.replace(to_replace="Urban", value="2")
loan_dataset = loan_dataset.replace(to_replace="Semiurban", value="1")
loan_dataset = loan_dataset.replace(to_replace="3+", value="4")


# ... (Your preprocessing code here)

# Separate features and target variable
X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

# Model Selection and Hyperparameter Tuning (Random Forest)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

rf_classifier = RandomForestClassifier(random_state=2)
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, Y_train)

best_rf_classifier = grid_search.best_estimator_

# Evaluate the model
X_train_prediction = best_rf_classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = best_rf_classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
percentage_approval = (X_test_prediction == 'Y').mean() * 100

# print('Percentage of loan approvals in the test data:', percentage_approval)

# Save the trained model
filename = 'trained_model1_rf.sav'
pickle.dump(best_rf_classifier, open(filename, 'wb'))


print('Accuracy on training data : ', training_data_accuracy)
print('Accuracy on test data : ', test_data_accuracy)
