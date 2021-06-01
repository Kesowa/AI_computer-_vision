import pickle
import os

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import (normalize_input_variable,encode_labels,build_dataset)

# building the input and output variables from database
X, Y = build_dataset(dataset_path = "dataset")
# preprocessing input and output variables
X_input_vars = normalize_input_variable(input_vars = X)
Y_output_vars = encode_labels(output_vars = Y)

# performing train test split
X_train, X_test, y_train, y_test = train_test_split(X_input_vars,
							                        Y_output_vars,
							                		test_size = 0.20,
							                		random_state = 101)


# Using SVM + Hyperparameter tuning to perform classification
C_values =  [0.1, 1, 10, 100, 1000]
accuracy = []
for c in C_values:
	model = SVC(C = c, kernel='linear')
	model.fit(X_train, y_train)
	Y_predicted = model.predict(X_test)
	acc = accuracy_score(Y_predicted, y_test)
	accuracy.append(acc)

best_C_index = np.argmax(accuracy)
best_C = C_values[best_C_index] 
print("Best C :", best_C, "| Accuracy: ", np.max(accuracy))

# training the classifier with the best C value
model = SVC(C = best_C, kernel='linear')
model.fit(X_train, y_train)

# saving the best model
pickle.dump(model, open('model/SVC.pkl', 'wb'))