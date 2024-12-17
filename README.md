# Iris Dataset - Logistic Regression Classifier

This project demonstrates a simple implementation of a **Logistic Regression** model using the popular **Iris Dataset**. The model predicts the species of iris flowers based on their sepal and petal dimensions.

## Project Description
The Iris dataset is a classic dataset in machine learning. It includes:
- **150 samples**
- **4 features** (sepal length, sepal width, petal length, petal width)
- **3 classes** of flowers:
  - Setosa
  - Versicolor
  - Virginica

This project uses Logistic Regression to classify iris species and evaluates the model's performance.

---

## Steps Involved

1. **Importing Libraries**
   - `pandas`, `numpy` for data manipulation
   - `sklearn` for machine learning model and utilities

2. **Loading the Iris Dataset**
   - Features (X): Sepal and petal dimensions
   - Target (y): Iris species

3. **Splitting Data**
   - Splits the dataset into **training** and **testing** sets with an 80-20 ratio.

4. **Model Training**
   - Trains a **Logistic Regression** model with a maximum of 200 iterations.

5. **Making Predictions**
   - Predicts species for the test data.

6. **Model Evaluation**
   - Calculates accuracy and displays the classification report.

7. **Sample Prediction**
   - Tests the model on a sample input: `[5.1, 3.5, 1.4, 0.2]`

---

## Code
Below is the main code snippet:
```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Single sample prediction
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)
print(f"Predicted Class for {sample}: {iris.target_names[prediction[0]]}")
```

---

## Requirements
- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`

To install dependencies, use:
```bash
pip install numpy pandas scikit-learn
```

---

## How to Run
1. Clone this repository:
   ```bash
   git clone <your-repo-link>
   ```
2. Navigate to the project directory:
   ```bash
   cd iris-logistic-regression
   ```
3. Run the Python script:
   ```bash
   python iris_classification.py
   ```
4. Check the accuracy, classification report, and test with a custom sample.

---

## Sample Output
```
Accuracy: 1.0

Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

Predicted Class for [[5.1 3.5 1.4 0.2]]: setosa
```

---

## License
This project is open-source and licensed under the MIT License.

---

## Author
Hitesh Kumar S

---

Happy Learning! 🌟
