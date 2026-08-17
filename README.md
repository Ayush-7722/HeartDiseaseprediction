Heart Disease Prediction

This project is a machine learning based heart disease prediction system. I worked with the dataset by first doing EDA and data preprocessing, then trained and compared different classification algorithms to find the model that performed best.

The final model can be used through a simple Streamlit web interface, where users can enter patient information and get a prediction.


Loaded and explored the dataset
Performed Exploratory Data Analysis (EDA)
Checked missing values and data types
Handled categorical features using encoding
Split the data into training and testing sets
Applied feature scaling where required
Trained multiple machine learning models
Compared models using Accuracy and F1 Score
Saved the trained model and preprocessing files using joblib
Built a basic Streamlit frontend for prediction
Machine Learning Algorithms

I tested the following algorithms:

Logistic Regression
K-Nearest Neighbors (KNN)
Naive Bayes
Decision Tree
Support Vector Machine (SVM)
Best Model

Logistic Regression performed the best among the models tested, with:

Accuracy: 86.96%
F1 Score: 88.57%

KNN was a close second, with 86.41% accuracy and 88.15% F1 score.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Joblib
Streamlit
