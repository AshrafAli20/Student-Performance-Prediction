
# ==========================================
# STUDENT PERFORMANCE PREDICTION
# USING LINEAR REGRESSION
# ==========================================

# 1. Import required libraries

import pandas as pd
import numpy as np
import pickle
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

warnings.filterwarnings("ignore")


# 2. Create a class

class StudentPerformanceModel:

    def __init__(self):

        self.model = LinearRegression()

        self.features = [
            'Hours Studied',
            'Previous Scores',
            'Extracurricular Activities',
            'Sleep Hours',
            'Sample Question Papers Practiced'
        ]

        self.target = 'Performance Index'

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    # 3. Load dataset

    def load_dataset(self):

        print("\nLoading Dataset...")

        self.df = pd.read_csv("Student_Performance.csv")

        print("\nDataset Loaded Successfully!")

        print("\nDataset Shape:")
        print(self.df.shape)

        print("\nDataset Columns:")
        print(self.df.columns.tolist())

        print("\nFirst Five Records:")
        print(self.df.head())

    # 4. Preprocess dataset

    def preprocess_data(self):

        print("\nPreprocessing Dataset...")

        self.df = self.df.drop_duplicates()

        self.df['Extracurricular Activities'] = (
            self.df['Extracurricular Activities']
            .map({'Yes': 1, 'No': 0})
        )

        self.df = self.df.dropna(
            subset=self.features + [self.target]
        )

        X = self.df[self.features]

        y = self.df[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )
        )

        print("\nPreprocessing Completed!")

        print("Training Data:", self.X_train.shape)

        print("Testing Data:", self.X_test.shape)

    # 5. Train model

    def train_model(self):

        print("\nTraining Linear Regression Model...")

        self.model.fit(
            self.X_train,
            self.y_train
        )

        print("\nModel Training Completed!")

    # 6. Evaluate model

    def evaluate_model(self):

        print("\nEvaluating Model...")

        predictions = self.model.predict(self.X_test)

        r2 = r2_score(
            self.y_test,
            predictions
        )

        mse = mean_squared_error(
            self.y_test,
            predictions
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            self.y_test,
            predictions
        )

        print("\nMODEL PERFORMANCE")

        print("R2 Score:", round(r2, 4))

        print("MSE:", round(mse, 4))

        print("RMSE:", round(rmse, 4))

        print("MAE:", round(mae, 4))

    # 7. Sample prediction

    def sample_prediction(self):

        sample = pd.DataFrame(
            [[7, 85, 1, 8, 5]],
            columns=self.features
        )

        prediction = self.model.predict(sample)

        print("\nSample Student Prediction:")

        print("Performance Index:", round(prediction[0], 2))

    # 8. Save trained model

    def save_model(self):

        with open("model.pkl", "wb") as file:

            pickle.dump(self.model, file)

        print("\nModel Saved Successfully!")


# 9. Main execution

if __name__ == "__main__":

    student_model = StudentPerformanceModel()

    student_model.load_dataset()

    student_model.preprocess_data()

    student_model.train_model()

    student_model.evaluate_model()

    student_model.sample_prediction()

    student_model.save_model()
