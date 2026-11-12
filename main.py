
# =====================================================
# STUDENT PERFORMANCE PREDICTION
# MULTIPLE LINEAR REGRESSION
# =====================================================

import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


class StudentPerformanceModel:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
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

    # 1. Load dataset
    def load_dataset(self):

        self.df = pd.read_csv(self.dataset_path)

        print("\n========== DATASET INFORMATION ==========")

        print("\nFirst five records:")
        print(self.df.head())

        print("\nDataset dimensions:")
        print(self.df.shape)

        print("\nMissing values:")
        print(self.df.isnull().sum())

        print("\nColumn information:")
        print(self.df.dtypes)

    # 2. Data preprocessing
    def preprocess_data(self):

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

        print("\n========== DATA SPLITTING ==========")

        print("Training features:", self.X_train.shape)
        print("Training target:", self.y_train.shape)

        print("Testing features:", self.X_test.shape)
        print("Testing target:", self.y_test.shape)

    # 3. Model training
    def train_model(self):

        print("\n========== MODEL TRAINING ==========")

        self.model.fit(self.X_train, self.y_train)

        print("Multiple Linear Regression trained successfully!")

    # 4. Model evaluation
    def evaluate_model(self):

        predictions = self.model.predict(self.X_test)

        r2 = r2_score(self.y_test, predictions)

        mse = mean_squared_error(
            self.y_test, predictions
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            self.y_test, predictions
        )

        print("\n========== MODEL EVALUATION ==========")

        print(f"R2 Score: {r2:.4f}")
        print(f"R2 Score Percentage: {r2 * 100:.2f}%")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")

    # 5. Sample prediction
    def sample_prediction(self):

        sample = pd.DataFrame(
            [[7, 85, 1, 8, 5]],
            columns=self.features
        )

        result = self.model.predict(sample)

        print("\n========== SAMPLE PREDICTION ==========")

        print(
            f"Predicted Performance Index: {result[0]:.2f}"
        )

    # 6. Save trained model
    def save_model(self):

        with open('model.pkl', 'wb') as file:

            pickle.dump(self.model, file)

        print("\nModel saved successfully as model.pkl")


# =====================================================
# MAIN PROGRAM
# =====================================================

if __name__ == "__main__":

    student_model = StudentPerformanceModel(
        'Student_Performance.csv'
    )

    student_model.load_dataset()

    student_model.preprocess_data()

    student_model.train_model()

    student_model.evaluate_model()

    student_model.sample_prediction()

    student_model.save_model()
