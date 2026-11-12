
# ==========================================
# STUDENT PERFORMANCE PREDICTION WEB APP
# ==========================================

# 1. Import libraries

from flask import Flask, render_template, request

import pandas as pd
import numpy as np
import pickle


# 2. Create application class

class StudentPerformanceApp:

    def __init__(self):

        self.app = Flask(__name__)

        # Load trained model

        with open("model.pkl", "rb") as file:

            self.model = pickle.load(file)

        print("Student Performance Model Loaded Successfully!")

        # Register routes

        self.app.add_url_rule(
            "/",
            "home",
            self.home,
            methods=["GET"]
        )

        self.app.add_url_rule(
            "/predict",
            "predict",
            self.predict,
            methods=["POST"]
        )

    # 3. Home page

    def home(self):

        return render_template(
            "index.html",
            prediction=None,
            error=None,
            form_data={}
        )

    # 4. Prediction functionality

    def predict(self):

        try:

            # Get form values

            hours = float(request.form["hours_studied"])

            previous_scores = float(
                request.form["previous_scores"]
            )

            extracurricular = int(
                request.form["extracurricular"]
            )

            sleep_hours = float(
                request.form["sleep_hours"]
            )

            sample_papers = float(
                request.form["sample_papers"]
            )

            form_data = request.form.to_dict()

            # Validate inputs

            if not 1 <= hours <= 9:

                raise ValueError(
                    "Hours studied must be between 1 and 9."
                )

            if not 40 <= previous_scores <= 99:

                raise ValueError(
                    "Previous scores must be between 40 and 99."
                )

            if extracurricular not in [0, 1]:

                raise ValueError(
                    "Please select a valid extracurricular option."
                )

            if not 4 <= sleep_hours <= 9:

                raise ValueError(
                    "Sleep hours must be between 4 and 9."
                )

            if not 0 <= sample_papers <= 9:

                raise ValueError(
                    "Sample papers must be between 0 and 9."
                )

            # Prepare input for prediction

            input_data = pd.DataFrame(
                [[
                    hours,
                    previous_scores,
                    extracurricular,
                    sleep_hours,
                    sample_papers
                ]],

                columns=[
                    'Hours Studied',
                    'Previous Scores',
                    'Extracurricular Activities',
                    'Sleep Hours',
                    'Sample Question Papers Practiced'
                ]
            )

            # Generate prediction

            result = self.model.predict(input_data)[0]

            prediction = round(
                float(np.clip(result, 10, 100)),
                2
            )

            # Return result to frontend

            return render_template(
                "index.html",
                prediction=prediction,
                error=None,
                form_data=form_data
            )

        except (ValueError, TypeError) as e:

            return render_template(
                "index.html",
                prediction=None,
                error=str(e),
                form_data=request.form.to_dict()
            )

        except Exception as e:

            print("Prediction Error:", e)

            return render_template(
                "index.html",
                prediction=None,
                error="Something went wrong. Please try again.",
                form_data=request.form.to_dict()
            )


# 5. Create Flask application

student_app = StudentPerformanceApp()

app = student_app.app


# 6. Run application

if __name__ == "__main__":

    app.run(debug=True)
