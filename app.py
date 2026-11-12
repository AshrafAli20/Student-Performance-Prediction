
# =====================================================
# STUDENT PERFORMANCE PREDICTION WEB APPLICATION
# =====================================================

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle


class StudentPerformanceApp:

    def __init__(self):

        self.app = Flask(__name__)

        # Load trained machine learning model
        with open('model.pkl', 'rb') as file:
            self.model = pickle.load(file)

        # Register application routes
        self.app.add_url_rule(
            '/',
            'home',
            self.home,
            methods=['GET']
        )

        self.app.add_url_rule(
            '/predict',
            'predict',
            self.predict,
            methods=['POST']
        )

    # Home page
    def home(self):

        return render_template(
            'index.html',
            prediction=None,
            error=None,
            form_data={}
        )

    # Prediction functionality
    def predict(self):

        form_data = request.form.to_dict()

        try:

            hours = float(form_data['hours_studied'])
            previous_scores = float(
                form_data['previous_scores']
            )

            extracurricular = int(
                form_data['extracurricular']
            )

            sleep_hours = float(
                form_data['sleep_hours']
            )

            sample_papers = int(
                form_data['sample_papers']
            )

            # Input validation
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
                    "Select a valid extracurricular option."
                )

            if not 4 <= sleep_hours <= 9:
                raise ValueError(
                    "Sleep hours must be between 4 and 9."
                )

            if not 0 <= sample_papers <= 9:
                raise ValueError(
                    "Sample papers must be between 0 and 9."
                )

            # Prepare input data
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

            return render_template(
                'index.html',
                prediction=prediction,
                error=None,
                form_data=form_data
            )

        except (ValueError, TypeError) as error:

            return render_template(
                'index.html',
                prediction=None,
                error=str(error),
                form_data=form_data
            )

        except Exception:

            return render_template(
                'index.html',
                prediction=None,
                error="Something went wrong. Please try again.",
                form_data=form_data
            )


# Create application object
student_app = StudentPerformanceApp()

app = student_app.app


if __name__ == '__main__':

    app.run(debug=True)
