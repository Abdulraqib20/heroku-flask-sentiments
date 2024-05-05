from flask import Flask, render_template, request
from model import sentiment_score, preprocess_text
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# Load the DataFrame
df = pd.read_csv('survey_data.csv')

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    if request.method == "POST":
        user_input = request.form.get("user_input")
        if user_input:
            # Preprocess the user input
            preprocessed_input = preprocess_text(user_input)
            # Perform sentiment analysis
            result = sentiment_score(preprocessed_input)
            sentiment = {
                'predicted_label': result.get('predicted_label', None),
                'confidence_percentage': result.get('confidence_percentage', None)
            }
    return render_template("index.html", sentiment=sentiment)


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if request.method == 'POST':
        # Get form data
        user_input = request.form.get('user_input')
        course_code = request.form.get('course_code')
        previous_exp = request.form.get('previous_exp')
        gender = request.form.get('gender')
        attendance = request.form.get('attendance')
        difficulty = request.form.get('difficulty')
        study_hours = request.form.get('study_hours')
        satisfaction = request.form.get('satisfaction')
        department = request.form.get('department')

        # Update the DataFrame with the new feedback
        new_feedback = pd.DataFrame({
            'course code': [course_code], 
            'feedback': [user_input],
            'previous experience': [previous_exp],
            'gender': [gender],
            'attendance': [attendance],
            'course difficulty': [difficulty],
            'study hours (per week)': [study_hours],
            'overall satisfaction': [satisfaction],
            'department': [department],
            'date': [datetime.now().date()],
            'time': [datetime.now().strftime('%H:%M:%S')],
            'hour': [datetime.now().hour],
        })

        df['date'] = pd.to_datetime(df['date'])
        df = pd.concat([df, new_feedback], ignore_index=True)

        # Save the updated dataset to the CSV file
        df.to_csv('survey_data.csv', index=False)

        # Perform sentiment analysis
        preprocessed_input = preprocess_text(user_input)
        result = sentiment_score(preprocessed_input)
        sentiment = {
            'predicted_label': result.get('predicted_label', None),
            'confidence_percentage': result.get('confidence_percentage', None)
        }

        # Render template with sentiment analysis result
        return render_template('index.html', sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)