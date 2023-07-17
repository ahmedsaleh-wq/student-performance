from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
app = Flask(__name__)

ensemble_model = joblib.load('ensemble_model.pkl')

age_mapping = {1: [18, 21], 2: [22, 25], 3: [26, 100]}
sex_mapping = {1: 'female', 2: 'male'}
graduated_h_school_type_mapping = {1: 'private', 2: 'state', 3: 'other'}
scholarship_type_mapping = {1: 'None', 2: '25%', 3: '50%', 4: '75%', 5: 'Full'}
additional_work_mapping = {1: 'Yes', 2: 'No'}
activity_mapping = {1: 'Yes', 2: 'No'}
partner_mapping = {1: 'Yes', 2: 'No'}
total_salary_mapping = {1: 'USD 135-200', 2: 'USD 201-270', 3: 'USD 271-340', 4: 'USD 341-410', 5: 'above 410'}
transport_mapping = {1: 'Bus', 2: 'Private car/taxi', 3: 'Bicycle', 4: 'Other'}
accomodation_mapping = {1: 'rental', 2: 'dormitory', 3: 'with family', 4: 'Other'}
mother_ed_mapping = {1: 'primary school', 2: 'secondary school', 3: 'high school', 4: 'university', 5: 'MSc.', 6: 'Ph.D.'}
farther_ed_mapping = {1: 'primary school', 2: 'secondary school', 3: 'high school', 4: 'university', 5: 'MSc.', 6: 'Ph.D.'}
siblings_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: '5 or above'}
parental_status_mapping = {1: 'married', 2: 'divorced', 3: 'died - one of them or both'}
mother_occup_mapping = {1: 'retired', 2: 'housewife', 3: 'government officer', 4: 'private sector employee', 5: 'self-employment', 6: 'other'}
father_occup_mapping = {1: 'retired', 2: 'government officer', 3: 'private sector employee', 4: 'self-employment', 5: 'other'}
weekly_study_hours_mapping = {1: 'None', 2: '<5 hours', 3: '6-10 hours', 4: '11-20 hours', 5: 'more than 20 hours'}
reading_non_scientific_mapping = {1: 'None', 2: 'Sometimes', 3: 'Often'}
reading_scientific_mapping = {1: 'None', 2: 'Sometimes', 3: 'Often'}
attendance_seminars_dep_mapping = {1: 'Yes', 2: 'No'}
impact_of_projects_mapping = {1: 'positive', 2: 'negative', 3: 'neutral'}
attendances_classes_mapping = {1: 'always', 2: 'sometimes', 3: 'never'}
preparation_midterm_company_mapping = {1: 'alone', 2: 'with friends', 3: 'not applicable'}
preparation_midterm_time_mapping = {1: 'closest date to the exam', 2: 'regularly during the semester', 3: 'never'}
taking_notes_mapping = {1: 'never', 2: 'sometimes', 3: 'always'}
listenning_mapping = {1: 'never', 2: 'sometimes', 3: 'always'}
discussion_improves_interest_mapping = {1: 'never', 2: 'sometimes', 3: 'always'}
flip_classrom_mapping = {1: 'not useful', 2: 'useful', 3: 'not applicable'}
grade_previous_mapping = {1: '<2.00', 2: '2.00-2.49', 3: '2.50-2.99', 4: '3.00-3.49', 5: 'above 3.49'}
grade_expected_mapping = {1: '<2.00', 2: '2.00-2.49', 3: '2.50-2.99', 4: '3.00-3.49', 5: 'above 3.49'}

course_id_mapping = {1: 'Course 1', 2: 'Course 2', 3: 'Course 3', 4: 'Course 4', 5: 'Course 5', 6: 'Course 6', 7: 'Course 7', 8: 'Course 8', 9: 'Course 9'}  
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    questionnaire_data = {
        'age': [int(request.form['age'])],
        'sex': [int(request.form['sex'])],
        'graduated_h_school_type': [int(request.form['graduated_h_school_type'])],
        'scholarship_type': [int(request.form['scholarship_type'])],
        'additional_work': [int(request.form['additional_work'])],
        'activity': [int(request.form['activity'])],
        'partner': [int(request.form['partner'])],
        'total_salary': [int(request.form['total_salary'])],
        'transport': [int(request.form['transport'])],
        'accomodation': [int(request.form['accomodation'])],
        'mother_ed': [int(request.form['mother_ed'])],
        'farther_ed': [int(request.form['farther_ed'])],
        'siblings': [int(request.form['siblings'])],
        'parental_status': [int(request.form['parental_status'])],
        'mother_occup': [int(request.form['mother_occup'])],
        'father_occup': [int(request.form['father_occup'])],
        'weekly_study_hours': [int(request.form['weekly_study_hours'])],
        'reading_non_scientific': [int(request.form['reading_non_scientific'])],
        'reading_scientific': [int(request.form['reading_scientific'])],
        'attendance_seminars_dep': [int(request.form['attendance_seminars_dep'])],
        'impact_of_projects': [int(request.form['impact_of_projects'])],
        'attendances_classes': [int(request.form['attendances_classes'])],
        'preparation_midterm_company': [int(request.form['preparation_midterm_company'])],
        'preparation_midterm_time': [int(request.form['preparation_midterm_time'])],
        'taking_notes': [int(request.form['taking_notes'])],
        'listenning': [int(request.form['listenning'])],
        'discussion_improves_interest': [int(request.form['discussion_improves_interest'])],
        'flip_classrom': [int(request.form['flip_classrom'])],
        'grade_previous': [int(request.form['grade_previous'])],
        'grade_expected': [int(request.form['grade_expected'])],
        'course_id': [int(request.form['course_id'])],
    }


    input_data = pd.DataFrame(questionnaire_data)

   
    prediction = ensemble_model.predict(input_data)
    grade_mapping = {
        0: 'Fail',
        1: 'DD',
        2: 'DC',
        3: 'CC',
        4: 'CB',
        5: 'BB',
        6: 'BA',
        7: 'AA'
    }
    predicted_grade = grade_mapping[prediction[0]]

    return render_template('result.html', prediction=predicted_grade)

if __name__ == '__main__':
    app.run(debug=True)
