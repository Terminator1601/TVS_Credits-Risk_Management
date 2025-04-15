from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import sqlite3
import os

current_Directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(current_Directory)

model = pickle.load(open(os.path.join(current_Directory, 'trained_model1_rf.sav'), 'rb'))

app = Flask(__name__, 
           template_folder=os.path.join(current_Directory, 'templates'),
           static_folder=os.path.join(current_Directory, 'static'))


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    applicant_name = request.form.get('Name')
    Gender = request.form.get('Gender')
    Marital_status = request.form.get('Married')
    Dependents = request.form.get('Dependents')
    Education = request.form.get('Education')
    Property = request.form.get('Property_Area')
    Employment = request.form.get('Employment')
    Loan_Term = request.form.get('Loan_Amount_Term')
    applicant_income = float(request.form.get('ApplicantIncome'))
    coapplicant_income = float(request.form.get('CoapplicantIncome'))
    loan_amount = float(request.form.get('LoanAmount'))
    credit_history = float(request.form.get('Credit_History'))

    input_data = np.array([Gender, Marital_status, Dependents, Education, Employment, applicant_income, coapplicant_income,
                          loan_amount, Loan_Term, credit_history, Property]).reshape(1, -1)
    prediction = model.predict(input_data)
    print(prediction)
    
    if prediction[0] == '0':
        status = 'n'
    else:
        status = 'y'

    db_path = os.path.join(current_Directory, "static", "database", "tvs_credit")
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute('SELECT MAX(loan_id) FROM userData')
    max_loan_id = cursor.fetchone()[0]
    loan_id = max_loan_id + 1 if max_loan_id is not None else 1
    
    query1 = "INSERT INTO userData (loan_id, name, gender, married_status, dependent, education, employment, property_area, a_income, co_income, loan_amount,loan_amt_term, credit_history,loan_status) VALUES (?,?, ?, ?, ?, ?, ?, ?,?, ?,?, ?, ?, ?)"
    cursor.execute(query1, (loan_id, applicant_name, Gender, Marital_status, Dependents, Education,
                   Employment, Property, applicant_income, coapplicant_income, loan_amount, Loan_Term, credit_history, status))
    connection.commit()
    connection.close()

    return render_template('after.html', status=prediction[0])


@app.route('/solution')
def solution():
    try:
        db_path = os.path.join(current_Directory, "static", "database", "tvs_credit")
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        cursor.execute("SELECT * FROM userData ORDER BY loan_id DESC LIMIT 1")
        latest_entry = cursor.fetchone()

        if latest_entry:
            loan_id, name, gender, married_status, dependent, education, employment, property_area, a_income, co_income, loan_amount, loan_amt_term, credit_history, loan_status = latest_entry
            connection.close()

            months = (loan_amt_term/30)
            money_to_be_lend = int((co_income + a_income) * months)

            return render_template('alternative.html',
                                credit_history=credit_history,
                                coapplicant_income=co_income,
                                loan_amt_term=loan_amt_term,
                                money_to_be_lend=money_to_be_lend)

        else:
            return jsonify({'error': 'No data found in the database.'}), 404

    except sqlite3.Error as e:
        return jsonify({'error': 'Database error: ' + str(e)}), 500


# This is required for local development, but Vercel will ignore it
if __name__ == "__main__":
    app.run() 