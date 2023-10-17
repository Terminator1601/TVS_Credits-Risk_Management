from flask import Flask, render_template, request, redirect
import pickle
import numpy as np
import sqlite3
import os

current_Directory = os.path.dirname(os.path.abspath(__file__))
print(current_Directory)

model = pickle.load(open('trained_model1_rf.sav', 'rb'))

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    # return max_loan_id + 1 if max_loan_id is not None else 1

    # loan_id = max_loan_id + 1 if max_loan_id is not None else 1
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

    # Prepare the input data for prediction
    input_data = np.array([Gender, Marital_status, Dependents, Education, Employment, applicant_income, coapplicant_income,
                          loan_amount, Loan_Term, credit_history, Property]).reshape(1, -1)
    # input_data_reshape = input_data_numpy.reshape(1, -1)
    prediction = model.predict(input_data)
    # percentage_approval = (X_test_prediction == 'Y').mean() * 100

    if prediction[0] == 0:
        status = 'n'
    else:
        status = 'y'

    # making connection with database

    connection = sqlite3.connect(
        current_Directory + "\\static\\database\\tvs_credit")
    cursor = connection.cursor()
    cursor.execute('SELECT MAX(loan_id) FROM userData')
    max_loan_id = cursor.fetchone()[0]
    # connection.close()
    loan_id = max_loan_id + 1 if max_loan_id is not None else 1
    cursor = connection.cursor()
    query1 = "INSERT INTO userData (loan_id, name, gender, married_status, dependent, education, employment, property_area, a_income, co_income, loan_amount,loan_amt_term, credit_history,loan_status) VALUES (?,?, ?, ?, ?, ?, ?, ?,?, ?,?, ?, ?, ?)"
    cursor.execute(query1, (loan_id, applicant_name, Gender, Marital_status, Dependents, Education,
                   Employment, Property, applicant_income, coapplicant_income, loan_amount, Loan_Term, credit_history, status))

    # query1="INSERT INTO userData VALUES({a_name},{gen}.{married},'{dependent}','{edu}','{emp}','{p_area}','{a_income}','{co_income}','{l_amt}','{c_score}')".format(a_name=applicant_name,gen=Gender,married=Marital_status,dependent=Dependents,edu=Education,emp=Employment,p_area=Property,a_income=applicant_income,co_income=coapplicant_income,l_amt=loan_amount,c_score=credit_history)
    # cursor.execute(query1)
    connection.commit()
    print(loan_amount, loan_id)

    return redirect(url_for('solutions', result=result, credit_history=credit_history, Loan_Term=Loan_Term, coapplicant_income=coapplicant_income))

    # print('Percentage of loan approvals in the test data:', percentage_approvall  )

    if prediction[0] == '0':
        result = 'rejected'
    else:
        result = 'approved'
    print(result)
    data_to_pass = {
        'credit_history': credit_history,
        'Loan_Term': Loan_Term,
        'coapplicant_income': coapplicant_income
    }

    return render_template('after.html', prediction_text=result)


@app.route('/solution')
def solution():
    result = request.args.get('result')
    credit_history = request.args.get('credit_history')
    Loan_Term = request.args.get('Loan_Term')
    coapplicant_income = request.args.get('coapplicant_income')

    # Render the 'alternative.html' template and pass the data for display
    return render_template('alternative.html', result=result, credit_history=credit_history, Loan_Term=Loan_Term, coapplicant_income=coapplicant_income)


if __name__ == "__main__":
    app.run(debug=True)
