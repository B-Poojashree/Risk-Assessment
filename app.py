from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Dummy user database
users = {}

# --- Load and prepare model using only key features ---
df = pd.read_csv("C://Users/pooja/Documents/financial_risk_assessment.csv")
df.dropna(subset=['Risk_Rating'], inplace=True)

features = ["Age", "Income", "Loan_Amount", "Credit_Score"]
df = df[features + ["Risk_Rating"]]

num_imputer = SimpleImputer(strategy='mean')
df[features] = num_imputer.fit_transform(df[features])

le = LabelEncoder()
df['Risk_Rating'] = le.fit_transform(df['Risk_Rating'])

X = df[features]
y = df['Risk_Rating']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('predict'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if username exists and password matches
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('predict'))  # This redirects to predict page
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm = request.form['confirm']

        if len(username) < 3:
            flash('Username must be at least 3 characters')
        elif password != confirm:
            flash('Passwords do not match')
        elif username in users:
            flash('Username already exists')
        else:
            users[username] = generate_password_hash(password)
            flash('Signup successful! Please login.')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    risk = None
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            income = float(request.form['income'])
            loan_amount = float(request.form['loan_amount'])
            credit_score = float(request.form['credit_score'])

            input_data = np.array([[age, income, loan_amount, credit_score]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            predicted_label = le.inverse_transform(prediction)[0]

            risk_map = {
                "High Risk": "High",
                "Low Risk": "Low",
                "Medium Risk": "Normal"
            }
            risk = risk_map.get(predicted_label, "Normal")
        except Exception as e:
            risk = f"Error: {e}"

    return render_template('predict.html', username=session['username'], risk=risk)

if __name__ == '__main__':
    app.run(debug=True)