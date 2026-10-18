from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
from reportlab.pdfgen import canvas
import pandas as pd
import pickle
import json
import seaborn as sns
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import matplotlib.pyplot as plt
import uuid

from groq import Groq

import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# Load the environment variables from the .env file 
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Access the variables
api_key = os.getenv("GROQ_API")
# db_url = os.getenv("DATABASE_URL")



client = Groq(api_key=api_key)
# Load model, encoders, and scaler
with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'rb') as model_file:
    loaded_model = pickle.load(model_file)

MODEL_COLUMNS = list(loaded_model.feature_names_in_)
print(type(loaded_model))
print(hasattr(loaded_model, "feature_importances_"))

with open(os.path.join(MODEL_DIR, 'encoder.pkl'), 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as scaler_file:
    scaler_data = pickle.load(scaler_file)


app = Flask(__name__)
app.secret_key = "customer_churn_ai_2026"

def make_prediction(input_data):

    print("===== make_prediction START =====")

    input_df = pd.DataFrame([input_data])

    print("DataFrame Created")

    # Columns ko training order me arrange karo
    input_df = input_df.reindex(columns=MODEL_COLUMNS)

    print("Columns Reindexed")

    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    print("Encoding Finished")

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numerical_cols] = scaler_data.transform(input_df[numerical_cols])

    print("Scaling Finished")

    print(input_df)
    print(input_df.dtypes)
    print(input_df.isna().sum())

    print("Predicting...")

    prediction = loaded_model.predict(input_df)[0]

    print("Prediction Done")

    probability = loaded_model.predict_proba(input_df)[0, 1]
    confidence = round(max(probability, 1 - probability) * 100, 2)

    return "Churn" if prediction == 1 else "No Churn", probability, confidence


def generate_ai_explanation(input_data):

    reasons = []

    if input_data["Contract"] == "Month-to-month":
        reasons.append("📅 Customer has a Month-to-Month contract which increases churn risk.")

    if input_data["MonthlyCharges"] > 80:
        reasons.append("💰 Monthly charges are high.")

    if input_data["tenure"] < 12:
        reasons.append("⏳ Customer tenure is less than 12 months.")

    if input_data["InternetService"] == "Fiber optic":
        reasons.append("🌐 Fiber Optic customers generally have a higher churn rate.")

    if input_data["PaymentMethod"] == "Electronic check":
        reasons.append("💳 Electronic Check payment method is linked with higher churn.")

    if len(reasons) == 0:
        reasons.append("✅ Customer profile appears stable with low churn indicators.")

    return reasons

def generate_recommendations(input_data):

    tips = []

    if input_data["Contract"] == "Month-to-month":
        tips.append("📅 Offer a One-Year Contract with Discount.")

    if input_data["MonthlyCharges"] > 80:
        tips.append("💰 Provide a Special Discount Offer.")

    if input_data["tenure"] < 12:
        tips.append("🎁 Give a Welcome Loyalty Benefit.")

    if input_data["PaymentMethod"] == "Electronic check":
        tips.append("💳 Encourage Auto Payment using Credit Card or Bank Transfer.")

    if input_data["TechSupport"] == "No":
        tips.append("🛠 Offer Free Technical Support for 30 Days.")

    if len(tips) == 0:
        tips.append("✅ No immediate retention action is required.")

    return tips

def generate_dashboard_charts(user_history):

    os.makedirs(os.path.join(BASE_DIR, "static", "charts"), exist_ok=True)

    churn = sum(1 for i in user_history if i["prediction"] == "Churn")
    no_churn = len(user_history) - churn

    plt.figure(figsize=(5,5))
    plt.pie(
        [churn, no_churn],
        labels=["Churn","No Churn"],
        autopct="%1.1f%%"
    )
    plt.title("Prediction Distribution")
    plt.savefig(os.path.join(BASE_DIR, "static", "charts", "pie_chart.png"))
    plt.close()

        # Feature Importance Chart
    if hasattr(loaded_model, "feature_importances_"):

        importance = loaded_model.feature_importances_

        feature_df = pd.DataFrame({
            "Feature": MODEL_COLUMNS,
            "Importance": importance
        })

        feature_df = feature_df.sort_values(
            by="Importance",
            ascending=False
        ).head(10)

        plt.figure(figsize=(8,5))

        plt.barh(
            feature_df["Feature"],
            feature_df["Importance"]
        )

        plt.title("Top 10 Important Features")
        plt.xlabel("Importance")
        plt.gca().invert_yaxis()

        plt.tight_layout()

        plt.savefig(os.path.join(BASE_DIR, "static", "charts", "feature_importance.png"))

        plt.close()

    probs = [i["probability"] for i in user_history]

    plt.figure(figsize=(7,4))
    plt.plot(range(1,len(probs)+1), probs, marker="o")
    plt.title("Prediction Probability")
    plt.xlabel("Prediction Number")
    plt.ylabel("Probability (%)")
    plt.grid(True)
    plt.savefig(os.path.join(BASE_DIR, "static", "charts", "probability_chart.png"))
    plt.close()

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

try:
    with open(os.path.join(STORAGE_DIR, "users.json"), "r") as file:
        users = json.load(file)
except:
    users = {}

if "admin@gmail.com" not in users:
    users["admin@gmail.com"] = "123456"

    with open(os.path.join(STORAGE_DIR, "users.json"), "w") as file:
        json.dump(users, file, indent=4)
try:
    with open(os.path.join(STORAGE_DIR, "history.json"), "r") as file:
        history = json.load(file)
except:
    history = []

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        if email in users and users[email] == password:
            session["user"] = email
            return redirect(url_for("index"))

        return render_template(
            "login.html",
            error="Invalid Email or Password"
        )

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        if email in users:
            return render_template(
                "register.html",
                error="Email already exists"
            )

        users[email] = password

        with open(os.path.join(STORAGE_DIR, "users.json"), "w") as file:
            json.dump(users, file, indent=4)

        return redirect(url_for("login"))

    return render_template("register.html")

@app.route('/', methods=['GET', 'POST'])
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    prediction = None
    probability = None
    confidence = None
    explanation = None
    recommendations = None
    if request.method == 'POST':
        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges']),
        }

        prediction, probability, confidence = make_prediction(input_data)
        session["form_data"] = input_data
        explanation = generate_ai_explanation(input_data)
        recommendations = generate_recommendations(input_data)

        session["prediction_result"] = {
            "prediction": prediction,
            "probability": probability,
            "confidence": confidence,
            "explanation": explanation,
            "recommendations": recommendations
        }

        session["last_prediction"] = {
            "prediction": prediction,
            "probability": round(probability * 100, 2),
            "confidence": confidence,
            "time": datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        }

        history.append({
            "user": session["user"],
            "prediction": prediction,
            "probability": round(probability * 100, 2),
            "time": datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        })

        with open(os.path.join(STORAGE_DIR, "history.json"), "w") as file:
            json.dump(history, file, indent=4)

    if request.method == "POST":
        return render_template(
            "result.html",
            prediction=prediction,
            probability=probability,
            confidence=confidence,
            explanation=explanation,
            recommendations=recommendations
    )

    return render_template(
    "churn.html",
    form_data=session.get("form_data", {})
    )
@app.route("/download_report")
def download_report():

    if "last_prediction" not in session:
        return redirect(url_for("index"))

    data = session["last_prediction"]

    filename = os.path.join(REPORT_DIR, "Customer_Churn_Report.pdf")

    c = canvas.Canvas(filename)

    c.setFont("Helvetica-Bold", 20)
    c.drawString(150, 800, "Customer Churn Prediction Report")

    c.setFont("Helvetica", 14)

    c.drawString(50, 740, f"Prediction : {data['prediction']}")
    c.drawString(50, 710, f"Probability : {data['probability']}%")
    c.drawString(50, 680, f"AI Confidence : {data['confidence']}%")
    c.drawString(50, 650, f"Generated On : {data['time']}")

    c.save()

    return send_file(
        filename,
        as_attachment=True,
        download_name="Customer_Churn_Report.pdf"
    )
@app.route("/explanation")
def explanation():

    if "prediction_result" not in session:
        return redirect(url_for("index"))

    return render_template(
        "explanation.html",
        data=session["prediction_result"]
    )


@app.route("/recommendation")
def recommendation():

    if "prediction_result" not in session:
        return redirect(url_for("index"))

    return render_template(
        "recommendation.html",
        data=session["prediction_result"]
    )
@app.route("/result")
def result():

    if "prediction_result" not in session:
        return redirect(url_for("index"))

    data = session["prediction_result"]

    return render_template(
        "result.html",
        prediction=data["prediction"],
        probability=data["probability"],
        confidence=data["confidence"]
    )

@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect(url_for("login"))

    try:
        with open(os.path.join(STORAGE_DIR, "history.json"), "r") as file:
            all_history = json.load(file)
    except:
        all_history = []

    user_history = [
        item for item in all_history
        if item["user"] == session["user"]
    ]

    total_predictions = len(user_history)
    # Dashboard Statistics
    today = datetime.now().strftime("%d-%m-%Y")

    today_predictions = sum(
        1 for item in user_history
        if item["time"].startswith(today)
    )

    churn_count = sum(
        1 for item in user_history
        if item["prediction"] == "Churn"
    )

    no_churn_count = total_predictions - churn_count
    
    retention_rate = 0
    if total_predictions > 0:
        retention_rate = round(
            (no_churn_count / total_predictions) * 100,
            2
        )

    avg_confidence = 0
    if total_predictions > 0:
        avg_confidence = round(
            sum(item["probability"] for item in user_history) / total_predictions,
            2
        )

    # Prediction Success Rate

    if total_predictions > 0:
        success_rate = round(
            (no_churn_count / total_predictions) * 100,
            2
        )
    else:
        success_rate = 0
    

    avg_probability = 0
    
    if total_predictions > 0:
        avg_probability = round(
            sum(item["probability"] for item in user_history) / total_predictions,
            2
        )

    # Customer Risk Level

    if avg_probability >= 70:
        risk_level = "🔴 High Risk"

    elif avg_probability >= 40:
        risk_level = "🟡 Medium Risk"

    else:
        risk_level = "🟢 Low Risk"

    accuracy = "Not Available"

    if hasattr(loaded_model, "score"):
        accuracy = "Trained Model"

    generate_dashboard_charts(user_history)

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        churn_count=churn_count,
        no_churn_count=no_churn_count,
        avg_probability=avg_probability,
        model_name=type(loaded_model).__name__,
        accuracy=accuracy,
        success_rate=success_rate,
        risk_level=risk_level,
        today_predictions=today_predictions,
        retention_rate=retention_rate,
        avg_confidence=avg_confidence,
    )

@app.route("/history")
def view_history():

    if "user" not in session:
        return redirect(url_for("login"))

    user_history = []

    try:
        with open(os.path.join(STORAGE_DIR, "history.json"), "r") as file:
            all_history = json.load(file)

        for item in all_history:
            if item["user"] == session["user"]:
                user_history.append(item)

    except:
        user_history = []

    user_history.reverse()

    return render_template(
        "history.html",
        history=user_history
    )

@app.route("/bulk_prediction", methods=["GET", "POST"])
def bulk_prediction():

    if request.method == "GET":
        return render_template("bulk_prediction.html")

    file = request.files["csv_file"]

    if file.filename == "":
        return "No File Selected"

    filename = secure_filename(file.filename)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(UPLOAD_DIR, filename)

    file.save(filepath)

    df = pd.read_csv(filepath)

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    df[numeric_cols] = df[numeric_cols].replace(r'^\s*$', 0, regex=True)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    print("CSV Columns:")
    print(df.columns)
    

    

    prediction_list = []

    probability_list = []

    for _, row in df.iterrows():

        customer = row.to_dict()

        customer.pop("customerID", None)
        customer.pop("Churn", None)

        customer = {}

        for col in MODEL_COLUMNS:
            customer[col] = row.get(col, None)


        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            value = customer[col]
            if pd.isna(value) or str(value).strip() == "":
                customer[col] = 0
            else:
                customer[col] = float(value)


        for col in encoders.keys():
            value = customer[col]
            if pd.isna(value) or str(value).strip() == "":
                customer[col] = encoders[col].classes_[0]
            elif value not in encoders[col].classes_:
                customer[col] = encoders[col].classes_[0]

        print("Row Number:", _)
        print(customer)

        prediction, probability, confidence = make_prediction(customer)

        print("Prediction Finished")

        prediction_list.append(prediction)
        probability_list.append(round(probability * 100, 2))

    df["Prediction"] = prediction_list

    df["Churn Probability (%)"] = probability_list

    output_path = os.path.join(UPLOAD_DIR, "Prediction_Result.csv")

    df.to_csv(output_path, index=False)

    return send_file(
        output_path,
        as_attachment=True,
        download_name="Prediction_Result.csv"
    )

@app.route("/chatbot", methods=["POST"])
def chatbot():

    data = request.get_json()
    message = data.get("message", "")

    chat_history = session.get("chat_history", [])

    chat_history.append({
        "role": "user",
        "content": message
    })

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages = [
                {
                     "role": "system",
                     "content": """You are an AI Customer Churn Assistant.

            - Explain customer churn in simple language.
            - If the user asks about the current dashboard or prediction, use the prediction available in chat context.
            - Answer customer churn questions professionally.
            - Also work as a general AI assistant for normal questions.
            - Maintain conversation history naturally."""
                    }
                ] + chat_history
            )

        reply = response.choices[0].message.content

        chat_history.append({
            "role": "assistant",
            "content": reply
        })

        session["chat_history"] = chat_history

    except Exception as e:
        reply = f"Groq Error: {str(e)}"

    return jsonify({
        "reply": reply
    })


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)