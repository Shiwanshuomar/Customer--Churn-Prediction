# Customer Churn Prediction

A Flask-based customer churn prediction application with ML prediction, dashboard charts, bulk CSV prediction, PDF reports, recommendations, and an optional Groq-powered chatbot.

## Project Structure

- `app.py` - Flask application and routes
- `model/` - trained model, encoder, and scaler
- `data/` - source dataset
- `templates/` - Jinja2 HTML templates
- `static/charts/` - dashboard chart assets
- `uploads/` - runtime uploaded/generated CSV files
- `storage/` - runtime users and prediction history
- `notebooks/` - model development notebook
- `reports/` - generated/report artifacts
- `Procfile` - deployment command

## Run locally

```bash
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

For the chatbot, create a `.env` file with:

```env
GROQ_API=your_api_key_here
```
