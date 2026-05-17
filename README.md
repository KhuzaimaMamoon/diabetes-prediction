# Diabetes Prediction: From Data to Deployment

MLOps SP26 Assignment 1 — End-to-end diabetes prediction pipeline covering data cleaning, EDA, model training, and FastAPI deployment.

---

## Project Description

This project uses a real-world medical dataset (`diabetes_unclean.csv`) to build a machine learning pipeline that predicts whether a patient is:
- **N** — Non-Diabetic  
- **P** — Pre-Diabetic  
- **Y** — Diabetic  

The pipeline covers data cleaning, exploratory data analysis (EDA), training 5 classification models, selecting the best one (Random Forest), and deploying it via a FastAPI REST API.

---

## Project Structure

```
diabetes_project/
├── data_model.ipynb          # Jupyter notebook: EDA, cleaning, model training
├── app.py                    # FastAPI application
├── diabetes_model.pkl        # Saved best model (Random Forest)
├── training_columns.pkl      # Column names used during training
├── label_encoder.pkl         # Label encoder for CLASS target
├── diabetes_clean.csv        # Cleaned dataset
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── screenshots/              # cURL response screenshots & EDA plots
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd diabetes_project
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the FastAPI Server

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.  
Auto-generated docs: `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint   | Description              |
|--------|------------|--------------------------|
| GET    | `/`        | Health check             |
| POST   | `/predict` | Predict diabetes class   |

### Input Schema (`/predict`)

```json
{
  "age": 65,
  "urea": 7.5,
  "cr": 52.0,
  "hba1c": 11.2,
  "chol": 6.1,
  "tg": 2.8,
  "hdl": 0.9,
  "ldl": 3.5,
  "vldl": 1.2,
  "bmi": 32.5,
  "gender": "M"
}
```

---

## Example cURL Commands

### Health Check
```bash
curl http://localhost:8000/
```

### Test 1 — Diabetic Patient
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 65, "urea": 7.5, "cr": 52.0, "hba1c": 11.2, "chol": 6.1, "tg": 2.8, "hdl": 0.9, "ldl": 3.5, "vldl": 1.2, "bmi": 32.5, "gender": "M"}'
```

### Test 2 — Non-Diabetic Patient
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 28, "urea": 4.2, "cr": 48.0, "hba1c": 5.1, "chol": 4.0, "tg": 1.2, "hdl": 1.8, "ldl": 2.1, "vldl": 0.6, "bmi": 22.0, "gender": "F"}'
```

### Test 3 — Invalid Gender (Validation Error)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "urea": 5.0, "cr": 50.0, "hba1c": 6.0, "chol": 5.0, "tg": 1.5, "hdl": 1.2, "ldl": 2.5, "vldl": 0.8, "bmi": 25.0, "gender": "X"}'
```

### Test 4 — Missing Fields (Validation Error)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 50, "urea": 5.0, "cr": 50.0}'
```

---

## Model Performance Comparison

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.9241   | 0.9177    | 0.9241 | 0.9194   |
| SVM                 | 0.8449   | 0.7138    | 0.8449 | 0.7738   |
| Decision Tree       | 0.9802   | 0.9800    | 0.9802 | 0.9800   |
| **Random Forest**   | **0.9868** | **0.9867** | **0.9868** | **0.9866** |
| KNN                 | 0.8779   | 0.8792    | 0.8779 | 0.8781   |

**Best Model: Random Forest** — highest accuracy, precision, recall, and F1-score on the 30% test split.

---

## Git Commit History

```
git commit -m "Initial commit: Add data cleaning notebook"
git commit -m "Add EDA visualizations (6 plots)"
git commit -m "Add model training and evaluation (5 models)"
git commit -m "Add FastAPI app with Pydantic validation"
git commit -m "Add cURL test examples and README"
```

---

## Screenshots

See the `screenshots/` folder for:
- EDA plots (gender distribution, age/BMI histograms, scatter plots, box plot)
- cURL command API responses
