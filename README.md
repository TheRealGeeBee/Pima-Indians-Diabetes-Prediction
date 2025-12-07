# Pima Indians Diabetes Prediction

This repository contains a **diabetes prediction web application** for female Pima Indians using machine learning. The app leverages a **Gradient Boosting Classifier** and provides **explainable predictions** using SHAP values.

---

## 🗂️ Project Overview

Diabetes is a growing health concern, and early detection is critical. This project uses the **Pima Indians Diabetes dataset** from Kaggle to train a predictive model that estimates the likelihood of diabetes based on key health features.

Key highlights of this project:

* **Dataset:** Kaggle's Pima Indians Diabetes dataset
* **Target Group:** Female Pima Indians
* **Machine Learning Model:** Gradient Boosting Classifier
* **Feature Selection:** Top 5 features selected using `SelectKBest` from `scikit-learn`
* **Selected Features:** `Glucose`, `BMI`, `Age`, `Insulin`, `Pregnancies`
* **Explainability:** SHAP values show how each feature contributes to the prediction
* **Deployment:** Flask web application with interactive HTML pages

---

## Features

* Input **individual health metrics** (top 5 features) via a web form
* View the **predicted likelihood of diabetes**
* Get the **confidence score** of the prediction
* See **feature contributions** via SHAP values for explainable AI

---

## Demo Screenshots

<img width="789" height="802" alt="Screenshot 2025-12-06 215742" src="https://github.com/user-attachments/assets/76282e29-1bd8-4f82-910e-491ce77c7d86" />


<img width="857" height="840" alt="Screenshot 2025-12-07 002654" src="https://github.com/user-attachments/assets/cb602f30-f3a6-453e-b233-13cb37060e3f" />

---


## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/TheRealGeeBee/pima-indians-diabetes-prediction.git
cd pima-indians-diabetes-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Flask server

```bash
python app.py
```

### 4. Access the app

Open your browser and navigate to:

```
http://127.0.0.1:5000/
```

---

## Model Details

* **Algorithm:** Gradient Boosting Classifier
* **Selected Features:** `Glucose`, `BMI`, `Age`, `Insulin`, `Pregnancies`
* **Data Preprocessing:** Numerical features scaled using `StandardScaler`
* **Explainability:** SHAP `Explainer.pkl` is used to generate feature importance for individual predictions

---

## 📂 Project Structure

```
pima-indians-diabetes-prediction/
│
├── app.py                  # Flask app
├── model.pkl               # Trained Gradient Boosting model
├── Explainer.pkl           # SHAP explainer for model interpretability
├── StandardScaler.pkl      # Scaler for input feature values
├── templates/              # HTML templates
│   ├── home_page.html      # Input form page
│   └── output_page.html    # Prediction result page
├── static/                 # CSS, JS, images
├── requirements.txt        # Python dependencies
└── README.md
```

---

## References

* [Kaggle: Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* [scikit-learn Gradient Boosting Classifier](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
* [SHAP Explainable AI](https://github.com/slundberg/shap)

---

## Future Improvements

* Expand to **predict for both genders**
* Include **more features** for higher accuracy
* Add **interactive visualizations** for SHAP explanations
* Deploy to **cloud (Heroku, AWS, etc.)** for public access

---

## License

This project is open-source and available under the MIT License.
