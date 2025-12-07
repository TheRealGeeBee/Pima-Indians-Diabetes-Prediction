from flask import Flask, request, render_template, redirect
import joblib as jb
import pandas as pd
import shap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

shap.initjs()
app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        Pregnancies = request.form["no_of_pregnancies"]
        Glucose = request.form["glucose_reading"]
        Insulin = request.form["insulin_reading"]
        BMI = request.form["BMI"]
        Age = request.form["Age"]

        scaler = jb.load("StandardScaler.pkl")
        model = jb.load("GBM_Classifier.pkl")
        explainer = jb.load("GBMExplainer.pkl")

        data = [[Pregnancies, Glucose, Insulin, BMI, Age]]
        inference_data = pd.DataFrame(scaler.transform(data),
                                      columns=["Pregnancies",
                                               "Glucose", "Insulin",
                                               "BMI", "Age"])
        print(inference_data)

        prediction = model.predict(inference_data)
        data_to_show = [] #This is the explainability feature we will show on the result page
        shap_values = explainer(inference_data)
        values_to_be_displayed = [round(x, 3) for x in shap_values[0].values]
        data_columns = ["Pregnancies", "Glucose", "Insulin",
                                               "BMI", "Age"]
        for i, j in zip(data_columns, values_to_be_displayed):
            value = [i, j]
            data_to_show.append(value)
        shap.plots.waterfall(shap_values[0], show=False)
        plt.savefig(r"static\waterfall.png", bbox_inches="tight")

        if prediction == 1:
            prediction_string = "Positive"
        else:
            prediction_string = "Negative"

        confidence = round(np.max(model.predict_proba(inference_data), axis=1)[0]*100, 1)
        return render_template("result_page.html", pred=prediction_string, confidence=confidence,
                               data=data_to_show)
    else:
        return render_template("home_page.html")

if __name__ == "__main__":
    app.run(debug=True)