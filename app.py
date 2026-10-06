from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

#Load trained model
with open("model.pkl", "rb") as file:
    model, encoder = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():  

    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    previous_marks = float(request.form["previous_marks"])

    #Make prediction
    prediction = model.predict([
        [study_hours, attendance, previous_marks]
    ])

    result = encoder.inverse_transform(prediction)[0]

    return render_template(
        "result.html",
        result=result,
        study_hours=study_hours,
        attendance=attendance,
        previous_marks=previous_marks
    )

if __name__ == "__main__":
    app.run(debug=True)