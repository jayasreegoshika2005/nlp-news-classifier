from flask import Flask, render_template, request
import pickle
import pytesseract
import cv2
from PIL import Image
from utils import preprocess_text

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_text", methods=["POST"])
def predict_text():
    text = request.form["news_text"]
    clean = preprocess_text(text)
    vector = vectorizer.transform([clean])
    prediction = model.predict(vector)[0]
    if prediction == 1:
        prediction = "World"
    elif prediction == 2:
        prediction = "Sports"
    elif prediction == 3:
        prediction = "Business"
    elif prediction == 4:
        prediction = "Sci/Tech"

    return render_template("index.html", prediction=prediction)
if __name__ == "__main__":
    app.run(debug=True)