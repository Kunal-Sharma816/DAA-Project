import numpy as np
import re
import pickle
import nltk
from flask import Flask, render_template, request, url_for
import requests

nltk.download("punkt")
nltk.download("stopwords")

# Load resume screening models
clf = pickle.load(open("clf.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

app = Flask(__name__)

# Resume screening function
def CleanResume(resume_text):
    cleanTxt = re.sub(r"http\S+\s", " ", resume_text)
    cleanTxt = re.sub(r"RT|cc", " ", cleanTxt)
    cleanTxt = re.sub(r"#\S+\s", " ", cleanTxt)
    cleanTxt = re.sub(r"@\S+", " ", cleanTxt)
    cleanTxt = re.sub(
        r"[%s]" % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", cleanTxt
    )
    cleanTxt = re.sub(r"[^\x00-\x7f]", " ", cleanTxt)
    cleanTxt = re.sub(r"\s+", " ", cleanTxt)
    return cleanTxt

# Categories mapping for resume screening
categories_mapping = {
    6: "Data Science",
    12: "HR",
    0: "Advocate",
    1: "Arts",
    24: "Web Designing",
    16: "Mechanical Engineer",
    22: "Sales",
    14: "Health and fitness",
    5: "Civil Engineer",
    15: "Java Developer",
    4: "Business Analyst",
    21: "SAP Developer",
    2: "Automation Testing",
    11: "Electrical Engineering",
    18: "Operations Manager",
    20: "Python Developer",
    8: "DevOps Engineer",
    17: "Network Security Engineer",
    19: "PMO",
    7: "Database",
    13: "Hadoop",
    10: "ETL Developer",
    9: "DotNet Developer",
    3: "Blockchain",
    23: "Testing",
}

@app.route("/", methods=["GET", "POST"])
def Index():
    return render_template("index.html")

@app.route("/Summarize", methods=["POST"])
def Summarize():
    API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
    headers = {"Authorization": "Bearer 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"}

    data = request.form["data"]
    length = request.form["length"]  # Get the selected length option
    
    # Set different max lengths based on user selection
    length_mapping = {
        "short": {"min": 30, "max": 70},
        "medium": {"min": 70, "max": 150},
        "long": {"min": 150, "max": 250}
    }
    
    params = length_mapping.get(length, length_mapping["medium"])
    
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    try:
        output = query({
            "inputs": data,
            "parameters": {"min_length": params["min"], "max_length": params["max"]},
        })[0]
        return render_template("index.html", summary_result=output["summary_text"])
    except Exception as e:
        return render_template("index.html", summary_error=str(e))
@app.route("/ScreenResume", methods=["POST"])
def ScreenResume():
    if 'resume' not in request.files:
        return render_template("index.html", resume_error="No file uploaded")
    
    uploaded_file = request.files['resume']
    
    if uploaded_file.filename == '':
        return render_template("index.html", resume_error="No file selected")
    
    try:
        resume_bytes = uploaded_file.read()
        resume_text = resume_bytes.decode("utf-8")
    except UnicodeDecodeError:
        resume_text = resume_bytes.decode("latin-1")
    
    cleaned_resume = CleanResume(resume_text)
    input_features = tfidf.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]
    
    category_name = categories_mapping.get(prediction_id, "Unknown")
    return render_template("index.html", resume_result=category_name)

if __name__ == "__main__":
    app.run(debug=True)