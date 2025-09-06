from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("spam_classifier.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    vect_msg = vectorizer.transform([message])
    
    prediction = model.predict(vect_msg)[0]
    probas = model.predict_proba(vect_msg)[0]
    spam_proba = probas[1]
    ham_proba = probas[0]

    if prediction == 1:
        label = "Spam ❌"
        confidence = spam_proba * 100
    else:
        label = "Ham ✅"
        confidence = ham_proba * 100

    result = f"{label}<br>Spam Confidence: {spam_proba * 100:.2f}%<br>Ham Confidence: {ham_proba * 100:.2f}%"
    print(result)  # Debug print

    return render_template("index.html", prediction=result)

# if __name__ == '__main__':
#     app.run(debug=True)
#     import webbrowser
# from threading import Timer

# def open_browser():
#     webbrowser.open_new("http://127.0.0.1:5000/")

# if __name__ == '__main__':
#     Timer(1, open_browser).start()
#     app.run(debug=False)

