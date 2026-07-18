from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("house_price_model.pkl")

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    area = float(data['area'])
    age = float(data['age'])
    bedrooms = float(data['bedrooms'])

    predicted_output = model.predict([[area, bedrooms, age]])

    print(predicted_output)
    return render_template('index.html', predicted_price=predicted_output[0])

@app.route("/profile")
def profile():
    return render_template('profile.html')

if __name__ == '__main__':
    app.run(debug=True)


