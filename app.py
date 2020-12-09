from flask import *
import joblib
##Loading the model
classifier = joblib.load('spam_message.pkl')
cv = joblib.load('vector_transform.pkl')


app = Flask(__name__)

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict', methods =['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        prediction = classifier.predict(vect)
    return render_template('result.html', prediction_message = prediction)


if __name__ == '__main__':
    app.run(debug = True)

