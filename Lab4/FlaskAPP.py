from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('/Users/nishkvaishnav/Documents/AIDI-2004-AI-In-Enterprisesystems/Lab4/model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('Front.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = [float(data['Length1']), float(data['Length2']), float(data['Length3']),
                float(data['Height']), float(data['Width'])]

    species = data['Species']
    if species == 'Bream':
        features += [1, 0, 0, 0, 0, 0]
    elif species == 'Roach':
        features += [0, 1, 0, 0, 0, 0]
    elif species == 'Pike':
        features += [0, 0, 1, 0, 0, 0]
    elif species == 'Perch':
        features += [0, 0, 0, 1, 0, 0]
    elif species == 'Smelt':
        features += [0, 0, 0, 0, 1, 0]
    elif species == 'Parkki':
        features += [0, 0, 0, 0, 0, 1]

    prediction = model.predict([features])[0]
    return f'The predicted weight is {prediction:.2f}g'


if __name__ == '__main__':
    app.run(debug=True)
