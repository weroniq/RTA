from flask import Flask
from flask import request
from flask import jsonify
import joblib

#create a flask
app = Flask(__name__)

# Create an API end point
@app.route('/api/v1.0/predict', methods=['GET'])
def get_prediction():

    # fixed acidity
    fixed_acidity = float(request.args.get('fa'))
    # volatile acidity
    volatile_acidity = float(request.args.get('va'))
    # citric acid
    citric_acid = float(request.args.get('ca'))
    # residual sugar
    residual_sugar = float(request.args.get('rs'))
    # chlorides
    chlorides = float(request.args.get('ch'))
    # free sulfur dioxide
    free_sulfur_dioxide = float(request.args.get('fsd'))
    # total sulfur dioxide
    total_sulfur_dioxide = float(request.args.get('tsd'))
    # density
    density = float(request.args.get('de'))
    # ph
    ph = float(request.args.get('ph'))
    # sulphates
    sulphates = float(request.args.get('su'))
    # alcohol
    alcohol = float(request.args.get('al'))

    # The features of the observation to predict
    features = [fixed_acidity,
                volatile_acidity,
                citric_acid,
                residual_sugar,
                chlorides,
                free_sulfur_dioxide,
                total_sulfur_dioxide,
                density,
                ph,
                sulphates,
                alcohol]

    # Load pickled model file
    model = joblib.load('model.pkl')

    # Predict the class using the model
    predicted_class = int(model.predict([features]))

    # Return a json object containing the features and prediction
    return jsonify(features=features, predicted_class=predicted_class)


if __name__ == '__main__':
    # Run the app at 0.0.0.0:8889
    app.run(port=8889,host='0.0.0.0')