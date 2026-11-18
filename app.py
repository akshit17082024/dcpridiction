from flask import Flask, render_template, request, flash, redirect
import os
import pickle
import traceback
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def model_path(name):
    # absolute path so the app works no matter which folder it is started from
    return os.path.join(BASE_DIR, 'models', name)

_KERAS_CACHE = {}

def get_keras_model(name):
    # imported lazily so a TensorFlow problem can't break the other pages.
    # The model is loaded once and then kept in memory (much faster next time).
    # NOTE: the *.keras files are the same trained networks as the old *.h5 files,
    # re-saved so that Keras 3 (TensorFlow 2.16+) can read them.
    if name not in _KERAS_CACHE:
        from tensorflow.keras.models import load_model
        _KERAS_CACHE[name] = load_model(model_path(name))
    return _KERAS_CACHE[name]


app = Flask(__name__)

# The models expect their inputs in exactly this order. Forms that send a hidden
# "disease" field are read by NAME in this order, so the page layout (sections,
# dropdowns, ...) can be arranged freely without changing the predictions.
FEATURE_ORDER = {
    'heart': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
              'exang', 'oldpeak', 'slope', 'ca', 'thal'],
    'liver': ['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
              'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
              'Albumin', 'Albumin_and_Globulin_Ratio', 'Gender_Male'],
    'kidney': ['age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc',
               'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane'],
}

def predict(values, dic):
    if len(values) == 8:
        model = pickle.load(open(model_path('diabetes.pkl'),'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 13:
        model = pickle.load(open(model_path('heart.pkl'),'rb'))
        values = np.asarray(values)
        # In the heart dataset the training label 1 means "NO heart disease" (verified:
        # more blocked vessels / exercise angina / ST depression all push it towards 0).
        # The result page treats 1 as "consult a doctor", so flip it here.
        return 1 - model.predict(values.reshape(1, -1))[0]
    elif len(values) == 18:
        model = pickle.load(open(model_path('kidney.pkl'),'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 10:
        model = pickle.load(open(model_path('liver.pkl'),'rb'))
        values = np.asarray(values)
        # The notebook re-labelled the data with replace([2,1],[1,0]), so in the trained
        # model 1 means "NO liver disease" and 0 means "liver patient" (verified with a
        # liver-patient row from the training data). The result page treats 1 as
        # "consult a doctor", so flip it here.
        return 1 - model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    if request.method != 'POST':
        return redirect('/')
    to_predict_dict = request.form.to_dict()
    disease = to_predict_dict.pop('disease', None)
    try:
        if disease in FEATURE_ORDER:
            to_predict_list = [float(to_predict_dict[k]) for k in FEATURE_ORDER[disease]]
        else:
            to_predict_list = [float(v) for v in to_predict_dict.values()]
    except (ValueError, KeyError):
        # the user typed something that is not a number (blank, letters, ...)
        return render_template("home.html", message="Please enter valid Data (numbers only)")
    try:
        pred = predict(to_predict_list, to_predict_dict)
        if pred is None:
            raise ValueError("unexpected number of inputs: %d" % len(to_predict_list))
    except Exception:
        # a real server-side problem (e.g. model failed to load) - show it in the
        # terminal instead of blaming the user's input
        traceback.print_exc()
        return render_template("home.html", message="Prediction failed on the server. Check the terminal for details.")

    names = {8: 'Diabetes', 13: 'Heart Disease', 18: 'Kidney Disease', 10: 'Liver Disease'}
    disease_name = names.get(len(to_predict_list), 'this condition')
    return render_template('predict.html', pred = pred, disease_name = disease_name)

@app.route("/malariapredict", methods = ['POST', 'GET'])
def malariapredictPage():
    if request.method != 'POST':
        return redirect('/malaria')
    pred = None
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('RGB')
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,3))
                img = img.astype(np.float64)
                model = get_keras_model("malaria.keras")
                pred = np.argmax(model.predict(img, verbose=0)[0])
        except Exception:
            traceback.print_exc()
            message = "Please upload a valid Image"
            return render_template('malaria.html', message = message)
    if pred is None:
        return render_template('malaria.html', message = "Please upload an Image")
    return render_template('malaria_predict.html', pred = pred)

@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if request.method != 'POST':
        return redirect('/pneumonia')
    pred = None
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,1))
                img = img / 255.0
                model = get_keras_model("pneumonia.keras")
                pred = np.argmax(model.predict(img, verbose=0)[0])
        except Exception:
            traceback.print_exc()
            message = "Please upload a valid Image"
            return render_template('pneumonia.html', message = message)
    if pred is None:
        return render_template('pneumonia.html', message = "Please upload an Image")
    return render_template('pneumonia_predict.html', pred = pred)

if __name__ == '__main__':
	app.run(debug = True)