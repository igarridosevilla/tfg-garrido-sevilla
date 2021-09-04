# -*- coding: utf-8 -*-

import pickle
import random
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

multinomial = pickle.load(open('static/models/model1.pkl', 'rb'))
linearsvc = pickle.load(open('static/models/model2.pkl', 'rb'))
randomforest = pickle.load(open('static/models/model2.pkl', 'rb'))
perceptron = pickle.load(open('static/models/model4.pkl', 'rb'))

predictions = 0

def prediction(text, model, patient):
	this_prediction = model.predict(patient)[0]
	global predictions
	predictions += this_prediction
	return f'{text} { "PRESENTA RIESGO" if (this_prediction == 1) else "NO PRESENTA RIESGO" } \n'


min_age = 20
max_age = 45
probability_man = 0.5
min_length = 15
max_length = 30
just_men = ["601"]
just_women = ["112,1"]

cie_data = pd.read_excel("static/data/02_CIE_tokenized_info.xlsx")[["CIE","TOKENIZED","RISKWORDS","CONDICIÓN MÉDICA"]]
cie_data['CIE'] = cie_data['CIE'].apply(str)
cie_data.fillna('[]', inplace=True)
cie_data['TOKENIZED'] = cie_data['TOKENIZED'].apply(eval)
cie_data['RISKWORDS'] = cie_data['RISKWORDS'].apply(eval)

def delete_risk_tokenized(df):
    return [w for w in df['TOKENIZED'] if w not in df['RISKWORDS']]

cie_data['NONRISKWORDS'] = cie_data.apply(delete_risk_tokenized, axis = 1)

def create_virtual_patient(cie, risk):
    return {
        "CIE": cie,
        "AGE": random.randint(min_age, max_age),
        "SEX": get_sex(cie),
        "RISK": risk,
        "TEXT": get_new_text(cie, risk),
    }

def get_sex(cie):
    if cie in just_women:
        return 'M'
    elif cie in just_men:
        return 'H'
    return random.choices(['M', 'H'], weights=[probability_man, 1 - probability_man], k=1)[0]
        

#cie: str; risk: int(0,1)
def get_new_text(cie, risk):
    ordinary_words = cie_data.loc[cie_data['CIE'] == cie]['NONRISKWORDS'].iloc[0]
    risk_words = cie_data.loc[cie_data['CIE'] == cie]['RISKWORDS'].iloc[0]
    
    text_length = random.randint(min_length, max_length)
    number_risk_words = random.randint(risk, int(text_length/2) * risk) # admite repetición, no necesario min

    text = random.choices(risk_words, k = number_risk_words)
    text += random.choices(ordinary_words, k = text_length - number_risk_words)
    
    random.shuffle(text)
    return " ".join(map(str, text))

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/predict', methods =['POST'])
def predict():
	patient_to_classify = request.form.getlist('observations')

	result = prediction('Multinomial:   ', multinomial, patient_to_classify)
	result += prediction('Linear SVC:    ', linearsvc, patient_to_classify)
	result += prediction('Random Forest: ', randomforest, patient_to_classify)
	result += prediction('Perceptron:    ', perceptron, patient_to_classify)
	global predictions
	result += f'Ensemble:       { "PRESENTA RIESGO" if (predictions > 2) else "NO PRESENTA RIESGO" } \n'
	predictions = 0
	return render_template('index.html',
		result = result)

@app.route('/generate', methods =['GET'])
def generate():
	cie = request.args.get("cielist")
	risk = 1 if request.args.get("risk_option") == 'Riesgo Vital' else 0

	return render_template('index.html',
		patient = [str(cie_data.loc[cie_data['CIE'] == cie]['CONDICIÓN MÉDICA'].iloc[0]), 
			str(create_virtual_patient(cie, risk)).replace(', ', '\n').replace('{', '').replace('\'', '').replace('}', '')])

if __name__ == '__main__':
   app.run()
    
    