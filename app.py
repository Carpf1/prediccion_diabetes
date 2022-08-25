from flask import Flask,render_template,request,redirect,jsonify
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

with open('c_diabetes.pkl','rb') as file:
    classifier=pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html',home_active='active')


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        num_embarazos=request.form.get('Pregnancies')
        Azucar=request.form.get('Glucose')
        Presion_diastolica=request.form.get('BloodPressure')
        Grasa_Triceps=request.form.get('SkinThickness')
        nivel_Insulina=request.form.get('InsulinLevel')
        Grasa_corporal=request.form.get('BodyMassIndex')
        Antecedentes_diabeticos=request.form.get('DiabetesPedigreeFunction')
        Edad=request.form.get('Age')
        
        data=np.array([[int(num_embarazos),int(Azucar),int(Presion_diastolica),int(Grasa_Triceps),int(nivel_Insulina),float(Grasa_corporal),float(Antecedentes_diabeticos),int(Edad)]])
        prediction=classifier.predict(data)

        context={
            'num_embarazos':num_embarazos,
            'Azucar':Azucar,
            'Presion_diastolica':Presion_diastolica,
            'Grasa_Triceps':Grasa_Triceps,
            'nivel_Insulina':nivel_Insulina,
            'Grasa_corporal':Grasa_corporal,
            'Antecedentes_diabeticos':Antecedentes_diabeticos,
            'Edad':Edad,
            'pred':prediction
        }        

        return render_template('prediction.html',context=context,pred_active='active')

    elif request.method=='GET':
        return redirect('/')

@app.route('/api')
def api():
    return render_template('api.html',api_active='active')

@app.route('/api/<int:num_embarazos>/<int:Azucar>/<int:Presion_diastolica>/<int:Grasa_Triceps>/<int:nivel_Insulina>/<float:Grasa_corporal>/<float:Antecedentes_diabeticos>/<int:Edad>')
def api_pred(num_embarazos,Azucar,Presion_diastolica,Grasa_Triceps,nivel_Insulina,Grasa_corporal,Antecedentes_diabeticos,Edad):
    data=np.array([[int(num_embarazos),int(Azucar),int(Presion_diastolica),int(Grasa_Triceps),int(nivel_Insulina),float(Grasa_corporal),float(Antecedentes_diabeticos),int(Edad)]])
    prediction=classifier.predict(data)

    result={
            'num_embarazos':num_embarazos,
            'Azucar':Azucar,
            'Presion_diastolica':Presion_diastolica,
            'Grasa_Triceps':Grasa_Triceps,
            'nivel_Insulina':nivel_Insulina,
            'Grasa_corporal':Grasa_corporal,
            'Antecedentes_diabeticos':Antecedentes_diabeticos,
            'Edad':Edad,
            'pred':bool(prediction[0])
        }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=4000,debug=True)       