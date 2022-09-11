from flask import Flask, render_template,request
import pickle
import numpy as np
import sklearn

#create the object of class
model=pickle.load(open('modelDecision.pkl','rb'))
app = Flask(__name__)

# creating the default route
@app.route('/')
#creating the index function
def index():
    return render_template('index.html')


@app.route('/prediction',methods=['POST'])
def prediction():
    WeightInPound=request.form.get("WeightInPound")
    BMI=request.form.get("BMI")
    #prediction
    result=model.predict([[WeightInPound,BMI]])
    print("result data:",result)

    return render_template('index.html',result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
