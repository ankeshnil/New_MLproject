 #   This code same as app.py file. it created because of aws deployment 

from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)  # this is our entry point for .ebextenstion (aws binstalk deployment)

app = application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':   # here we show the from where we put new data
        return render_template('home.html')
    
    else:         # here in post we do all thing like capture the data, scaling data and make prediction
        data = CustomData(
            # this line is just organizing user input into a clean format before sending it to the prediction pipeline.
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race/ethnicity'),
            parental_level_of_education=request.form.get('parental level of education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test preparation course'),
            reading_score=float(request.form.get('reading score')),
            writing_score=float(request.form.get('writing score'))
        )
        
        pred_df = data.get_data_as_dataframe()  # here i convert the avpbe dada in data frame
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)  # predict() function write in predict_pipeline.py fiile
        
        return render_template('home.html', results = results[0]) 
    

if __name__ == "__main__":
    app.run(host= "0.0.0.0")
    
