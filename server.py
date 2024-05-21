#flask
from flask import Flask, render_template, request, redirect, url_for, session

#modules
from modules.config import app, db
from modules.tpot_pipeline import tpot_pipeline

#excepciones
from sqlalchemy.orm.exc import NoResultFound 

#otros
import pandas as pd
import numpy as np

pipeline = tpot_pipeline()
input_data = pd.DataFrame([input_features])
input_data = np.array(input_features).reshape(1, -1)
prediction = pipeline.predict(input_data)
print(f"Predicted output: {prediction[0]}")

with app.app_context():
    db.create_all()
    
@app.route("/", methods=['GET', 'POST'])
def raiz():  
    return render_template("main.html")