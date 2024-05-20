#flask
from flask import Flask, render_template, request, redirect, url_for, session

#modules
from modules.config import app, db
from modules.model import prediction

#excepciones
from sqlalchemy.orm.exc import NoResultFound 

with app.app_context():
    db.create_all()
    
@app.route("/", methods=['GET', 'POST'])
def raiz():  