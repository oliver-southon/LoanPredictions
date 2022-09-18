from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from models.forms import LoanForm, DummyForm
from models.funcs import *
from modelling.new_modelling import display_similar_clients, make_models, make_preds, local_explanations, make_train
import random
from secret_key import sk

import re

app = Flask(__name__)
app.config['SECRET_KEY'] = sk
Bootstrap(app)
x_train, y_train = make_train()
xgb_model, dt_model, svc_model, explainer = make_models(x_train, y_train)                                 

@app.route('/', methods=["POST", "GET"])
def index():
    form = LoanForm()
    results = None
    plot_url = None
    submitted = False

    if form.validate_on_submit():
        submitted = True
        pred = [list(float(i) for i in list(form.data.values())[:-2])]
        print(pred)
        results = make_preds(xgb_model, dt_model, svc_model, pred)

        local_explanations(explainer, xgb_model, dt_model, svc_model, pred)

        display_similar_clients(xgb_model, dt_model, svc_model, pred, x_train, y_train)

    return render_template('index.html', form=form, submitted=submitted, results=results)

@app.route('/explanations', methods=["POST", "GET"])
def explanations():
    return render_template('explanations.html')

@app.route('/features', methods=["POST", "GET"])
def features():
    return render_template('features.html')

@app.route('/modelling')
def modelling():
    return render_template('modelling.html')

if __name__ == "__main__":
    app.run(debug=True)