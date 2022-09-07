from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from models.forms import LoanForm
from models.funcs import *
from modelling.modelling import make_models, make_preds
import random
from secret_key import sk

import re

app = Flask(__name__)
app.config['SECRET_KEY'] = sk
Bootstrap(app)
lr, rfc, dtc = make_models()

@app.route('/', methods=["POST", "GET"])
def index():
    form = LoanForm()
    results = None
    plot_url = None
    submitted = False

    if form.validate_on_submit():
        submitted = True
        pred = [list(float(i) for i in list(form.data.values())[:-2])]
        results = make_preds(lr, rfc, dtc, pred)
        plot_url = example_run()
    return render_template('index.html', form=form, plot_url=plot_url, submitted=submitted, results=results)

@app.route('/explanations', methods=["POST", "GET"])
def explanations():
    return render_template('explanations.html')

@app.route('/modelling')
def modelling():
    return render_template('modelling.html')

if __name__ == "__main__":
    app.run(debug=True)