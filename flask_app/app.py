from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from models.forms import LoanForm
import random
from secret_key import sk

app = Flask(__name__)
app.config['SECRET_KEY'] = sk
Bootstrap(app)


@app.route('/')
def index():
    form = LoanForm()
    result = None
    if form.validate_on_submit():
        result = random.choice(["yes", "no"])
    return render_template('index.html', form=form, result=result)

if __name__ == "__main__":
    app.run(debug=True)