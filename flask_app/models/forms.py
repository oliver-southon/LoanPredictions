from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField, DecimalField, DecimalRangeField, validators

class LoanForm(FlaskForm):
    f_acid = DecimalRangeField("Fixed Acidity", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=9)
    v_acid = DecimalRangeField("Volatile Acidity", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=0.59)
    c_acid = DecimalRangeField("Citric Acidity", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=0.83)
    r_sugar = DecimalRangeField("Residual Sugar", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=33.2)
    chlorides = DecimalRangeField("Volatile Acidity", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=0.178)
    fs_dioxide = DecimalRangeField("Free Sulfur Dioxide", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=145.5)
    ts_dioxide = DecimalRangeField("Total Sulfur Dioxide", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=224.5)
    density = DecimalRangeField("density", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=1.01305)
    ph = DecimalRangeField("pH", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=3.27)
    sulphates = DecimalRangeField("Sulphates", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=0.65)
    alcohol = DecimalRangeField("Alcohol", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=11.1)
    submit = SubmitField('Submit')
