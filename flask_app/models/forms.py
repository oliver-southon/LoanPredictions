from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField, DecimalField, DecimalRangeField, validators
import pandas as pd

fd = pd.read_csv("data/feature_details.csv", index_col="feature")

feature_names = fd.index.values # Get list of feature names

class LoanForm(FlaskForm):
    fields = []
    for i in range(0,23):
        name = str(feature_names[i])
        fields.append(DecimalRangeField(name, [validators.DataRequired(), validators.NumberRange(min=fd.at[name, "min"], max=fd.at[name, "max"])], render_kw={'style': 'width: 90%', 'id': f'{i}'}, default=round(fd.at[name, "mean"],2))) 

    f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23 = fields
    submit = SubmitField('Submit')

class DummyForm(FlaskForm):
    field1 = DecimalRangeField('field1', [validators.DataRequired(), validators.NumberRange(min=0, max=5)], render_kw={'style': 'width: 90%'}, default=0) 
    field2 = DecimalRangeField('field2', [validators.DataRequired(), validators.NumberRange(min=0, max=5)], render_kw={'style': 'width: 90%'}, default=0) 
    submit = SubmitField('Submit')
