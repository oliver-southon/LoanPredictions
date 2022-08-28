from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField, DecimalField, DecimalRangeField, validators

class LoanForm(FlaskForm):
    attr1 = DecimalRangeField("Attr1", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=5)
    attr2 = DecimalRangeField("Attr2", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=5)
    attr3 = DecimalRangeField("Attr3", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=5)
    attr4 = DecimalRangeField("Attr4", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=5)
    attr5 = DecimalRangeField("Attr5", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=5)
    attr6 = DecimalRangeField("Attr6", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=5)
    attr7 = DecimalRangeField("Attr7", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=5)
    attr8 = DecimalRangeField("Attr8", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=5)
    attr9 = DecimalRangeField("Attr9", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=5)
    attr10 = DecimalRangeField("Attr10", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=5)
    attr11 = DecimalRangeField("Attr11", [validators.DataRequired()], render_kw={'style': 'width: 90%'}, default=5)
    submit = SubmitField('Submit')
