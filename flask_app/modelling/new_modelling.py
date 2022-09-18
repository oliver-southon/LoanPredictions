import shap
import lime
import lime.lime_tabular

from html2image import Html2Image

import pandas as pd
import numpy as np
import random
import seaborn as sns
from matplotlib import pyplot as plt
import io
import base64
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, make_scorer, accuracy_score, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV


import xgboost as xgb
from xgboost import plot_importance

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import linear_model

from sklearn.svm import SVC

from aix360.algorithms.protodash import ProtodashExplainer

import warnings
warnings.simplefilter(action ='ignore')

def make_train():
    heloc = pd.read_csv("data/heloc_dataset.csv")
    df= heloc.copy()
    df.replace({"Bad":1, "Good":0}, inplace=True)

    X = df.drop('RiskPerformance', axis=1)
    y = df['RiskPerformance']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=8)
    return X_train, y_train

def make_models(X_train, y_train):
    dt_model = DecisionTreeClassifier(max_leaf_nodes = 12, random_state=0, criterion='entropy')
    dt_model = dt_model.fit(X_train, y_train)

    svc_model = SVC(kernel='rbf', probability=True)
    svc_model= svc_model.fit(X_train, y_train)

    xgb_model = xgb.XGBClassifier(objective="reg:logistic", 
                                seed=70, 
                                col_sample_bytree=0.4,
                                learning_rate= 0.01, 
                                max_depth= 5,
                                n_estimators=500,
                                alpha=10)

    xgb_model = xgb_model.fit(X_train.values, y_train)

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, 
                                                   feature_names=X_train.columns, 
                                                   verbose=True, 
                                                   class_names=['Approved', 'Denied'],
                                                   mode='classification')

    return xgb_model, dt_model, svc_model, explainer

def make_preds(xgb, dt, svm, pred):
    results = {}
    results["xgb"] = xgb.predict(pred)[0]
    results["dt"] = dt.predict(pred)[0]
    results["svm"] = svm.predict(pred)[0]
    new_results = _translate_results(results)
    return new_results

def _translate_results(results):
    for key, val in results.items():
        if val == 0:
            results[key] = True
        else:
            results[key] = False

    total_good = sum(1 for v in results.values() if v == True)
    if total_good == 3:
        outcome = "yes"
    elif total_good == 2:
        outcome = "maybe"
    else:
        outcome = "no"
    results["outcome"] = outcome
    results["total_approved"] = total_good
    return results

def local_explanations(explainer, xgb, dt, svc, pred):
    # plot_urls = []
    for model in [xgb, dt, svc]:
        exp = _local_explanation(explainer, model, pred)
        # img = io.BytesIO()
        model_name = f'{model}'.split('=')[0]
        exp.save_to_file(f'temp/{model_name}.html')

        hti = Html2Image(size=(500,500), output_path='static/')

        hti.screenshot(html_file=f'temp/{model_name}.html', save_as=f'{model_name}.png')
        # g.savefig(img, format='png')
        # plt.close()
        # img.seek(0)
        # plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        # plot_urls.append(plot_url)
    # return plot_urls

def _local_explanation(explainer, model, pred):
    pred_series = pd.Series(np.asarray(pred[0]), index=['ExternalRiskEstimate', 'MSinceOldestTradeOpen',
       'MSinceMostRecentTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades',
       'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec',
       'PercentTradesNeverDelq', 'MSinceMostRecentDelq',
       'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades',
       'NumTradesOpeninLast12M', 'PercentInstallTrades',
       'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days',
       'NetFractionRevolvingBurden', 'NetFractionInstallBurden',
       'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance',
       'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance'])
    explanation = explainer.explain_instance(pred_series, model.predict_proba)
    return explanation

def display_similar_clients(xgb, dt, svc, user_input, X_train, y_train):
    for ml_model in [xgb, dt, svc]:
        model_name = f'{ml_model}'.split('=')[0]
        g = _display_similar_clients(ml_model, user_input, X_train, y_train)
        print(g)
        g.to_html(f'temp/{model_name}_similar.html')

        hti = Html2Image(size=(800,600), output_path='static/')
        hti.screenshot(html_file=f'temp/{model_name}_similar.html', save_as=f'{model_name}_similar.png')

def _display_similar_clients(ml_model, user_input, X_train, y_train): 
  
    '''
    take in user's input (as a list) for prediction; 
    
    '''
    user_input = user_input[0]
    print("PRINTING USER INPUT FOR FUNC", user_input)
    # sourcing from X_train and y_train, put all 'good' samples together, all 'bad' samples together
    existing_client_profiles = pd.concat([y_train, X_train], axis=1)

    good_client_con = existing_client_profiles.copy()['RiskPerformance'] == 0.0
    good_client_profiles = X_train.loc[good_client_con, :]

    bad_client_con = existing_client_profiles.copy()['RiskPerformance'] == 1.0
    bad_client_profiles = X_train.loc[bad_client_con, :]

    # make prediction on user input

    index = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen',
        'MSinceMostRecentTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades',
        'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec',
        'PercentTradesNeverDelq', 'MSinceMostRecentDelq',
        'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades',
        'NumTradesOpeninLast12M', 'PercentInstallTrades',
        'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days',
        'NetFractionRevolvingBurden', 'NetFractionInstallBurden',
        'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance',
        'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']
    pred_dummy = ml_model.predict(np.array(user_input).reshape(1,-1))[0]
    dummy_applicant = pd.concat([pd.Series(user_input, index=index), pd.Series({'RiskPerformance':pred_dummy})], axis=0)
    dummy_applicant = pd.DataFrame(dummy_applicant)
    dummy_applicant = dummy_applicant.T.replace({'RiskPerformance':{0.0:'pred: approve', 1.0:'pred: decline'}}).T
    dummy_applicant = dummy_applicant.rename(columns={0:'User Input'})


    # implement protodash
    ## if prediction is 1.0, stack the dummy applicant with exsiting 'bad' applicants
    ## if prediction is 0.0, stack the dummy applicant with existing 'good' applicants

    if pred_dummy == 1.0:

        explainer = ProtodashExplainer()
        W, S, SetValues = explainer.explain(np.array(user_input).reshape(1,-1), bad_client_profiles.values, m=3)

        similar_clients = bad_client_profiles.iloc[S, :]
        similar_clients = similar_clients.reset_index(drop=True)
        y_bad = y_train[bad_client_con]
        similar_clients['RiskPerformance'] = y_bad.iloc[S].values
        similar_clients = similar_clients.replace({'RiskPerformance':{0.0:'paid', 1.0:'default'}})
        similar_clients = similar_clients.rename(index={0:'Existing Client 1', 1:'Existing Client 2', 2:'Existing Client 3'})
        similar_clients = similar_clients.T
        similar_clients = pd.concat([dummy_applicant, similar_clients], axis=1)

    else:

        explainer = ProtodashExplainer()
        W, S, SetValues = explainer.explain(np.array(user_input).reshape(1,-1), good_client_profiles.values, m=3)

        similar_clients = good_client_profiles.iloc[S, :]
        similar_clients = similar_clients.reset_index(drop=True)
        y_good = y_train[good_client_con]
        similar_clients['RiskPerformance'] = y_good.iloc[S].values
        similar_clients = similar_clients.replace({'RiskPerformance':{0.0:'paid', 1.0:'default'}})
        similar_clients = similar_clients.rename(index={0:'Existing Client 1', 1:'Existing Client 2', 2:'Existing Client 3'})
        similar_clients = similar_clients.T
        similar_clients = pd.concat([dummy_applicant, similar_clients], axis=1)


    slice_idx = pd.IndexSlice
    slice_ = slice_idx[slice_idx[:'PercentTradesWBalance'], slice_idx['Existing Client 1':'Existing Client 3']]
    return similar_clients.style.apply(lambda x: (abs((x+1e-10)/(similar_clients['User Input'][:-1]+1e-10) - 1) <= 0.2).map({True:'background-color: yellow', False: ''}), subset=slice_).set_precision(0)



