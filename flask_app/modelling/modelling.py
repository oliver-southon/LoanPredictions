from json import load
import sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

def make_models():
    df = pd.read_csv("data/winequality_white.csv", header=0, sep=';')
    df["quality_category"] = pd.cut(df["quality"], bins=[0, 5, 9], labels=["poor", "good"])
    df.drop(["quality"], axis=1, inplace=True)


    X = df[df.columns[:-1]]
    y = df["quality_category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    normalise = MinMaxScaler()
    fit = normalise.fit(X_train)
    X_train = fit.transform(X_train)
    X_test = fit.transform(X_test)

    lr = LogisticRegression(random_state=1)
    lr.fit(X_train, y_train)

    rfc = RandomForestClassifier(random_state=1)
    rfc.fit(X_train, y_train)

    dtc = DecisionTreeClassifier(random_state=1)
    dtc.fit(X_train, y_train)

    return lr, rfc, dtc

def make_preds(lr, rfc, dtc, pred):
    results = {}
    results["lr"] = lr.predict(pred)[0]
    results["rfc"] = rfc.predict(pred)[0]
    results["dtc"] = dtc.predict(pred)[0]
    print(results)
    new_results = _translate_results(results)
    print(new_results)
    return new_results

def _translate_results(results):
    for key, val in results.items():
        if val == "good":
            results[key] = True
        else:
            results[key] = False

    total_good = sum(1 for v in results.values() if v == True)
    if total_good >= 2:
        outcome = True
    else:
        outcome = False
    results["outcome"] = outcome
    return results

if __name__ == "__main__":
    lr, rfc, dtc = make_models()

    pred = [[9.0, 0.59, 0.83, 33.2, 0.18, 145.5, 224.5, 1.01, 3.27, 0.65, 11.1]]

    make_preds(lr, rfc, dtc, pred)
