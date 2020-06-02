from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

def train_model(X,y):
    model = linear_model.LogisticRegression(max_iter=10000)
    y_pred = cross_val_predict(model, X, y, cv=5)
    score = accuracy_score(y,y_pred)
    print("Accuracy = {0:.4f}".format(score))
    model.fit(X, y)
    return model, score
