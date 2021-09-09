"""
Purpose

Stores dictionaries containing default sets of models to try fitting over

"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
#from xgboost import XGBClassifier
from sklearn.svm import SVC


MODELS = {
    "Classification": {
        "LR": {
            "Classifier": LogisticRegression()
        },
        "LDA": {
            "Classifier": LinearDiscriminantAnalysis()
        },
        "KNN": {
            "Classifier": KNeighborsClassifier()
        },
        "CART": {
            "Classifier": DecisionTreeClassifier()
        },
        "NB": {
            "Classifier": GaussianNB()
        },
        "RF": {
            "Classifier": RandomForestClassifier(),
            "Parameters": {
                "n_estimators": [16, 32]
            }
        },
        "ADA": {
            "Classifier": AdaBoostClassifier(),
            "Parameters": {
                "n_estimators": [16, 32]
            }
        },
#        "XGB": {
#            "Classifier": XGBClassifier(verbosity=0)
#        },
        "GB": {
            "Classifier": GradientBoostingClassifier(),
            "Parameters": {
                "n_estimators": [16, 32],
                "learning_rate": [0.8, 1.0]
            }
        },
        "SVM": {
            "Classifier": SVC(),
            "Parameters": {
                "kernel": ["rbf"],
                "C": [1, 10],
                "gamma": [0.001, 0.0001]
            }
        }
    }
}
