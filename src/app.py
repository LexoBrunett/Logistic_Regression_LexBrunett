from utils import db_connect
engine = db_connect()

# your code here

import pandas as pd

bank_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv", sep = ";")
bank_data = bank_data.drop_duplicates().reset_index(drop = True)
bank_data.head(20)

#Null information
bank_data.isnull().sum()

# Min-Max scaler

from sklearn.preprocessing import MinMaxScaler

bank_data["job_n"] = pd.factorize(bank_data["job"])[0]
bank_data["marital_n"] = pd.factorize(bank_data["marital"])[0]
bank_data["education_n"] = pd.factorize(bank_data["education"])[0]
bank_data["default_n"] = pd.factorize(bank_data["default"])[0]
bank_data["housing_n"] = pd.factorize(bank_data["housing"])[0]
bank_data["loan_n"] = pd.factorize(bank_data["loan"])[0]
bank_data["contact_n"] = pd.factorize(bank_data["contact"])[0]
bank_data["month_n"] = pd.factorize(bank_data["month"])[0]
bank_data["day_of_week_n"] = pd.factorize(bank_data["day_of_week"])[0]
bank_data["poutcome_n"] = pd.factorize(bank_data["poutcome"])[0]
bank_data["y_n"] = pd.factorize(bank_data["y"])[0]
num_variables = ["job_n", "marital_n", "education_n", "default_n", "housing_n", "loan_n", "contact_n", "month_n", "day_of_week_n", "poutcome_n",
                 "age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y_n"]

scaler = MinMaxScaler()
scal_features = scaler.fit_transform(bank_data[num_variables])
bank_data_scal = pd.DataFrame(scal_features, index = bank_data.index, columns = num_variables)
bank_data_scal.head()

# Feature selection

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

X = bank_data_scal.drop("y_n", axis = 1)
y = bank_data_scal["y_n"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

selection_model = SelectKBest(chi2, k = 5)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

X_train_sel.head()

X_train_sel["y_n"] = list(y_train)
X_test_sel["y_n"] = list(y_test)
X_train_sel.to_csv("../data/processed/clean_train.csv", index = False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index = False)

train_data = pd.read_csv("../data/processed/clean_train.csv")
test_data = pd.read_csv("../data/processed/clean_test.csv")

train_data.head()

X_train = train_data.drop(["y_n"], axis = 1)
y_train = train_data["y_n"]
X_test = test_data.drop(["y_n"], axis = 1)
y_test = test_data["y_n"]

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

#optimization

from sklearn.model_selection import GridSearchCV

hyperparams = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

grid = GridSearchCV(model, hyperparams, scoring = "accuracy", cv = 10)
grid

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grid.fit(X_train, y_train)

print(f"Best hyperparameters: {grid.best_params_}")

model = LogisticRegression(C = 0.1, penalty = "l2", solver = "liblinear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

accuracy_score(y_test, y_pred)

