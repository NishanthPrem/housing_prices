#%% Importing the Libraries

import pandas as pd
import numpy as np

#%% Loading the train set

df = pd.read_csv('train.csv')
X_train = df.iloc[:, :-1].values
y_train = df.iloc[:, -1].values

#%% Loading the Test set

df_test = pd.read_csv('test.csv')
X_test = df.iloc[:, :-1].values
y_test = df.iloc[:, -1].values

#%% Filling the missing values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_train[:,3:4])
X_train[:,3:4] = imputer.transform(X_train[:,3:4])

imputer.fit(X_train[:,6:7])
X_train[:,6:7] = imputer.transform(X_train[:,6:7])

#%% Feature scaling

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X_train[['MSZoning']] = np.array(ct.fit_transform(X_train))

#%% Applying Random Forest algorithm

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(random_state=1)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score

mae = mean_absolute_error(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
