#%% Importing the Libraries

import pandas as pd

#%% Loading the train set

df = pd.read_csv('train.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#%% Loading the Test

df_test = pd.read_csv('test.csv')
X_test = df.iloc[:, :-1].values
y_test = df.iloc[:, -1].values

#%% Skipping the feature scaling and missing values



#%% Applying Random Forest algorithm

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(random_state=1)
regressor.fit(X, y)

y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score

mae = mean_absolute_error(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
