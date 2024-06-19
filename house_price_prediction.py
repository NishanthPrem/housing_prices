#%% Importing the Libraries

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

#%% Loading the train set

df_train = pd.read_csv('train.csv')
benchmark_features = ['YearBuilt', 'MoSold', 'LotArea', 'BedroomAbvGr']

X_train = df_train[benchmark_features]
y_train = df_train['SalePrice']


#%% Loading the Test set

df_test = pd.read_csv('test.csv')
X_test = df_test[benchmark_features]


#%% Handle missing data

num_imputer = SimpleImputer(strategy='median')
X_train = num_imputer.fit_transform(X_train)
X_test = num_imputer.transform(X_test)


#%% Feature scaling

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), slice(0, len(benchmark_features))),
    ],
    remainder='passthrough' )

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

#%% Applying Random Forest algorithm

regressor = RandomForestRegressor(random_state=1)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#%% Submission File

submission = pd.DataFrame({
    'Id': df_test['Id'],
    'SalePrice': y_pred
})

submission.to_csv('submission.csv', index=False)