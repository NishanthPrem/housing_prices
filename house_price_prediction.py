#%% Importing the Libraries

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

#%% Loading the train set

df_train = pd.read_csv('train.csv')
X_train = df_train.drop(columns=['SalePrice'])
y_train = df_train['SalePrice']

#%% Loading the Test set

df_test = pd.read_csv('test.csv')
X_test= df_test.copy()

#%% Preprocessing

numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

#%% Handle missing data

num_imputer = SimpleImputer(strategy='median')
X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

#%% Feature scaling

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols) 
    ],
    remainder='passthrough'  
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

#%% Applying Random Forest algorithm

regressor = RandomForestRegressor(random_state=1)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#%% Generating the submission file
submission = pd.DataFrame({
    'Id': df_test['Id'],
    'SalePrice': y_pred
})

submission.to_csv('submission.csv', index=False)