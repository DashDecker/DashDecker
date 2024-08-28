# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:13:47 2023

@author: dashi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from hyperopt import fmin, tpe, hp
import gridstatus
import joblib


plt.style.use('fivethirtyeight')
caiso = gridstatus.CAISO()

first_df = pd.read_csv('C:/Users/dashi/OneDrive/Desktop/CAISO Projects/Data/LMP Data/2022_Lmp_Data.csv')
second_df = pd.read_csv('C:/Users/dashi/OneDrive/Desktop/CAISO Projects/Data/LMP Data/2023_Lmp_Data.csv')
main_df = pd.concat([first_df, second_df])
main_df = main_df.set_index('Time')
main_df.index = pd.to_datetime(main_df.index)
main_df.head()
main_df.shape


# Overall LMP Data Plot

color_pal = sns.color_palette()
main_df.plot(style='.', figsize=(15, 5), color=color_pal[0], title='CAISO LMP Data Since 2022')

train = main_df.loc[main_df.index < '01-01-2023']
test = main_df.loc[main_df.index >= '01-01-2023']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('01-01-2023', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()


def create_features(df):
    """
    Create time series features based on time series index.

    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['season'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    # df['dayofmonth'] = df.index.day
    # df['weekofyear'] = df.index.isocalendar().week
    return df

main_df = create_features(main_df)
main_df.head()


def add_lags(df):
    target_map = main_df['LMP'].to_dict()
    df['1 Year Lag'] = (df.index - pd.Timedelta('364 days')).map(target_map) 
    return df

main_df = add_lags(main_df)
main_df.head()
main_df.tail()
main_df['1 Year Lag']



# Initial TimeSeriesSplit

tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
main_df = main_df.sort_index()

fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)
fold = 0

for train_index, val_index in tss.split(main_df):
    train = main_df.iloc[train_index]
    test = main_df.iloc[val_index]
    
    train['LMP'].plot(ax=axs[fold], label='Training Set',
                       title=f'Data Train/Test Split Fold {fold}')
    test['LMP'].plot(ax=axs[fold], label='Test Set')
    axs[fold].axvline(test.index.min(), color='black', ls='--')
    fold += 1
    
plt.show()


# MODEL CREATION

tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
main_df = main_df.sort_index()

fold = 0
preds = []
scores = []

for train_index, val_index in tss.split(main_df):
    train = main_df.iloc[train_index]
    test = main_df.iloc[val_index]
    
    train = create_features(train)
    test = create_features(test)
    
    FEATURES = ['hour', 'dayofweek', 'season', 'month', 'year', 'dayofyear',
                '1 Year Lag']
    TARGET = 'LMP'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, 
                           learning_rate=0.008, max_depth=1, reg_alpha=0.43,
                           reg_lambda= 0.66754, colsample_bytree=0.61947,
                           subsample=0.10949, gamma=0.18349)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=20)


    y_pred = reg.predict(X_test)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)
    

print(f'Score across folds {np.mean(scores):0.4f}')
print(f'Fold scores:{scores}')


fi = pd.DataFrame(reg.feature_importances_, index=X_train.columns,
             columns=['Importance'])
fi.sort_values('Importance').plot(kind='barh', title='Feature Importance')


# First Optimization

# Define the objective function to minimize (negative RMSE)
def first_model_objective(parameters):
    n_estimators = 1000
    
    reg = xgb.XGBRegressor(
        n_estimators = n_estimators,
        learning_rate = parameters['learning_rate'],
        max_depth = int(parameters['max_depth']),
        subsample = parameters['subsample'],
        colsample_bytree = parameters['colsample_bytree'],
        gamma = parameters['gamma'],
        reg_alpha = parameters['reg_alpha'],
        reg_lambda = parameters['reg_lambda']
        )

    reg.fit(X_train, y_train, eval_set = [(X_test, y_test)], verbose=0)
    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return -rmse

# Define the search space
param_space = {
    # 'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'subsample': hp.uniform('subsample', 0.1, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
    'gamma': hp.uniform('gamma', 0, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
}

# Perform Bayesian optimization
best = fmin(fn=first_model_objective, space=param_space, algo=tpe.suggest,
            max_evals=100, verbose=1)

# Print the best hyperparameters
print("Best Hyperparameters:", best)




#Forecast on Test Set


test['prediction'] = reg.predict(X_test)
main_df.merge(test[['prediction']], how='left', left_index=True, right_index=True)

ax = main_df[['LMP']].plot(figsize=(15, 5))
main_df['prediction'] = reg.predict(create_features(main_df[FEATURES]))
main_df.loc[main_df.index.isin(test.index), 'prediction'].plot(ax=ax, style='.')
plt.legend(['Real Data', 'Predictions'])
ax.set_title('Raw Data vs. Predictions')
plt.show()


main_df.loc[(main_df.index > '04-10-23') & (main_df.index < '04-17-23')]['LMP'] \
    .plot(figsize=(15, 5), title='One Week Data')
main_df.loc[(main_df.index > '04-10-23') & (main_df.index < '04-17-23')]['prediction'] \
    .plot(style='.')
plt.legend(['Real Data', 'Predictions'])
plt.show()

score = np.sqrt(mean_squared_error(test['LMP'], test['prediction']))
print(f'RMSE Score on test set: {score:0.2f}')


test['error'] = np.abs(test[TARGET] - test['prediction'])
test['date'] = test.index.date
test.groupby('date')['error'].mean().sort_values(ascending=False).head(10)
test.groupby('date')['error'].mean().sort_values(ascending=True).head(10)





main_df = create_features(main_df)

FEATURES = ['hour', 'dayofweek', 'season', 'month', 'year', 'dayofyear',
          '1 Year Lag']
TARGET = 'LMP'

X_all = main_df[FEATURES]
y_all = main_df[TARGET]


reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', objective='reg:squarederror',
                       n_estimators=1000, 
                       max_depth=1, 
                       learning_rate=0.00676, 
                       gamma=0.0027, 
                       reg_alpha=0.33764, 
                       reg_lambda=0.66704, 
                       colsample_bytree=0.98508,
                       subsample=0.76613
                       )
reg.fit(X_all, y_all, eval_set=[(X_all, y_all)], verbose=20)  


# Create future preds DataFrame

main_df.index.max()
future = pd.date_range('2023-11-01', '2023-11-30', freq='1h')
future_df = pd.DataFrame(index=future)
future_df['isFuture'] = True
main_df['isFuture'] = False
predictions_df = pd.concat([main_df, future_df])
predictions_df = create_features(predictions_df)
predictions_df = add_lags(predictions_df)
predictions_df

preds_features = predictions_df.query('isFuture').copy()
preds_features
preds_features.columns
preds_features[['LMP', 'Interval Start', 'Interval End']]

preds_features['Future Predictions'] = reg.predict(preds_features[FEATURES])
preds_features[['Future Predictions', 'hour', 'dayofyear']]

preds_features['Future Predictions'].plot(figsize=(10, 5), color=color_pal[4],
                                             ms=1, lw=1, title='Future Predictions')
plt.show()


joblib.dump(reg, 'LMP_model.joblib')


def future_preds_objective(parameters):
    n_estimators = 1000
    
    reg = xgb.XGBRegressor(
        n_estimators = n_estimators,
        learning_rate = parameters['learning_rate'],
        max_depth = int(parameters['max_depth']),
        subsample = parameters['subsample'],
        colsample_bytree = parameters['colsample_bytree'],
        gamma = parameters['gamma'],
        reg_alpha = parameters['reg_alpha'],
        reg_lambda = parameters['reg_lambda']
        )

    reg.fit(X_all, y_all, eval_set = [(X_all, y_all)], verbose=0)

    y_pred = reg.predict(X_all)
    rmse = np.sqrt(mean_squared_error(y_all, y_pred))
    
    return -rmse

# Define the search space
param_space = {
    # 'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'subsample': hp.uniform('subsample', 0.1, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
    'gamma': hp.uniform('gamma', 0, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
}

# Perform Bayesian optimization
best = fmin(fn=future_preds_objective, space=param_space, algo=tpe.suggest, 
            max_evals=50, verbose=1)

# Print the best hyperparameters
print("Best Hyperparameters:", best)


