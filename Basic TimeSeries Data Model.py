# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:45:53 2023

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
# from sklearn.metrics import precision_score

plt.style.use('fivethirtyeight')
caiso = gridstatus.CAISO()

df = pd.read_csv('C:/Users/dashi/OneDrive/Desktop/CAISO Projects/Data/Load Data/Load_Data.csv')
df = df.set_index('Time')
df.index = pd.to_datetime(df.index)
df.head()
df.shape


df['Load'].plot(kind='hist', bins=500)
df.query('Load > 53000').plot(figsize=(15, 5), style='.')
df.query('Load > 53000').copy()

load_threshold = 53000
df = df[df['Load'] <= load_threshold]


# Overall Load Data Plot

color_pal = sns.color_palette()
df.plot(style='.', figsize=(15, 5), color=color_pal[0], title='CAISO Load Data Since 2021')


train = df.loc[df.index < '01-01-2023']
test = df.loc[df.index >= '01-01-2023']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('01-01-2023', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()

df.loc[(df.index > '04-10-2022') & (df.index < '04-17-2022')].plot()


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
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)
df.head()


def add_lags(df):
    target_map = df['Load'].to_dict()
    df['1 Year Lag'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['2 Year Lag'] = (df.index - pd.Timedelta('728 days')).map(target_map) 
    return df

df = add_lags(df)
df.head()
df.tail()
df[['1 Year Lag', '2 Year Lag']]


#Visualize feature to target relationship

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='hour', y='Load')
ax.set_title('Hourly Load')
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='month', y='Load', palette='Reds')
ax.set_title('Monthly Load')
plt.show()


# Initial TimeSeriesSplit

tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
df = df.sort_index()

fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)
fold = 0

for train_index, val_index in tss.split(df):
    train = df.iloc[train_index]
    test = df.iloc[val_index]
    
    train['Load'].plot(ax=axs[fold], label='Training Set',
                       title=f'Data Train/Test Split Fold {fold}')
    test['Load'].plot(ax=axs[fold], label='Test Set')
    axs[fold].axvline(test.index.min(), color='black', ls='--')
    fold += 1
    
plt.show()


# MODEL CREATION

tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
df = df.sort_index()

fold = 0
preds = []
scores = []

for train_index, val_index in tss.split(df):
    train = df.iloc[train_index]
    test = df.iloc[val_index]
    
    train = create_features(train)
    test = create_features(test)
    
    FEATURES = ['hour', 'dayofweek', 'season', 'month', 'year', 'dayofyear',
                '1 Year Lag', '2 Year Lag']
    TARGET = 'Load'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, 
                           learning_rate=0.001)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=50)


    y_pred = reg.predict(X_test)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)
    

print(f'Score across folds {np.mean(scores):0.4f}')
print(f'Fold scores:{scores}')


fi = pd.DataFrame(reg.feature_importances_, index=X_train.columns,
             columns=['Importance'])
fi.sort_values('Importance').plot(kind='barh', title='Feature Importance')


#Forecast on Test Set


test['prediction'] = reg.predict(X_test)
# df.merge(test[['prediction']], how='left', left_index=True, right_index=True)

ax = df[['Load']].plot(figsize=(15, 5))
df['prediction'] = reg.predict(create_features(df[FEATURES]))
df.loc[df.index.isin(test.index), 'prediction'].plot(ax=ax, style='.')
plt.legend(['Real Data', 'Predictions'])
ax.set_title('Raw Data vs. Predictions')
plt.show()


df.loc[(df.index > '04-10-23') & (df.index < '04-17-23')]['Load'] \
    .plot(figsize=(15, 5), title='One Week Data')
df.loc[(df.index > '04-10-23') & (df.index < '04-17-23')]['prediction'] \
    .plot(style='.')
plt.legend(['Real Data', 'Predictions'])
plt.show()

score = np.sqrt(mean_squared_error(test['Load'], test['prediction']))
print(f'RMSE Score on test set: {score:0.2f}')


test['error'] = np.abs(test[TARGET] - test['prediction'])
test['date'] = test.index.date
test.groupby('date')['error'].mean().sort_values(ascending=False).head(10)
test.groupby('date')['error'].mean().sort_values(ascending=True).head(10)


# Better cross validation & add more features
# Add weather forecast, traffic data, & holidays


# precision_score(test['Load'], test['prediction'], zero_division=0)



df = create_features(df)

FEATURES = ['hour', 'dayofweek', 'season', 'month', 'year', 'dayofyear',
          '1 Year Lag', '2 Year Lag']
TARGET = 'Load'

X_all = df[FEATURES]
y_all = df[TARGET]


reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', objective='reg:squarederror',
                       n_estimators=1000, 
                       max_depth=1, 
                       learning_rate=0.01065, 
                       gamma=0.47482, 
                       reg_alpha=0.75910, 
                       reg_lambda=0.29588, 
                       colsample_bytree=0.27434,
                       subsample=0.78153
                       )
reg.fit(X_all, y_all, eval_set=[(X_all, y_all)], verbose=20)  


# Create future preds DataFrame

df.index.max()
future = pd.date_range('2023-11-01', '2023-11-30', freq='1h')
future_df = pd.DataFrame(index=future)
future_df['isFuture'] = True
df['isFuture'] = False
main_df = pd.concat([df, future_df])
main_df = create_features(main_df)
main_df = add_lags(main_df)
main_df

future_w_features = main_df.query('isFuture').copy()
future_w_features
future_w_features.columns
future_w_features[['Load', 'Interval Start', 'Interval End']]

future_w_features['Future Predictions'] = reg.predict(future_w_features[FEATURES])
future_w_features[['Future Predictions', 'hour', 'dayofyear']]

future_w_features['Future Predictions'].plot(figsize=(10, 5), color=color_pal[4],
                                             ms=1, lw=1, title='Future Predictions')
plt.show()


# Save Model
# JobLib Dump for Model Storage

joblib.dump(reg, 'Load_Since_2021_model.joblib')


# Daily Model Updates

def model_updates(reg, new_data):
    
    X_new = new_data[FEATURES]
    y_new = new_data[TARGET]
    
    reg.fit(X_new, y_new)
    
    return reg



reg_new = xgb.XGBRegressor()
reg_new.load_model('CAISO_Load_Data_Model.json')



# Bayesian Optimization



# Define the objective function to minimize (negative RMSE)
def objective(parameters):
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
best = fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=50, verbose=1)

# Print the best hyperparameters
print("Best Hyperparameters:", best)





# Define the objective function
def nonfuture_objective(params, df):
    learning_rate = params['learning_rate']
    print(f'Trying learning_rate={learning_rate}')

    tss = TimeSeriesSplit(n_splits=5, test_size=24 * 365 * 1, gap=24)
    df = df.sort_index()

    fold = 0
    scores = []

    for train_index, val_index in tss.split(df):
        train = df.iloc[train_index]
        test = df.iloc[val_index]

        train = create_features(train)
        test = create_features(test)

        FEATURES = ['hour', 'dayofweek', 'season', 'month', 'year', 'dayofyear',
                    '1 Year Lag', '2 Year Lag']
        TARGET = 'Load'

        x_train = train[FEATURES]
        y_train = train[TARGET]

        x_test = test[FEATURES]
        y_test = test[TARGET]

        reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50,
                               learning_rate=learning_rate)
        reg.fit(x_train, y_train,
                eval_set=[(x_train, y_train), (x_test, y_test)],
                verbose=0)

        y_pred = reg.predict(x_test)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(score)

    avg_score = np.mean(scores)
    print(f'Average RMSE: {avg_score:0.4f}')

    return avg_score

# Define the search space
search_space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.1)),
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    # Add more hyperparameters as needed
}

# Run the optimization
best = fmin(fn=lambda params: nonfuture_objective(params, df),
            space=search_space,
            algo=tpe.suggest,
            max_evals=20,  # Adjust as needed
            verbose=1)

print(f'Best hyperparameters: {best}')








start = pd.Timestamp("November 1, 2023").normalize()
end = pd.Timestamp('November 16, 2023').normalize()
load_df = caiso.get_load(start, end=end)

load_df['Time'] = load_df['Time'].dt.tz_localize(None)

comb_df = pd.concat([future_w_features, load_df])
comb_df = comb_df.set_index('Time')
comb_df[['Load', 'Future Predictions']]

comb_df 

summary = comb_df[['Load', 'Future Predictions']].describe()
print(summary)



comps = pd.date_range('2023-11-01', '2023-11-30', freq='1h')
comp_df = comb_df.groupby(comb_df.index).mean().reindex(comps)
comp_df.shape
comp_df.columns

comp_df[['Load', 'Future Predictions']][:300]



