from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd


def read_data(TRAIN_DIR, nrows=None):
    data_train = pd.read_csv(TRAIN_DIR,
                             parse_dates=["pickup_datetime",
                                          "dropoff_datetime"],
                             nrows=nrows)
    data_train = data_train.drop(['dropoff_datetime'], axis=1)
    data_train.loc[:, 'store_and_fwd_flag'] = data_train['store_and_fwd_flag'].map({'Y': True,
                                                                                    'N': False})
    data_train = data_train[data_train.trip_duration < data_train.trip_duration.quantile(0.99)]

    xlim = [-74.03, -73.77]
    ylim = [40.63, 40.85]
    data_train = data_train[(data_train.pickup_longitude > xlim[0]) & (data_train.pickup_longitude < xlim[1])]
    data_train = data_train[(data_train.dropoff_longitude > xlim[0]) & (data_train.dropoff_longitude < xlim[1])]
    data_train = data_train[(data_train.pickup_latitude > ylim[0]) & (data_train.pickup_latitude < ylim[1])]
    data_train = data_train[(data_train.dropoff_latitude > ylim[0]) & (data_train.dropoff_latitude < ylim[1])]

    return data_train


def train_xgb(X_train, labels):
    Xtr, Xv, ytr, yv = train_test_split(X_train.values,
                                        labels,
                                        test_size=0.2,
                                        random_state=0)

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xv, label=yv)

    evals = [(dtrain, 'train'), (dvalid, 'valid')]

    params = {
        'min_child_weight': 1, 'eta': 0.166,
        'colsample_bytree': 0.4, 'max_depth': 9,
        'subsample': 1.0, 'lambda': 57.93,
        'booster': 'gbtree', 'gamma': 0.5,
        'silent': 1, 'eval_metric': 'rmse',
        'objective': 'reg:linear',
    }

    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=227,
                      evals=evals, early_stopping_rounds=60, maximize=False,
                      verbose_eval=10)

    print('Modeling RMSLE %.5f' % model.best_score)
    return model


def predict_xgb(model, X_test):
    dtest = xgb.DMatrix(X_test.values)
    ytest = model.predict(dtest)
    X_test['trip_duration'] = np.exp(ytest) - 1
    return X_test[['trip_duration']]
