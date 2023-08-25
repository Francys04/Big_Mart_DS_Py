from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from Src.pre_processing import X, Y

'''Splitting the data into training data and testing data'''

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

'''Machine Learning Model Training'''

# XGBoost Regressor
regressor = XGBRegressor(eta=0.1, nrounds=1000, max_depth=8, colsample_bytree=0.5, scale_pos_weight=1.1,
                         booster='gbtree',
                         metric='multi:softmax')
# # plotting
regressor.fit(X_train, Y_train)

# '''predicting on training data'''
# training_data_prediction = regressor.predict(X_train)
#
# # R squared value
# r2_train = metrics.r2_score(Y_train, training_data_prediction)
# print('R Square value : ', r2_train)
