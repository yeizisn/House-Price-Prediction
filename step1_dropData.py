import numpy as np 
import pandas as pd 
from sklearn.linear_model import Ridge,Lasso,ElasticNet, BayesianRidge, LassoLarsIC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import Imputer,RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator,TransformerMixin, RegressorMixin, clone
#import xgboost as xgb
import lightgbm as lgb

trainSet = pd.read_csv('train.csv')
testSet = pd.read_csv('test.csv')

#drop missing data after that 63 predictors remain
train_missing = trainSet.isnull().sum().sort_values(ascending=False)
drop_index = (train_missing[train_missing>1]).index

trainSet = trainSet.drop(drop_index,axis=1)
trainSet = trainSet.drop(trainSet.loc[trainSet['Electrical'].isnull()].index)

testSet = testSet.drop(drop_index,axis=1)
test_Id = testSet['Id']
testSet = testSet.drop('Id',axis=1)


trainSet = trainSet.drop('Id',axis=1)
SalePrice = np.log(trainSet['SalePrice'])

trainSet = trainSet.drop('SalePrice',axis=1)
#61 predictors now

all_data = pd.concat((trainSet,testSet),axis=0)

#dummy encoder
#220 predictors now
all_data = pd.get_dummies(all_data)
columns = all_data.columns
imp = Imputer(missing_values='NaN',strategy="most_frequent",axis=0)
all_data = imp.fit_transform(all_data)

trainSet = all_data[:1459]
testSet = all_data[1459:]


trainSet = pd.DataFrame(trainSet,columns=columns)

testSet = pd.DataFrame(testSet,columns=columns)


print trainSet.isnull()
#imputing the test data
'''
MSZoning         4
BsmtHalfBath     2
BsmtFullBath     2
Functional       2
Utilities        2
Exterior2nd      1
KitchenQual      1
GarageCars       1
GarageArea       1
BsmtFinSF1       1
SaleType         1
TotalBsmtSF      1
BsmtUnfSF        1
BsmtFinSF2       1
Exterior1st      1
'''


#choose predictor by random forest
n_folds = 5
def rmsle_cv(model):
	kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(trainSet.values)
	rmse = np.sqrt(-cross_val_score(model, trainSet.values, SalePrice, scoring="neg_mean_squared_error",cv=kf))
	return rmse


# lasso = make_pipeline(RobustScaler(),Lasso(alpha=0.005,random_state=1))
# score_lasso = rmsle_cv(lasso)
# print('\nLasso Score : {:.4f} ({:.4f})\n'.format(score_lasso.mean(),score_lasso.std()))

# ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.005,l1_ratio=0.9, random_state=3))
# score_ENet = rmsle_cv(ENet)
# print('\nElasticNet Score : {:.4f} ({:.4f})\n'.format(score_ENet.mean(),score_ENet.std()))

# KRR = KernelRidge(alpha=0.6,kernel="polynomial",degree=2,coef0=2.5)
# score_KRR = rmsle_cv(KRR)
# print('\nKernelRidge Score : {:.4f} ({:.4f})\n'.format(score_KRR.mean(),score_KRR.std()))

# GBoost = GradientBoostingRegressor(n_estimators=3000,learning_rate=0.05,
# 									max_depth=4,max_features='sqrt',
# 									min_samples_leaf=15,min_samples_split=10,
# 									loss='huber', random_state=5)
# score_GBoost = rmsle_cv(GBoost)
# print('\nGradient Boosting  Score : {:.4f} ({:.4f})\n'.format(score_GBoost.mean(),score_GBoost.std()))





class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, \
#                              learning_rate=0.05, max_depth=3, \
#                              min_child_weight=1.7817, n_estimators=2200,\
#                              reg_alpha=0.4640, reg_lambda=0.8571,\
#                              subsample=0.5213, silent=1,\
#                              random_state=7, nthread = -1)


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000,learning_rate=0.05,
									max_depth=4,max_features='sqrt',
									min_samples_leaf=15,min_samples_split=10,
									loss='huber', random_state=5)
lasso = make_pipeline(RobustScaler(),Lasso(alpha=0.005,random_state=1))


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

stacked_averaged_models.fit(trainSet.values, SalePrice)
stacked_train_pred = stacked_averaged_models.predict(trainSet.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(testSet.values))
print(rmsle(SalePrice, stacked_train_pred))



model_lgb.fit(trainSet, SalePrice)
lgb_train_pred = model_lgb.predict(trainSet)
lgb_pred = np.expm1(model_lgb.predict(testSet.values))
print(rmsle(SalePrice, lgb_train_pred))
#prediction.

ensemble = lgb_pred*0.30

sub = pd.DataFrame()
sub['Id'] = test_Id
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv', index=False)


