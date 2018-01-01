################EDA part######################
import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
from scipy.stats import norm, skew #for some statistics

'''
#plot the correlation matrix
#correlation matrix
df_train = pd.read_csv('train.csv')
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True,cmap="YlGnBu");
plt.show()
'''

#The top 10 important predictors:
'''
(*) for qualititive
1.OverallQual (*)
2.GrLivArea
3.TotalBsmtSF
4.2ndFlrSF
5.BsmtFinSF
6.1stFlrSF
7.GarageCars(*)
8.GarageArea
9.LotArea
10.YearBuilt(*)
'''

'''

columns = ['GrLivArea','TotalBsmtSF','2ndFlrSF','BsmtFinSF1','1stFlrSF','GarageArea','LotArea','SalePrice']

Subtrain = df_train[columns]
corrmat = Subtrain.corr()
k = 7 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(Subtrain[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,cmap="YlGnBu")
plt.title("Correlation Matrix")
plt.show()
'''


'''
#box plot overallqual/saleprice gragecar/saleprice and YearOfBuild/SalePrice
for var in vars:
	data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
	f, ax = plt.subplots(figsize=(8, 6))
	fig = sns.boxplot(x=var, y="SalePrice", data=data)
	fig.axis(ymin=0, ymax=800000);
	plt.show()
'''


'''
#plot GrLivArea&SalePrice Without Outlier
#Deleting outliers

#Check the graphic again
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'],color="C9")
plt.ylabel('SalePrice', fontsize=16)
plt.xlabel('GrLivArea', fontsize=16)
plt.title("Without Outlier GrLivArea & SalePrice",fontsize=16)
plt.show()
'''


'''
#plot GrLivArea&SalePrice With Outlier
#Deleting outliers
train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index
#Check the graphic again
fig, ax = plt.subplots()
for i in range(1459):
	if (i!=523 and i!=1298):
		ax.scatter(x = train['GrLivArea'].values[i], y = train['SalePrice'].values[i],color="C9")
	else:
		ax.scatter(x = train['GrLivArea'].values[i], y = train['SalePrice'].values[i],color='red')

plt.ylabel('SalePrice', fontsize=16)
plt.xlabel('GrLivArea', fontsize=16)
plt.title("Original GrLivArea & SalePrice",fontsize=16)
plt.show()
'''



'''
#plot the skewed data (make them normrized)
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
'''

'''
# plot comparison of predicted result for xgboost and lasso
test = pd.read_csv('Subtest.csv')
lasso_result = pd.read_csv("Lasso_output.csv")
xgb_result = pd.read_csv("xgb_output.csv")
price = test["SalePrice"]
original_price = np.asarray((sorted(price[0:100])))
lasso_price = np.asarray(sorted(lasso_result["SalePrice"][0:100]))
xgb_price = np.asarray(sorted(xgb_result["SalePrice"][0:100]))

# original_price = sorted(np.log(price[0:49]))
# lasso_price = sorted(np.log(lasso_result["SalePrice"][0:49]))
# xgb_price = sorted(np.log(xgb_result["SalePrice"][0:49]))

#x = np.arange(1,51)
# price = sorted(price)
# pred_price = sorted(pred_price)
x = np.arange(1,101)

line1 = plt.scatter(x,original_price,c='C9',label='Actual SalePrice')
#line2 = plt.scatter(x,abs(lasso_price - original_price)/original_price, c='tomato',label="Predicted SalePrice(lasso)")
line3 = plt.scatter(x,xgb_price, c='tomato',label="Predicted SalePrice(XGBoost)",alpha=0.35)
plt.legend(handles=[line1,line3])
plt.title("Predicted SalePrice vs. Actual SalePrice (XGBoost)")
plt.grid()
plt.show()
'''


