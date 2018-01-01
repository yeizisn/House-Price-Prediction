import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# 81 columns = 1 ID + 1 saleprice + 
all_data= pd.read_csv("train.csv")
testSet = pd.read_csv("test.csv")

#checking for missing data
'''
NAs = pd.concat([trainSet.isnull().sum(),testSet.isnull().sum()], axis=1,keys=['Train','Test'])
NAs =  NAs[NAs.sum(axis=1) >0]
NAs.sort_values(['Train','Test'],ascending=[True,False],inplace=True)



fig = plt.figure()
ax = fig.add_subplot(111)

ax2 = ax.twinx()
NAs.Train.plot(kind="bar",ax=ax, width=0.4,alpha=0.35,position=1.2)
NAs.Test.plot(kind="bar",ax=ax2, width=0.4, position=0)
ax.grid(axis='x',linestyle='dashed')

fig2 = plt.figure()
lt_NAs = NAs[0:18]
ax3 = fig2.add_subplot(111)
ax4 = ax3.twinx()
ax3.set_ylim(0,17)
ax4.set_ylim(0,17)
lt_NAs.Train.plot(kind="bar",ax=ax3, width=0.4,alpha=0.35,position=1.2)
lt_NAs.Test.plot(kind="bar",ax=ax4, width=0.4, position=0)
ax3.grid(axis='x',linestyle='dashed')
plt.show()
'''

'''
#plot the count for missing value for the train set 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
f, ax = plt.subplots(figsize=(16, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=16)
plt.ylabel('Percent of missing values', fontsize=16)
plt.title('Percent missing data by feature', fontsize=16)
plt.show()
'''





