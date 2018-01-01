# House Price
----
## Parameter Selection
*  missing data

              |  Train | Test
------------- |--------|-----  
MSZoning       |   0    | 4.0
BsmtFullBath   |   0    | 2.0
BsmtHalfBath   |   0    | 2.0
Functional     |   0    | 2.0
Utilities      |   0    | 2.0
BsmtFinSF1      |  0    | 1.0
BsmtFinSF2      |  0    | 1.0
BsmtUnfSF       |  0    | 1.0
Exterior1st     |  0    | 1.0
Exterior2nd     |  0   |  1.0
GarageArea      |  0   |  1.0
GarageCars      |  0   |  1.0
KitchenQual     |  0   |  1.0
SaleType         | 0    | 1.0
TotalBsmtSF     |  0    | 1.0
Electrical      |  1    | 0.0
MasVnrType      |  8  |  16.0
MasVnrArea     |   8   | 15.0
BsmtCond       |  37   | 45.0
BsmtQual       |  37  |  44.0
BsmtFinType1    | 37  |  42.0
BsmtExposure   |  38  |  44.0
BsmtFinType2  |   38  |  42.0
GarageCond     |  81  |  78.0
GarageFinish  |   81  |  78.0
GarageQual    |   81  |  78.0
GarageYrBlt   |   81  |  78.0
GarageType    |   81  |  76.0
LotFrontage   |  259  | 227.0
FireplaceQu   |  690 |  730.0
Fence        |  1179 | 1169.0
Alley        |  1369 | 1352.0
MiscFeature  |  1406 | 1408.0
PoolQC      |   1453 | 1456.0

*  Correlation Matrix
The following picture are the correlation matrix:
<div align=center>
<img src="/Users/Song/Documents/2017/clemsonCourses/Applieddatascience/project/house_price/corr_matrix.png" width=600>
</div>

<div align=center>
<img src="/Users/Song/Documents/2017/clemsonCourses/Applieddatascience/project/house_price/top10_quanti.png" width=400>
</div>

*  Drop missing data base on correlation matrix
`OverallQual` ,`GrLivArea`, `GarageCars`,`GarageArea`...`YearBuilt` are important to `SalePrice`.

	* `GarageX` drop
	
	`GarageCars` and `GarageArea` have high correlation with each other, and `GarageCars`is more important to `SalePrice`, so we drop `GarageArea` and other `GarageX` in missing data form.
	
	* `BsmtX` drop
	
	Since we will retain `TotalBsmtSF`, so we will drop all the `BsmtX` in missing data form. 
	
	* `MasVnrX` drop
	
	They have high correlation with `YearBuilt` and `OverallQual` which are already considered. So we won't lose information.
	
	* `Electrical`
	We will drop the row of the missing data.
	
After these steps, there will be no missing data in the training set.
