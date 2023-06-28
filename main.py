#importing some necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Now we need load the dataset
df= pd.read_csv("E:/TCS ION PROJECT/Housing.csv")
# explore the dataset
df.shape
df.shape
df.describe()
# Data Cleaning
print(df.isnull().sum())
print("Missing Values are :",df.isnull().values.any())
# Check for Duplicate values 
df[df.duplicated(keep=False)]
lb =LabelEncoder()
cat_data = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
for i in cat_data:
    df[i] = lb.fit_transform(df[i])
df.head()
dms = pd.get_dummies(df[["furnishingstatus"]])
dms.head()
df=pd.concat([df, dms], axis=1)
df.rename({'semi-furnished': 'semi_furnished'}, axis=1, inplace=True)
df.head(10)
df=df.drop(['furnishingstatus'], axis=1)
df.head()
df.hist(figsize=(20,15))
sns.pairplot(df, palette='bright', kind="reg")
sns.pairplot(df[['price', 'area']], palette='bright', kind="reg")
fig, axs = plt.subplots(1, len(df.columns), figsize=(20,10))

for i, ax in enumerate(axs.flat):
    ax.boxplot(df.iloc[:,i])
    ax.set_title(df.columns[i], fontsize=20, rotation=90)
    ax.tick_params(axis='y', labelsize=14)
    
plt.tight_layout()
l=['bedrooms','bathrooms','stories']
for i in l:
    plt.figure(figsize=(20,5))
    sns.set_style(style='darkgrid')
    sns.countplot(data=df,x=i)
    plt.title(i + ' Distribution')
    plt.xlabel(i)
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.show()
    df.corr()["price"].sort_values(ascending=False).to_frame()
    c=['area','bathrooms','airconditioning','stories','parking','bedrooms','prefarea']
for i in c:
    print(df[['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement']])
    sns.regplot(x='area', y='price', data=df)
    x = df.drop('price', axis = 1)
y = df['price']
x.head()
y.head()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 101)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(f"""
Shape of X Train: {x_train.shape}
Shape of Y Train: {y_train.shape}
Shape of X Test:  {x_test.shape}
Shape of Y Test:  {y_test.shape}
""")
linear_reg= LinearRegression()
linear_reg.fit(x_train, y_train)
linear_reg_mod=linear_reg.fit(x_train, y_train)
y_train_pred = linear_reg_mod.predict(x_train)
y_train_pred = pd.DataFrame(y_train_pred)
R2 = metrics.r2_score(y_train , y_train_pred)
MAE = metrics.mean_absolute_error(y_train , y_train_pred)
MSE = metrics.mean_squared_error(y_train , y_train_pred)
RMSE = np.sqrt(MSE)
pd.DataFrame([R2, MAE, MSE, RMSE], index=['R2 Score', 'MAE', 'MSE', 'RMSE'], columns=['Training Scores'])
y_pred = linear_reg_mod.predict(x_test)
y_pred = pd.DataFrame(y_pred)
R2 = metrics.r2_score(y_test , y_pred)
MAE = metrics.mean_absolute_error(y_test, y_pred)
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
pd.DataFrame([R2, MAE, MSE, RMSE], index=['R2 Score', 'MAE', 'MSE', 'RMSE'], columns=['Testing Scores'])
Prediction = {'Actual Price' : np.array(y_test), 
            'Predicted Price' : np.array(y_pred).flatten()
           }

Prediction = pd.DataFrame(Prediction)
print('Prediction using Linear Regression Model')
Prediction.head(10)
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              
plt.xlabel('y_test', fontsize=18)                          
plt.ylabel('y_pred', fontsize=16) 
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)
lasso.fit(x_train, y_train)
y_train_pred = lasso.predict(x_train)
y_train_pred = pd.DataFrame(y_train_pred)
R2 = metrics.r2_score(y_train , y_train_pred)
MAE = metrics.mean_absolute_error(y_train , y_train_pred)
MSE = metrics.mean_squared_error(y_train , y_train_pred)
RMSE = np.sqrt(MSE)
pd.DataFrame([R2, MAE, MSE, RMSE], index=['R2 Score', 'MAE', 'MSE', 'RMSE'], columns=['Training Scores'])
y_pred = lasso.predict(x_test)
y_pred = pd.DataFrame(y_pred)
R2 = metrics.r2_score(y_test , y_pred)
MAE = metrics.mean_absolute_error(y_test, y_pred)
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
pd.DataFrame([R2, MAE, MSE, RMSE], index=['R2 Score', 'MAE', 'MSE', 'RMSE'], columns=['Testing Scores'])
Prediction = {'Actual Price' : np.array(y_test), 
            'Predicted Price' : np.array(y_pred).flatten()
           }

Prediction = pd.DataFrame(Prediction)
print('Prediction using LASSO Regression Model')
Prediction.head()
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.5)
ridge.fit(x_train, y_train)
y_train_pred = ridge.predict(x_train)
y_train_pred = pd.DataFrame(y_train_pred)
R2 = metrics.r2_score(y_train , y_train_pred)
MAE = metrics.mean_absolute_error(y_train , y_train_pred)
MSE = metrics.mean_squared_error(y_train , y_train_pred)
RMSE = np.sqrt(MSE)
pd.DataFrame([R2, MAE, MSE, RMSE], index=['R2 Score', 'MAE', 'MSE', 'RMSE'], columns=['Training Scores'])
y_pred = ridge.predict(x_test)
y_pred = pd.DataFrame(y_pred)
R2 = metrics.r2_score(y_test , y_pred).round(5)
MAE = metrics.mean_absolute_error(y_test, y_pred).round(2)
MSE = metrics.mean_squared_error(y_test, y_pred).round(2)
RMSE = np.sqrt(MSE).round(2)
pd.DataFrame([R2, MAE, MSE, RMSE], index=['R2 Score', 'MAE', 'MSE', 'RMSE'], columns=['Testing Scores'])
Prediction = {'Actual Price' : np.array(y_test), 
            'Predicted Price' : np.array(y_pred).flatten().round(2)
           }

Prediction = pd.DataFrame(Prediction)
print('Prediction using Ridge Regression Model')
Prediction.head()
from sklearn.ensemble import RandomForestRegressor
RFRmodel = RandomForestRegressor(n_estimators=100,max_depth = 10, min_samples_split=10,random_state=42,criterion='mse')
RFRmodel.fit(x_train,y_train)
y_train_pred = RFRmodel.predict(x_train)
y_test_pred = RFRmodel.predict(x_test)
R2 = metrics.r2_score(y_train , y_train_pred)
MAE = metrics.mean_absolute_error(y_train , y_train_pred)
MSE = metrics.mean_squared_error(y_train , y_train_pred)
RMSE = np.sqrt(MSE)
pd.DataFrame([R2, MAE, MSE, RMSE], index=['R2 Score', 'MAE', 'MSE', 'RMSE'], columns=['Training Scores'])
R2 = metrics.r2_score(y_test , y_test_pred)
MAE = metrics.mean_absolute_error(y_test, y_test_pred)
MSE = metrics.mean_squared_error(y_test, y_test_pred)
RMSE = np.sqrt(MSE)
pd.DataFrame([R2, MAE, MSE, RMSE], index=['R2 Score', 'MAE', 'MSE', 'RMSE'], columns=['Testing Scores'])
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
knn.fit(x_train,y_train)
y_train_pred = knn.predict(x_train)
y_test_pred = knn.predict(x_test)
R2 = metrics.r2_score(y_train , y_train_pred)
MAE = metrics.mean_absolute_error(y_train , y_train_pred)
MSE = metrics.mean_squared_error(y_train , y_train_pred)
RMSE = np.sqrt(MSE)
pd.DataFrame([R2, MAE, MSE, RMSE], index=['R2 Score', 'MAE', 'MSE', 'RMSE'], columns=['Training Scores'])
R2 = metrics.r2_score(y_test , y_test_pred)
MAE = metrics.mean_absolute_error(y_test, y_test_pred)
MSE = metrics.mean_squared_error(y_test, y_test_pred)
RMSE = np.sqrt(MSE)
pd.DataFrame([R2, MAE, MSE, RMSE], index=['R2 Score', 'MAE', 'MSE', 'RMSE'], columns=['Testing Scores'])
linear_reg_mod.intercept_
linear_reg_mod.coef_
from joblib import dump
dump(linear_reg_mod , 'ipa.pkl')