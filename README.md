# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Adhithya M R
RegisterNumber:  212222240002
*/

import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])



```

## Output:
### Data head ():

![239682377-b82352d4-d32b-417f-b084-9bcc6679d6d4](https://github.com/AdhithyaMR/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118834761/f9a49600-b7a7-4a49-8a49-17c18914ad73)

### Data info:
![239682386-f86626d8-e411-4d23-b827-c9be33f3f986](https://github.com/AdhithyaMR/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118834761/ba0eebdd-edd8-4fef-83b4-a392e2db6886)

### isnull() and sum():
![239682392-9f764145-534a-4aaa-8573-d9a98b255e71](https://github.com/AdhithyaMR/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118834761/7addb2ab-dd67-4333-a39f-d7baa834320d)


### Data.head() for salary:



![239682444-cb24a583-388f-48e2-a09a-a45e6128c12b](https://github.com/AdhithyaMR/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118834761/38a52619-80d1-4053-aa4a-83af96e6599c)

### MSE value:

![239682475-ce29115f-52b7-417d-857e-1309409ddd1f](https://github.com/AdhithyaMR/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118834761/1e3dcdce-90d1-474c-aa4e-d9e7ae59b227)

### r2 value:

![239682499-7cf9ec95-7b09-4250-9a50-09b12bd94308](https://github.com/AdhithyaMR/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118834761/626630aa-20d3-4678-bc49-d8d665cbc87e)
### Data prediction:

![239682529-cdd07dd7-6bfb-494f-9fe3-378d69126a31](https://github.com/AdhithyaMR/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118834761/857920b0-b9b2-4ebd-bc2a-f823b65311f4)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
