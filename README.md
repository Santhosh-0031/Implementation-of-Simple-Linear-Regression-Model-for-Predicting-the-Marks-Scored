# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
### SANTHOSH KUMAR 
### 212223100051
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
### Dataset
![Screenshot 2024-08-21 211158](https://github.com/user-attachments/assets/2955a6d7-b094-4f83-b178-6ac8f478a78d)

### Head Values
![Screenshot 2024-08-21 211212](https://github.com/user-attachments/assets/1ee74d39-6a59-4b01-9968-0ebcd5e8f8d7)

### Tail Values
![Screenshot 2024-08-21 211222](https://github.com/user-attachments/assets/d29ff13d-0fb5-4a2d-83de-4e4f134ad0e1)

### X and Y values
![Screenshot 2024-08-21 211253](https://github.com/user-attachments/assets/df9c57ab-657f-4dfe-b9f1-0921007c7d38)

### Predication values of X and Y
![Screenshot 2024-08-21 211306](https://github.com/user-attachments/assets/cc67692f-a055-4138-bb8c-c35ec9c1f0b0)

### MSE,MAE and RMSE
![Screenshot 2024-08-21 211400](https://github.com/user-attachments/assets/c53013ce-aa8f-4ec2-bcb6-9bbad39ea4cb)

### Training Set
![Screenshot 2024-08-21 211324](https://github.com/user-attachments/assets/60374597-4980-43d9-b121-6ef645805e91)

### Testing Set
![Screenshot 2024-08-21 211348](https://github.com/user-attachments/assets/81b4f695-8065-4412-a9dc-1dddf3d18cdd)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
