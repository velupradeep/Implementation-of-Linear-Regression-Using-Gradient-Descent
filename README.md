# EX-03 Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:PRADEEP V
RegisterNumber:212223240119
*/
```

```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
    
data=pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:,:-2].values)
print(X)

X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta = linear_regression(X1_Scaled,Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value:{pre}")

```
## Output:
# X Values
![326158720-9bf60fa0-af77-42cf-a31d-ea158d181abb](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/99be1b19-edab-4d57-9a99-430b5837a919)
# Y Values
![326158727-3be61adc-d7e3-4d8c-97b4-3e0ac9547133](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/5ac27f83-360b-4036-8d95-3f216ae2d19c)
# X Scaled Values
![326158733-c0b84bd5-3d9e-45f2-8f24-7908acecff74](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/b4caf973-460c-432e-a1f9-f92c6c2619ee)
# Y Scaled Values
![326158906-9d167518-03cc-4e2c-b206-93ac4cf31ba6](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/6ed269dd-7022-4b26-b1bf-74580f8b4824)
# Predicted Value
![326158770-9c78ef45-c44a-4873-8ff1-262becf88045](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/79210ae5-9af9-4728-9b93-41ac79974118)





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
