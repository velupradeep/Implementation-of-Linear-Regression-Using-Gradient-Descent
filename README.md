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

# head
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/25f8427a-92ac-4609-98a3-251bb4a7884d)
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/dbdf3f39-e313-43f8-b838-d22f7da1468d)
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/77ec1c72-8be8-4e3f-9c96-5a388ec96bd4)
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/25c28cf7-98b4-49b0-98f7-24749f4d8c2d)
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/ed6747c9-0030-448f-bbd8-abc1928ed654)
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/2fe7e68f-f5fe-4f57-8e8b-96805c8315d8)
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/348aa47e-c4e2-4cb4-a9d5-709c329d03c5)
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/8a5d0c72-c459-4ede-897a-c5b64e285325)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
