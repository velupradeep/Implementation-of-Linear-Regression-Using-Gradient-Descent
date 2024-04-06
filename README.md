                                               #NAME:PRADEEP V
                                               #REG N0:212223240119

# Implementation-of-Linear-Regression-Using-Gradient-Descent

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

/*
Program to implement the linear regression using gradient descent.
Developed by:PRADEEP V
RegisterNumber:212223240119
*/
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


## Output:
![ml 1](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/ead19855-c333-4652-a136-a78eead7573e)


![ml 2](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/6e85c3fd-857a-488d-bb8e-19b754c6901f)

![ml 4](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/4ca1a305-ae5b-420b-b252-ebf2c2b0b4e2)

![ml 4](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/b0e2e359-9ede-49e2-9903-0ae91630f6d6)


![ml 6](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/5b57b617-1d3d-44e4-b64f-f31f5b748a6c)

![ml 7](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/5cf93d2d-a852-4372-8910-c16c27fb2854)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
