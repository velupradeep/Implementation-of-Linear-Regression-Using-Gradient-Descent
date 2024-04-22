# EX-03 Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

   ```


   


   ```

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

```






```
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

```




























```


## Output:
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/c73e1aa6-408d-41b3-8a63-feb0dc522f36)
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/36f75dc5-b6a5-4407-aa26-873131207a6c)
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/5389cca8-021d-47de-99c6-252329ac64df)
```


```
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/7f43a500-c741-4e0b-8180-c2da84bbb166)
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/fb229d58-7592-4c58-9103-ef247a768cd2)
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/fb8ef7ef-20be-483c-8555-5ee4647a30b8)
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/183da932-bff4-4414-b8e0-c5368bc63174)
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/46e6b800-4fa7-4189-907d-fad1906f9285)
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/79226fd0-2a7a-4ed6-abb5-6fee48daea0a)
































































































## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
