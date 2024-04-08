

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
![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/077d0ce5-8426-4ee0-8fe3-81e5d81c15dc)



![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/303ce4d4-5455-4d97-97f6-767b80caddf9)

```





```


![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/a13d70c6-25ea-4e8f-bd50-07138782e3bc)


![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/82f521df-7448-405d-bfa3-f70a8f3a3130)


![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/02604f35-2ad1-435b-820b-ac691bf19679)


![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/3431241b-f5c2-4e1d-ba6e-3fb2eab825f2)


![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/f992952f-c0c8-4b67-b117-2a541abbaf85)


![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/4f45bac5-f15a-41e6-af56-207217a62af7)


![image](https://github.com/velupradeep/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150329341/92e63318-5ecf-439c-aa54-d6bac3b4a8ed)














## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
