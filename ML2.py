#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data=pd.read_csv('ex1data1.txt',header=None)
plt.scatter(data[0],data[1])
data_n=data.values
m=len(data_n[:,0])
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)

regressor=LinearRegression()
regressor.fit(X,y)

theta0=regressor.intercept_
theta1=regressor.coef_

plt.plot(X,theta1*X+theta0)
plt.show()


# In[ ]:




