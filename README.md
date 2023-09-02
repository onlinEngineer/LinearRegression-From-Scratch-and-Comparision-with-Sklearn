
# LinearRegression-From-Scratch-and-Comparision-with-Sklearn

### LINEAR REGRESSION FUNDAMENTALS IN MACHINE LEARNING
---
#### Hypothesis

In Machine Learning, Hypothesis space is a manageable subset of all possible solutions. [4] 
```math
ℎ^𝑤(𝑥)=Σ_{j=0}\; 𝑤_𝑗𝑥=𝑤^𝑡𝑥
```
#### Loss Function

Loss function shows that the difference between input and output values. If the predicted values are different too much from actual values, that’s mean the model is not good enough. 

```math
𝑆𝑞𝑢𝑎𝑟𝑒𝑑\; 𝑒𝑟𝑟𝑜𝑟: 𝐿 ( 𝑦 ,ℎ(𝑥) ) = ( 𝑦^i − ℎ(𝑥^i) )^2

```

#### Empirical loss 

Empirical loss is the average loss over the data points.[1] 



Where N is the length of the data
```math
(1/𝑁) Σ_{i=1}^{N}\; 𝐿((𝑥_𝑖,𝑦_𝑖),ℎ(∙))
```
#### Training
Used for minimizing the empirical loss to finding the best predictor. In training, A certain part of the data is taken for training and tested. If the loss is high, new data point are added or the model is changed. [4]

#### Expected Loss
Because the training set is optimized, the empirical loss may not be representative of how well the model will perform on new samples. Therefore, the model is tested on a new set which is not used in during the training.[4]


### PROCEDURE
---
#### Preparation

First, we import the data (Sklearn.diabetes) we will use, and the libraries such as numpy, matplotlib. After that we split diabetes_X and diabetes_y for 80% X values for training and 20% y values for testing. There must be four data which are named training and testing for X, training, and testing for y.
We used training data for train to data, also we used the data we trained for testing.


#### Direct solution

We calculate the coefficient and intercept by using direct solution formula. Our coefficient is 957.00838947 and Intercept is 152.08225581

![Screenshot 2023-09-02 212341](https://github.com/onlinEngineer/LinearRegression-From-Scratch-and-Comparision-with-Sklearn/assets/70773825/579b3dde-1209-4670-a906-cde9f5c29f96)

 

#### Gradient Solution
Our step size (learning rate) is 0.1, iteration is 100000. When iteration size increased, the cost values are reducing as we already expected. The coefficient is 957.00838899, Intercept is 152.08225581, Cost is 4124.82.

![image](https://github.com/onlinEngineer/LinearRegression-From-Scratch-and-Comparision-with-Sklearn/assets/70773825/fd36714b-eec0-4e15-89df-66e0648a1ba5)


#### Coefficient Graph
As the number of iterations increases, the coefficient value increases rapidly up to a certain point and after approximately 20000 iterations, it slows down in the same course. So, We can determine the our coefficient value.

<img src="https://github.com/onlinEngineer/LinearRegression-From-Scratch-and-Comparision-with-Sklearn/assets/70773825/61d8415e-244d-4616-a58e-499fc0588e1e" alt="Resim 1" width="50%">


#### Cost Graph
As the number of iterations increases, the cost value are rapidly decreases up to a certain point and after 10000 iterations, the values are starting to change very slowly.

<img src="https://github.com/onlinEngineer/LinearRegression-From-Scratch-and-Comparision-with-Sklearn/assets/70773825/d3d7e625-d2c0-463b-b50f-4b2aba605783" alt="Resim 1" width="50%">


### Sklearn Calculations Comparision
---
The coefficient, intercept, MSE and R2 score are values as the same as we calculated except for the slightly deviation in coefficient.

#### My Calculation

![image](https://github.com/onlinEngineer/LinearRegression-From-Scratch-and-Comparision-with-Sklearn/assets/70773825/c7c1c898-9e69-4ce0-bd29-fbed0b056651)
![image](https://github.com/onlinEngineer/LinearRegression-From-Scratch-and-Comparision-with-Sklearn/assets/70773825/7d290d1a-89e8-44e2-8820-a2ddd29fda08)


#### Sklearn Calculation
![image](https://github.com/onlinEngineer/LinearRegression-From-Scratch-and-Comparision-with-Sklearn/assets/70773825/c555f1b8-5b75-4461-ac21-25713348ffc7)


### Linear Regression Graphs

#### My Graph

<img src="https://github.com/onlinEngineer/LinearRegression-From-Scratch-and-Comparision-with-Sklearn/assets/70773825/07eb0fc2-0908-443f-8b07-c0ad2eef34af" alt="Resim 1" width="50%">


#### Sklearn Graph

<img src="https://github.com/onlinEngineer/LinearRegression-From-Scratch-and-Comparision-with-Sklearn/assets/70773825/bf952dd5-7e89-457b-a193-377afcf134ce" alt="Resim 2" width="50%">



### CONCLUSION 
---
In this assignment, we leant what is linear regression, what benefit does it give us, what the fundamentals of the regression, how we apply the regression etc.
Linear Regression is an algorithm to find the relationship between two variables. These are called independent and dependent variables. Dependent variables denoted by “y”, as understood from the name, it is an output variable. So, there are some variables enter which is name independent variables, denoted by “X”, a linear regression model and sets the value of y. So, we build our model by calculating the relationship between the input and output values as well. There are two methods to construct a model in regression. One of them is direct solution. This method finds the coefficient and intercept directly but not always working. Because to apply this method, our matrix multiplication must be square matrix to get inverse of them. If we do not always have a square matrix, we cannot apply this method.
Other method is gradient descent method. Gradient descent is an iterative optimization algorithm to find the minimum of a function. Gradient descent widely used algorithms in machine learning, mainly because it can be applied to any function to optimize it. [5] In this method, we can calculate all values such as coefficient, intercept, empirical loss, MRE (Cost) etc. The method uses learning rate and iteration. In our experiment, when we increased the learning rate to 0.1 and increased the iteration number, we got a smoother result.
In both methods, our model was formed as follows. 𝑦=957.00838899∗x+152.08225581
In this model, the coefficient value is 957.00838899, and the intercept value is 152.08225581. This model forms the linear regression line in our graph. So, when we give x a number, this result will give us the closest value to the real value of y.



### REFERANCES
1. Sanjay Lall and Stephen Boyd, Supervised Learning via Empirical Risk Minimization, EE104 Stanford University, EE104 Spring 2018
2. Mohit Gupta, geeksforgeeks.org, ML | Linear Regression, 13 Sep 2018
3. Deepanshi, analyticsvidhya.com, All you need to know about your first Machine Learning model – Linear Regression, May 25, 2021
4. Dr. Bahadir K. Gunturk, Introduction, Elements of Machine Learning, Istanbul Medipol University
5. Adarsh Menon, towardsdatascience.com, Linear Regression using Gradient Descent, Sep 16, 2018
