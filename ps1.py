import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def direct_sol(X,y): #Question 1
    print(" --- Direct Solution --- ")
    ones = np.ones(len(X))
    one_X = np.c_[ones,X]

    # ((X^T*X)^-1)*(X^T*y) -- direct solution
    directSolution=np.matmul(np.linalg.inv(np.matmul(one_X.T,one_X)),np.matmul(one_X.T,y))


    print(f"Coefficient : {directSolution[1]}\nIntercept : {directSolution[0]}")
    return directSolution


def gradient(X,x_test,y,y_test,step_Size=0.1,iteration=1000): #Question 2 and 3

    coef=np.zeros(1)
    intcp = np.zeros(1)
    N=len(X)
    w=[]
    cost_values=[]

    for i in range(iteration):

        # linear equation
        equation = X.dot(coef) + intcp

        # error
        error = y - equation

        # gradient of the cost
        cost = np.sum((error ** 2) / len(error))

        # finding coefficient
        coef = coef - ((step_Size * -1.0 * np.dot(error,X))/ N)

        # finding intercept
        intcp = intcp - ((step_Size * -1.0 * np.sum(error))/N)

        if i % 10000 == 0: print(f"Iteration {i}: {cost}")

        # adding a list all the w(coefficient) and cost values.
        w.append(coef)
        cost_values.append(cost)

    print("")

    # testing the model
    equation = x_test.dot(coef) + intcp
    error = y_test - equation
    cost=np.sum((error ** 2)/len(error))
    print("--- Gradient Descent Solution ---")
    print("Coefficient: ", coef)
    print("Intercept: ",intcp)
    print("Mean squared error: %.2f" % cost)
    print("")


    print("******** Question 3 *********") #Question 3

    print("Plotting W Values")
    plt.plot(np.arange(iteration), w)
    plt.title("Coefficient Determination")
    plt.xlabel("Iteration")
    plt.ylabel("Coefficient")
    plt.show()
    print("Plotted W Values")

    print("Plotting Cost Values")
    plt.plot(np.arange(iteration), cost_values)
    plt.title("Empirical Cost")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()
    print("Plotted Cost Values")


def sklinear_model(X,x_test,y,y_test): # Question 4
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X,y)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(x_test)

    print(" --- Sklearn Calculation --- ")
    # The coefficients
    print('Coefficients: ', regr.coef_)
    print('Intercept: ', regr.intercept_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, diabetes_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, diabetes_y_pred))

    # Plot outputs
    plt.scatter(x_test, diabetes_y_test, color='black')
    plt.plot(x_test, diabetes_y_pred, color='blue', linewidth=3)
    plt.title("Sklearn Linear Regression Graph")
    plt.xticks(())
    plt.yticks(())

    plt.show()


if __name__ == '__main__':

    # load database
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    # determining the attribute
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # getting 80% X values for training and 20% y values for testing
    split = round(len(diabetes_X) * 0.20)

    # get first 80% of data X up to last 20%
    diabetes_X_train = diabetes_X[:-split]
    # get last 20% of data X from the end of the list
    diabetes_X_test = diabetes_X[-split:]

    # get first 80% of data Y up to last 20%
    diabetes_y_train = diabetes_y[:-split]
    # get last 20% of data Y from the end of the list
    diabetes_y_test = diabetes_y[-split:]

    print("")
    print("******** Question 1 *********")
    w0,w1= direct_sol(diabetes_X_train,diabetes_y_train) #Question 1
    print("*****************************")
    print("")
    step_size = 0.1
    iteration = 100000
    print("")
    print("******** Question 2 and 3 *********") #Question 2 and 3
    gradient(diabetes_X_train, diabetes_X_test,diabetes_y_train,diabetes_y_test, step_size, iteration)
    print("*****************************")
    print("")

    print("******** Question 4 *********")

    sklinear_model(diabetes_X_train,diabetes_X_test,diabetes_y_train,diabetes_y_test)

    print("*****************************")

    print("")
    print("******** Question 5 *********") #Question 5
    predicts = []

    # prediction process using linear equation that we created
    for i in range(len(diabetes_X_test)):
        predicts.append(w0+w1*diabetes_X_test[i])

    print(" --- My Calculation --- ")

    # comparing our prediction with real data to find the r2 score
    print('Coefficient of determination: %.2f' % r2_score(diabetes_y_test, predicts))
    print("|Note| : Coefficient,Intercept and MSE are alread calculated by using Gradient Descent Technique in Question 2")
    # Plot outputs
    plt.scatter(diabetes_X_test,diabetes_y_test)
    plt.title("My Linear Regression Graph")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(diabetes_X_test,predicts)
    plt.show()


    print("*****************************")

