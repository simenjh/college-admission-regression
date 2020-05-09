import numpy as np
from sklearn.model_selection import train_test_split
import data_processing
import dataplot



def college_admission(dataset_file, iterations=50, learning_rate=0.1, plot_X=False, plot_cost=False):
    dataset = data_processing.read_dataset(dataset_file)
    X, y = data_processing.preprocess(dataset)
    n = X.shape[1]

    if plot_X:
        dataplot.plot_X(X, y) ## Plotting the data after dimensionality reduction using PCA

    parameters = init_parameters(n)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)

    cost_train_vals, parameters = train_college_admission(X_train, y_train, parameters, iterations, learning_rate)

    cost_test = compute_test_cost(X_test, y_test, parameters)
    
    print(f"Training cost: {cost_train_vals[-1]}")
    print(f"Test cost: {cost_test}")

    if plot_cost:
        dataplot.plot_cost(cost_train_vals) 
    






    

def init_parameters(n):
    W = np.zeros((1, n))
    b = 0
    return {"W": W, "b": b}
    
    

def train_college_admission(X_train, y_train, parameters, iterations, learning_rate):
    m_train = X_train.shape[0]
    J_vals = []

    for i in range(iterations):
        y_pred = compute_y_pred(X_train, parameters)
        J = compute_cost(y_pred, y_train, m_train)

        J_vals.append(J)
        
        gradients = compute_gradients(X_train, y_train, y_pred, m_train)
        update_parameters(parameters, gradients, learning_rate)
    return J_vals, parameters
    


            

def compute_y_pred(X, parameters):
    return np.dot(parameters["W"], X.T) + parameters["b"]


def compute_cost(y_pred, y, m):
    return (1 / (2 * m)) * np.sum((y_pred - y.T)**2)


def compute_gradients(X_train, y_train, y_pred, m_train):
    dW = (1 / m_train) * np.dot(y_pred - y_train.T, X_train)
    db = (1 / m_train) * np.sum(y_pred - y_train.T)
    return {"dW": dW, "db": db}


def update_parameters(parameters, gradients, learning_rate):
   parameters["W"] -= learning_rate * gradients["dW"]
   parameters["b"] -= learning_rate * gradients["db"]


def compute_test_cost(X_test, y_test, parameters):
    m_test = X_test.shape[0]
    y_pred = compute_y_pred(X_test, parameters)
    return compute_cost(y_pred, y_test, m_test)
    
