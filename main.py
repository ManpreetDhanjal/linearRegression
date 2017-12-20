import numpy as np
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
from decimal import *

# computes the design matrix
def compute_design_matrix(X, centers, spreads):
    basis_func_outputs = np.exp(np.sum(np.matmul(X - centers, spreads) * (X - centers), axis=2)/(-2)).T
    return np.insert(basis_func_outputs, 0, 1, axis=1)

# computes closed form solution
def closed_form_sol(L2_lambda, design_matrix, output_data): 
    return np.linalg.solve(L2_lambda * np.identity(design_matrix.shape[1]) + np.matmul(design_matrix.T, design_matrix)
        ,np.matmul(design_matrix.T, output_data) ).flatten()

# computes the validation error by creating design matrix for validation data and calculating Erms
def validation_error(validation_input, validation_output, centers, spreads, Wml):
    basis_func = compute_design_matrix(validation_input, centers, spreads)
    predicted_output = np.matmul(basis_func, Wml.T)
    return get_rms_error(predicted_output, validation_output)

# calculates the error while training
def train_error(design_matrix, train_output,Wml):
    predicted_output = np.matmul(design_matrix, Wml.T)
    return get_rms_error(predicted_output, train_output)

# computes the stochastic gradient
def SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix, output_data, weights):
    N, _ = design_matrix.shape
    for epoch in range(num_epochs):
        for i in range(int(N/minibatch_size)):
            lower_bound = i * minibatch_size 
            upper_bound = min((i+1)*minibatch_size, N)
            Phi = design_matrix[lower_bound : upper_bound, :]
            t = output_data[lower_bound : upper_bound, :]
            
            error = train_error(Phi, output_data, weights)
            E_D = np.matmul((np.matmul(Phi, weights.T)-t).T, Phi)
            E = (E_D + L2_lambda * weights)/minibatch_size
            weights = weights - learning_rate * E 
            print(np.linalg.norm(E))
    return (error, weights)

# divides the data set into train(80%), validation(10%) and test(10%)
def divide_dataset(input_set, output_set):
    lower_bound = 0;
    upper_bound = int(input_set.shape[0] * 0.80)

    train_input = input_set[lower_bound:upper_bound,:]
    train_output = output_set[lower_bound:upper_bound,:]

    lower_bound = upper_bound + 1
    upper_bound = int(input_set.shape[0] * 0.90)

    validation_input = input_set[lower_bound:upper_bound, :]
    validation_output = output_set[lower_bound:upper_bound,:]

    lower_bound = upper_bound + 1
    upper_bound = input_set.shape[0]

    test_input = input_set[lower_bound:upper_bound, :]
    test_output = output_set[lower_bound:upper_bound, :]

    return (train_input, train_output, validation_input, validation_output, test_input, test_output)

# returns the variance for a given input set
def get_spreads(input_set):
    var = np.zeros((input_set.shape[1], input_set.shape[1]))
    for i in range(input_set.shape[1]):
        var[i,i] = np.var(input_set[:,i])
    return var

# calculates centers using kmeans method and spreads for those clusters
def get_centers_and_spread(input_set, M):
    kmeans = cluster.KMeans(n_clusters = M).fit(input_set)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    centers = centers[:,np.newaxis,:]
    spreads = np.zeros((M, input_set.shape[1], input_set.shape[1]))
    
    # get variance for the points in respective clusters and stack in M x D x D matrix
    for i in range(M):
        inputs = input_set[np.where(labels==i)]
        sigma = get_spreads(inputs)
        spreads[i,:,:] = np.linalg.pinv(sigma)
    return (centers, spreads)

# adds a new axis to input values
def format_input(input_set):
    return input_set[np.newaxis,:,:]

# calculate the rms error
def get_rms_error(predicted_output, output_set):
    NoOfTestingInp = output_set.shape[0]
    E_D = np.zeros([NoOfTestingInp,1])
    predicted_output = predicted_output[:,np.newaxis]
    E_D = np.square(output_set - predicted_output)
    E = 0.5 * E_D.sum()
    return (2*E/NoOfTestingInp)**0.5

# train using early stop in stochastic gradient
def check_early_stop(design_matrix, output_data, validation_input_stack, validation_output, centers, spreads):
    
    M = centers.shape[0]
    # number of times to check if the validation error is worsening
    patience_num = 10
    total_steps = 0
    Wml = 0
    optimal_steps = 0
    count = 0
    
    # randomly initiate weights
    weights = np.random.rand(1,M+1)
    v_min = float('inf')
    
    num_epochs = 10
    learning_rate = 0.1
    L2_lambda = 0.1
    i=0
    minibatch_size = design_matrix.shape[0]

    # return the optimum values when count exceeds the patience number
    while count < patience_num:
        # keeps track of number of epochs
        total_steps = total_steps + num_epochs

        # get the weights by running num_epoch times and get the validation error
        _, weights = SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix, output_data, weights)
        v_error = validation_error(validation_input_stack, validation_output, centers, spreads, weights)
        
        # save weights and optimal steps when minimum validation error is obtained
        if v_error < v_min:
            count = 0
            Wml = weights
            optimal_steps = total_steps
            v_min = v_error
        else:
            count = count+1
    return optimal_steps, Wml.flatten()

# train the closed form solution
def train_closed_form(input_set, output_set, L2_lambda):
    train_input, train_output, validation_input, validation_output, test_input, test_output = divide_dataset(input_set, output_set)
    
    M = 2
    centers_min = np.zeros(1)
    error_arr1 = np.zeros(24)
    spreads_min = np.zeros(1)
    weights = np.zeros(1)
    v_min = float('inf')
    design_matrix = np.zeros(1)
    
    train_input_stack = format_input(train_input)
    validation_input_stack = format_input(validation_input)
    
    i=0

    # grid search for M
    # start with M = 4 and increment by 2 in each iteration 
    while M < 50:
        M = M + 2;
        
        # get centers using k-means
        centers, spreads = get_centers_and_spread(train_input, M)
        design_matrix = compute_design_matrix(train_input_stack, centers, spreads)

        # compute the weights and calculate validation error
        Wml = closed_form_sol(L2_lambda, design_matrix, train_output)
        cur_validation_error = validation_error(validation_input_stack, validation_output, centers, spreads, Wml)

        # save centers and spreads when we found the minimum valdiation error
        if cur_validation_error < v_min:
            centers_min = centers
            v_min = cur_validation_error
            spreads_min = spreads
        error_arr1[i] = cur_validation_error
        i=i+1
    
    design_matrix = compute_design_matrix(train_input_stack, centers_min, spreads_min)
    Wml = closed_form_sol(L2_lambda, design_matrix, train_output)
    return (centers_min, spreads_min, Wml,error_arr1)

# test the closed form solution by giving test input and outputs as parameters
def test_closed_form(test_input, test_output, centers_min, Wml, spreads):
    test_input_stack = format_input(test_input)
    test_design_matrix = compute_design_matrix(test_input_stack, centers_min, spreads)
    predicted_output = np.matmul(test_design_matrix, Wml.T)
    return predicted_output

# feed the whole dataset input, dataset output and centers and spreads to train using stochastic gradient descent
def train_SGD(input_set, output_set, centers, spreads):
    train_input, train_output, validation_input, validation_output, test_input, test_output = divide_dataset(
        input_set, output_set)
    
    train_input_stack = format_input(train_input)
    validation_input_stack = format_input(validation_input)
    
    design_matrix = compute_design_matrix(train_input_stack, centers, spreads)
    
    number, weights = check_early_stop(
        design_matrix, train_output, validation_input_stack, validation_output, centers, spreads)
    
    return (number, weights)

# input the test inputs, test outputs, model parameters to predict the output
def test_SGD(test_input, test_output, centers, Wml, spreads):
    test_input_stack = format_input(test_input)
    test_design_matrix = compute_design_matrix(test_input_stack, centers, spreads)
    predicted_output = np.matmul(test_design_matrix, Wml.T)
    return predicted_output

def main():
    # load the input and output data

    # call train_closed_form() to train the linear regression model
    # call test_closed_form() to test for new input values

    # call train_SGD() to train using stochastic gradient descent method
    # call test_SGD() to test for new input values

    input_data = np.loadtxt('/Users/manpreetdhanjal/Downloads/input.csv', delimiter=',')
    output_data = np.loadtxt('/Users/manpreetdhanjal/Downloads/output.csv', delimiter=',').reshape([-1,1])

    letor_input = np.genfromtxt('/Users/manpreetdhanjal/Downloads/Querylevelnorm_X.csv', delimiter=',')
    letor_output = np.genfromtxt('/Users/manpreetdhanjal/Downloads/Querylevelnorm_t.csv').reshape([-1,1])

    L2_lambda = 0.1
    ### synthetic data
    train_input, train_output, validation_input, validation_output, test_input, test_output = divide_dataset(input_data, output_data)
    # centers, spreads, Wml, error_arr = train_closed_form(input_data, output_data, L2_lambda)
    # predicted_output = test_closed_form(test_input, test_output, centers, Wml, spreads)

    # use centers and spreads obtained from train_closed_form
    # you can also randomly inititalise your own centers and spreads 
    #M = 34
    #centers, spreads = get_centers_and_spread(train_input, M)
    #number, weights = train_SGD(input_data, output_data, centers, spreads)
    #predicted_output = test_SGD(test_input, test_output, centers, weights, spreads))

    ### LeTOR data
    train_input, train_output, validation_input, validation_output, test_input, test_output = divide_dataset(letor_input, letor_output)
    # centers, spreads, Wml, error_arr = train_closed_form(letor_input, letor_output, L2_lambda)
    # predicted_output = test_closed_form(test_input, test_output, centers, Wml, spreads)
    M = 32
    centers, spreads = get_centers_and_spread(train_input, M)
    number, weights = train_SGD(letor_input, letor_output, centers, spreads)
    predicted_output = test_SGD(test_input, test_output, centers, weights, spreads)

    plt.plot(test_output)
    plt.plot(predicted_output)
    plt.show()

if __name__ == '__main__':main()