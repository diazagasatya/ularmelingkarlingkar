'''
Created on June 16, 2017
Linear Regression using Gradient Descent finding the line of best fit of CO2 emmisions from 1958 - 2017
@author: diazagasatya

What is a gradient? 
    is the rise over run of points = slope 
'''
from numpy import *

# y = mx + b
# m is slope, b is y-int
def compute_error_for_line_given_points(b,m,points):
    # Estimate how bad our line is so we can update it every time-step (minimize the distance of our loss/error in comparison to the points) 
    totalError = 0
    #iteratively compute all points that we have from 0 to number of points we have in the data table)
    for i in range(0, len(points)):
        # x-values horizontal and y-values are vertical
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (m * x + b)) **2
        # divided by the total number of points that we have to compute 1 number
    return totalError / float(len(points))
        
def step_gradient(b_current, m_current, points, learningRate):
    # Gradient descent - a direction(up or down) or a tangent line into which our line will move towards to in respect to the smallest error
    # find the smallest possible error we will find the ideal b & m values and then plug them to the y = m x + b (we will get the line of best fit)
    # partial derivative = we're calculating a partial derivative of b and partial derivative of m
    # where should the graph move in respect to the smallest error
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]
                       
    
# Calculating the optimal b and m value
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # Going to start of as 0 (we have to learn these values from the gradient descent)
    b = starting_b
    m = starting_m
    
    # iterate 1000 times
    #get those optimal value from using step_gradient, using the points from data 'data.csv' use array() to convert the columns to an array
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    
    # Return the optimal values
    return [b,m]


# Pull our data set : collection of points on x-y values
def run():
    points = genfromtxt("data.csv", delimiter=",")
    # Hyper-parameter : what we use as tuning knobs for how fast our model will learn (too high will never converge and too low will be too slow)
    learning_rate = 0.0000001
    
    # y = mx + b (m = the ideal slope that we want, b = y-intercept)
    initial_b = 0
    initial_m = 0
    
    # How many times do we want for it to calculate
    num_iterations = 5000
    
    # Output the optimal value
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    print "Running..."
    
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))
    
if __name__ == '__main__':
    run()