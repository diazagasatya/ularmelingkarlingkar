# The optimal values of m and b can be actually calculated with way less effort than doing a linear regression.
# this is just to demonstarate gradient descent
# LET'S FIND THE RELATIONSHIP OF LAND AVG TEMP AND OCEANANDLAND AVG TEMP IN 100 YEARS

from numpy import *

# DEFINE: compute error for line given points taking b, m and points as parameters)
# y = mx + b
# m is slope, b is y-intercept
# Estimate how bad our line is so we can update it every time-step (minimize the distance of our loss/error in comparison to the points)
def compute_error_for_line_given_points(b,m,points):
    # initialize total_error
    total_error = 0
    # Iteratively compute all points that we have from 0 to number of points we have in the data table
    for i in range(0,len(points)):
        # x-values horizontal and y-values are vertical
        x = points[i,0]
        y = points[i,1]
        # compute the sum of error equations
        total_error += (y - (m * x + b)) **2
    # Return the value by dividing the sum of error equation with the number of points in the data set to get the average
    return total_error / float(len(points))

# DEFINE: gradient decent
# will return the new value of b and m
# Where the graph should move in respect to the smallest error
# Gradient descent - a direction(up or down) or a tangent line into which our line will move towards to in respect to the smallest error
# partial derivative = we're calculating a partial derivative of b and partial derivative of m
# where should the graph move in respect to the smallest error
def step_gradient(b,m,points,learning_rate):
    # initiate the b and m gradient first, define the value of number of points that are going to be used
    b_gradient = 0
    m_gradient = 0
    N = len(points)
    for i in range(0,len(points)):
        x = [i,0]
        y = [i,1]
    # find the b and m gradient using the partial derivative to find the direction the line should move
        b_gradient = -(2/N) (y - (m * x + b))
        m_gradient = -(2/N) * x * (y - ((m * x) + b))
    # find the smallest possible error we will find the ideal b & m values and then plug them to the y = m x + b (we will get the line of best fit)
    new_b = b - (learning_rate * b_gradient)
    new_m = m - (learning_rate * m_gradient)
    # return the new b and m after it got negated with the new gradient
    return [new_b,new_m]


# DEFINE : gradient descent runner
# Calculating the optimal b and m value
def gradient_descent_runner(points, learning_rate, starting_b, starting_m, num_of_iterations):
    # b and m Going to start of as 0 (we have to learn these values from the gradient descent)
    b = starting_b
    m = starting_m
    
    # iterate 1000 times
    #get those optimal value from using step_gradient, using the points from data 'data.csv' use array() to convert the columns to an array
    for i in range(num_of_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    
    # Return the optimal values
    return [b,m]


# DEFINE: run()
def run():
# Pull our data set : collection of points on x-y values
    points = genfromtxt("GlobalTemperatures.csv", delimiter=",")
    # define learning rate, initial b and initial m, num of iterations,
    learning_rate = 0.000001 # Hyper-parameter : what we use as tuning knobs for how fast our model will learn (too high will never converge and too low will be too slow)
    # y = mx + b (m = the ideal slope that we want, b = y-intercept)
    b_initial = 0
    m_initial = 0
    num_of_iterations = 10000 # How many times do we want for it to calculate
    # Output the optimal value
    # Print the initial value of b and m and error
  # Output the optimal value
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(b_initial, m_initial, compute_error_for_line_given_points(b_initial, m_initial, points))
    print "Running..."
    [b, m] = gradient_descent_runner(points, learning_rate, b_initial, m_initial, num_of_iterations)
    # Print the number of iterations, value b and m and error
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_of_iterations, b, m, compute_error_for_line_given_points(b_initial, m_initial, points)))

if __name__ == '__main__':
    run()