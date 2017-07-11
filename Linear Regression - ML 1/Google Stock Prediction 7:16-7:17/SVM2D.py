"""
	Machine Learning - Creating a Support Vector Machine
		-	What is Machine Learning? Creating an objective function through modifications of coefficients
			within. 
		-	How do we find the optimal coefficients? We need to teach the computer to find the optimal 
			weights(coefficients) by minimizing errors. 
		-	But what is the error? The error that we are going to use is the sum of squarred error
		-	How do we teach the computer to find the objective function?
			We need to calculate the error for each point and use gradient descent to find the local
			minima of the error and move the equation to its optimal position
		-	How do we move the function? We use the partial derivative to find the optimal coefficients
		-	What is the objective? We are going to create a line of best fit for GOOGLE STOCKS july16/17

Let's get to it!
"""
#Import numpy for math calculations, pandas for reading a csv file and matplot for graphing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Step 1 let's define the data

#Input data as the form of(X,Y)
dataset = pd.read_csv('GOOGL.csv')
data = dataset.as_matrix()

#how many values in the data?
numInstance = data.shape[0]

#Let's find the line of best fit between the two variables with GRADIENT DESCENT
#By doing so, in the end we will be able to somewhat predict google's stock based upon it's volume
#Line of best fit: Y=mx+c where m is the slope and c is the y-intercept
#We are going to find the optimal m and b that will create the line, thus predicting will be as simple
#as plugging in the x value thus getting the y as the prediction. Vice Versa

#Sum of squared error: the current m and b don't know which is our current fitting line
#The real data, that corresponds to 'target' on the equation of the image
#this function will return the error
#this is used to measure the quality of the fitting line, but will not use to the optimization process
#of Gradient Descent.
#This is the magnitude of the error; which are the distance between the points and the line
def sumOfErrorEquation(m,b,data):
	totalError=0.0

	for i in range(numInstance):
		#define the x and y value(y = the target point that you want in comparison to your current)
		adjustedClose = data[i,0]
		volume = data[i,1]
		#Follow the sum of squared error equation programmatically
		totalError += (volume - (m * adjustedClose + b)) **2

	return totalError/numInstance

#Let's create the gradient descent
#Create the initial line, update the line parameters in a way that makes the error SMALLER
#it implies that we need a direction, a way to descent the error valley. 
#compute the partial derivative of m and b. Get the direction that decreases the gradient to (0 error)
#if the derivative of m/b is -ve that means the optimal gradient must be increased in order to reach
#its optimal gradient. And if the derivative of m/b is positive, this means that the optimal gradient
#should be decreases from it to reach its optimal number (local minima).
#update m and b accordingly
#What do we need? the current 'm' and 'b' to know which is our current fitting line. the real data = target
#it will return the new 'BETTER' and updated m and b
def gradientDescent(m,b,data,learningRate):
	N = numInstance
	m_grad = 0
	b_grad = 0

	for i in range(numInstance):
		adjustedClose = data[i,0]
		volume = data[i,1]
		#Follow the partial derivative equation programmatically
		dm = -((2/N) * adjustedClose * (volume - (m * adjustedClose + b))) 
		db = -((2/N) * (volume - (m * adjustedClose + b)))

	#Update the current m and b
	m_grad += dm
	b_grad += db

	#Set the new 'BETTER' updated m and b
	new_m = m - (learningRate * m_grad)
	new_b = b - (learningRate * b_grad)

	return new_m,new_b

#After defining the gradient descent, we have to train the model
#In order to train the model we need to call the gradient descent method N number of times
#Until the graph reach its local minima and find the optimal function
#Define the number of iterations that you want the model to be train to to reach its optimal f(x)
#Every time-step it will modify m and b until it reaches its minimal gradient descent
#Display each timestep and print it to the console to see the error decreasing over time
def gradientDescentNSteps(starting_m,starting_b,data,iteration,learningRate):
	print('Starting line: y = %.6fx + %.6f - Error: %.6f\n' % 
		(starting_m,starting_b,sumOfErrorEquation(starting_m,starting_m,data)))
	m = starting_m
	b = starting_b
	display_freq = iteration
	#train the model how many steps to get the optimal m and b
	for i in range(iteration):
		m,b = gradientDescent(m,b,data,learningRate)
		if(i % display_freq >= 0):
			sse = sumOfErrorEquation(m,b,data)
			print('At step %d - Line: y = %.2fx + %.2f - Error: %.2f' %(i+1,m,b,sse))

	print('\nBest line: y = %.2fx + %.2f - Error: %.2f' %(m,b,sse))
	return m,b

#Test the model to find the optimal function
def main():
	#Lets do 10 steps of Gradient Descent from main
	M_STARTING = 0
	B_STARTING = 0
	ITERATIONS = 3000
	learningRate = 0.000001

	m_best,b_best = gradientDescentNSteps(M_STARTING,B_STARTING,data,ITERATIONS,learningRate)

	m=m_best
	b=b_best

	#Let's plot the data using matplotlib
	fig = plt.figure(figsize=(10,5))
	ax = fig.add_subplot(111)
	ax.set_title('Google Adj.Close vs Volume (July16-17)')
	ax.scatter(x=data[:,0],y=data[:,1],label='Data')
	plt.plot(data[:,0],m*data[:,0] + b, color = 'red',label='Out Fitting Line')
	ax.set_xlabel('Adj.Close')
	ax.set_ylabel('Volume')
	ax.legend(loc='best')
	plt.show()

#main
if __name__ == "__main__":
    main()

