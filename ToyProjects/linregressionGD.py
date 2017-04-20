# This is from a live tutorial with Siraj
# https://github.com/llSourcell/linear_regression_live


#%matplotlib inline
from numpy import *
import matplotlib.pyplot as plt

def compute_error_for_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[0, 1]
        totalError += (y - (m * x + b)) **2
        return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    #gradient descent
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


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_interations):
    b = starting_b
    m = starting_m

    for i in range(num_interations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b,m]

def run():
    points = genfromtxt('data_linregressionGD.csv', delimiter=',')
    #hyperparameter
    learning_rate = 0.0001
    #y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print(b)
    print(m)
    plt.scatter(points[:,0], points[:,1], c='b', alpha=0.4)

    pointsX_max = points[:,0].max(axis=0)
    pointsX_min = points[:,0].min(axis=0)
    xticks = linspace(pointsX_min, pointsX_max, 100)
    plt.plot(xticks, m*xticks+b)

    for i in range(5):
        newPointX = random.uniform(pointsX_min, pointsX_max)
        newPointY = m*newPointX+b
        plt.scatter(newPointX, newPointY, c='r', alpha=0.4)

if __name__ == '__main__':
    run()
