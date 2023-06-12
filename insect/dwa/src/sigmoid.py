import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the sigmoid function with two parameters: (x-x0)/k
def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k*(x-x0)))

# Define your data points

# FOR minus range (adv vel)
xdata = np.array([0.0,  0.5,  1.0, 1.5, 2.0, 2.5, 3.0])
ydata = np.array([0.01, 0.05, 0.2, 0.3, 0.5, 0.7, 1])

# FOR minus/plus range (side vel)
xdata = np.array([-2.0, -1.5, -1.0, -0.5, 0.0,  0.5,  1.0, 1.5, 2.0])
ydata = np.array([0.00, 0.05, 0.1, 0.4, 0.5, 0.6, 0.8, 0.98, 1])


# Provide initial guess for x0 and k
p0 = [np.median(xdata), 1]

# Fit the sigmoid function to the data
popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')

# Print the optimal parameters
print("x0, k = ", popt)

# Plot the original data
plt.scatter(xdata, ydata, label='data')

# Plot the fitted sigmoid function
x = np.linspace(-6, 6, 50)
y = sigmoid(x, *popt)
plt.plot(x, 2*y-1, label='fit')
plt.legend()
plt.show()
