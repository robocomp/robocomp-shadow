import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting

# Define the optimum (nearly the final center)
x_opt, y_opt = 1.0, 0.11

# Define the error function (a quadratic, paraboloid)
def error_function(x, y):
    return (x - x_opt)**2 + (y - y_opt)**2

# Create a mesh grid covering the region of interest.
x_vals = np.linspace(-110, 10, 400)
y_vals = np.linspace(-10, 110, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = error_function(X, Y)

# Provided trajectory data (centers)
centers = np.array([
    [-100.0,    100.0],
    [ -49.62,   49.9985],
    [-24.4299,  24.9971],
    [-11.8348,  12.4953],
    [ -5.53707,  6.24189],
    [ -2.38767,  3.11032],
    [ -0.811287, 1.53468],
    [ -0.0171305,0.727028],
    [  0.402647, 0.286028],
    [  0.691534, 0.0237682],
    [  0.9613,  -0.0271282],
    [  1.01293,  0.0892517],
    [  0.990277, 0.113356],
    [  1.00831,  0.108203],
    [  0.993064, 0.110049],
    [  1.00585,  0.109617],
    [  0.995071, 0.109842],
    [  1.00416,  0.109836],
    [  0.996489, 0.109883],
    [  1.00297,  0.109901]
])

# Provided error values at each center
error_values = np.array([
    20153.1,
     5038.29,
     1259.6,
      314.924,
       78.7559,
       19.7151,
        4.95695,
        1.27175,
        0.358949,
        0.141908,
        0.0728552,
        0.0228923,
        0.0161333,
        0.0174394,
        0.0169355,
        0.0170341,
        0.0169654,
        0.01696,
        0.0169427,
        0.0169346
])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the error surface.
surf = ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis', rstride=10, cstride=10)

# Plot the minimisation trajectory as a red line with markers.
ax.plot(centers[:, 0], centers[:, 1], error_values, 'ro-', markersize=5, label='Trajectory')

# (Optional) If you wish to include arrows, reduce their frequency or comment out the following block.
# for i in range(len(centers) - 1):
#     dx = centers[i+1, 0] - centers[i, 0]
#     dy = centers[i+1, 1] - centers[i, 1]
#     dz = error_values[i+1] - error_values[i]
#     ax.quiver(centers[i, 0], centers[i, 1], error_values[i],
#               dx, dy, dz, arrow_length_ratio=0.1, color='k')

# Mark the final center (the optimum) as a blue point.
ax.scatter(centers[-1, 0], centers[-1, 1], error_values[-1], color='b', s=50, label='Optimum')

# Label axes and add a title.
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Error')
ax.set_title('3D Error Surface with Minimisation Trajectory')
ax.legend()

plt.show()

