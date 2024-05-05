import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lorenz(t, Y, sigma, rho, beta):
    """
    defining the lorenz equations
    """
    x,y,z = Y
    dxdt = sigma * (y-x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# common parameter values
sigma = 10
rho = 28
beta = 8/3

# initial conditions
initials = [1, 1, 1]
t_span = (0, 50) 

sol = solve_ivp(lorenz, t_span, initials, args=(sigma, rho, beta), dense_output=True) # dense_output computes continuous solution

t = np.linspace(0, 50, 10000)
x,y,z = sol.sol(t)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x,y,z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz System')
plt.show()





