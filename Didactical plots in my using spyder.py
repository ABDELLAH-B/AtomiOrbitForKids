
"""
Created on Sat Aug  2 17:43:34 2025

@author: AYLAL
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Constants
alpha = 1 / 137
pi = np.pi

# System of equations
def system(vars):
    theta0, e = vars

    # Avoid division by zero or undefined values
    if not (0 < e < 1 and 0 < theta0 < pi):
        return [10, 10]

    # Equation 1: Time fraction
    lhs1 = theta0 + (e * np.sin(theta0)) / (1 + e * np.cos(theta0))
    rhs1 = (2 * pi * alpha) / np.sqrt(1 - e**2)
    eq1 = lhs1 - rhs1

    # Equation 2: Area fraction
    integrand = lambda theta: 1 / (1 + e * np.cos(theta))**2
    integral, _ = quad(integrand, -theta0, theta0)
    lhs2 = integral
    rhs2 = (2 * pi * alpha) / (1 - e**2)**(3/2)
    eq2 = lhs2 - rhs2

    return [eq1, eq2]

# Initial guess
initial_guess = [np.radians(5), 0.9]

# Solve system
solution = fsolve(system, initial_guess)
theta0_sol, e_sol = solution
theta0_deg = np.degrees(theta0_sol)

# Now plot the resulting ellipse and circle
a = 1  # semi-major axis
theta = np.linspace(0, 2 * np.pi, 1000)

# Elliptical orbit
r_ellipse = a * (1 - e_sol**2) / (1 + e_sol * np.cos(theta))
x_ellipse = r_ellipse * np.cos(theta)
y_ellipse = r_ellipse * np.sin(theta)

# Circular orbit (e = 0)
r_circle = np.full_like(theta, a)
x_circle = r_circle * np.cos(theta)
y_circle = r_circle * np.sin(theta)

# α zone (±theta0)
theta_zone = np.linspace(-theta0_sol, theta0_sol, 300)
r_zone = a * (1 - e_sol**2) / (1 + e_sol * np.cos(theta_zone))
x_zone = r_zone * np.cos(theta_zone)
y_zone = r_zone * np.sin(theta_zone)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_ellipse, y_ellipse, label=f"Elliptical Orbit (e ≈ {e_sol:.4f})", color='blue')
plt.plot(x_circle, y_circle, label="Circular Orbit (e = 0)", linestyle='--', color='gray')
plt.plot(x_zone, y_zone, label=f"α-Zone (θ₀ ≈ {theta0_deg:.2f}°)", color='red', linewidth=2)

plt.plot(0, 0, 'ko', label="Nucleus")
plt.gca().set_aspect('equal')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Elliptical vs Circular Orbit with α-Zone Highlighted")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

(theta0_deg, e_sol)
