import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define constants
g = 9.81  # Acceleration due to gravity (m/s^2)
L0 = 1.0  # Initial pendulum length (m)
omega0 = 0.0  # Initial angular velocity (rad/s)
theta0 = np.pi / 4.0  # Initial angle (radians)

# Initialize simulation parameters
current_length = L0  # Initialize current_length
total_time = 10.0  # Total simulation time (s)
SAMPLE_RATE = 1000  # Number of time steps

# Function to compute the length of the pendulum as a function of angle, angular acceleration, and time
def length_function(t, y):
    theta, omega = y
    # MATLAB-like algorithm for changing length
    if current_length > 0:
        if theta > 0 and omega > 0:
            current_length += (0.87 * 0.3048) * (total_time / SAMPLE_RATE)
        elif theta > 0 and omega < 0:
            current_length -= (2.79 * 0.3048) * (total_time / SAMPLE_RATE)
        elif theta < 0 and omega < 0:
            current_length += (0.87 * 0.3048) * (total_time / SAMPLE_RATE)
        elif theta < 0 and omega > 0:
            current_length -= (2.79 * 0.3048) * (total_time / SAMPLE_RATE)
    else:
        print("Pendulum length reached 0. Simulation ended.")
        return 0  # Return 0 to end the simulation
    
    return current_length

# Function representing the second-order differential equation
def pendulum_ode(t, y):
    theta, omega = y
    L = length_function(t, y)
    theta_dot = omega
    omega_dot = -(g / L) * np.sin(theta)
    return [theta_dot, omega_dot]

# Time span for the simulation
t_span = (0.0, total_time)

# Initial conditions
initial_conditions = [theta0, omega0]

# Solve the differential equation
sol = solve_ivp(pendulum_ode, t_span, initial_conditions, t_eval=np.linspace(t_span[0], t_span[1], SAMPLE_RATE))

# Extract results
t = sol.t
theta, omega = sol.y

# Plot the pendulum's motion
plt.figure(figsize=(8, 6))
plt.plot(t, theta, label='Angle (radians)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (radians)')
plt.title('2D Pendulum with Variable Length (MATLAB-like Length Update)')
plt.legend()
plt.grid()
plt.show()
