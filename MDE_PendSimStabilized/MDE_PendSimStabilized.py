import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Pendulum:
    def __init__(self, length_func, mass, air_resistance=False):
        self.length_func = length_func
        self.mass = mass
        self.air_resistance = air_resistance

    def equations_of_motion(self, t, state):
        theta, omega = state
        g = 9.81  # acceleration due to gravity

        if self.air_resistance:
            k = 0.1  # air resistance coefficient
            alpha = -k * omega
        else:
            alpha = 0

        # Evaluate the length function at the current time
        length = self.length_func(t)

        # Update the length-dependent term in the equation
        theta_double_dot = (-g / length) * np.sin(theta) - (alpha / (self.mass * length**2))
        return [omega, theta_double_dot]

def simulate_pendulum(length_func, mass, air_resistance=False, initial_conditions=[np.pi / 4, 0], t_span=(0, 10)):
    pendulum = Pendulum(length_func, mass, air_resistance)
    solution = solve_ivp(
        fun=pendulum.equations_of_motion,
        t_span=t_span,
        y0=initial_conditions,
        t_eval=np.linspace(t_span[0], t_span[1], 1000)
    )
    return solution

# Define a function that specifies the changing length over time
def pendulum_length(t):
    # Example: Linear increase in length over time
    return 1.0 + 0.1 * t  # Increase length by 0.1 meters per second initially

# Parameters
pendulum_mass = 0.1  # Mass remains constant
epsilon = np.deg2rad(2.8)  # Convert 2.8 degrees to radians
alpha = np.deg2rad(4.47)  # Convert 4.47 deg/s to radians/s
initial_conditions = [np.pi / 6, 0]  # Initial angle and angular velocity
simulation_time = (0, 29)  # Start and end time

# Simulation loop
length = (30.0 * 0.3048)  # Initial pendulum length converted to meters
total_time = simulation_time[1] - simulation_time[0]  # Total simulation time

while True:
    # Simulate the pendulum with the dynamic length function
    solution = simulate_pendulum(pendulum_length, pendulum_mass, initial_conditions=initial_conditions, t_span=simulation_time)

    # Prepare arrays to store real-time data for plotting
    t_data = []
    theta_data = []
    omega_data = []
    alpha_data = []
    length_data = []  # Store length values

    # Initialize variables to track the current time and length
    current_time = simulation_time[0]
    current_length = length

    # Plot the results with real-time updates
    plt.figure(figsize=(12, 8))
    plt.ion()  # Turn on interactive mode for real-time plotting

    for t_, state in zip(solution.t, solution.y.T):
        t_data.append(t_)
        theta, omega = state
        alpha = (-9.81 / current_length) * np.sin(theta)  # Calculate angular acceleration in real-time

        # Store data for real-time plotting
        theta_data.append(theta)
        omega_data.append(omega)
        alpha_data.append(alpha)
        length_data.append(current_length)  # Store length values

        # Check conditions and modify length in real-time
        if current_length > 0:  # Ensure length is positive
            if np.isclose(theta, epsilon) and omega < -alpha:
                current_length += (0.87 * 0.3048) * (total_time / 1000)  # Increase length by 0.87 ft/s
            elif theta < -epsilon and omega >= -alpha:
                current_length -= (2.79 * 0.3048) * (total_time / 1000)  # Decrease length by 2.79 ft/s
            elif np.isclose(theta, -epsilon) and omega > alpha:
                current_length += (0.87 * 0.3048) * (total_time / 1000)  # Increase length by 0.87 ft/s
            elif theta > epsilon and omega <= alpha:
                current_length -= (2.79 * 0.3048) * (total_time / 1000)  # Decrease length by 2.79 ft/s
        else:
            print("Pendulum length reached 0. Simulation ended.")
            break

        # Update the length function for the next time step
        def pendulum_length(t):
            return current_length

        # Update the time for the next time step
        current_time = t_

        # Plot real-time data
        plt.clf()

        # Plot Angle
        plt.subplot(411)
        plt.plot(t_data, theta_data)
        plt.xlabel('Time')
        plt.ylabel('Angle (radians)')
        plt.title('Pendulum Angle')
        plt.grid(True)

        # Plot Angular Velocity
        plt.subplot(412)
        plt.plot(t_data, omega_data)
        plt.xlabel('Time')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Angular Velocity')
        plt.grid(True)

        # Plot Angular Acceleration
        plt.subplot(413)
        plt.plot(t_data, alpha_data)
        plt.xlabel('Time')
        plt.ylabel('Angular Acceleration (rad/s^2)')
        plt.title('Angular Acceleration')
        plt.grid(True)

        # Plot Length (as a scatter plot)
        plt.subplot(414)
        plt.scatter(t_data, length_data, s=10, c='r')  # Scatter plot for length with larger and red dots
        plt.xlabel('Time')
        plt.ylabel('Length')
        plt.title('Pendulum Length')
        plt.grid(True)

        plt.tight_layout()
        plt.pause(0.01)  # Pause for a short time to update the plot

    plt.ioff()  # Turn off interactive mode after the simulation

    # Ask the user for input to continue or quit the simulation
    user_input = input("Enter 'q' to quit or any other key to continue: ")
    if user_input == 'q':
        break
