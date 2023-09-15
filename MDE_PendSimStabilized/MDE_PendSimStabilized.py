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
initial_conditions = [np.pi / 6, 0]  # Initial angle (30 degrees) and angular velocity (0 by default)
simulation_time = (0, 10)  # Start and end time
length_change_rate = 0.01  # Rate of change of pendulum length (meters per second)

# Simulation loop
length = 1.0  # Initial pendulum length (meters)
while True:
    # Simulate the pendulum with the dynamic length function
    solution = simulate_pendulum(pendulum_length, pendulum_mass, initial_conditions=initial_conditions, t_span=simulation_time)

    # Prepare arrays to store real-time data for plotting
    t_data = []
    theta_data = []
    omega_data = []
    alpha_data = []

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

        # Check conditions and modify length in real-time
        if theta < -0.1 and omega >= -0.1:
            current_length -= length_change_rate  # Shorten the pendulum
        elif theta > 0.1 and omega <= 0.1:
            current_length += length_change_rate  # Lengthen the pendulum

        # Update the length function for the next time step
        def pendulum_length(t):
            return current_length

        # Update the time for the next time step
        current_time = t_

        # Plot real-time data
        plt.clf()
        plt.subplot(411)
        plt.plot(t_data, theta_data)
        plt.xlabel('Time')
        plt.ylabel('Angle (radians)')
        plt.title('Pendulum Angle')

        plt.subplot(412)
        plt.plot(t_data, omega_data)
        plt.xlabel('Time')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Angular Velocity')

        plt.subplot(413)
        plt.plot(t_data, alpha_data)
        plt.xlabel('Time')
        plt.ylabel('Angular Acceleration (rad/s^2)')
        plt.title('Angular Acceleration')

        plt.subplot(414)
        plt.plot(t_data, [pendulum_length(t_) for t_ in t_data])
        plt.xlabel('Time')
        plt.ylabel('Length')
        plt.title('Pendulum Length')

        plt.tight_layout()
        plt.pause(0.01)  # Pause for a short time to update the plot

    plt.ioff()  # Turn off interactive mode after the simulation

 


   # Theoretically (has yet to be tested) will allow us to modify the below mid simulation and continue the graph going
    #user_input = input("Enter 'q' to quit or 'c' to change parameters: ")
    #if user_input == 'q':
    #    break
    #elif user_input == 'c':
    #    pendulum_length = float(input("Enter new pendulum length: "))
    #    pendulum_mass = float(input("Enter new pendulum mass: "))
    #    air_resistance = bool(int(input("Enter 0 for no air resistance, 1 for air resistance: ")))
    #    initial_conditions[0] = float(input("Enter new initial angle (in radians): "))

