import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Pendulum:
    def __init__(self, length, mass, air_resistance=False):
        self.length = length
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

        theta_double_dot = (-g / self.length) * np.sin(theta) - (alpha / (self.mass * self.length**2))
        return [omega, theta_double_dot]

def simulate_pendulum(length, mass, air_resistance=False, initial_conditions=[np.pi / 4, 0], t_span=(0, 10)):
    pendulum = Pendulum(length, mass, air_resistance)
    solution = solve_ivp(
        fun=pendulum.equations_of_motion,
        t_span=t_span,
        y0=initial_conditions,
        t_eval=np.linspace(t_span[0], t_span[1], 1000)
    )
    return solution

def plot_pendulum_results(solution):
    t = solution.t
    theta = solution.y[0]
    omega = solution.y[1]
    alpha = np.gradient(omega, t)  # Calculate angular acceleration
    length = pendulum_length + pendulum_length * (1 - np.cos(theta))

    plt.figure(figsize=(12, 8))

    #Plot Angle
    plt.subplot(411)
    plt.plot(t, theta)
    plt.xlabel('Time')
    plt.ylabel('Angle (radians)')
    plt.title('Pendulum Angle')

    #Plot Angle Velocity
    plt.subplot(412)
    plt.plot(t, omega)
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocity')

    #Plot Angle Acceleration
    plt.subplot(413)
    plt.plot(t, alpha)
    plt.xlabel('Time')
    plt.ylabel('Angular Acceleration (rad/s^2)')
    plt.title('Angular Acceleration')

    #Plot Pend Length
    plt.subplot(414)
    plt.plot(t, length)
    plt.xlabel('Time')
    plt.ylabel('Length')
    plt.title('Pendulum Length')

    plt.tight_layout()
    plt.show()

#Initial Parameters
pendulum_length = 1.0 #is in meters (m)
pendulum_mass = 0.1 #is in killograms (Kg)
initial_conditions = [np.pi / 4, 0]  # Initial angle and angular velocity
simulation_time = (0, 10)  # Start and end time

# Simulation loop
while True:
    # Simulate the pendulum
    solution = simulate_pendulum(pendulum_length, pendulum_mass, initial_conditions=initial_conditions, t_span=simulation_time)

    # Plot the results
    plot_pendulum_results(solution)

    # Algorithm to modify pendulum length 
    # Replace this with desired algorithm

    angular_acceleration = np.gradient(solution.y[1], solution.t)  # Calculate angular acceleration

    #test algorithm (key note: have to still work out how to access angular acceleration)
    if solution.y[1][-1] < -1.0:
        pendulum_length *= 1.1  # Increase length by 10% if angular velocity is less than -1.0
    elif solution.y[1][-1] > 1.0:
        pendulum_length *= 0.9  # Decrease length by 10% if angular velocity is greater than 1.0
        
    #the data is accessed via a 2D array
    #solution.y[0] coresponds to Angle
    #solution.y[1] corresponds to Angular Velocity
    #solution.y[1][-1] corresponds to final angular velocity
    #[-1] corresponds to the final entry (or the entry closest to the most recent time)
    #same deal, just now to call for angular acceleration it is angular_acceleration[-1]

    # Update initial conditions for the next cycle
    initial_conditions = [solution.y[0][-1], solution.y[1][-1]]

    # Theoretically (has yet to be tested) will allow us to modify the below mid simulation and continue the graph going
    user_input = input("Enter 'q' to quit or 'c' to change parameters: ")
    if user_input == 'q':
        break
    elif user_input == 'c':
        pendulum_length = float(input("Enter new pendulum length: "))
        pendulum_mass = float(input("Enter new pendulum mass: "))
        air_resistance = bool(int(input("Enter 0 for no air resistance, 1 for air resistance: ")))
        initial_conditions[0] = float(input("Enter new initial angle (in radians): "))

