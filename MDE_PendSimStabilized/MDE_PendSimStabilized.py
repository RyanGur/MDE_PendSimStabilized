import numpy as np
from scipy.integrate import solve_ivp
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import itertools

SAMPLE_RATE = 1000

class Pendulum:
    def __init__(self, length_func, mass, air_resistance=False):
        self.length_func = length_func
        self.mass = mass
        self.air_resistance = air_resistance

    def equations_of_motion(self, t, state):
        theta, theta_dot = state
        g = 9.81  # acceleration due to gravity

        if self.air_resistance:
            k = 0.1  # air resistance coefficient
            alpha = -k * theta_dot
        else:
            alpha = 0

        # Evaluate the length function at the current time
        length = self.length_func(t)

        # Calculate the second derivative of theta (theta_double_dot)
        theta_double_dot = (-g / length) * np.sin(theta) + (alpha / (self.mass * length**2))

        return [theta_dot, theta_double_dot]

def simulate_pendulum(length_func, mass, air_resistance=False, initial_conditions=[np.pi / 4, 0], t_span=(0, 10)):
    pendulum = Pendulum(length_func, mass, air_resistance)
    solution = solve_ivp(
        fun=pendulum.equations_of_motion,
        t_span=t_span,
        y0=initial_conditions,
        t_eval=np.linspace(t_span[0], t_span[1], SAMPLE_RATE)
    )
    return solution

# Define a function that specifies the changing length over time
def pendulum_length(t):
    # Example: Linear increase in length over time
    return 1.0 + 0.1 * t  # Increase length by 0.1 meters per second initially

# Parameters
pendulum_mass = 0.1  # Mass remains constant
initial_conditions = [np.pi / 12, 0]  # Initial angle and angular velocity
simulation_time = (0, 29)  # Start and end time
total_time = simulation_time[1] - simulation_time[0]  # Total simulation time

# Define parameter search space using skopt's Real and Integer dimensions
param_space = [
    Real(0.5, 5.0, name='epsilon1'),
    Real(0.5, 5.0, name='epsilon2'),
    Real(0.1, 10.0, name='alpha'),
    Real(0.1, 2.0, name='increase_length'),
    Real(0.1, 2.0, name='decrease_length')
]

# Initialize Bayesian optimization
opt = BayesSearchCV(
    simulate_pendulum,
    param_space,
    n_iter=50,  # Number of optimization steps
    random_state=0,
    n_jobs=-1  # Use all available CPU cores
)

@use_named_args(param_space)
def objective(**params):
    print("Trying parameters:", params)
    epsilon1 = params['epsilon1']
    epsilon2 = params['epsilon2']
    alpha = params['alpha']
    increase_length = params['increase_length']
    decrease_length = params['decrease_length']

    # Simulation loop
    length = (30.0 * 0.3048)  # Initial pendulum length converted to meters

    # Simulate the pendulum with the dynamic length function
    solution = simulate_pendulum(pendulum_length, pendulum_mass, initial_conditions=initial_conditions, t_span=simulation_time)

    # Initialize variables to track the angle peaks and length change
    max_angle1 = None
    min_angle1 = None
    max_angle2 = None
    min_angle2 = None
    current_length = length
    success = True

    # Check conditions and modify length in real-time
    for t_, state in zip(solution.t, solution.y.T):
        theta, omega = state

        if max_angle1 is None or theta > max_angle1:
            max_angle1 = theta
        if min_angle1 is None or theta < min_angle1:
            min_angle1 = theta

        if current_length > 0 and current_length >= length:  # Ensure length is positive and not increasing
            if theta < epsilon1 and omega < -alpha:
                current_length += increase_length * (total_time / SAMPLE_RATE)  # Increase length
            elif theta < -epsilon2 and omega >= -alpha:
                current_length -= decrease_length * (total_time / SAMPLE_RATE)  # Decrease length
            elif theta > -epsilon1 and omega > alpha:
                current_length += increase_length * (total_time / SAMPLE_RATE)  # Increase length
            elif theta > epsilon2 and omega <= alpha:
                current_length -= decrease_length * (total_time / SAMPLE_RATE)  # Decrease length
            else:
                current_length -= decrease_length * (total_time / SAMPLE_RATE)  # Default: Decrease length
        else:
            print("Pendulum length reached 0 or increased. Simulation ended.")
            success = False
            break

        # Update the length function for the next time step
        def pendulum_length(t):
            return current_length

        # Update the time for the next time step
        current_time = t_

    # Calculate the difference between the absolute values of the first and second peaks
    if success:
        if max_angle1 is not None and max_angle2 is not None:
            peak_diff = abs(max_angle1) - abs(max_angle2)
            return -peak_diff  # We want to maximize the peak difference (negative because it's a minimization problem)

    return np.inf  # Return a large value for unsuccessful simulations

# Perform Bayesian optimization
opt.fit([0], [0])  # Initialize the optimizer

# Find the best parameters
best_params = opt.best_params_
best_peak_diff = -opt.best_value_  # Convert back to positive since we maximized the negative peak difference

# Save the best parameters to a text file
if best_params is not None:
    with open("best_parameters.txt", "w") as file:
        file.write(f"Best Parameters: epsilon1 = {best_params['epsilon1']}, epsilon2 = {best_params['epsilon2']}, "
                   f"alpha = {best_params['alpha']}, increase_length = {best_params['increase_length']}, "
                   f"decrease_length = {best_params['decrease_length']}, Best Peak Difference = {best_peak_diff}")
        print(f"Best Parameters: epsilon1 = {best_params['epsilon1']}, epsilon2 = {best_params['epsilon2']}, "
              f"alpha = {best_params['alpha']}, increase_length = {best_params['increase_length']}, "
              f"decrease_length = {best_params['decrease_length']}, Best Peak Difference = {best_peak_diff}")
else:
    print("No suitable parameters found.")
