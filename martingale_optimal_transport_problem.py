import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt



# Define the cost function for the European call option
def call_option_payoff(s, k):
    return np.maximum(s - k, 0)

#-------------------------------------------------------------------------------------------------------
# Parameters for the initial and final distributions
mean_initial, std_initial = 100, 10
mean_final, std_final = 110, 15
num_points = 20

# Generate discretized points for the distributions
points_initial = np.linspace(mean_initial - 3*std_initial, mean_initial + 3*std_initial, num_points)
points_final = np.linspace(mean_final - 3*std_final, mean_final + 3*std_final, num_points)

# Generate the probability densities for a normal distribution at these points
initial_distribution = np.exp(-(points_initial - mean_initial)**2 / (2 * std_initial**2)) / (std_initial * np.sqrt(2 * np.pi))
final_distribution = np.exp(-(points_final - mean_final)**2 / (2 * std_final**2)) / (std_final * np.sqrt(2 * np.pi))

# Normalize to ensure they sum to 1
initial_distribution /= np.sum(initial_distribution)
final_distribution /= np.sum(final_distribution)

# Plot the initial and final distributions
plt.figure(figsize=(10, 6))
plt.plot(points_initial, initial_distribution, label='Initial Distribution', color='blue')
plt.plot(points_final, final_distribution, label='Final Distribution', color='red')
plt.title('Initial and Final Distributions of the Underlying Asset')
plt.xlabel('Asset Price')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
# Define the strike price
strike_price = (mean_initial + mean_final) / 2  # Midpoint between the two means

# Define the range for asset prices
# Extending to 4 standard deviations from the mean of each distribution
min_price = min(mean_initial - 4*std_initial, mean_final - 4*std_final)
max_price = max(mean_initial + 4*std_initial, mean_final + 4*std_final)
asset_prices = np.linspace(min_price, max_price, num_points)
#-------------------------------------------------------------------------------------------------------


# Define the transport matrix variable
transport_matrix = cp.Variable((num_points, num_points), nonneg=True)

# Martingale constraint: expected value of next asset price equals current price
tolerance = 7
martingale_constraints = []
for i in range(num_points-1):
    martingale_constraints.append(cp.abs(asset_prices[i+1] - asset_prices[i]) <= tolerance)

# Constraints: marginals must match the given distributions
marginal_constraints = [
    cp.sum(transport_matrix, axis=1) == initial_distribution,
    cp.sum(transport_matrix, axis=0) == final_distribution
]

# Objective: Maximize the expected payoff
payoff_matrix = call_option_payoff(asset_prices.reshape(-1, 1), strike_price)
objective = cp.Maximize(cp.sum(cp.multiply(payoff_matrix, transport_matrix)))

# Define and solve the problem
problem = cp.Problem(objective, martingale_constraints + marginal_constraints)
problem.solve()

# The solution gives an upper bound on the option price
option_price_upper_bound = problem.value
