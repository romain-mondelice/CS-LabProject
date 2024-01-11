import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def black_scholes_call(S, K, r, t, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    return call_price

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
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
# Define the transport matrix variable
transport_matrix = cp.Variable((num_points, num_points), nonneg=True)

# Martingale constraint: expected value of next asset price equals current price
tolerance = 10
martingale_constraints = []
for i in range(num_points-1):
    expected_next_price = cp.sum(cp.multiply(points_final, transport_matrix[i, :]))
    martingale_constraints.append(cp.abs(expected_next_price - points_initial[i]) <= tolerance)

# Constraints: marginals must match the given distributions
marginal_constraints = [
    cp.sum(transport_matrix, axis=1) == initial_distribution,
    cp.sum(transport_matrix, axis=0) == final_distribution
]

# Objective: Maximize the expected payoff
payoff_matrix = call_option_payoff(points_final.reshape(-1, 1), strike_price)
objective = cp.Maximize(cp.sum(cp.multiply(payoff_matrix, transport_matrix)))

# Define and solve the problem
problem = cp.Problem(objective, martingale_constraints + marginal_constraints)
problem.solve()

# The solution gives an upper bound on the option price
option_price_upper_bound = problem.value
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
# Parameters for Black-Scholes
S = (mean_initial + mean_final) / 2  # Current price of the underlying asset
K = strike_price  # Strike price
r = 0.01  # Risk-free interest rate (1%)
t = 1    # Time to expiration (1 year)
sigma = np.mean([std_initial, std_final]) / S  # Volatility (average of the std deviations normalized by current price)

black_scholes_price = black_scholes_call(S, K, r, t, sigma)

print(f"The market black scholes price of option would be: ${black_scholes_price:.4f}")
print(f"The upper bound of option would be: ${option_price_upper_bound:.4f}")




#-------------------------------------------------------------------------------------------------------




