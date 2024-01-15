import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm
from scipy.stats import lognorm

#------------------------------------------------------------------------------------------
# This function discretizes the support of the density function into n intervals
def deterministic_discretization(density_func, support, n, k):
    points = np.linspace(support[0], support[1], n+1)
    weights = np.zeros(n)
    for i in range(n):
        weights[i], _ = quad(density_func, points[i], points[i+1], args=(k))
    weights /= np.sum(weights) # Normalize the weights
    return (points[:-1] + points[1:])/2, weights

def rho_k(x, k):
    if x <= 0:
        return 0
    else:
        return np.exp(-(np.log(x) + k**2/2)**2 / (2*k**2)) / (x * k * np.sqrt(2*np.pi))

def lookback_option_cost(x, y, z, lambda_term):
    return np.maximum(x, y, z) - lambda_term

def asian_option_cost(x, y, z, lambda_term):
    return ((x + y + z) / 3 - lambda_term**2)

def KL_divergence(p, q):
    # Calculate the KL divergence between p and q
    return np.sum(p * np.log(p / q) - p + q, where=(p != 0))

def project_onto_C1(p, alpha):
    alpha = np.asarray(alpha)
    for i in range(p.shape[0]):
        row_sum = np.sum(p[i, :])
        if row_sum > 0:
            p[i, :] *= (alpha[i] / row_sum)
    return p

def project_onto_C2(p, beta, k):
    beta_k = np.asarray(beta[k])
    for j in range(p.shape[1]):
        col_sum = np.sum(p[:, j])
        if col_sum > 0:
            p[:, j] *= (beta_k[j] / col_sum)
    return p

def project_onto_C2plus(p, x, y, alpha):
    p_new = np.copy(p)
    for k in range(len(x)):
        expected_value = np.dot(p_new[k, :], y)
        adjustment_factor = (alpha[k] * x[k] / expected_value) if expected_value != 0 else 0
        p_new[k, :] *= adjustment_factor
        row_sum = np.sum(p_new[k, :])
        if row_sum != 0:
            p_new[k, :] /= row_sum
    return p_new

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
# Define the support for rho_k(x) which in practice might be truncated to a reasonable interval
k_values = [1, 2, 3]
n = 100  # Number of intervals
support = (0.01, 10)  # Truncated support for the density function

# Example values for discretization
m = 100
n = 3 # Trading dates

discretizations = {}
for k in k_values:
    points, weights = deterministic_discretization(rho_k, support, n, k)
    discretizations[k] = (points, weights)
    
for k, (points, weights) in discretizations.items():
    print(f"Discretization for k={k}:")
    print(f"Points: {points}")
    print(f"Weights: {weights}\n")
    
plt.figure(figsize=(14, 6))
for k, (points, weights) in discretizations.items():
    plt.plot(points, weights, label=f'k={k}', marker='o')

plt.title('Discretized Density Functions for Different k')
plt.xlabel('x')
plt.ylabel('Probability Weights')
plt.legend()
plt.grid(True)
plt.show()
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
# Bregman projection
# Assuming k_values corresponds to different times or conditions
n = 100

# Define the parameters for the normal distribution
mu = np.mean(support)
sigma = (support[1] - support[0]) / 10
points = np.linspace(support[0], support[1], n)
pdf_values = norm.pdf(points, mu, sigma)
alpha = pdf_values / np.sum(pdf_values)

# Beta is a dictionary where each trading date k has an associated distribution
mus = [np.log(0.9), np.log(1.0), np.log(1.1)]
sigma = 0.25

# Initialize beta as an empty dictionary
beta = {}
for k, mu in zip(k_values, mus):
    points = np.linspace(support[0], support[1], n)
    pdf_values = lognorm.pdf(points, sigma, scale=np.exp(mu))
    beta[k] = pdf_values / np.sum(pdf_values)

# Probability matrix p is an n x n matrix
p = np.ones((n, n)) / (n * n)

x_points = np.linspace(support[0], support[1], n)
alpha_k = np.ones(m)

# Begin the iterative projection process with the corrected functions
num_iterations = 1000
tolerance = 1e-6
for k in range(3):
    print(f"Trading day {k+1}...")
    p_prev = p.copy()
    p = project_onto_C1(p, alpha)
    p = project_onto_C2(p, beta, k+1)
    for iteration in range(num_iterations):
        p = project_onto_C2plus(p, x_points, alpha_k, alpha)
        divergence = KL_divergence(p, p_prev)
        if divergence < tolerance:
            print(f'Converged after {iteration+1} iterations with divergence = {divergence}')
            break
    else:
        print(f'Did not converge after {num_iterations} iterations.')
#------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------
# Visualization
# Compute the heat map for the optimizer p
plt.imshow(p, cmap='Blues', aspect='auto')
plt.colorbar(label='Probabilities')
plt.xlabel('S2')
plt.ylabel('S1')
plt.title('Joint Distribution of S1 and S2')
plt.tight_layout()  # Adjusts plot so that they fit into the figure area.
plt.show()
#------------------------------------------------------------------------------------------


