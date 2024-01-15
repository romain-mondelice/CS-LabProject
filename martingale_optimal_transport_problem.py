import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

#------------------------------------------------------------------------------------------
def deterministic_discretization(density_func, support, n):
    """
    Discretize a probability density function using deterministic method.

    Parameters:
    density_func: Function to compute the probability density.
    support: Tuple of (min, max) values representing the support of the density.
    n: Number of intervals for discretization.

    Returns:
    A tuple of support points and corresponding weights (probabilities).
    """
    points = np.linspace(support[0], support[1], n + 1)
    weights = np.array([(density_func((points[i] + points[i+1]) / 2) * (points[i+1] - points[i])) for i in range(n)])
    # Normalize weights to ensure they sum up to 1
    weights /= np.sum(weights)
    return (points[:-1] + points[1:]) / 2, weights

def rho_k(x, k=3):
    if x <= 0:
        return 0
    return np.exp(-np.log(x)**2 / (2 * k**2)) / (x * np.sqrt(2 * np.pi * k**2))

def integral_tail(large_value, k=3):
    # The upper limit for integration is set to np.inf for the integration to infinity
    result, _ = quad(lambda x: rho_k(x, k), large_value, np.inf)
    return result

def lookback_option_cost(x, y, z, lambda_term):
    return np.maximum(x, y, z) - lambda_term

def asian_option_cost(x, y, z, lambda_term):
    return ((x + y + z) / 3 - lambda_term**2)

def KL_divergence(p, q):
    # Calculate the KL divergence between p and q
    return np.sum(p * np.log(p / q) - p + q, where=(p != 0))

def project_onto_C1(p, alpha):
    """
    Projects a probability matrix p onto the constraint set C1, 
    where each row sum must equal the corresponding entry in alpha.
    """
    row_sums = np.sum(p, axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    return (p / row_sums) * alpha[:, np.newaxis]

def project_onto_C2(p, beta):
    """
    Projects a probability matrix p onto the constraint set C2, 
    where each column sum must equal the corresponding entry in beta.
    """
    col_sums = np.sum(p, axis=0, keepdims=True)
    # Avoid division by zero
    col_sums[col_sums == 0] = 1
    return (p / col_sums) * beta

def project_onto_C2plus(p, xi, yj, alpha, i):
    """
    Projects a probability matrix p onto the constraint set C2+l, 
    which involves the martingale condition for a specific row i.
    """
    # Compute the expected value of y given x_i
    expected_value = np.sum(p[i, :] * yj) / alpha[i][0] if alpha[i][0] != 0 else 0
    print("expected_value >> ", expected_value)
    # Compute the scale factor for row i to satisfy the martingale property
    scale_factor = xi[i] / expected_value if expected_value != 0 else 0
    print("scale_factor >> ", scale_factor)
    # Update row i
    p_new = np.copy(p)
    p_new[i, :] *= scale_factor
    print(p_new)
    # Adjust the row to ensure the sum matches alpha_i
    row_sum = np.sum(p_new[i, :])
    if row_sum != 0:
        p_new[i, :] *= alpha[i] / row_sum
    return p_new
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
# Define the support for rho_k(x) which in practice might be truncated to a reasonable interval
epsilon = 1e-2
k = 3
large_value = 1000
integral_tail_value = integral_tail(large_value, k)
print(integral_tail_value)

support = (epsilon, large_value)  # epsilon > 0 to avoid log(0), large_value large enough to approximate infinity

# Example values for discretization
m = 100
n = 3

# Support points and weights for each marginal distribution
support_points = []
weights = []
for k in range(1, n+1):
    sp, w = deterministic_discretization(lambda x: rho_k(k, x), support, m)
    support_points.append(sp)
    weights.append(w)
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
# Bregman projection
# Initialize the joint probability matrix p
p = np.full((m, n), fill_value=1/(m*n))

# Define alpha and beta based on the weights obtained from discretization
alpha = np.array(weights)
alpha = alpha.reshape(m, n)
beta = np.array(weights)
beta = beta.reshape(m, n)

# Set the number of iterations for the Bregman projection
num_iterations = 1000
tolerance = 1e-6

# Begin the iterative projection process
for iteration in range(num_iterations):
    p_prev = p.copy()
    p = project_onto_C1(p, alpha)
    p = project_onto_C2(p, beta)
    for i in range(m):
        p = project_onto_C2plus(p, support_points[i], support_points, alpha, i)
    
    # Check for convergence using KL divergence
    divergence = KL_divergence(p, p_prev)
    if divergence < tolerance:
        print(f'Converged after {iteration} iterations with divergence = {divergence}')
        break
#------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------
# Visualization
# Compute the heat map for the optimizer p
plt.figure(figsize=(10, 5))
plt.imshow(p, cmap='hot', interpolation='nearest')
plt.colorbar(label='Probability Value')
plt.title('Heat Map of the Joint Distribution p')
plt.xlabel('Index j')
plt.ylabel('Index i')
plt.show()

#------------------------------------------------------------------------------------------


