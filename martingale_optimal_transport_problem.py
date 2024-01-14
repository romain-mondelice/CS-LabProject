import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------
def cost_function(x):
    return np.exp(-x)

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
    expected_value = np.sum(p[i, :] * yj) / alpha[i] if alpha[i] != 0 else 0
    # Compute the scale factor for row i to satisfy the martingale property
    scale_factor = xi[i] / expected_value if expected_value != 0 else 0
    # Update row i
    p_new = np.copy(p)
    p_new[i, :] *= scale_factor
    # Adjust the row to ensure the sum matches alpha_i
    row_sum = np.sum(p_new[i, :])
    if row_sum != 0:
        p_new[i, :] *= alpha[i] / row_sum
    return p_new
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
# Define 2 discrete probability distributions
# Since it's a test purpose and that we don't have any specific values for alpha_i, beta_j, delta_xi, and delta_yj are provided,
# we will generate two discrete probability distributions with random values for demonstration purposes.

# For simplicity, we'll assume m = n = 5 (5-point distributions)
m = 100  # Number of points in the first distribution
n = 100  # Number of points in the second distribution

# Generate random weights (alpha_i and beta_j) for the two distributions
# and normalize them so they sum up to 1 to represent probabilities
alpha = np.random.rand(m)
alpha /= alpha.sum()  # Normalize to get probabilities that sum to 1

beta = np.random.rand(n)
beta /= beta.sum()  # Normalize to get probabilities that sum to 1

# Generate random support points (xi and yj) for the two distributions
xi = np.random.rand(m)
yj = np.random.rand(n)

# Create the discrete probability distributions
mu = [(a, x) for a, x in zip(alpha, xi)]
nu = [(b, y) for b, y in zip(beta, yj)]
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
# Initialize your probability matrix p, for example with uniform distribution
p = np.full((m, n), fill_value=1/(m*n))

# Set the number of iterations for the Bregman projection
num_iterations = 100

# Begin the iterative projection process
tolerance=1e-6
p_prev = p
for iteration in range(num_iterations):
    print("Iteration: ", iteration)
    p = project_onto_C1(p, alpha)
    p = project_onto_C2(p, beta)
    for i in range(m):
        p = project_onto_C2plus(p, xi, yj, alpha, i)
    if KL_divergence(p, p_prev) < tolerance:
        break
    p_prev = p
#------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------
# Visualization
P_n_i = np.array([np.sum(p[i, :] * cost_function((i - np.arange(n)) / n)) for i in range(m)])

# Plot the expected values P_n,i
plt.figure(figsize=(10, 5))
plt.plot(P_n_i, label='P_n,i')
plt.xlabel('i')
plt.ylabel('P_n,i')
plt.title('Expected values P_n,i')
plt.legend()
plt.show()

# Create a heat map for the optimizer p for n = 100
plt.figure(figsize=(10, 5))
plt.imshow(p, cmap='hot', interpolation='nearest')
plt.colorbar(label='Probability Value')
plt.title('Heat map of the optimizer p for n = 100')
plt.xlabel('j')
plt.ylabel('i')
plt.show()
#------------------------------------------------------------------------------------------


