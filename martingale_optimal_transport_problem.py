import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

#------------------------------------------------------------------------------------------
def cost_function(x):
    return np.exp(-x)

def KL_divergence(p, q):
    # Calculate the KL divergence between p and q
    return np.sum(p * np.log(p / q) - p + q, where=(p != 0))

def project_onto_C1(p, alpha):
    row_sums = np.sum(p, axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    return (p / row_sums) * alpha[:, np.newaxis]

def project_onto_C2(p, beta):
    col_sums = np.sum(p, axis=0, keepdims=True)
    # Avoid division by zero
    col_sums[col_sums == 0] = 1
    return (p / col_sums) * beta

def project_onto_C2plus(p, xi, yj, alpha, i):
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
# Parameters for the Gaussian distributions
mean1, std1 = 0, 1  # Initial Gaussian distribution (mean, std)
mean2, std2 = 1, 1.5  # Final Gaussian distribution (mean, std)

# Discretization parameters
num_points = 100  # Number of points to discretize the distribution
range_min, range_max = -5, 5  # Range for discretization

# Discretize the Gaussian distributions
points = np.linspace(range_min, range_max, num_points)
alpha = np.exp(-(points - mean1)**2 / (2 * std1**2)) / (std1 * np.sqrt(2 * np.pi))
beta = np.exp(-(points - mean2)**2 / (2 * std2**2)) / (std2 * np.sqrt(2 * np.pi))

# Normalize to make them valid probability distributions
alpha /= np.sum(alpha)
beta /= np.sum(beta)
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
m = 100  # Number of points in the first distribution
n = 100  # Number of points in the second distribution

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
plt.figure(figsize=(10, 8))
plt.imshow(p, cmap='Blues', aspect='auto')
plt.colorbar(label='Transport Quantity')
plt.xlabel('Final Distribution Points (S2)')
plt.ylabel('Initial Distribution Points (S1)')
plt.title('Optimal Transport Plan')
plt.tight_layout()
plt.show()
#------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
# Animation
def update(frame):
    ax.clear()
    # Plot the initial and final distributions
    ax.plot(points, alpha, color='green', label='Initial Distribution')
    ax.plot(points + range_max, beta, color='red', label='Final Distribution')

    # Visualize the transport for a subset of points at each frame
    step = len(points) // num_frames
    start, end = step * frame, step * (frame + 1)
    
    for i in range(start, end):
        for j in range(len(points)):
            if p[i, j] > threshold:  # Only visualize significant transports
                ax.arrow(points[i], alpha[i], range_max + points[j] - points[i], beta[j] - alpha[i], 
                         alpha=0.3, length_includes_head=True, head_width=0.02, color='blue')
    
    ax.legend()
    ax.set_xlim([range_min, 2 * range_max])
    ax.set_ylim([0, max(alpha.max(), beta.max())])
    ax.set_title('Step: {}'.format(frame))

# Create the figure and axes
fig, ax = plt.subplots()
threshold = 0.00001  # Threshold for visualizing transport
num_frames = 30    # Number of frames in the animation

ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)
ani.save('MOT_transport_animation.mp4', writer='ffmpeg')
plt.show()
#-------------------------------------------------------------------------------------------------------
