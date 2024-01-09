import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

#-------------------------------------------------------------------------------------------------------
# Function to interpolate between two distributions
def interpolate_distributions(dist1, dist2, weight):
    return (1 - weight) * dist1 + weight * dist2

# Initialize function for the animation
def init():
    line.set_ydata([np.nan] * len(points))
    return line,

# Animate function
def animate(i):
    weight = i / num_frames
    updated_dist = interpolate_distributions(prob_dist1, prob_dist2, weight)
    line.set_ydata(updated_dist)
    return line,
#-------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------
# Parameters for the Gaussian distributions
mean1, std1 = 0, 1  # Initial Gaussian distribution (mean, std)
mean2, std2 = 1, 1.5  # Final Gaussian distribution (mean, std)

# Discretization parameters
num_points = 100  # Number of points to discretize the distribution
range_min, range_max = -5, 5  # Range for discretization

# Discretize the Gaussian distributions
points = np.linspace(range_min, range_max, num_points)
prob_dist1 = np.exp(-(points - mean1)**2 / (2 * std1**2)) / (std1 * np.sqrt(2 * np.pi))
prob_dist2 = np.exp(-(points - mean2)**2 / (2 * std2**2)) / (std2 * np.sqrt(2 * np.pi))

# Normalize to make them valid probability distributions
prob_dist1 /= np.sum(prob_dist1)
prob_dist2 /= np.sum(prob_dist2)

#Plot the two Gaussian distributions
plt.figure(figsize=(10, 6))
plt.plot(points, prob_dist1, label='Initial Gaussian (mean=0, std=1)')
plt.plot(points, prob_dist2, label='Final Gaussian (mean=1, std=1.5)')
plt.title('Gaussian Distributions')
plt.xlabel('Points')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()
#-------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------
# Set up the optimization variables
# Define the cost matrix
cost_matrix = np.abs(points.reshape(-1, 1) - points.reshape(1, -1))

# Variable for the transport plan
transport_plan = cp.Variable((num_points, num_points), nonneg=True)

# Objective: minimize the total transport cost
objective = cp.Minimize(cp.sum(cp.multiply(cost_matrix, transport_plan)))

# Constraints
constraints = [cp.sum(transport_plan, axis=1) == prob_dist1,
               cp.sum(transport_plan, axis=0) == prob_dist2]

# Define and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Check the status and print the results
if problem.status not in ['infeasible', 'unbounded']:
    print("Optimal transport problem solved successfully.")
    print(f"Total transport cost: {problem.value}")
    optimal_transport_plan = transport_plan.value
else:
    print(f"Problem status: {problem.status}")

plt.figure(figsize=(10, 8))
sns.heatmap(optimal_transport_plan, cmap='viridis')
plt.title('Heatmap of Optimal Transport Plan')
plt.xlabel('Final Distribution Points')
plt.ylabel('Initial Distribution Points')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(cost_matrix, cmap='coolwarm', alpha=0.5)
sns.heatmap(optimal_transport_plan, cmap='viridis', alpha=0.7)
plt.title('Cost Matrix with Optimal Transport Plan Overlay')
plt.xlabel('Final Distribution Points')
plt.ylabel('Initial Distribution Points')
plt.show()
#-------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------
# Set up the figure
fig, ax = plt.subplots()
line, = ax.plot(points, prob_dist1, color='blue')  # This will be the evolving line
ax.plot(points, prob_dist1, color='green', label='Initial Distribution')  # Static line for initial distribution
ax.plot(points, prob_dist2, color='red', label='Final Distribution')  # Static line for final distribution
plt.legend()

num_frames = 30
ani = FuncAnimation(fig, animate, frames=num_frames, init_func=init, blit=True)
ani.save('distribution_evolution.mp4', writer='ffmpeg')
plt.show()
#-------------------------------------------------------------------------------------------------------