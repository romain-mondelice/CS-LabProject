import numpy as np
import matplotlib.pyplot as plt

def generate_samples(n):
    X = np.random.uniform(-1, 1, n)
    G = np.random.choice([-1, 1], n, p=[0.5, 0.5])
    Y = X + G
    return X, Y

def plot_heatmap(X, Y):
    plt.figure(figsize=(6, 6))
    plt.hist2d(X, Y, bins=50, range=[[-2, 2], [-2, 2]], cmap='Blues')
    plt.colorbar(label='Count')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Heatmap of (X, Y)')
    plt.show()

def compute_expectations(X, Y):
    c_xy = np.mean(np.abs(X - Y))
    y_squared = np.mean(Y**2)
    x_squared = np.mean(X**2)
    return c_xy, y_squared, x_squared

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))

def bregman_projection(P, mu, nu, max_iter=100, tol=1e-6):
    for _ in range(max_iter):
        P_prev = P.copy()
        
        # Project onto the constraint set defined by mu
        P = P * mu[:, np.newaxis] / np.sum(P, axis=1)[:, np.newaxis]
        
        # Project onto the constraint set defined by nu
        P = P * nu[np.newaxis, :] / np.sum(P, axis=0)[np.newaxis, :]
        
        # Check for convergence
        if kl_divergence(P, P_prev) < tol:
            break
    
    return P

def main():
    n = 100000  # Number of samples
    X, Y = generate_samples(n)

    # Discretize the support of X and Y
    x_range = np.linspace(-1, 1, 100)
    y_range = np.linspace(-2, 2, 200)

    # Compute the empirical distributions of X and Y
    mu, _ = np.histogram(X, bins=x_range, density=True)
    nu, _ = np.histogram(Y, bins=y_range, density=True)

    # Initialize the coupling matrix P
    P = np.outer(mu, nu)

    # Perform Bregman projection
    P_star = bregman_projection(P, mu, nu)

    # Compute expectations using the optimal coupling P*
    x_grid, y_grid = np.meshgrid(x_range[:-1], y_range[:-1], indexing='ij')
    c_xy = np.sum(np.abs(x_grid - y_grid) * P_star)
    y_squared = np.sum(y_grid**2 * P_star)
    x_squared = np.sum(x_grid**2 * P_star)

    print(f"E[|X - Y|] = {c_xy:.4f}")
    print(f"E[Y^2 - X^2] = {y_squared - x_squared:.4f}")

    # Plot heatmap
    plot_heatmap(X, Y)

if __name__ == "__main__":
    main()