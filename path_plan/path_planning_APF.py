import numpy as np
import matplotlib.pyplot as plt

# Define the grid and goal position
grid_size = 100
goal_position = (80, 80)
max_step_size = 5
# Generate random obstacles
num_obstacles = 10
obstacle_positions = np.random.randint(0, grid_size, size=(num_obstacles, 2))

# Define parameters for potential field
k_att = 0.1  # Attractive potential constant
k_rep = 100  # Repulsive potential constant
d_rep = 10   # Distance threshold for repulsion

# Define function to calculate attractive potential
def attractive_potential(position):
    return 0.5 * k_att * np.linalg.norm(np.array(position) - np.array(goal_position))**2

# Define function to calculate repulsive potential
def repulsive_potential(position):
    rep_potential = 0
    for obstacle_pos in obstacle_positions:
        distance = np.linalg.norm(np.array(position) - np.array(obstacle_pos))
        if 0 < distance < d_rep :
            rep_potential += 0.5 * k_rep * (1 / distance - 1 / d_rep)**2
    return rep_potential


def update_obstacles():
    for i in range(num_obstacles):
        # Generate random step size in both x and y directions
        step_x = np.random.randint(-max_step_size, max_step_size + 1)
        step_y = np.random.randint(-max_step_size, max_step_size + 1)

        # Update obstacle position with the random step
        new_x = obstacle_positions[i, 0] + step_x
        new_y = obstacle_positions[i, 1] + step_y

        # Ensure the new position is within the grid bounds
        new_x = np.clip(new_x, 0, grid_size - 1)
        new_y = np.clip(new_y, 0, grid_size - 1)

        # Update obstacle position
        obstacle_positions[i, 0] = new_x
        obstacle_positions[i, 1] = new_y


# Define function to calculate total potential
def total_potential(position):
    return attractive_potential(position) + repulsive_potential(position)

# Define function to calculate gradient of potential field
def calculate_gradient(position):
    delta = 0.1
    gradient_x = (total_potential((position[0] + delta, position[1])) - total_potential((position[0] - delta, position[1]))) / (2 * delta)
    gradient_y = (total_potential((position[0], position[1] + delta)) - total_potential((position[0], position[1] - delta))) / (2 * delta)
    return np.array([gradient_x, gradient_y])

# Define function to update position based on gradient descent
def update_position(position, learning_rate):
    gradient = calculate_gradient(position)
    new_position = position - learning_rate * gradient
    return new_position

# Perform gradient descent to navigate towards the goal
max_iterations = 100
learning_rate = 1.0
current_position = np.array([grid_size // 2 + 40, grid_size // 2])  # Start at the center
trajectory = [current_position]

for _ in range(max_iterations):
    current_position = update_position(current_position, learning_rate)
    trajectory.append(current_position)

# Plot the potential field and trajectory
X, Y = np.meshgrid(np.arange(0, grid_size), np.arange(0, grid_size))
update_obstacles()
potential_field = np.zeros_like(X)
for i in range(grid_size):
    for j in range(grid_size):
        potential_field[i, j] = total_potential((i, j))

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, potential_field, levels=20, cmap='viridis')
plt.plot(goal_position[0], goal_position[1], 'ro', label='Goal')
plt.plot(obstacle_positions[:, 0], obstacle_positions[:, 1], 'kx', label='Obstacles')
plt.plot([pos[0] for pos in trajectory], [pos[1] for pos in trajectory], 'r-', label='Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Potential Field Navigation with Dynamic Obstacles')
plt.colorbar(label='Potential')
plt.legend()
plt.grid(True)
plt.show()
