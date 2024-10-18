import numpy as np
from scipy.integrate import solve_ivp

def equations_set1(t, y, applied_force):
    # Equations when distance s is above the threshold
    # y[0] = position of particle 1 (x1)
    # y[1] = velocity of particle 1 (v1)
    # y[2] = position of particle 2 (x2)
    # y[3] = velocity of particle 2 (v2)

    x1, v1, x2, v2 = y

    dx1dt = v1
    dv1dt = applied_force / m1  # Force on particle 1
    dx2dt = v2
    dv2dt = 0  # No force on particle 2

    return [dx1dt, dv1dt, dx2dt, dv2dt]

def equations_set2(t, y, applied_force):
    # Equations when distance s is below the threshold
    # These might be different depending on your problem
    x1, v1, x2, v2 = y

    dx1dt = v1
    dv1dt = applied_force / m1  # Force on particle 1
    dx2dt = v2
    dv2dt = 0  # No force on particle 2

    return [dx1dt, dv1dt, dx2dt, dv2dt]

def event(t, y):
    # Event function to detect when s crosses the threshold
    x1, v1, x2, v2 = y
    s = abs(x1 - x2)
    return s - s_threshold

event.terminal = True  # Stop the integration when this event occurs
event.direction = 0  # Detect zero crossing in any direction

def solve_particles(applied_force, y0, t_span, s_threshold, m1):
    def dynamics(t, y):
        x1, v1, x2, v2 = y
        s = abs(x1 - x2)
        if s > s_threshold:
            return equations_set1(t, y, applied_force)
        else:
            return equations_set2(t, y, applied_force)

    sol = solve_ivp(dynamics, t_span, y0, events=event, dense_output=True)

    return sol

# Initial conditions and parameters
m1 = 1.0  # Mass of particle 1
applied_force = 1.0  # Applied force on particle 1
s_threshold = 1.0  # Distance threshold
y0 = [0, 0, 2, 0]  # Initial conditions: [x1, v1, x2, v2]
t_span = (0, 10)  # Time span for the integration

sol = solve_particles(applied_force, y0, t_span, s_threshold, m1)

import matplotlib.pyplot as plt

# Plot the results
t = np.linspace(t_span[0], t_span[1], 300)
z = sol.sol(t)

plt.plot(t, z.T[:, 0], label='x1 (Particle 1)')
plt.plot(t, z.T[:, 2], label='x2 (Particle 2)')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Positions of Particles over Time')
plt.show()
