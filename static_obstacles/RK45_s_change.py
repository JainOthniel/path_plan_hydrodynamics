# import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from functools import partial
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt
# import pandas as pd
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})
rc('text', usetex=True)

def event_nan_inf(t, y, p, aa2_col_num, dimm):

    # Check for NaN or infinite values in the state variables
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        # print("Debugging event_nan_inf:")
        # print("t:", t)
        # print("y:", y)
        # print("p:", p)
        # print("aa2_col_num:", aa2_col_num)
        # print("dimm:", dimm)
        # print("Event condition met: NaN or infinite value detected in y")
        return 0  # Terminate integration
    else:
        # print("Event condition not met: No NaN or infinite value detected in y")
        return 1 # Continue integration

# checks s foe far field
def event_s_check_far(t,state,*args):

    Xox, Xoz, Xdx, Xdz = state
    r = np.sqrt((Xox - Xdx) ** 2 + (Xoz - Xdz) ** 2)
    p, a, dim = args
    a2 = a
    a1, rho, Fz, Fx = p
    s = 2 * r / (a2 + a1)
    return s - 2.0001


event_s_check_far.terminal = True  # Stop the integration when this event occurs
event_s_check_far.direction = -1  # Detect zero crossing in any direction


def solve_traj(initial_cond, t_span,s_threshold,*args):
    p, a, dim, max_steps_size,switch_count = args
    a2 = a
    a1, rho, Fz, Fx = p
    #switch_count = 0  # counts the number of times the functions switches : help in adding perturbations

    def dynamics_wrapper(t, y):
        return dynamics(t, y, switch_count)

    def dynamics(t, y, switch_count):
        Xox, Xoz, Xdx, Xdz = y
        # if np.isnan(y.any()):print(y)
        r = np.sqrt((Xox - Xdx) ** 2 + (Xoz - Xdz) ** 2)
        s = 2 * r / (a2 + a1)
        args1 = p, a2, dim
        # when runninng RK45 manually use functool partial to enteradditioanal arguments

        if s > s_threshold:
            if switch_count != 0:
                switch_count = 0
            # diff_eq_with_args_far = partial(diff_eq, p=args1[0], a=args1[1], dimm=args1[2])
            return diff_eq(t,y,*args1)
        elif s <= s_threshold:
            if switch_count == 0:
                # y += perturb
                switch_count += 1 # Adding perturbations
            # diff_eq_with_args_near = partial(diff_eq_near, p=args1[0], a=args1[1], dimm=args1[2])
            return diff_eq_near(t,y,*args1)

    sol = RK45(dynamics_wrapper, t_span[0], initial_cond,t_span[1],max_step=max_steps_size)
    # sol = RK45(dynamics, t_span[0], initial_conditions, t_span[1])
    return sol

# dim = bb1, bb2, bb3, dd1 ,dd2, dd3, dd4, ee1, ee2, ll1, ll2, ll3, mm0, mm1, mm2
def diff_eq(t,state,p,a,dim):
    Xox, Xoz, Xdx, Xdz = state
    r = np.sqrt((Xox - Xdx) ** 2 + (Xoz - Xdz) ** 2)
    # p,a,dim = args
    a2 = a
    a1, rho, Fz, Fx = p
    la = a2 / a1
    x = Xox - Xdx
    z = Xoz - Xdz
    s = (2 * r) / (a2 + a1)
    # if s < 2.0001: s = 2.0001
    term1B11 = (68 * la ** 5) / ((1 + la) ** 6 * s ** 6)
    term2B11 = (32 * la ** 3 * (10 - 9 * la ** 2 + 9 * la ** 4)) / ((1 + la) ** 8 * s ** 8)
    term3B11 = (192 * la ** 5 * (35 - 18 * la ** 2 + 6 * la ** 4)) / ((1 + la) ** 10 * s ** 10)

    B11 = 1 - (term1B11 + term2B11 + term3B11)

    term1A11 = (60 * la ** 3) / ((1 + la) ** 4 * s ** 4)
    term2A11 = (60 * la ** 3 * (8 - la ** 2)) / ((1 + la) ** 6 * s ** 6)
    term3A11 = (32 * la ** 3 * (20 - 123 * la ** 2 + 9 * la ** 4)) / ((1 + la) ** 8 * s ** 8)
    term4A11 = (64 * la ** 2 * (175 + 1500 * la - 426 * la ** 2 + 18 * la ** 4)) / ((1 + la) ** 10 * s ** 10)
    A11 = B11 - (term1A11 - term2A11 + term3A11 + term4A11)

    term1L = 1
    term2L = -3 / ((1 + la) * s)
    term3L = 4 * (1 + la ** 2) / ((1 + la) ** 3 * s ** 3)
    term4L = -60 * la ** 3 / ((1 + la) ** 4 * s ** 4)
    term5L = 32 * la ** 3 * (15 - 4 * la ** 2) / ((1 + la) ** 6 * s ** 6)
    term6L = -2400 * la ** 3 / ((1 + la) ** 7 * s ** 7)
    term7L = -192 * la ** 3 * (5 - 22 * la ** 2 + 3 * la ** 4) / ((1 + la) ** 8 * s ** 8)
    term8L = 1920 * la ** 3 * (1 + la ** 2) / ((1 + la) ** 9 * s ** 9)
    term9L = -256 * la ** 5 * (70 - 375 * la - 120 * la ** 2 + 9 * la ** 3) / ((1 + la) ** 10 * s ** 10)
    term10L = -1536 * la ** 3 * (10 - 151 * la ** 2 + 10 * la ** 4) / ((1 + la) ** 11 * s ** 11)
    L = term1L + term2L + term3L + term4L + term5L + term6L + term7L + term8L + term9L + term10L

    term1M = 1
    term2M = -3 / (2 * (1 + la) * s)
    term3M = -2 * (1 + la ** 2) / ((1 + la) ** 3 * s ** 3)
    term4M = -68 * la ** 5 / ((1 + la) ** 6 * s ** 6)
    term5M = -32 * la ** 3 * (10 - 9 * la ** 2 + 9 * la ** 4) / ((1 + la) ** 8 * s ** 8)
    term6M = -192 * la ** 5 * (35 - 18 * la ** 2 + 6 * la ** 4) / ((1 + la) ** 10 * s ** 10)
    term7M = -16 * la ** 3 * (560 - 553 * la ** 2 + 560 * la ** 4) / ((1 + la) ** 11 * s ** 11)
    M = term1M + term2M + term3M + term4M + term5M + term6M + term7M

    A12 = (A11 - L) * ((1 + la) / 2)
    B12 = (B11 - M) * ((1 + la) / 2)
    # dXodt = (Fz / (3 * np.pi * rho * (a1 + a2))) * (((A12 - B12) / r ** 2) * (Xoz - Xdz) * (Xox - Xdx))
    # dZodt = (Fz / (3 * np.pi * rho * (a1 + a2))) * ((((A12 - B12) / r ** 2) * ((Xoz - Xdz) ** 2)) + B12)
    # dXddt = (Fz / (6 * np.pi * rho * a1)) * (((A11 - B11) / r ** 2) * (Xoz - Xdz) * (Xox - Xdx))
    # dZddt = (Fz / (6 * np.pi * rho * a1)) * ((((A11 - B11) / r ** 2) * ((Xoz - Xdz) ** 2)) + B11)
    dx_ddt = (1 / (6 * np.pi * rho * a1)) * (
            A11 * (Fx * x + Fz * z) * x / r ** 2 + B11 * Fx - B11 * (Fx * x + Fz * z) * x / r ** 2)
    dz_ddt = (1 / (6 * np.pi * rho * a1)) * (
            A11 * (Fx * x + Fz * z) * z / r ** 2 + B11 * Fz - B11 * (Fx * x + Fz * z) * z / r ** 2)
    dx_odt = (1 / (3 * np.pi * rho * (a1 + a2))) * (
            A12 * (Fx * x + Fz * z) * x / r ** 2 + B12 * Fx - B12 * (Fx * x + Fz * z) * x / r ** 2)
    dz_odt = (1 / (3 * np.pi * rho * (a1 + a2))) * (
            A12 * (Fx * x + Fz * z) * z / r ** 2 + B12 * Fz - B12 * (Fx * x + Fz * z) * z / r ** 2)
    return [dx_odt, dz_odt, dx_ddt, dz_ddt]


def diff_eq_near(t,state,p,a,dim):
    Xox, Xoz, Xdx, Xdz= state
    r = np.sqrt((Xox - Xdx) ** 2 + (Xoz - Xdz) ** 2)
    # p, a, dim = args
    a2 = a
    a1, rho, Fz, Fx = p
    la = a2 / a1
    x = Xox - Xdx
    z = Xoz - Xdz
    s = (2 * r) / (a2 + a1)
    # if s < 2.0001: s = 2.000101
    # print(state)
    # Define coefficients
    # d1, d2, d3, d4 = 0.7750, 0.9306, -0.900, -2.0
    # b1, b2, b3, e1, e2 = 0.8905, 5.772, 7.070, 6.043, 6.327
    # l1, l2, l3 = 2.0, -1.80, -4.0
    # m0, m1, m2 = 0.4021, 2.967, 5.088
    b1, b2, b3, d1, d2, d3, d4, e1, e2, l1, l2, l3, m0, m1, m2 = dim
    xi_value = s - 2
    # if xi_value <= 0:
    #     print(xi_value)
    #     xi_value = 0.01
    # Calculate the A_11 value with the error term
    A11 = d1 + d2 * xi_value + d3 * xi_value ** 2 * np.log(1 / xi_value) + d4 * xi_value ** 2

    # Calculate the B_11 value
    B11 = (b1 * (np.log(1 / xi_value)) ** 2 + b2 * np.log(1 / xi_value) + b3) / \
          ((np.log(1 / xi_value)) ** 2 + e1 * np.log(1 / xi_value) + e2)

    # Calculate the L value
    L = l1 * xi_value + l2 * xi_value ** 2 * np.log(1 / xi_value) + l3 * xi_value ** 2

    # Calculate the M value
    M = (m0 * (np.log(1 / xi_value)) ** 2 + m1 * np.log(1 / xi_value) + m2) / \
        ((np.log(1 / xi_value)) ** 2 + e1 * np.log(1 / xi_value) + e2)

    # if s - 2 <= 0: print(f's - 2 = {s - 2}\nA11 = {A11}\nB11 = {B11}\nL = {L}\nM ={M}')

    A12 = (A11 - L) * ((1 + la) / 2)
    B12 = (B11 - M) * ((1 + la) / 2)
    dx_ddt = (1 / (6 * np.pi * rho * a1)) * (
            A11 * (Fx * x + Fz * z) * x / r ** 2 + B11 * Fx - B11 * (Fx * x + Fz * z) * x / r ** 2)
    dz_ddt = (1 / (6 * np.pi * rho * a1)) * (
            A11 * (Fx * x + Fz * z) * z / r ** 2 + B11 * Fz - B11 * (Fx * x + Fz * z) * z / r ** 2)
    dx_odt = (1 / (3 * np.pi * rho * (a1 + a2))) * (
            A12 * (Fx * x + Fz * z) * x / r ** 2 + B12 * Fx - B12 * (Fx * x + Fz * z) * x / r ** 2)
    dz_odt = (1 / (3 * np.pi * rho * (a1 + a2))) * (
            A12 * (Fx * x + Fz * z) * z / r ** 2 + B12 * Fz - B12 * (Fx * x + Fz * z) * z / r ** 2)
    # if s - 2 <= 0: print(f's - 2 = {s - 2}\nA11 = {A11}\nB11 = {B11}\nA12 = {A12}\nB12 ={B12}')
    # if s - 2 <= 0: print(f's - 2 = {s - 2}\nuxo = {dx_odt}\nuzo = {dz_odt}\nuxd = {dx_ddt}\nuzd ={dz_ddt}')
    return [dx_odt, dz_odt, dx_ddt, dz_ddt]



def velo(Xox, Xoz, Xdx, Xdz, pp, a, dimm):
    p = pp
    dim = dimm
    dx_odt, dz_odt, dx_ddt, dz_ddt = diff_eq_near(Xox, Xoz, Xdx, Xdz, p, a, dim)

    uud = np.sqrt(dx_ddt ** 2 + dz_ddt ** 2)
    uuo = np.sqrt(dx_odt ** 2 + dz_odt ** 2)

    return [uuo, uud]

sep_dis = (0, 0.005, 0.01, 0.1, 0.25, 0.8, 1.8, 2.5, 2.7, 2.9, 3, 3.3, 3.6, 3.8, 4, 5, 7, 11)
size_ratio = (0.125, 0.25, 0.5, 1, 2, 4, 8)
aa1 = 2.5
aa2 = (2.5 / 8, 2.5 / 4, 2.5 / 2, 2.5, 2.5 * 2, 2.5 * 4, 2.5 * 8)
rrho = 0.1
FFz = -0.005
FFz_ne = -0.005
FFx = 0
pp = aa1, rrho, FFz, FFx
p_ne = aa1, rrho, FFz_ne, FFx
bb1 = (0.9942, 0.9729, 0.9272, 0.8905, 0.7642, 0.4734, 0.2377)
bb2 = (1.536, 3.843, 5.611, 5.772, 5.020, 3.710, 2.704)
bb3 = (-1.544, 0.3421, 4.404, 7.070, 5.604, 1.894, -0.840)
dd1 = (0.99968, 0.9951, 0.9537, 0.7750, 0.4768, 0.2488, 0.1250)
dd2 = (-0.00018, 0.0088, 0.1514, 0.9306, 2.277, 3.610, 5.620)
dd3 = (-0.003, -0.026, -0.194, -0.900, -2.188, -4.061, -8.500)
dd4 = (0, 0, -0.3, -2.0, -4.5, -6.4, -9.2)
ee1 = (1.518, 3.795, 5.600, 6.043, 5.600, 3.795, 1.518)
ee2 = (-1.536, 0.323, 4.179, 6.327, 4.179, 0.323, -1.536)
ll1 = (0.3205, 0.6040, 1.173, 2.000, 2.788, 3.759, 5.658)
ll2 = (-0.458, -0.679, -1.115, -1.800, -2.649, -4.224, -8.557)
ll3 = (0.356, -0.320, -1.63, -4.00, -5.17, -6.48, -9.16)
mm0 = (0.0108, 0.0596, 0.2142, 0.4021, 0.4077, 0.2451, 0.1148)
mm1 = (0.721, 1.391, 2.274, 2.967, 3.352, 3.097, 2.602)
mm2 = (-0.396, 0.440, 2.750, 5.088, 4.777, 1.918, -0.696)
Xox = 63.5
Xoz = 59.5
Xdx = 63.5
Xdz = 99.5

# t_initial = 0
# t_final = 1_20_000
# t_span = (t_initial, t_final)
# num_steps = 2_40_000  # Adjust as needed
#
# # Generate time points
# t_values = np.linspace(t_span[0], t_span[1], num_steps)

# Initialize solution arrays

t_initial = 0
t_final = 10_000
t_span = (t_initial, t_final)
t_span_ne = (t_initial, -t_final)
step_size = 0.1
num_steps = int(t_final / step_size)  # Adjust as needed
Xox_values = np.full((num_steps, len(aa2)),np.nan)#len(sep_dis)))
Xoz_values = np.full((num_steps, len(aa2)),np.nan)#len(sep_dis)))
Xdx_values = np.full((num_steps, len(aa2)),np.nan)#len(sep_dis)))
Xdz_values = np.full((num_steps, len(aa2)),np.nan)#len(sep_dis)))
Xox_values_ne = np.full((num_steps, len(aa2)),np.nan)#len(sep_dis)))
Xoz_values_ne = np.full((num_steps, len(aa2)),np.nan)#len(sep_dis)))
Xdx_values_ne = np.full((num_steps, len(aa2)),np.nan)#len(sep_dis)))
Xdz_values_ne = np.full((num_steps, len(aa2)),np.nan)#len(sep_dis)))
# Generate time points
t_values = np.arange(t_span[0], t_span[1], step_size)
t_values_ne = np.arange(t_span_ne[0], t_span_ne[1], -step_size)


# Xox_values = np.full((num_steps, len(sep_dis)), np.nan)
# Xoz_values = np.full((num_steps, len(sep_dis)), np.nan)
# Xdx_values = np.full((num_steps, len(sep_dis)), np.nan)
# Xdz_values = np.full((num_steps, len(sep_dis)), np.nan)
# Xox_values_ne = np.full((num_steps, len(sep_dis)), np.nan)
# Xoz_values_ne = np.full((num_steps, len(sep_dis)), np.nan)
# Xdx_values_ne = np.full((num_steps, len(sep_dis)), np.nan)
# Xdz_values_ne = np.full((num_steps, len(sep_dis)), np.nan)
#
# Xdx_values_rel = np.full((num_steps, len(sep_dis)), np.nan)
# Xdz_values_rel = np.full((num_steps, len(sep_dis)), np.nan)
# Xdx_values_ne_rel = np.full((num_steps, len(sep_dis)), np.nan)
# Xdz_values_ne_rel = np.full((num_steps, len(sep_dis)), np.nan)

#for index, IP in enumerate(sep_dis):
# Time span and number of steps
# Initial conditions
# Loop through each column of sep_dis
plt.figure(figsize=(9, 7))
plt.style.use('dark_background')
colormap = plt.cm.get_cmap('RdYlBu',4)
for i,s_threshold in enumerate([3, 2.8, 2.5, 2.3, 2.1, 2.015]):
    # for i, per in enumerate([0.001, 0.0001, 0.00001, 0.000001]):
        for col_num in [3]:  # range(0, len(aa2)):#len(sep_dis)):
            # col_num = 0
            Xox = 0.04 * (aa2[col_num] + aa1) / 2 + 63.5
            # Xox = 0.1 + 63.5
            s_thers = s_threshold
            s_stop = 2.000
            a2 = aa2[col_num]
            # Set initial conditions
            initial_conditions = [Xox, Xoz, Xdx, Xdz]
            # Xox_values[0, col_num], Xoz_values[0, col_num], Xdx_values[0, col_num], Xdz_values[
            #     0, col_num] = initial_conditions

            dimm = (bb1[col_num], bb2[col_num], bb3[col_num], dd1[col_num], dd2[col_num], dd3[col_num], dd4[col_num]
                    , ee1[col_num], ee2[col_num], ll1[col_num], ll2[col_num], ll3[col_num], mm0[col_num], mm1[col_num],
                    mm2[col_num])
            # solution = solve_ivp(diff_eq, t_span, initial_conditions, args=(pp, aa2[col_num], dimm), method='RK45',
            #                      t_eval=t_values, rtol=rtoll, atol=atoll,
            #                      events=lambda t1, y, p=pp, aa=aa2[col_num], dim=dimm: event_nan_inf(t, y, pp, aa, dim))
            # solution = solve_traj(initial_conditions,t_span,t_values,s_thres,pp,aa2[col_num],dimm)
            # args1 = pp, a2, dimm, step_size
            # diff_eq_with_args = partial(diff_eq, p=args1[0], a=args1[1], dimm=args1[2])
            # #when runninng RK45 manually use functool partial to enteradditioanal arguments
            # solution = RK45(diff_eq_with_args, t_span[0], initial_conditions, t_span[1], max_step=step_size)
            switch_count_1 = 0
            solution = solve_traj(initial_conditions, t_span, s_thers, pp, aa2[col_num], dimm, step_size,
                                  switch_count_1)
            print(switch_count_1)
            # solution = RK45(diff_eq, t_span[0], initial_conditions, t_span[1],max_step=step_size,args=args1)
            # print('passed initial stage')
            # Initialize arrays to store the results
            t_values = []
            state_values = []

            # Integrate the system and check for the event condition
            while solution.status == 'running':
                solution.step()
                t_values.append(solution.t)
                state_values.append(solution.y)
                # Calculate s at the current step
                Xox_1, Xoz_1, Xdx_1, Xdz_1 = solution.y
                r = np.sqrt((Xox_1 - Xdx_1) ** 2 + (Xoz_1 - Xdz_1) ** 2)
                s = 2 * r / (aa2[col_num] + aa1)
                # print(s)
                # Check if s is less than the threshold
                if s - s_stop < 0 or np.isnan(s-s_stop) or np.isinf(s-s_stop):
                    print(f"Integration halted because s fell below the threshold {s_stop} at t={solution.t}.")
                    # print(f"Integration halted because s fell below the threshold at t={solution.t}.")
                    break
            print(switch_count_1)
            # Xox_values[:len(state_values), col_num] = state_values[:,0]
            # Xoz_values[:len(state_values), col_num] = state_values[:,1]
            # Xdx_values[:len(state_values), col_num] = state_values[:,2]
            # Xdz_values[:len(state_values), col_num] = state_values[:,3]
            # for t, state in zip(t_values, state_values):
            #     print(f"t={t}: {state}")

            # mask = ~np.isnan(Xdx_values[:, col_num]) & ~np.isnan(Xdz_values[:, col_num])
            # last_index = np.where(mask == 1)[0][-1]
            # initial_conditions_ne = [Xox_values[last_index, col_num], Xoz_values[last_index, col_num],
            #                          Xdx_values[last_index, col_num], Xdz_values[last_index, col_num]]
            # solution_ne = solve_traj(initial_conditions_ne, t_span_ne, t_values_ne, s_thres, pp, aa2[col_num], dimm)
            # solution_ne = solve_ivp(diff_eq, t_span_ne, initial_conditions_ne, args=(pp_ne, aa2[col_num], dimm), method='RK45',
            #                         t_eval=t_values_ne, rtol=rtoll, atol=atoll,
            #                         events=lambda t1, y, p=pp, aa=aa2[col_num], dim=dimm: event_nan_inf(t, y, pp, aa, dim))
            # # events=lambda t2, y, pp_ne, aa=aa2[col_num], dim=dimm: event_nan_inf(t2, y, pp_ne, aa, dim),
            #
            # # Xox_values_ne[:len(solution_ne.y[0]) // 10, col_num] = solution_ne.y[0][::10]
            # # Xoz_values_ne[:len(solution_ne.y[1]) // 10, col_num] = solution_ne.y[1][::10]
            # # Xdx_values_ne[:len(solution_ne.y[2]) // 10, col_num] = solution_ne.y[2][::10]
            # # Xdz_values_ne[:len(solution_ne.y[3]) // 10, col_num] = solution_ne.y[3][::10]
            #
            # Xox_values_ne[:len(solution_ne.y[0]), col_num] = solution_ne.y[0][:]
            # Xoz_values_ne[:len(solution_ne.y[1]), col_num] = solution_ne.y[1][:]
            # Xdx_values_ne[:len(solution_ne.y[2]), col_num] = solution_ne.y[2][:]
            # Xdz_values_ne[:len(solution_ne.y[3]), col_num] = solution_ne.y[3][:]

            # RK4 integration
            # for i in range(1, num_steps):
            #     r = np.sqrt((Xox_values[i - 1, col_num] - Xdx_values[i - 1, col_num]) ** 2 +
            #                 (Xoz_values[i - 1, col_num] - Xdz_values[i - 1, col_num]) ** 2)
            #
            #     s = 2 * r / (aa2[col_num] + aa1)
            #     if s >= 3:
            #         h = t_values[i] - t_values[i - 1]
            #         k1x, k1z, k1y, k1w = diff_eq(Xox_values[i - 1, col_num], Xoz_values[i - 1, col_num],
            #                                      Xdx_values[i - 1, col_num], Xdz_values[i - 1, col_num], p, aa2[col_num])
            #         k2x, k2z, k2y, k2w = diff_eq(Xox_values[i - 1, col_num] + 0.5 * h * k1x,
            #                                      Xoz_values[i - 1, col_num] + 0.5 * h * k1z,
            #                                      Xdx_values[i - 1, col_num] + 0.5 * h * k1y,
            #                                      Xdz_values[i - 1, col_num] + 0.5 * h * k1w, p, aa2[col_num])
            #
            #         k3x, k3z, k3y, k3w = diff_eq(Xox_values[i - 1, col_num] + 0.5 * h * k2x,
            #                                      Xoz_values[i - 1, col_num] + 0.5 * h * k2z,
            #                                      Xdx_values[i - 1, col_num] + 0.5 * h * k2y,
            #                                      Xdz_values[i - 1, col_num] + 0.5 * h * k2w, p, aa2[col_num])
            #
            #         k4x, k4z, k4y, k4w = diff_eq(Xox_values[i - 1, col_num] + h * k3x,
            #                                      Xoz_values[i - 1, col_num] + h * k3z,
            #                                      Xdx_values[i - 1, col_num] + h * k3y,
            #                                      Xdz_values[i - 1, col_num] + h * k3w, p, aa2[col_num])
            #
            #         Xox_values[i, col_num] = Xox_values[i - 1, col_num] + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            #         Xoz_values[i, col_num] = Xoz_values[i - 1, col_num] + (h / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
            #         Xdx_values[i, col_num] = Xdx_values[i - 1, col_num] + (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
            #         Xdz_values[i, col_num] = Xdz_values[i - 1, col_num] + (h / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
            #     elif s < 3:
            #         h = t_values[i] - t_values[i - 1]
            #         k1x, k1z, k1y, k1w = diff_eq_near(Xox_values[i - 1, col_num], Xoz_values[i - 1, col_num],
            #                                           Xdx_values[i - 1, col_num], Xdz_values[i - 1, col_num],
            #                                           p, aa2[col_num], dim)
            #         k2x, k2z, k2y, k2w = diff_eq_near(Xox_values[i - 1, col_num] + 0.5 * h * k1x,
            #                                           Xoz_values[i - 1, col_num] + 0.5 * h * k1z,
            #                                           Xdx_values[i - 1, col_num] + 0.5 * h * k1y,
            #                                           Xdz_values[i - 1, col_num] + 0.5 * h * k1w, p, aa2[col_num], dimm)
            #         k3x, k3z, k3y, k3w = diff_eq_near(Xox_values[i - 1, col_num] + 0.5 * h * k2x,
            #                                           Xoz_values[i - 1, col_num] + 0.5 * h * k2z,
            #                                           Xdx_values[i - 1, col_num] + 0.5 * h * k2y,
            #                                           Xdz_values[i - 1, col_num] + 0.5 * h * k2w, p, aa2[col_num], dimm)
            #         k4x, k4z, k4y, k4w = diff_eq_near(Xox_values[i - 1, col_num] + h * k3x,
            #                                           Xoz_values[i - 1, col_num] + h * k3z,
            #                                           Xdx_values[i - 1, col_num] + h * k3y,
            #                                           Xdz_values[i - 1, col_num] + h * k3w, p, aa2[col_num], dimm)
            #
            #         Xox_values[i, col_num] = Xox_values[i - 1, col_num] + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            #         Xoz_values[i, col_num] = Xoz_values[i - 1, col_num] + (h / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
            #         Xdx_values[i, col_num] = Xdx_values[i - 1, col_num] + (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
            #         Xdz_values[i, col_num] = Xdz_values[i - 1, col_num] + (h / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)

            colum = 3
            state_values = np.array(state_values)
            offset_Obstacle = {state_values[0, 0] - state_values[-1, 0]}
            offset_Microrobot = {state_values[0, 2] - state_values[-1, 2]}
            print(f'offset Obstacle = {state_values[0, 0] - state_values[-1, 0]}')
            print(f'offset Microrobot = {state_values[0, 2] - state_values[-1, 2]}')
            # if os.path.exists(filepath) and os.path.exists(filepath_1):
            # Read the CSV file into a pandas DataFrame
            # df_p = pd.read_csv(filepath)
            # df_d = pd.read_csv(filepath_1)
            # Create a single plot
            # plt.figure(figsize=(8, 6))
            # num_pairs = df.shape[1] // 3
            # Plot results
            # x_column = df_d.iloc[:, colum]
            # y_column = df_d.iloc[:, colum + 2]
            # plt.plot(y_column, x_column, label='Trajectory of Obstacle Numerical', linestyle='dashed')

            ####################commanded out for testing##########3333
            # plt.figure(figsize=(8, 6))
            # Mask the NaN values
            # mask = ~np.isnan(Xdx_values[:, colum]) & ~np.isnan(Xdz_values[:, colum])
            # plt.plot(Xoz_values[mask, colum], Xox_values[mask, colum],color='red', label='Trajectory of Obstacle Analytical ')
            # plt.plot(Xdz_values[mask, colum], Xdx_values[mask, colum],color='blue', label='Trajectory of microrobot Analytical ')
            # mask = ~np.isnan(Xdx_values[:, colum]) & ~np.isnan(Xdz_values[:, colum])
            plt.plot(state_values[:, 1], state_values[:, 0], color=colormap(i), label=f'{s_thers}')
            plt.plot(state_values[:, 3], state_values[:, 2], color=colormap(i), linestyle='--')
            # plt.plot([], [], label=f'Obstacle offset = {offset_Obstacle}')
            # plt.plot([], [], label=f'Microrobot offset = {offset_Microrobot}')
            # mask_ne = ~np.isnan(Xdx_values_ne[:, colum]) & ~np.isnan(Xdz_values_ne[:, colum])
            # plt.plot(Xoz_values_ne[mask_ne, colum], Xox_values_ne[mask_ne, colum],color='red',linestyle='--', label='Trajectory of Obstacle Analytical -ve ')
            # plt.plot(Xdz_values_ne[mask_ne, colum], Xdx_values_ne[mask_ne, colum], color='blue',linestyle='--', label='Trajectory of microrobot Analytical -ve ')

            # plt.plot(-Xoz_values[:, colum] + Xdz_values[:, colum], -Xox_values[:, colum] + Xdx_values[:, colum], label='Trajectory of Obstacle Analytical ')
            # plt.plot(Xdz_values[:, colum], Xdx_values[:, colum], label='Trajectory of microrobot Analytical ')
            #
            # plt.plot(Xoz_values_ne[:, colum], Xox_values_ne[:, colum],
            #          label='Trajectory of Obstacle Analytical -ve ')
            # plt.plot(Xdz_values_ne[:, colum], Xdx_values_ne[:, colum],
            #          label='Trajectory of microrobot Analytical -ve ')
            # plt.plot(Xdz_values_rel[:, colum], Xdz_values_rel[:, colum],
            #          label='Trajectory of Obstacle Analytical -ve ')
            # plt.plot(Xdz_values_ne_rel[:, colum], Xdx_values_ne_rel[:, colum],
            #          label='Trajectory of microrobot Analytical -ve ')
            # plt.legend(frameon=False)
            # # plt.yticks(np.arange(61,80,1))
            # # plt.xticks(np.arange(0,128,5))
            # # plt.xlim(5,65)
            # # plt.ylim(63,78)
            # plt.xlabel('Z')
            # plt.ylabel('X')
            # # plt.title(f'Trajectory of b = {size_ra}')
            # plt.title(f'Trajectory s = {s_thres}, step size = {step_size}, s_stop ={2.0000} perturb={0.001}')
            # plt.tight_layout()
            # Assuming mask is your boolean mask
            # last_index = np.where(mask == 1)[0][-1]
            # 66.83574757492379
            # 10.97410360864194
            # 61.45849933409127
            # -48.84374661209797
            # print(f' o_offset = {abs(Xox_values[last_index, colum] - Xox_values[0, colum])}\n'
            #       f'd_offset = {abs(Xdx_values[last_index, colum] - Xdx_values[0, colum])}')
            # print(last_index)
            # print(Xox_values[last_index, colum])  # 66.83574757492379
            # print(Xoz_values[last_index, colum])  # 10.97410360864194
            # print(Xdx_values[last_index, colum])  # 61.45849933409127
            # print(Xdz_values[last_index, colum])  # -48.84374661209797
            # plt.show()
plt.legend(frameon=False)
# plt.yticks(np.arange(61,80,1))
# plt.xticks(np.arange(0,128,5))
# plt.xlim(5,65)
# plt.ylim(63,78)
plt.xlabel('Z')
plt.ylabel('X')
# plt.title(f'Trajectory of b = {size_ra}')
plt.title(f'Trajectory - step size = {step_size}, s_stop ={2.0000}')
plt.tight_layout()
plt.savefig(f'Traj_plot_s.png')
plt.clf()
plt.show()
# Define file names for saving
# cwd = os.getcwd()
# Xox_ana = os.path.join(cwd, 'Xox_values_sz_ratio_new_RK45.csv')
# Xoz_ana = os.path.join(cwd, 'Xoz_values_sz_ratio_new_RK45.csv')
# Xdx_ana = os.path.join(cwd, 'Xdx_values_sz_ratio_new_RK45.csv')
# Xdz_ana = os.path.join(cwd, 'Xdz_values_sz_ratio_new_RK45.csv')
#
# # Save Xox_values to CSV
# np.savetxt(Xox_ana, Xox_values[::10], delimiter=",")
#
# # Save Xoz_values to CSV
# np.savetxt(Xoz_ana, Xoz_values[::10], delimiter=",")
#
# # Save Xdx_values to CSV
# np.savetxt(Xdx_ana, Xdx_values[::10], delimiter=",")
#
# # Save Xdz_values to CSV
# np.savetxt(Xdz_ana, Xdz_values[::10], delimiter=",")
