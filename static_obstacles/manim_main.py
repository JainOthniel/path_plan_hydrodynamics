from manim import *

""" main file in path_plan_hydrodynamics """

import numpy as np 
from parameters import Parameters
from mobility_functions import Mobility_Functions
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.patches import Circle
# Disable scientific notation
np.set_printoptions(suppress=True)



def velocity_robot(t, r_d, param: Parameters, mobility_fun: Mobility_Functions) -> np.ndarray:
    
    param.position_robot = r_d
    dimen_distance = (2 * param.distance_cal) / (param.radius_obstacle + param.radius_robot)
    mobility_fun.dimensionless_distance = dimen_distance
    inter_act = interactions(mobility_fun, param)
    U_stokes = param.Force / (6 * np.pi *0.1 * param.radius_robot)
    del_U = np.dot(param.Force, inter_act) - U_stokes
    
    return U_stokes + del_U.sum(axis=0)


def interactions(mobility_fun: Mobility_Functions, param: Parameters) -> np.ndarray:
    return b_alp_beta(mobility_fun.A11, mobility_fun.B11, param) - (b_alp_beta(mobility_fun.A12, mobility_fun.B12, param) \
                        @ b_alp_beta(mobility_fun.A21, mobility_fun.B21, param)) \
                        @ np.linalg.inv(b_alp_beta(mobility_fun.A22, mobility_fun.B22, param))
    

def broad_cast(shape_array: np.ndarray, reshape_array) -> np.ndarray:
    return np.broadcast_to(shape_array[:, np.newaxis, np.newaxis], reshape_array.shape)

def b_alp_beta(A: np.ndarray, B: np.ndarray, param: Parameters):
    
    const = 1 / ( 3 * np.pi * param.viscosity * (param.radius_robot + param.radius_obstacle))
    
    A_alp_beta = broad_cast(A,param.outer_product) * param.outer_product /\
    np.power(param.distance_cal.reshape((param.position_array.shape[0], 1, 1)), 2)
    
    B_alp_beta = broad_cast(B, param.outer_product) * (param.unit_tensor_mat - param.outer_product /\
    np.power(param.distance_cal.reshape((param.position_array.shape[0], 1, 1)), 2))

    return const * (A_alp_beta + B_alp_beta)


def main():
    
    a_d = 5
    a_o = 5
    r_d = np.array([10, 50,10])
    r_o = np.array([[20, 40, 20], [40, 60, 10], [60, 40, 5], [80, 60, 20], [100, 60, 10], [120, 40, 5], [150, 60, 20], [190, 60, 10]]) 
    # r_o = np.array([[20, 40, 20], [40, 60, 40], [60, 40], [80, 60], [100, 60], [120, 40], [150, 60], [190, 60]]) 
    F = np.array([0.005,0,0])
    r = r_o - r_d 
    param = Parameters(viscosity=0.1, radius_robot=a_d, radius_obstacle=a_o, position_robot=r_d, position_obstacle=r_o, Force=F)
    s = 2 * param.distance_cal / (a_o + a_d)
    la = np.tile( [a_o / a_d], (s.shape[0]))
    mob_fun = Mobility_Functions(dimensionless_distance=s, lamda=la)
    vel=[]
    inter = interactions(mob_fun, param)
    vel = velocity_robot(0 ,r_d, param, mob_fun)
    U_stokes = param.Force / (6 *np.pi *0.1* 5)
    U_robo_magn = np.linalg.norm(vel,ord=2,axis=0)
    tspan = (0,6_00_000)
    t_eval = np.arange(0,6_00_000,1)


class RobotTrajectoryScene(ThreeDScene):
    def construct(self):
        # Set up the 3D axes
        axes = ThreeDAxes(
            x_range=[0, 200, 50], y_range=[30, 65, 10], z_range=[0, 30, 5],
            axis_config={"include_tip": False}
        )

        # Add title
        title = Text('3D Trajectory of Robot and Positions of Obstacles').move_to([100, 50, 55])

        # Add labels for the axes
        labels = VGroup(
            Text("X").move_to([200, 0, 0]),
            Text("Y").move_to([0, 65, 0]),
            Text("Z").move_to([0, 0, 30]),
        )

        # Obstacles represented as red spheres
        obstacle_positions = [[50, 45, 15], [120, 55, 20], [175, 60, 10]]  # Example positions
        obstacle_radius = 5  # Radius of the obstacles

        obstacle_spheres = VGroup()
        for pos in obstacle_positions:
            sphere = Surface(
                lambda u, v: np.array([
                    obstacle_radius * np.cos(u) * np.sin(v) + pos[0],
                    obstacle_radius * np.sin(u) * np.sin(v) + pos[1],
                    obstacle_radius * np.cos(v) + pos[2]
                ]),
                u_range=[0, 2 * PI], v_range=[0, PI],
                checkerboard_colors=[RED],
                fill_opacity=0.5
            )
            obstacle_spheres.add(sphere)

        # Trajectory of the robot as a blue line (example points)
        trajectory_points = [
            [0, 35, 5], [50, 45, 10], [100, 50, 20], [150, 60, 25], [200, 65, 30]
        ]
        trajectory = Line3D(trajectory_points[0], trajectory_points[1], color=BLUE)
        for i in range(1, len(trajectory_points) - 1):
            next_segment = Line3D(trajectory_points[i], trajectory_points[i + 1], color=BLUE)
            trajectory.add(next_segment)

        # Add everything to the scene
        self.add(axes, title, labels, obstacle_spheres, trajectory)

        # Set the camera view
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)

        # Animate the scene
        self.begin_ambient_camera_rotation(rate=0.2)  # Slowly rotate the camera
        self.wait(10)

# To run this scene, save the code in a file and run it using:
# manim -pql filename.py RobotTrajectoryScene
