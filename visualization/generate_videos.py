#!/usr/bin/env python3
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = dir_path + '/../src'
import sys
sys.path.insert(1, src_path + '/python')
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
import matplotlib.patches as patches
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from param_utils import readParam, generateSampleIndices

rng = np.random.default_rng(42)
cmap_act = LinearSegmentedColormap.from_list('yellows', colors=('xkcd:white', 'xkcd:gold'), N=2)

TIME_STEPS = readParam('TIME_STEPS')
TIME_WRITE = readParam('TIME_WRITE')
TIME_SYNC = readParam('TIME_SYNC')
NUM_FILES = int(TIME_STEPS / TIME_WRITE) + 1
NUM_CONTROL_STEPS = int(TIME_STEPS / TIME_SYNC) + 1
DEFECT_ORDER_THRESH = readParam('DEFECT_ORDER_THRESH')  # lower is stricter
OUTPUT_FILE_PREFIX = readParam('OUTPUT_FILE_PREFIX')
OUTPUT_FILE_PATH = src_path + '/cpp/' + OUTPUT_FILE_PREFIX
GEOM_NAME = readParam('GEOM_NAME')
OBSTACLE_COLOR = 'xkcd:silver'
QUIVER_COLOR = 'xkcd:gunmetal'
GOAL_COLOR = 'xkcd:green'
NEW_DEFECT_DETECTION_THRESH = 20  # in lattice units

I = readParam('I')
J = readParam('J')
max_index = I * J - 1

quiver_width = 0.001 if GEOM_NAME.startswith('CROSS_MAZE') else 0.0015
defect_marker_size = 130 if GEOM_NAME.startswith('CROSS_MAZE') else 200
goal_ring_radius = 20 if GEOM_NAME.startswith('CROSS_MAZE') else 16
goal_ring_thickness = 5 if GEOM_NAME.startswith('CROSS_MAZE') else 4

stills_index_list = TIME_SYNC * np.array(list(range(NUM_CONTROL_STEPS)))

fig1, ax1 = plt.subplots()
fig1.set_figheight(4)
fig1.set_figwidth(4)
fig1.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
fig2, ax2 = plt.subplots()
fig2.set_figheight(4)
fig2.set_figwidth(4)
fig2.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None) 

sample_indices = generateSampleIndices()

x_defects_list = {}
y_defects_list = {}
defect_colors_list = {}
nearest_plus_defect_history = []
nearest_plus_defect_history_x_list = {}
nearest_plus_defect_history_y_list = {}


def plotGeometryObstacles(ax, xmin, xmax, ymin, ymax):
    if GEOM_NAME == 'FREE':
        plotFreeGeometryObstacles(ax, xmin, xmax, ymin, ymax)
    elif GEOM_NAME == 'SINGLE_CHANNEL':
        plotSingleChannelGeometryObstacles(ax, xmin, xmax, ymin, ymax)
    elif GEOM_NAME == 'T_JUNCTION':
        plotTJunctionGeometryObstacles(ax, xmin, xmax, ymin, ymax)
    elif GEOM_NAME == 'CROSS_JUNCTION':
        plotCrossJunctionGeometryObstacles(ax, xmin, xmax, ymin, ymax)
    elif GEOM_NAME in ['CROSS_MAZE_CW', 'CROSS_MAZE_CCW']:
        plotCrossMazeGeometryObstacles(ax, xmin, xmax, ymin, ymax)
    else:
        raise ValueError(f'Unsupported geometry name {GEOM_NAME}')


def plotFreeGeometryObstacles(ax, xmin, xmax, ymin, ymax):
    # no obstacles in the free geometry
    pass


def plotSingleChannelGeometryObstacles(ax, xmin, xmax, ymin, ymax):
    chan_bot_frac = 9 / 21
    chan_top_frac = 12 / 21

    rect_bot = patches.Polygon(np.vstack((np.array([xmin, ymin]),
                                               np.array([xmin, ymin + chan_bot_frac * (ymax - ymin)]),
                                               np.array([xmax, ymin + chan_bot_frac * (ymax - ymin)]),
                                               np.array([xmax, ymin]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_bot)

    rect_top = patches.Polygon(np.vstack((np.array([xmin, ymin + chan_top_frac * (ymax - ymin)]),
                                                np.array([xmax, ymin + chan_top_frac * (ymax - ymin)]),
                                                np.array([xmax, ymax]),
                                                np.array([xmin, ymax]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_top)


def plotTJunctionGeometryObstacles(ax, xmin, xmax, ymin, ymax):
    h_chan_bot_frac = 9 / 21
    h_chan_top_frac = 12 / 21
    v_chan_left_frac = 18 / 21

    rect_top = patches.Polygon(np.vstack((np.array([xmin, ymin + h_chan_top_frac * (ymax - ymin)]),
                                               np.array([xmin + v_chan_left_frac * (xmax - xmin), ymin + h_chan_top_frac * (ymax - ymin)]),
                                               np.array([xmin + v_chan_left_frac * (xmax - xmin), ymax]),
                                               np.array([xmin, ymax]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_top)

    rect_bot = patches.Polygon(np.vstack((np.array([xmin, ymin]),
                                               np.array([xmin + v_chan_left_frac * (xmax - xmin), ymin]),
                                               np.array([xmin + v_chan_left_frac * (xmax - xmin), ymin + h_chan_bot_frac * (ymax - ymin)]),
                                               np.array([xmin, ymin + h_chan_bot_frac * (ymax - ymin)]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_bot)


def plotCrossJunctionGeometryObstacles(ax, xmin, xmax, ymin, ymax):
    h_chan_bot_frac = 9 / 21
    h_chan_top_frac = 12 / 21
    v_chan_left_frac = 9 / 21
    v_chan_right_frac = 12 / 21

    rect_top_left = patches.Polygon(np.vstack((np.array([xmin, ymin + h_chan_top_frac * (ymax - ymin)]),
                                               np.array([xmin + v_chan_left_frac * (xmax - xmin), ymin + h_chan_top_frac * (ymax - ymin)]),
                                               np.array([xmin + v_chan_left_frac * (xmax - xmin), ymax]),
                                               np.array([xmin, ymax]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_top_left)

    rect_top_right = patches.Polygon(np.vstack((np.array([xmin + v_chan_right_frac * (xmax - xmin), ymin + h_chan_top_frac * (ymax - ymin)]),
                                                np.array([xmax, ymin + h_chan_top_frac * (ymax - ymin)]),
                                                np.array([xmax, ymax]),
                                                np.array([xmin + v_chan_right_frac * (xmax - xmin), ymax]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_top_right)

    rect_bot_right = patches.Polygon(np.vstack((np.array([xmin + v_chan_right_frac * (xmax  - xmin), ymin]),
                                                np.array([xmax, ymin]),
                                                np.array([xmax, ymin + h_chan_bot_frac * (ymax - ymin)]),
                                                np.array([xmin + v_chan_right_frac * (xmax - xmin), ymin + h_chan_bot_frac * (ymax - ymin)]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_bot_right)

    rect_bot_left = patches.Polygon(np.vstack((np.array([xmin, ymin]),
                                               np.array([xmin + v_chan_left_frac * (xmax - xmin), ymin]),
                                               np.array([xmin + v_chan_left_frac * (xmax - xmin), ymin + h_chan_bot_frac * (ymax - ymin)]),
                                               np.array([xmin, ymin + h_chan_bot_frac * (ymax - ymin)]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_bot_left)


def plotCrossMazeGeometryObstacles(ax, xmin, xmax, ymin, ymax):
    north_h_chan_top_frac = 24 / 33
    north_h_chan_bot_frac = 21 / 33

    west_v_chan_left_frac = 9 / 33
    west_v_chan_right_frac = 12 / 33

    south_h_chan_top_frac = 12 / 33
    south_h_chan_bot_frac = 9 / 33

    east_v_chan_left_frac = 21 / 33
    east_v_chan_right_frac = 24 / 33

    rect_top_left = patches.Polygon(np.vstack((np.array([xmin, ymin + north_h_chan_top_frac * (ymax - ymin)]),
                                               np.array([xmin + west_v_chan_left_frac * (xmax - xmin), ymin + north_h_chan_top_frac * (ymax - ymin)]),
                                               np.array([xmin + west_v_chan_left_frac * (xmax - xmin), ymax]),
                                               np.array([xmin, ymax]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_top_left)

    rect_top_right = patches.Polygon(np.vstack((np.array([xmin + east_v_chan_right_frac * (xmax - xmin), ymin + north_h_chan_top_frac * (ymax - ymin)]),
                                                np.array([xmin + east_v_chan_right_frac * (xmax - xmin), ymax]),
                                                np.array([xmax, ymax]),
                                                np.array([xmax, ymin + north_h_chan_top_frac * (ymax - ymin)]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_top_right)

    rect_bot_right = patches.Polygon(np.vstack((np.array([xmax, ymin + south_h_chan_bot_frac * (ymax - ymin)]),
                                                np.array([xmax, ymin]),
                                                np.array([xmin + east_v_chan_right_frac * (xmax - xmin), ymin]),
                                                np.array([xmin + east_v_chan_right_frac * (xmax - xmin), ymin + south_h_chan_bot_frac * (ymax - ymin)]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_bot_right)

    rect_bot_left = patches.Polygon(np.vstack((np.array([xmin, ymin]),
                                               np.array([xmin, ymin + south_h_chan_bot_frac * (ymax - ymin)]),
                                               np.array([xmin + west_v_chan_left_frac * (xmax - xmin), ymin + south_h_chan_bot_frac * (ymax - ymin)]),
                                               np.array([xmin + west_v_chan_left_frac * (xmax - xmin), ymin]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_bot_left)

    rect_mid_left = patches.Polygon(np.vstack((np.array([xmin, ymin + south_h_chan_top_frac * (ymax - ymin)]),
                                               np.array([xmin + west_v_chan_left_frac * (xmax - xmin), ymin + south_h_chan_top_frac * (ymax - ymin)]),
                                               np.array([xmin + west_v_chan_left_frac * (xmax - xmin), ymin + north_h_chan_bot_frac * (ymax - ymin)]),
                                               np.array([xmin, ymin + north_h_chan_bot_frac * (ymax - ymin)]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_mid_left)

    rect_mid_right = patches.Polygon(np.vstack((np.array([xmin + east_v_chan_right_frac * (xmax - xmin), ymin + south_h_chan_top_frac * (ymax - ymin)]),
                                               np.array([xmax, ymin + south_h_chan_top_frac * (ymax - ymin)]),
                                               np.array([xmax, ymin + north_h_chan_bot_frac * (ymax - ymin)]),
                                               np.array([xmin + east_v_chan_right_frac * (xmax - xmin), ymin + north_h_chan_bot_frac * (ymax - ymin)]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_mid_right)

    rect_mid_top = patches.Polygon(np.vstack((np.array([xmin + west_v_chan_right_frac * (xmax - xmin), ymin + north_h_chan_top_frac * (ymax - ymin)]),
                                               np.array([xmin + east_v_chan_left_frac * (xmax - xmin), ymin + north_h_chan_top_frac * (ymax - ymin)]),
                                               np.array([xmin + east_v_chan_left_frac * (xmax - xmin), ymax]),
                                               np.array([xmin + west_v_chan_right_frac * (xmax - xmin), ymax]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_mid_top)

    rect_mid_bot = patches.Polygon(np.vstack((np.array([xmin + west_v_chan_right_frac * (xmax - xmin), ymin]),
                                               np.array([xmin + east_v_chan_left_frac * (xmax - xmin), ymin]),
                                               np.array([xmin + east_v_chan_left_frac * (xmax - xmin), ymin + south_h_chan_bot_frac * (ymax - ymin)]),
                                               np.array([xmin + west_v_chan_right_frac * (xmax - xmin), ymin + south_h_chan_bot_frac * (ymax - ymin)]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_mid_bot)

    rect_center = patches.Polygon(np.vstack((np.array([xmin + west_v_chan_right_frac * (xmax - xmin), ymin + south_h_chan_top_frac * (ymax - ymin)]),
                                               np.array([xmin + east_v_chan_left_frac * (xmax - xmin), ymin + south_h_chan_top_frac * (ymax - ymin)]),
                                               np.array([xmin + east_v_chan_left_frac * (xmax - xmin), ymin + north_h_chan_bot_frac * (ymax - ymin)]),
                                               np.array([xmin + west_v_chan_right_frac * (xmax - xmin), ymin + north_h_chan_bot_frac * (ymax - ymin)]))), facecolor=OBSTACLE_COLOR, alpha=1)
    ax.add_patch(rect_center)


def computeAngleDiff(startAngle, endAngle):
    min_index = np.argmin([np.abs(endAngle - startAngle), np.abs(endAngle - startAngle + np.pi), np.abs(endAngle - startAngle - np.pi)])

    if min_index == 0:
        return endAngle - startAngle
    elif min_index == 1:
        return endAngle - startAngle + np.pi
    else:
        return endAngle - startAngle - np.pi


def updateOrientationFigure(i):
    global nearest_plus_defect_history
    global nearest_plus_defect_history_x_list
    global nearest_plus_defect_history_y_list

    ax2.clear()
    df_orientation = pd.read_csv(OUTPUT_FILE_PATH + 'orientation_' + str(i * TIME_WRITE) + '.dat', sep=' ', header=None)
    x = df_orientation.iloc[:, 0]
    y = df_orientation.iloc[:, 1]
    order = df_orientation.iloc[:, 2]
    angle = df_orientation.iloc[:, 3]
    angle[angle < 0] += np.pi  # director goes from 0 to np.pi, not 0 to 2 * np.pi
    angle[angle > np.pi] -= np.pi
    cos = np.cos(angle)
    sin = np.sin(angle)

    x_defects = x.loc[order <= DEFECT_ORDER_THRESH]
    y_defects = y.loc[order <= DEFECT_ORDER_THRESH]

    filtered_x_defects = []
    filtered_y_defects = []
    defect_colors = []
    plus_defects_list = []

    for x_defect, y_defect in zip(x_defects.to_numpy(), y_defects.to_numpy()):
        i0 = x_defect + y_defect * I
        i1 = x_defect - 1 + (y_defect + 1) * I
        i2 = x_defect + (y_defect + 1) * I
        i3 = x_defect + 1 + (y_defect + 1) * I
        i4 = x_defect - 1 + y_defect * I
        i5 = x_defect + 1 + y_defect * I
        i6 = x_defect - 1 + (y_defect - 1) * I
        i7 = x_defect + (y_defect - 1) * I
        i8 = x_defect + 1 + (y_defect - 1) * I
        defect_order = order[i0]

        if all(np.array([i1, i2, i3, i4, i5, i6, i7, i8]) < max_index) and all(np.array([i1, i2, i3, i4, i5, i6, i7, i8]) > 0):
            if defect_order < order[i1] and defect_order < order[i2] and defect_order < order[i3] and defect_order < order[i4] and defect_order < order[i5] and defect_order < order[i6] and defect_order < order[i7] and defect_order < order[i8]:
                winding_number = 0
                winding_number += computeAngleDiff(angle[i5], angle[i3])
                winding_number += computeAngleDiff(angle[i3], angle[i2])
                winding_number += computeAngleDiff(angle[i2], angle[i1])
                winding_number += computeAngleDiff(angle[i1], angle[i4])
                winding_number += computeAngleDiff(angle[i4], angle[i6])
                winding_number += computeAngleDiff(angle[i6], angle[i7])
                winding_number += computeAngleDiff(angle[i7], angle[i8])
                winding_number += computeAngleDiff(angle[i8], angle[i5])
                winding_number = np.round(winding_number / (2 * np.pi), decimals=1)
                
                if winding_number == 0.5:
                    filtered_x_defects.append(x_defect)
                    filtered_y_defects.append(y_defect)
                    defect_colors.append('xkcd:red')
                    plus_defects_list.append([x_defect, y_defect])
                elif winding_number == -0.5:
                    filtered_x_defects.append(x_defect)
                    filtered_y_defects.append(y_defect)
                    defect_colors.append('xkcd:azure')

    df_activity = pd.read_csv(OUTPUT_FILE_PATH + 'activity_' + str(i * TIME_WRITE) + '.dat', sep=' ', header=None)
    activity = df_activity.iloc[:, 2]
    ax2.imshow(activity.to_numpy().reshape(J, I), cmap=cmap_act, alpha=0.5, origin='lower', extent=[0, I, 0, J])

    im = ax2.quiver(x[sample_indices], y[sample_indices], cos[sample_indices], sin[sample_indices], pivot='mid', width=quiver_width, scale=50, color=QUIVER_COLOR, headlength=0, headaxislength=0)  # headless quivers for nematics

    xmin, xmax = np.min(x), np.max(x) + 1
    ymin, ymax = np.min(y), np.max(y) + 1
    plotGeometryObstacles(ax2, xmin, xmax, ymin, ymax)

    df_goal = pd.read_csv(OUTPUT_FILE_PATH + 'goal_' + str(i * TIME_WRITE) + '.dat', sep=' ', header=None)
    goal_horiz_frac = df_goal.iloc[0, 0]
    goal_vert_frac = df_goal.iloc[0, 1]
    goal_x, goal_y = goal_horiz_frac * I, goal_vert_frac * J
    goal_ring = patches.Wedge((goal_x, goal_y), goal_ring_radius, 0, 360, width=goal_ring_thickness, color=GOAL_COLOR)
    ax2.add_patch(goal_ring)

    if len(plus_defects_list) > 0:
        nearest_plus_defect_index = np.argmin(np.linalg.norm(np.vstack(plus_defects_list) - np.array([goal_x, goal_y]), axis=1)) if len(plus_defects_list) > 1 else 0
        nearest_plus_defect = plus_defects_list[nearest_plus_defect_index]

        if len(nearest_plus_defect_history) > 0 and np.linalg.norm(np.linalg.norm(np.array(nearest_plus_defect) - np.array(nearest_plus_defect_history[-1]))) > NEW_DEFECT_DETECTION_THRESH:
            nearest_plus_defect_history = []  # clear history

        nearest_plus_defect_history.append(nearest_plus_defect)
        nearest_plus_defect_history_array = np.array(nearest_plus_defect_history)
        nearest_plus_defect_history_x, nearest_plus_defect_history_y = nearest_plus_defect_history_array[:, 0], nearest_plus_defect_history_array[:, 1]
        nearest_plus_defect_history_x_list[str(i)] = nearest_plus_defect_history_x
        nearest_plus_defect_history_y_list[str(i)] = nearest_plus_defect_history_y
        ax2.plot(nearest_plus_defect_history_x, nearest_plus_defect_history_y, c='xkcd:red')

    x_defects_list[str(i)] = filtered_x_defects
    y_defects_list[str(i)] = filtered_y_defects
    defect_colors_list[str(i)] = defect_colors
    im = ax2.scatter(filtered_x_defects, filtered_y_defects, s=defect_marker_size, c=defect_colors)
    ax2.set_axis_off()

    timestep_idx = i * TIME_WRITE

    if timestep_idx in stills_index_list:
        sync_timestep = int(timestep_idx / TIME_SYNC)
        fig2.savefig(dir_path + f'/output/stills/orientation_{GEOM_NAME}_{sync_timestep}.pdf', dpi=600)

    return im,


def updateVelocityFigure(i):
    ax1.clear()
    df_velocity = pd.read_csv(OUTPUT_FILE_PATH + 'velocity_' + str(i * TIME_WRITE) + '.dat', sep=' ', header=None)
    x = df_velocity.iloc[:, 0]
    y = df_velocity.iloc[:, 1]
    vx = df_velocity.iloc[:, 3]
    vy = df_velocity.iloc[:, 4]

    df_activity = pd.read_csv(OUTPUT_FILE_PATH + 'activity_' + str(i * TIME_WRITE) + '.dat', sep=' ', header=None)
    activity = df_activity.iloc[:, 2]
    ax1.imshow(activity.to_numpy().reshape(J, I), cmap=cmap_act, alpha=0.5, origin='lower')

    im = ax1.quiver(x[sample_indices], y[sample_indices], vx[sample_indices], vy[sample_indices], pivot='mid', scale=0.08, scale_units='width', width=quiver_width, color=QUIVER_COLOR)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    plotGeometryObstacles(ax1, xmin, xmax, ymin, ymax)

    df_goal = pd.read_csv(OUTPUT_FILE_PATH + 'goal_' + str(i * TIME_WRITE) + '.dat', sep=' ', header=None)
    goal_horiz_frac = df_goal.iloc[0, 0]
    goal_vert_frac = df_goal.iloc[0, 1]
    goal_ring = patches.Wedge((goal_horiz_frac * I, goal_vert_frac * J), goal_ring_radius, 0, 360, width=goal_ring_thickness, color=GOAL_COLOR)
    ax1.add_patch(goal_ring)
 
    if str(i) in nearest_plus_defect_history_x_list:
        nearest_plus_defect_history_x = nearest_plus_defect_history_x_list[str(i)]
        nearest_plus_defect_history_y = nearest_plus_defect_history_y_list[str(i)]
        ax1.plot(nearest_plus_defect_history_x, nearest_plus_defect_history_y, c='xkcd:red')

    im = ax1.scatter(x_defects_list[str(i)], y_defects_list[str(i)], s=defect_marker_size, c=defect_colors_list[str(i)])
    ax1.set_axis_off()

    timestep_idx = i * TIME_WRITE

    if timestep_idx in stills_index_list:
        sync_timestep = int(timestep_idx / TIME_SYNC)
        fig1.savefig(dir_path + f'/output/stills/velocity_{GEOM_NAME}_{sync_timestep}.pdf', dpi=600)

    return im,


# NOTE: Order is important
orientation_animation_fig = animation.FuncAnimation(fig2, updateOrientationFigure, frames=NUM_FILES, interval=60, blit=True, repeat_delay=2,)
orientation_animation_fig.save(dir_path + '/output/orientation.mp4', writer='ffmpeg', dpi=600, savefig_kwargs=dict(facecolor='xkcd:white'))
velocity_animation_fig = animation.FuncAnimation(fig1, updateVelocityFigure, frames=NUM_FILES, interval=60, blit=True, repeat_delay=2,)
velocity_animation_fig.save(dir_path + '/output/velocity.mp4', writer='ffmpeg', dpi=600, savefig_kwargs=dict(facecolor='xkcd:white'))