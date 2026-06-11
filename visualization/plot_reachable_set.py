#!/usr/bin/env python3
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = dir_path + '/../src'
reachability_path = dir_path + '/../reachability'
output_path = dir_path + '/output'
import sys
sys.path.insert(1, src_path + '/python')
import numpy as np
import pandas as pd
import pickle
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from concave_hull import concave_hull
from param_utils import readParam, generateSampleIndices

rng = np.random.default_rng(42)

DEFECT_ORDER_THRESH = readParam('DEFECT_ORDER_THRESH')  # lower is stricter
OUTPUT_FILE_PREFIX = readParam('OUTPUT_FILE_PREFIX')
OUTPUT_FILE_PATH = src_path + '/cpp/' + OUTPUT_FILE_PREFIX
GEOM_NAME = readParam('GEOM_NAME')
OBSTACLE_COLOR = 'xkcd:silver'
QUIVER_COLOR = 'xkcd:gunmetal'
GOAL_COLOR = 'xkcd:green'

I = readParam('I')
J = readParam('J')
max_index = I * J - 1

defect_marker_size = 130 if GEOM_NAME.startswith('CROSS_MAZE') else 200

fig, ax = plt.subplots()
fig.set_figheight(4)
fig.set_figwidth(4)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax.axis('equal')

sample_indices = generateSampleIndices()

x_defects = []
y_defects = []
defect_colors = []

# NOTE: Make sure to change this if you change initial defect positions in addDefect() in defect.c
if GEOM_NAME == 'FREE':
    # initial +1/2 defect
    x_defects.append(53)
    y_defects.append(210)
    defect_colors.append('xkcd:red')

    # initial -1/2 defect
    x_defects.append(385)
    y_defects.append(210)
    defect_colors.append('xkcd:azure')
elif GEOM_NAME == 'T_JUNCTION':
    # initial +1/2 defect
    x_defects.append(253)
    y_defects.append(210)
    defect_colors.append('xkcd:red')

    # initial -1/2 defect
    x_defects.append(166)
    y_defects.append(210)
    defect_colors.append('xkcd:azure')
elif GEOM_NAME == 'CROSS_JUNCTION':
    # initial +1/2 defect
    x_defects.append(53)
    y_defects.append(210)
    defect_colors.append('xkcd:red')

    # initial -1/2 defect
    x_defects.append(385)
    y_defects.append(210)
    defect_colors.append('xkcd:azure')
else:
    raise ValueError(f'Unsupported geometry name {GEOM_NAME}')


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

    rect_bot_right = patches.Polygon(np.vstack((np.array([xmin + v_chan_right_frac * (xmax - xmin), ymin]),
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


def plotGeometryAndReachableSet(ax):
    # NOTE: make sure you have run LBM simulation with parameter WRITE_VIZ_DATA set to TRUE at least for timestep 0
    df_orientation = pd.read_csv(OUTPUT_FILE_PATH + 'orientation_0.dat', sep=' ', header=None)
    x = df_orientation.iloc[:, 0]
    y = df_orientation.iloc[:, 1]
    order = df_orientation.iloc[:, 2]
    angle = df_orientation.iloc[:, 3]
    angle[angle < 0] += np.pi  # director goes from 0 to np.pi, not 0 to 2 * np.pi
    angle[angle > np.pi] -= np.pi
    cos = np.cos(angle)
    sin = np.sin(angle)

    # NOTE: order of plotting is intentional
    # plot initial director field
    ax.quiver(x[sample_indices], y[sample_indices], cos[sample_indices], sin[sample_indices], pivot='mid', width=0.0015, scale=50, color=QUIVER_COLOR, headlength=0, headaxislength=0)  # headless quivers for nematics

    # plot reachable set concave hull
    with open(reachability_path + '/reached_positions.pp', 'rb') as f:
        reached_positions = pickle.load(f)

    reached_positions_array = np.concatenate(list(reached_positions.values()))
    concave_hull_points = concave_hull(reached_positions_array, concavity=3)
    ax.add_patch(Polygon(concave_hull_points, alpha=1, facecolor='xkcd:light teal', linewidth=1, edgecolor='xkcd:light teal'))

    # plot reachable set sample points
    ax.scatter(*zip(*reached_positions_array), s=2, c='xkcd:teal')

    # plot initial defect positions
    ax.scatter(x_defects, y_defects, s=defect_marker_size, c=defect_colors)

    ax.set_axis_off()
    plt.xlim(0, I)
    plt.ylim(0, J)

    # plot geometry obstacles
    xmin, xmax = np.min(x), np.max(x) + 1
    ymin, ymax = np.min(y), np.max(y) + 1
    plotGeometryObstacles(ax, xmin, xmax, ymin, ymax)


plotGeometryAndReachableSet(ax)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig(output_path + '/reachable_set.pdf', dpi=800, bbox_inches='tight', pad_inches=0)