#!/usr/bin/env python3
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = dir_path + '/../src'
test_output_path = dir_path + '/../learning/test/output'
output_path = dir_path + '/output'
import sys
sys.path.insert(1, src_path + '/python')
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.rcParams['font.family'] = 'Arial'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['font.size'] = 19
from param_utils import readParam

rng = np.random.default_rng(42)

OUTPUT_FILE_PREFIX = readParam('OUTPUT_FILE_PREFIX')
OUTPUT_FILE_PATH = src_path + '/cpp/' + OUTPUT_FILE_PREFIX

# Line colors for figure
COLOR_RL = '#E74C3C'
COLOR_STATIC = '#2980B9'
COLOR_DYNAMIC = '#1ABC9C'

episode_duration = 20
penalty = 500

ax = plt.gca()
fig = ax.figure
fig.set_size_inches(6, 5.7)
ax.set_position([0.15, 0.12, 0.8, 0.75])
ax.minorticks_on()
ax.tick_params(axis='both', which='minor', bottom=True, top=True, left=True, right=True, direction='in')
ax.tick_params(axis='both', which='major', bottom=True, top=True, left=True, right=True, direction='in')
ax.set_xticks(np.arange(0, episode_duration + 1, 4))
ax.set_xticks(np.arange(0, episode_duration + 1, 1), minor=True)

with open(test_output_path + '/goal_position.pp', 'rb') as f:
    goal_position = pickle.load(f)

with open(test_output_path + '/defect_positions_rl.pp', 'rb') as f:
    defect_positions_rl = pickle.load(f)

with open(test_output_path + '/defect_positions_static.pp', 'rb') as f:
    defect_positions_static = pickle.load(f)

with open(test_output_path + '/defect_positions_dynamic.pp', 'rb') as f:
    defect_positions_dynamic = pickle.load(f)

goal_x, goal_y = goal_position
rl_errors = [np.sqrt((goal_x - defect_x) ** 2 + (goal_y - defect_y) ** 2) for (defect_x, defect_y) in defect_positions_rl if defect_x is not None and defect_y is not None]
static_errors = [np.sqrt((goal_x - defect_x) ** 2 + (goal_y - defect_y) ** 2) for (defect_x, defect_y) in defect_positions_static if defect_x is not None and defect_y is not None]
dynamic_errors = [np.sqrt((goal_x - defect_x) ** 2 + (goal_y - defect_y) ** 2) for (defect_x, defect_y) in defect_positions_dynamic if defect_x is not None and defect_y is not None]
rl_timesteps = list(range(len(rl_errors)))
static_timesteps = list(range(len(static_errors)))
dynamic_timesteps = list(range(len(dynamic_errors)))
rl_duration = len(rl_errors) - 1
static_duration = len(static_errors) - 1
dynamic_duration = len(dynamic_errors) - 1

# Time-averaged integral of absolute error (IAE) using right Riemann sums
IAE_rl = np.sum(rl_errors[1:]) / rl_duration
IAE_static = np.sum(static_errors[1:]) / static_duration
IAE_dynamic = np.sum(dynamic_errors[1:]) / dynamic_duration

# Time-averaged integral of absolute error with early termination penalty (IAEP)
IAEP_rl = (np.sum(rl_errors[1:]) + (episode_duration - rl_duration) * penalty) / episode_duration
IAEP_static = (np.sum(static_errors[1:]) + (episode_duration - static_duration) * penalty) / episode_duration
IAEP_dynamic = (np.sum(dynamic_errors[1:]) + (episode_duration - dynamic_duration) * penalty) / episode_duration

print(f'IAE_rl : {IAE_rl}')
print(f'IAE_static : {IAE_static}')
print(f'IAE_dynamic : {IAE_dynamic}')
print(f'IAEP_rl : {IAEP_rl}')
print(f'IAEP_static : {IAEP_static}')
print(f'IAEP_dynamic : {IAEP_dynamic}')

static_label = 'Static'
dynamic_label = 'Rule-based'
rl_label = 'RL'

static_name_handle, = plt.plot(static_timesteps, static_errors, linestyle='--', linewidth=2, marker='o', markersize=8, c=COLOR_STATIC, label=f'{static_label}')
static_IAE_handle = mlines.Line2D([], [], color='none', label=f'{IAE_static:>2.1f}')
#dynamic_name_handle, = plt.plot(dynamic_timesteps, dynamic_errors, linestyle=':', linewidth=3, marker='o', markersize=8, c=COLOR_DYNAMIC, label=f'{dynamic_label}')
#dynamic_IAE_handle = mlines.Line2D([], [], color='none', label=f'{IAE_dynamic:>2.1f}')
rl_name_handle, = plt.plot(rl_timesteps, rl_errors, linestyle='-', linewidth=3, marker='o', markersize=8, c=COLOR_RL, label=f'{rl_label}')
rl_IAE_handle = mlines.Line2D([], [], color='none', label=f'{IAE_rl:>2.1f}')

if static_duration < rl_duration:
    plt.plot(
    static_timesteps[-1], static_errors[-1],
    marker='x',
    markersize=16,
    color='black',
    markeredgewidth=2,
    linestyle='none')

if dynamic_duration < rl_duration:
    plt.plot(
    dynamic_timesteps[-1], dynamic_errors[-1],
    marker='x',
    markersize=16,
    color='black',
    markeredgewidth=2,
    linestyle='none')

ymin, ymax = plt.ylim()
plt.ylim(ymin, max(1.05 * ymax, 105))  # NOTE: This makes sure 100 appears as one of the tick marks (with some space above). This preserves spacing / figure scaling since the y-axis will always contain at least one 3 digit tick mark.

#plt.title('Cross junction (bottom channel)', fontweight='bold', fontsize=24, pad=20)
#plt.title('Cross junction (right channel)', fontweight='bold', fontsize=24, pad=20)
#plt.title('Cross junction (top channel)', fontweight='bold', fontsize=24, pad=20)
#plt.title('T-junction (top channel)', fontweight='bold', fontsize=24, pad=20)
plt.title('T-junction (bottom channel)', fontweight='bold', fontsize=24, pad=20)
#plt.title('Free geometry (8-way patterns)', fontweight='bold', fontsize=24, pad=20)
plt.xlabel('Time / ' + rf'$\tau$')
plt.ylabel('Defect distance to goal (l.u.)')

col1_name_handle, = plt.plot([], [], ' ', label='Controller')
col2_name_handle, = plt.plot([], [], ' ', label='IAE')
handles = [col1_name_handle, rl_name_handle, static_name_handle, col2_name_handle, rl_IAE_handle, static_IAE_handle]
#handles = [col1_name_handle, rl_name_handle, dynamic_name_handle, col2_name_handle, rl_IAE_handle, dynamic_IAE_handle]
labels = [h.get_label() for h in handles]
legend = plt.legend(handles=handles, labels=labels, ncols=2, columnspacing=0, fontsize=19, loc='upper right', borderaxespad=0.8)
legend.get_frame().set_boxstyle('square')
legend.get_frame().set_edgecolor('xkcd:black')
legend.get_frame().set_linewidth(0.8)

for text in legend.get_texts():
    if text.get_text() in [col1_name_handle.get_label(), col2_name_handle.get_label()]:
        text.set_fontweight('bold')

plt.savefig(output_path + '/integral_error_plot.pdf', dpi=800)