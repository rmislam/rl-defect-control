#!/usr/bin/env python3
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = dir_path + '/../src'
import sys
sys.path.insert(1, dir_path + '/../environments')
sys.path.insert(1, src_path + '/python')
import atexit
import pickle
from active_nematic_multibinary_env import ActiveNematicEnv
#from active_nematic_discrete_env import ActiveNematicEnv

SEND_PIPE_PATH = src_path + '/cpp/python_done'
RECV_PIPE_PATH = src_path + '/cpp/c_done'

def exit_handler():
    os.remove(SEND_PIPE_PATH)
    os.remove(RECV_PIPE_PATH)
    print('Removed named pipes')

atexit.register(exit_handler)

os.mkfifo(SEND_PIPE_PATH)
os.mkfifo(RECV_PIPE_PATH)


if __name__ == "__main__":
    env = ActiveNematicEnv(SEND_PIPE_PATH, RECV_PIPE_PATH)
    env.initSim()

    num_iterations = 10000
    env.reset()
    stepIdx = 0
    initial_defect_position = env.getNearestDefect()
    reached_positions = {'step_0': [initial_defect_position]}
    realized_trajectories = []
    trajectory = [initial_defect_position]

    for iter_idx in range(num_iterations):
        if iter_idx % 10 == 0:
            print(f'iter_idx: { iter_idx }', flush=True)

        action = env.action_space.sample()
        _, _, is_done, is_trunc, _ = env.step(action)
        stepIdx += 1

        if not is_trunc:
            key = f'step_{stepIdx}'

            if key not in reached_positions:
                reached_positions[key] = []

            current_defect_position = env.getNearestDefect()
            reached_positions[key].append(current_defect_position)
            trajectory.append(current_defect_position)

        if is_done or is_trunc:
            env.reset()
            stepIdx = 0
            realized_trajectories.append(trajectory)
            trajectory = [initial_defect_position]

    # Save position data
    with open('reached_positions.pp', 'wb') as f:
        pickle.dump(reached_positions, f)

    with open('realized_trajectories.pp', 'wb') as f:
        pickle.dump(realized_trajectories, f)

    print("Completed reachability estimation")
    env.endSim()
    env.close()
