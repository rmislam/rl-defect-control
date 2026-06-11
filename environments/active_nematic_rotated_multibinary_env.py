import os
dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = dir_path + '/../src'
import sys
sys.path.insert(1, src_path + '/python')
import numpy as np
import pandas as pd
import gymnasium as gym
from typing import Optional
from scipy.ndimage import gaussian_filter
from param_utils import readParam

I_obs = readParam('I')
J_obs = readParam('J')
I_model = readParam('I_model')  # Note that I_model must be less than or equal to I_obs (same for J)
J_model = readParam('J_model')
ALPHA = readParam('ALPHA')


class ActiveNematicEnv(gym.Env):
    def __init__(self, send_pipe_path, recv_pipe_path):
        self._send_pipe_path = send_pipe_path
        self._recv_pipe_path = recv_pipe_path
        self._is_done = False
        self._px = None
        self._py = None
        self._allowed_primitives_list = []
        self.loadAllowedPrimitivesList()
        self._goal_frac_i = None
        self._goal_frac_j = None
        self._goal_field = None
        self._nearest_defect_i = None
        self._nearest_defect_j = None
        self._I_start = 0
        self._I_end = I_model
        self._J_start = 0
        self._J_end = J_model
        self._rot_multiple = 0
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(6, I_model, J_model), dtype=np.float32)  # depth of 6: [velocity_x, velocity_y, Q_xx, Q_xy, defects_field, goal_field]
        self.action_space = gym.spaces.MultiBinary(len(self._allowed_primitives_list))

    def setQuadrant(self, quadrant):
        assert quadrant in ['BL', 'BR', 'TL', 'TR'], "quadrant argument must be one of 'BL', 'BR', 'TL', or 'TR'"

        if quadrant == 'BL':    # bottom left
            self._I_start = 0
            self._I_end = I_model
            self._J_start = 0
            self._J_end = J_model
        elif quadrant == 'BR':  # bottom right
            self._I_start = I_obs - I_model
            self._I_end = I_obs
            self._J_start = 0
            self._J_end = J_model
        elif quadrant == 'TL':  # top left
            self._I_start = 0
            self._I_end = I_model
            self._J_start = J_obs - J_model
            self._J_end = J_obs
        elif quadrant == 'TR':  # top right
            self._I_start = I_obs - I_model
            self._I_end = I_obs
            self._J_start = J_obs - J_model
            self._J_end = J_obs

    def setRotation(self, rot_multiple):
        self._rot_multiple = rot_multiple

    def _extractSubmatrix(self, matrix):
        submatrix_prerot = matrix[self._I_start:self._I_end, self._J_start:self._J_end, :]
        submatrix = np.rot90(submatrix_prerot, k=self._rot_multiple)
        return submatrix

    def _injectSubmatrix(self, matrix, submatrix):
        submatrix_prerot = np.rot90(submatrix, k=-self._rot_multiple)
        matrix[self._I_start:self._I_end, self._J_start:self._J_end] = submatrix_prerot

    def _rotateIndices(self, i, j, rot_multiple):
        if rot_multiple > 0:
            for _ in range(rot_multiple):
                i, j = self._rotateIndicesBy90(i, j)
        elif rot_multiple < 0:
            for _ in range(-rot_multiple):
                i, j = self._rotateIndicesBy90(i, j, reverse=True)
        # if rot_multiple == 0, do nothing

        return i, j

    def _rotateIndicesBy90(self, i, j, reverse=False):
        if reverse:
            i = I_model - 1 - i
            i, j = j, i
        else:
            i, j = j, i
            i = I_model - 1 - i

        return i, j

    def loadAllowedPrimitivesList(self, filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat'):
        self._allowed_primitives_list = []

        with open(filepath, 'rb') as f:
            for line_idx, line in enumerate(f):
                line_tokens = line.decode('utf-8').strip().split()

                if line_idx == 0:
                    self._px, self._py = [int(token) for token in line_tokens[2:4]]
                else:
                    self._allowed_primitives_list.append([primitive.tolist() for primitive in np.array(line_tokens).reshape(-1, 2).astype(int)])

    def _isPointInActivityPattern(self, i, j, pattern_vector):
        for primitive_idx, is_primitive_on in enumerate(pattern_vector):
            if is_primitive_on:
                primitives = self._allowed_primitives_list[primitive_idx]

                for primitive in primitives:
                    x_idx, y_idx = primitive

                    if i >= self._px * x_idx and i < self._px * (x_idx + 1):
                        if j >= self._py * y_idx and j < self._py * (y_idx + 1):
                            return True

        return False

    def _writeActivity(self, px, py, pattern_vector):
        with open(src_path + '/cpp/input/activity.dat', 'w') as f:
            activity_model = np.zeros((I_model, J_model))
            activity_obs = np.zeros((I_obs, J_obs))

            for i in range(I_model):
                for j in range(J_model):
                    activity_model[i, j] = ALPHA if self._isPointInActivityPattern(i, j, pattern_vector) else 0

            self._injectSubmatrix(activity_obs, activity_model)

            for j in range(J_obs):
                for i in range(I_obs):
                    f.write(f'{i} {j} {activity_obs[i, j]:.6f}\n')

    def _readReward(self):
        with open(src_path + '/cpp/output/reward.dat', 'r') as f:
            reward = np.double(f.readline().rstrip().split()[0])

        return reward

    def setGoal(self, goal=None):
        if goal is not None:
            self._goal_frac_i = goal[0]
            self._goal_frac_j = goal[1]
        else:
            self._goal_frac_i = np.round(0.1 * np.random.random() + 0.7, 2)  # 0.7 to 0.8
            self._goal_frac_j = 0.5

        goal_map = np.zeros((I_model, J_model), dtype=np.float32)
        goal_map[int(np.round(self._goal_frac_i * I_model)), int(np.round(self._goal_frac_j * J_model))] = 1.0
        self._goal_field = gaussian_filter(goal_map, sigma=1.5)
        max_goal_value = np.abs(self._goal_field).max()

        # Renormalize goal_field to lie between -1 and 1 for better gradients and signal strength
        if max_goal_value > np.finfo(np.float32).eps:
            self._goal_field /= max_goal_value

        self._writeGoal()

    def _writeGoal(self):
        with open(src_path + '/cpp/input/goal.param', 'w') as f:
            i_prerot, j_prerot = self._goal_frac_i * I_model, self._goal_frac_j * J_model
            i, j = self._rotateIndices(i_prerot, j_prerot, rot_multiple=-self._rot_multiple)
            goal_frac_i_obs = (self._I_start + i) / I_obs
            goal_frac_j_obs = (self._J_start + j) / J_obs
            f.write(f'{goal_frac_i_obs:.2f} {goal_frac_j_obs:.2f}\n')
    
    def _readIsTruncAndSetNearestDefect(self):
        is_trunc = False

        with open(src_path + '/cpp/output/defect_obs.dat', 'r') as f:
            tokens = np.array(f.readline().rstrip().split(), dtype=np.float32)

            if int(tokens[0]) == -1:  # no defect was detected in simulation
                is_trunc = True
                self._nearest_defect_i = None
                self._nearest_defect_j = None
            else:
                tokens_list = [tokens[:2]]

                for line in f.readlines():  # all lines after the first line
                    tokens = np.array(line.rstrip().split(), dtype=np.float32)
                    tokens_list.append(tokens[:2])

                # find nearest defect from tokens_list
                diff_vector_array = np.array(tokens_list) - np.array([self._goal_frac_i * I_obs, self._goal_frac_j * J_obs])
                nearest_defect_idx = np.argmin([np.linalg.norm(diff_vector) for diff_vector in diff_vector_array])
                nearest_defect = tokens_list[nearest_defect_idx]
                self._nearest_defect_i = nearest_defect[0]
                self._nearest_defect_j = nearest_defect[1]

        return is_trunc

    def getNearestDefect(self):
        return self._nearest_defect_i, self._nearest_defect_j

    def _readObservations(self):
        velocity_df = pd.read_csv(src_path + '/cpp/output/velocity.dat', sep='\s+', header=None)
        velocity_field_obs = 1e4 * velocity_df.iloc[:, [3, 4]].to_numpy().reshape(I_obs, J_obs, 2, order='F')  # multiply by 1e4 because std is about 1e-4
        velocity_field = self._extractSubmatrix(velocity_field_obs)

        orientation_df = pd.read_csv(src_path + '/cpp/output/orientation.dat', sep='\s+', header=None)
        orientation_field_obs = orientation_df.iloc[:, [2, 3]].to_numpy().reshape(I_obs, J_obs, 2, order='F')
        orientation_field = self._extractSubmatrix(orientation_field_obs)

        defects_df = pd.read_csv(src_path + '/cpp/output/defects.dat', sep='\s+', header=None)
        defects_field_obs = gaussian_filter(defects_df.iloc[:, [2]].to_numpy().reshape(I_obs, J_obs, 1, order='F'), sigma=1.5)
        defects_field = self._extractSubmatrix(defects_field_obs)
        max_defect_value = np.abs(defects_field).max()

        # Renormalize defects_field to lie between -1 and 1 for better gradients and signal strength
        if max_defect_value > np.finfo(np.float32).eps:
            defects_field /= max_defect_value

        is_trunc = self._readIsTruncAndSetNearestDefect()
        return np.moveaxis(np.dstack([velocity_field, orientation_field, defects_field, self._goal_field]), -1, 0), is_trunc  # move channels axis to front

    def _readIsTrunc(self):
        with open(src_path + '/cpp/output/defect_obs.dat', 'r') as f:
            tokens = np.array(f.readline().rstrip().split(), dtype=np.float32)
            is_trunc = int(tokens[0]) == -1  # no defect was detected in simulation

        return is_trunc

    def initSim(self):
        print('Waiting for simulation to be initialized...')
        with open(self._recv_pipe_path, 'r') as recv_pipe:
            recv_pipe.read()
            print('Simulation has been initialized')

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.setGoal(goal=options.get('goal', None) if options else None)
        super().reset(seed=seed)

        with open(self._send_pipe_path, 'w') as send_pipe:
            send_pipe.write('1')  # 0 is continue, 1 is reset

        with open(self._recv_pipe_path, 'r') as recv_pipe:
            data = recv_pipe.read()
            self._is_done = True if data == '1' else False

        obs, _ = self._readObservations()
        return obs, {}

    def step(self, action):
        self._writeActivity(self._px, self._py, action)

        with open(self._send_pipe_path, 'w') as send_pipe:
            send_pipe.write('0')  # 0 is continue, 1 is reset

        with open(self._recv_pipe_path, 'r') as recv_pipe:
            data = recv_pipe.read()
            self._is_done = True if data == '1' else False

        reward = self._readReward()
        obs, is_trunc = self._readObservations()
        return obs, reward, self._is_done, is_trunc, {}

    def getState(self):
        state, _ = self._readObservations()
        return state

    def endSim(self):
        while not self._is_done:
            self._is_done = self.step(np.zeros(self.action_space.n))[2]

        with open(self._send_pipe_path, 'w') as send_pipe:
            send_pipe.write('0')  # 0 is continue, so calling this after the last time step will end the sim
