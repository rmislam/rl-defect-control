import os
dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = dir_path + '/../../src'
import numpy as np

class BaselineDynamicAgent():
    def __init__(self):
        self._px = None
        self._py = None
        self._allowed_primitives_list = []
        self.loadAllowedPrimitivesList()

    def loadAllowedPrimitivesList(self, filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat'):
        with open(filepath, 'rb') as f:
            for line_idx, line in enumerate(f):
                line_tokens = line.decode('utf-8').strip().split()

                if line_idx == 0:
                    self._px, self._py = [int(token) for token in line_tokens[2:4]]
                else:
                    primitives = line_tokens[:-1]
                    deg_angle = np.float32(line_tokens[-1])
                    self._allowed_primitives_list.append([[primitive.tolist() for primitive in np.array(primitives, dtype=int).reshape(-1, 2)], deg_angle])

        self._allowed_primitives_list.append([[], 0])  # add an action corresponding to no activity

    def predict(self, state):
        # State contains (1) the position of the +1/2 defect nearest the goal, and (2) the goal position
        defect_x, defect_y = state['defect_pos']
        goal_x, goal_y = state['goal_pos']
        dist_to_goal_list = []

        for pattern in self._allowed_primitives_list:
            primitives, deg_angle = pattern

            if len(primitives) == 0:
                # No activity pattern. Assume defect stays where it is (not necessarily a justified assumption)
                dist_to_goal = np.linalg.norm([goal_x - defect_x, goal_y - defect_y])
            else:
                local_dx = (len(primitives) - 1) * self._px

                cos_theta = np.cos(np.deg2rad(deg_angle))
                sin_theta = np.sin(np.deg2rad(deg_angle))
                rotmat = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=np.float32)
                dx, dy = rotmat.dot(np.array([local_dx, 0], dtype=np.float32))

                new_x = defect_x + dx
                new_y = defect_y + dy

                dist_to_goal = np.linalg.norm([goal_x - new_x, goal_y - new_y])

            dist_to_goal_list.append(dist_to_goal)

        return np.argmin(dist_to_goal_list)
