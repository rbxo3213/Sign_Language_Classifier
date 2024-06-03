import os
import numpy as np

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def vector_normalization(joint):
    parent_joint = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :2]
    child_joint = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :2]
    v = child_joint - parent_joint
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
    angle = np.arccos(np.einsum('nt,nt->n', v[:-1], v[1:]))
    angle = np.degrees(angle)
    angle_label = np.array([angle], dtype=np.float32)
    return v, angle_label
