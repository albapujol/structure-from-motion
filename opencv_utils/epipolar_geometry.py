import numpy as np
import cv2


def estimate_fundamental_matrix(p_set, q_set):
    return cv2.findFundamentalMat(p_set[:, :-1], q_set[:, :-1])[0]



def estimate_fundamental_matrix_ransac(p_set, q_set, th=2.0):
    return estimate_fundamental_matrix(p_set[:,:-1].ravel(), q_set[:,:-1].ravel())


def evaluate_camera_matrices(p, q, Pp, Pqs):
    return None

def estimate_camera_matrix_from_fundamental(F, p, q):
    return None

def triangulate(ps, qs, Pp, Pq):
    return None

def estimate_camera_matrices(p_set, q_set):
    """
    Estimate the camera matrix given a set of correspondences
    :param p_set: Corresponding points on image #1
    :param q_set: Corresponding points on image #2
    :param K: Intrinsic camera matrix
    :param imshape: Image size
    :return:
    """
    p_h = np.concatenate((p_set, np.ones((len(p_set), 1))), axis=1)
    q_h = np.concatenate((q_set, np.ones((len(p_set), 1))), axis=1)
    F = estimate_fundamental_matrix_ransac(p_h, q_h)
    Pp, Pq = estimate_camera_matrix_from_fundamental(F, p_set[0], q_set[0])
    return Pp, Pq


