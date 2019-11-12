import numpy as np


def estimate_fundamental_matrix(p_set, q_set):
    """
    Estimate the fundamental matrix using the normalized 8-point algorithm
    https://en.wikipedia.org/wiki/Eight-point_algorithm
    :param p_set: Points on image P
    :param q_set: Points on image q
    :return: Fundamental matrix
    """
    # Normalize points
    p_normv = np.linalg.norm(p_set, axis=-1)[:, np.newaxis]
    q_normv = np.linalg.norm(q_set, axis=-1)[:, np.newaxis]
    p_norm = p_set / p_normv
    q_norm = q_set / q_normv

    # Row function for homogenoeus linear equation
    row_f = lambda p, q: np.array([p[0] * q[0], p[0] * q[1], p[0],
                                   p[1] * q[0], p[1] * q[1], p[1],
                                   q[0], q[1], 1])

    # Build Y matrix
    Y = np.zeros((6, len(p_set)))
    for i, p, q in (p_norm, q_norm):
        Y[i] = row_f(p, q)

    # Compute SVD
    _, _, Vt = np.linalg.svd(Y)

    # Create fundamental matrix
    F_r = np.reshape(Vt[2, :], (3, 3))

    # Enforce internal constraint
    U, D, Vt = np.linalg.svd(F_r)
    D[-1, -1]: 0
    F = U @ D @ Vt

    # Denormalize
    F = p_normv.T @ F @ q_normv
    return F


def evaluate_camera_matrices(p, q, Ps, imshape):
    """
    Evaluate the different options for camera matrix
    :param p: Test point #1
    :param q: Test point #2
    :param Ps: List containing the possible camera matrices
    :param imshape: Image size
    :return: Correct camera matrix
    """
    Pp = np.concatenate((np.eye(3), np.zeros(3).T), axis=1)
    for Pq in Ps:
        pos3D = triangulate(p, q, Pp, Pq, imshape)
        pos3D = pos3D / pos3D[3]
        poscam1 = Pp @ pos3D
        poscam2 = Pq @ pos3D
        if poscam1[2] > 0 and poscam2[2] > 0:
            return Pq


def estimate_camera_matrices_from_fundamental(K, F):
    """
    Estimat the camera matrix between two point sets using the fundamental matrix.
    :param K: Intrinsic camera matrix
    :param F: Fundamental matrix
    :return: Camera matrix candidates
    """
    # Estimate essential matrix
    E = K.T @ F @ K

    # Compute candidate camera matrices
    [U, _, Vt] = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ Vt
    if np.linalg.det(R1) < 0:
        R1 = -R1
    R2 = U @ W.T @ Vt
    if np.linalg.det(R2) < 0:
        R2 = -R2
    return [
        K @ np.concatenate((R1, U[:, 2]), axis=1),
        K @ np.concatenate((R1, -U[:, 2]), axis=1),
        K @ np.concatenate((R2, U[:, 2]), axis=1),
        K @ np.concatenate((R2, -U[:, 2]), axis=1)
    ]


def triangulate(p, q, Pp, Pq, imshape):
    """
    Traingulate a 2D pair p, q given their point coordinates and the camera matrices
    :param p: Point #1
    :param q: Point #2
    :param Pp: Camera matrix for point #1
    :param Pq: Camera matrix for point #2
    :param imshape: Image size
    :return: 3D coordinates of the triangulated point
    """
    H = np.array([[2.0 / float(imshape[0]), 0, -1],
                  [0, 2.0 / float(imshape[1]), -1],
                  [0, 0, 1]])

    # Normalize coordinates
    if len(p) == 2:
        p = np.array([p[0], p[1], 1])
    else:
        p = p / p[2]
    if len(q) == 2:
        q = np.array([q[0], q[1], 1])
    else:
        q = q / q[2]

    # Homogeneous coordinates
    p = H @ p
    q = H @ q
    Pp = H @ Pp
    Pq = H @ Pq

    # Compute matrix A
    A = np.zeros((4, 4))
    A[0, :] = p[0] * Pp[2, :]
    A[1, :] = p[1] * Pp[2, :]
    A[2, :] = q[0] * Pq[2, :]
    A[3, :] = q[1] * Pq[2, :]

    _, _, Vt = np.linalg.svd(A)
    return Vt[-1, :]


def estimate_camera_matrix(p_set, q_set, K, imshape):
    """
    Estimate the camera matrix given a set of correspondences
    :param p_set: Corresponding points on image #1
    :param q_set: Corresponding points on image #2
    :param K: Intrinsic camera matrix
    :param imshape: Image size
    :return:
    """
    F = estimate_fundamental_matrix(p_set, q_set)
    Ps = estimate_camera_matrices_from_fundamental(K, F)
    P = evaluate_camera_matrices(p_set[0], q_set[0], Ps, imshape)
    return P

