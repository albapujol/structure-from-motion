import numpy as np


def normalize(points):
    """
    Normaize 2D points in homogeneous coordinates
    Based on a code by Peter Kovesi
    https://github.com/carandraug/PeterKovesiImage
    :param points: Image points in homogeneous coordinates
    :return: T, normalized points
    """
    c = points.mean(axis=0, keepdims=True)
    c[0, 2] = 0
    outpoints = points-c
    meandist = np.sqrt(outpoints[:, 0]**2 + outpoints[:, 1]**2).mean()
    scale = np.sqrt(2) / meandist
    T = np.array([[scale, 0, -scale*c[0, 0]], [0, scale, -scale*c[0, 1]], [0, 0, 1]])
    outpoints = (T @ outpoints.T).T
    return T, outpoints


def estimate_fundamental_matrix(p_set, q_set):
    """
    Estimate the fundamental matrix using the normalized 8-point algorithm
    https://en.wikipedia.org/wiki/Eight-point_algorithm
    :param p_set: Points on image P
    :param q_set: Points on image q
    :return: Fundamental matrix
    """
    p_h = np.concatenate((p_set, np.ones((len(p_set), 1))), axis=1)
    q_h = np.concatenate((q_set, np.ones((len(p_set), 1))), axis=1)
    # Normalize points
    Tp, p_norm = normalize(p_h)
    Tq, q_norm = normalize(q_h)

    # Row function for homogenoeus linear equation
    row_f = lambda p, q: np.array([p[0] * q[0], p[0] * q[1], p[0],
                                   p[1] * q[0], p[1] * q[1], p[1],
                                   q[0], q[1], 1])

    # Build Y matrix
    Y = np.zeros((len(p_set), 9))
    for i, (p, q) in enumerate(zip(p_norm, q_norm)):
        Y[i, :] = row_f(p, q)

    # Compute SVD
    _, _, Vt = np.linalg.svd(Y)

    # Create fundamental matrix
    F_r = np.reshape(Vt[2, :], (3, 3))

    # Enforce internal constraint
    U, D, Vt = np.linalg.svd(F_r)
    Dm = np.array([[D[0], 0, 0], [0, D[1], 0], [0, 0, 0]])
    F = U @ Dm @ Vt

    # Denormalize
    F = Tp.T @ F @ Tq

    return F


def evaluate_camera_matrices(p, q, Pp, Pqs):
    """
    Evaluate the different options for camera matrix
    :param p: Test point #1
    :param q: Test point #2
    :param Ps: List containing the possible camera matrices
    :param imshape: Image size
    :return: Correct camera matrix
    """
    for Pq in Pqs:
        pos3D = triangulate(p, q, Pp, Pq)
        print(pos3D)
        print(Pp)
        print(Pq)
        pos3D = pos3D / pos3D[2]
        poscam1 = Pp.T @ pos3D.T
        poscam2 = Pq.T @ pos3D.T
        print(poscam1)
        print(poscam2)
        if poscam1[2] > 0 and poscam2[2] > 0:
            return Pq


def estimate_camera_matrices_from_fundamental(K,  F):
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
        K @ np.concatenate((R1, U[:, 2:]), axis=1),
        K @ np.concatenate((R1, -U[:, 2:]), axis=1),
        K @ np.concatenate((R2, U[:, 2:]), axis=1),
        K @ np.concatenate((R2, -U[:, 2:]), axis=1)
    ]


def triangulate(ps, qs, Pp, Pq):
    """
    Traingulate a 2D pair p, q given their point coordinates and the camera matrices
    :param p: Point #1
    :param q: Point #2
    :param Pp: Camera matrix for point #1
    :param Pq: Camera matrix for point #2
    :param imshape: Image size
    :return: 3D coordinates of the triangulated point
    """

    # Compute matrix A
    A = []
    for p, q in zip(ps, qs):
        A.append(p * Pp[2, :] - Pp[0, :])
        A.append(q * Pq[2, :] - Pq[1, :])

    # Calculate best point
    A = np.array(A)
    u, d, vt = np.linalg.svd(A)
    X = vt[-1, 0:3] / vt[-1, 3]  # normalize
    return X


def estimate_camera_matrices(p_set, q_set, kp, kq, pshape, qshape):
    """
    Estimate the camera matrix given a set of correspondences
    :param p_set: Corresponding points on image #1
    :param q_set: Corresponding points on image #2
    :param K: Intrinsic camera matrix
    :param imshape: Image size
    :return:
    """
    Kp = build_K(kp, pshape[0], pshape[1])
    Kq = build_K(kq, qshape[0], qshape[1])
    F = estimate_fundamental_matrix(p_set, q_set)
    Pqs = estimate_camera_matrices_from_fundamental(Kq, F)
    Pp = Kp @ np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    Pq = evaluate_camera_matrices(p_set[0], q_set[0], Pp, Pqs)
    return Pp, Pq


def build_K(f, imsh0, imsh1):
    return np.array([[f, 0, float(imsh0)/2], [0, f, float(imsh1)/2], [0, 0, 1]])
