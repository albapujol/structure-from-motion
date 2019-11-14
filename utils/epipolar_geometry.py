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
    # Normalize points
    # p_norm, q_norm = p_set, q_set

    Tp, p_norm = normalize(p_set)
    Tq, q_norm = normalize(q_set)

    # Row function for homogenoeus linear equation
    row_f = lambda p, q: np.array([q[0] * p[0], q[1] * p[0], p[0],
                                   q[0] * p[1], q[1] * p[1], p[1],
                                   q[0], q[1], 1])

    # Build Y matrix
    Y = np.zeros((len(p_set), 9))
    for i, (p, q) in enumerate(zip(p_norm, q_norm)):
        Y[i, :] = row_f(p, q)

    # Compute SVD
    _, _, Vt = np.linalg.svd(Y)

    # Create fundamental matrix
    F_r = np.reshape(Vt[-1, :], (3, 3)).T

    # Enforce internal constraint
    U, D, Vt = np.linalg.svd(F_r)
    Dm = np.array([[D[0], 0, 0], [0, D[1], 0], [0, 0, 0]])
    F = U @ Dm @ Vt

    # Denormalize
    F = Tp.T @ F @ Tq
    return F


def compute_fundamental_inliers(p_set, q_set, F, th):
    qtFp = np.zeros((len(p_set)))
    for n in range(len(p_set)):
        qtFp[n] = q_set[n].T @ F @ p_set[n]
    Fp = F @ p_set.T
    Ftq =F.T @ q_set.T
    d = qtFp**2 / (Fp[0, :]**2 + Fp[1, :]**2 + Ftq[0, :]**2 + Ftq[1, :]**2)
    return np.argwhere(abs(d) < th).flatten()


def estimate_fundamental_matrix_ransac(p_set, q_set, th=2.0):
    it = 0
    max_it = 1000
    best_inliers = []
    p = 0.999
    while it < max_it:
        random_idxes = np.random.randint(len(p_set), size=8)
        p_set_it = p_set[random_idxes]
        q_set_it = q_set[random_idxes]
        F_it = estimate_fundamental_matrix(p_set_it, q_set_it)
        inliers = compute_fundamental_inliers(p_set, q_set, F_it, th)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
        fracinliers = len(best_inliers) / len(p_set)
        pNoOutliers = 1 - fracinliers**8
        pNoOutliers = max(np.spacing(1), pNoOutliers)
        pNoOutliers = min(1 - np.spacing(1), pNoOutliers)
        max_it = min(max_it, np.log(1 - p) / np.log(pNoOutliers))
        it = it + 1
    print(best_inliers)
    p_set_final = p_set[best_inliers]
    q_set_final = q_set[best_inliers]
    F = estimate_fundamental_matrix(p_set_final, q_set_final)
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
        pos3D = triangulate2(p, q, Pp, Pq)
        pos3D = pos3D / pos3D[2]
        poscam1 = Pp.T @ pos3D.T
        poscam2 = Pq.T @ pos3D.T
        if poscam1[2] > 0 and poscam2[2] > 0:
            return Pq


def estimate_camera_matrix_from_fundamental(F, p, q):
    """
    Estimat the camera matrix between two point sets using the fundamental matrix.
    :param K: Intrinsic camera matrix
    :param F: Fundamental matrix
    :return: Camera matrix candidates
    """
    # Compute candidate camera matrices
    [U, _, Vt] = np.linalg.svd(F)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ Vt
    if np.linalg.det(R1) < 0:
        R1 = -R1
    R2 = U @ W.T @ Vt
    if np.linalg.det(R2) < 0:
        R2 = -R2
    print(U[:, 2:])
    print(R1)
    print(R2)
    Pp = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    Pqs = [np.concatenate((R1, U[:, 2:]), axis=1),
           np.concatenate((R1, -U[:, 2:]), axis=1),
           np.concatenate((R2, U[:, 2:]), axis=1),
           np.concatenate((R2, -U[:, 2:]), axis=1)]
    Pq = evaluate_camera_matrices(p, q, Pp, Pqs)
    print("Evaluated")
    print(Pp)
    print(Pq)
    return Pp, Pq


def triangulate2(ps, qs, Pp, Pq):
    """
    Traingulate a 2D pair p, q given their point coordinates and the camera matrices
    :param p: Point #1
    :param q: Point #2
    :param Pp: Camera matrix for point #1
    :param Pq: Camera matrix for point #2
    :return: 3D coordinates of the triangulated point
    """
    # Compute matrix A
    A = []
    A.append(ps[1] * Pp[2, :] - Pp[0, :])
    A.append(ps[0] * Pp[2, :] - Pp[1, :])
    A.append(qs[1] * Pq[2, :] - Pq[0, :])
    A.append(qs[0] * Pq[2, :] - Pq[1, :])
    # Calculate best point
    A = np.array(A)

    u, d, vt = np.linalg.svd(A)
    X = vt[-1, 0:3] / vt[-1, 3]  # normalize
    return X


def triangulateN(points, Ps):
    """
    Traingulate a 2D set of points p, q given their point coordinates and the camera matrices
    :param points: 2D points on the different imags
    :param Ps: Camera matrices for each point
    :return: 3D coordinates of the triangulated point
    """
    # Compute matrix A
    A = []
    for ps, Pp in zip(points, Ps):
        A.append(-ps[1] * Pp[2, :] + Pp[0, :])
        A.append(ps[0] * Pp[2, :] - Pp[1, :])
    # Calculate best point
    A = np.array(A)
    u, d, vt = np.linalg.svd(A)
    X = vt[-1, 0:3] / vt[-1, 3]  # normalize
    return X



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
    print(F)
    # return
    Pp, Pq = estimate_camera_matrix_from_fundamental(F, p_set[0], q_set[0])
    return Pp, Pq


