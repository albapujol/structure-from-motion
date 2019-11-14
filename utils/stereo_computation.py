import numpy as np
from .epipolar_geometry import estimate_fundamental_matrix
from PIL import Image

def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def  stereorectify(F, ph, qh):
    e2 = nullspace(F).flatten()
    mirror = e2[0] < 0

    d = max(np.sqrt(e2[0]**2+e2[1]**2), 1e-7)
    alpha = e2[0]/d
    beta = e2[1]/d
    R = np.array([[alpha, beta, 0], [-beta, alpha, 0], [0, 0, 1]])
    e2 = R @ e2

    if abs(e2[2]) < 1e-6*abs(e2[0]):
        invf = 0
    else:
        invf = -e2[2]/e2[0]
    G = np.array([[1, 0, 0], [0, 1, 0], [invf, 0, 1]])
    H2 = G @ R

    e2 = nullspace(F).flatten()
    e2_x = np.array([[0, -e2[2], e2[1]],
                     [e2[2], 0, e2[0]],
                     [-e2[1], e2[0], 0]])
    e2_111 = np.array([[e2[0], e2[0], e2[0]],
                       [e2[1], e2[1], e2[1]],
                       [e2[2], e2[2], e2[2]]])
    H0 = H2 @ (e2_x @ F + e2_111)

    A = (H0 @ ph.T).T
    A = A / (A[:, 2].reshape(10, 1))
    B = H2 @ qh.T
    B = B[1, :]

    X, _, _, _ = np.linalg.lstsq(A, B, rcond=-1)

    Ha = np.array([[X[0], X[1], X[2]], [0, 1, 0], [0, 0, 1]])
    H1 = Ha @ H0

    if mirror:
        mm = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        H1 = mm @ H1
        H2 = mm @ H2

    return H1, H2


def apply_H(im, H):
    inv_homography_matrix = np.linalg.inv(H)
    inv_homography_matrix /= inv_homography_matrix[2, 2]
    homography_param = inv_homography_matrix.ravel()
    print(H)

    transformed_img = Image.fromarray(im).transform(
        size=im.shape[0:2],
        method=Image.PERSPECTIVE,
        data=homography_param,
        resample=Image.BICUBIC
    )
    return np.array(transformed_img)

def stereo_computation(left_im, right_im, p, q, min_disp, max_disp, winsize, cost_f):
    h, w = left_im.shape
    F = estimate_fundamental_matrix(p, q)
    Hp, Hq = stereorectify(F, p, q, w, h)

    measures = np.zeros(8, 3)
    measures[0,:] = Hp * np.array([0, 0, 1]).T
    measures[1,:] = Hp * np.array([h, 0, 1]).T
    measures[2,:] = Hp * np.array([0, w, 1]).T
    measures[3,:] = Hp * np.array([h, w, 1]).T
    measures[4,:] = Hq * np.array([0, 0, 1]).T
    measures[5,:] = Hq * np.array([h, 0, 1]).T
    measures[6, :] = Hq * np.array([0, w, 1]).T
    measures[7,:] = Hq * np.array([h, w, 1]).T
    measures = measures / measures[:, 3]

    corners = np.zeros((1, 4))
    corners[0] = min(measures[:, 1])
    corners[1] = max(measures[:, 1])
    corners[2] = min(measures[:, 2])
    corners[3] = max(measures[:, 3])
    corners = corners.round()

    left_im_r = apply_H_v2(left_im, Hp, corners)
    right_im_r = apply_H_v2(right_im, Hq, corners)

    winstep = np.ceil(winsize / 2) - 1
    dispmap = np.zeros((h, w))

    for i in range(winstep, h - winstep - 2):
        for j in range(winstep, w - winstep - 2):
            patchim1 = left_im[i - winstep:i + winstep, j - winstep: j + winstep]
            mincost = np.inf
            maxcorr = 0
            aux = 1
            dispstep1 = max_disp
            dispstep2 = max_disp
            if j - winstep - max_disp < 0:
                dispstep1 = j - winstep - 1
            if j + winstep + max_disp > w-1:
                dispstep2 = j + winstep - w
            for jj in range(-dispstep1, dispstep2):
                patchim2 = right_im[i - winstep:i + winstep, j - winstep + jj: j + winstep + jj]
                if cost_f == 'ssd':
                    cost[aux] = sum(sum((patchim1 - patchim2)^2))
                    if cost[aux] < mincost:
                        mincost = cost[aux]
                        mincol = j + jj
                    aux = aux + 1
                elif cost_f == 'ncc': # Normalized cross correlation
                    corr[aux] = ncc(patchim1, patchim2)
                    if corr[aux] > maxcorr:
                        maxcorr = corr[aux]
                        mincol = j + jj
            if abs(j - mincol) >= min_disp:
                dispmap[i,j] = abs(j - mincol)
            else:
                dispmap[i,j] = min_disp
            if min_disp > 0:
                dispmap[dispmap < min_disp] = min_disp

