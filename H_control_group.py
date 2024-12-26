import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import scipy.linalg
from cvxopt import solvers, matrix
import cvxpy as cp
import control as ct

seed = 1
np.random.seed(seed + 123)
dim_y = 3
dim_u = 1
dim_z = 6
pre_length = 10
N = 10
num_points = 50
delta_x = 4. / 50
solvers.options['show_progress'] = False
ref_theta = np.arange(0, 5., delta_x) * np.pi * 2
#ref_trajectory = np.vstack([ref_theta, np.sin(ref_theta) * (4 * np.pi - ref_theta) / (4 * np.pi), np.zeros([dim_y - 2, len(ref_theta)])])
ref_trajectory = np.vstack([ref_theta, ref_theta + 3 * np.sin(ref_theta / 2), np.zeros([dim_y - 2, len(ref_theta)])])
#plt.plot(ref_trajectory[0, :], ref_trajectory[1, :])
#plt.show()

k1 = 2
k2 = 3
k3 = 1
b1 = 3
b2 = 4
b3 = 2
m1 = 1
m2 = 2
m3 = 10


A = [[0., 1, 0, 0, 0, 0],
     [-k1/m1, -b1/m1, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [k2/m2, 0, -k2/m2, -b2/m2, 0, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, k3/m3, 0, -k3/m3, -b3/m3]]
B = [[0.],
     [1/m1],
     [0],
     [0],
     [0],
     [0]]
C = [[0., 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 1., 0],
     [0, 0, 0, 0, 0, 1]]
D = [[0.],
     [0],
     [0]]

A = np.array(A)
B = np.array(B)
C = np.array(C)
D = np.array(D)

sys = ct.StateSpace(A, B, C, D)

para = ct.matlab.c2d(sys, 1)

A_ = np.array(para.A)
B_ = np.array(para.B)
C_ = np.array(para.C)

# A_ = [[0.600423599106272, 0.232544157934830, 0, 0, 0, 0],
#       [-0.465088315869659, -0.0972088746982169, 0, 0, 0, 0],
#       [0.111856466835638, 0.0236553831899867, 0.617658533181963, 0.337980175656172, 0, 0],
#       [0.121679321448113, 0.0408903172656778, -0.506970263484258, -0.0583018181303809, 0, 0],
#       [0.00127419895283969, 0.000177866407692774, 0.0424189350747677, 0.00955432747389353, 0.953556782433585, 0.891325802576332],
#       [0.00442143092156121, 0.000740599729761365, 0.0748010890467929, 0.0233102801269806, -0.0891325802576333, 0.775291621918319],]
# B_ = [[0.199788200446864], [0.232544157934830], [0.00779534438518723], [0.0236553831899867], [0.0000336142721880195], [0.000177866407692774]]
# C_ = [[0., 0., 1., 0., 0., 0.], [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 1.]]
# A_ = np.array(A_)
# B_ = np.array(B_)
# C_ = np.array(C_)

A = np.hstack([np.vstack([A_, C_ @ A_]), np.zeros([dim_z + dim_y, dim_y])])
B = np.vstack([B_, C_ @ B_])

def cal_matrices(x_kshort, ref_tra):
    Q = np.eye(dim_z + dim_y)
    Q[:dim_z, :] = 0
    Q[dim_z + 2:, :] = 0
    F = np.eye(dim_z + dim_y)
    R = np.eye(dim_u) * 0.0000000001

    n = A.shape[0]
    p = B.shape[1]

    M = np.vstack((np.eye((n)), np.zeros((N*n,n))))
    C = np.zeros(((N+1)*n,N*p))
    tmp = np.eye(n)

    for i in range(N):
        rows = i * n + n
        C[rows:rows+n,:] = np.hstack((np.dot(tmp, B), C[rows-n:rows, 0:(N-1)*p]))
        tmp = np.dot(A, tmp)
        M[rows:rows+n,:] = tmp

    Q_bar_be = np.kron(np.eye(N), Q)
    Q_bar = scipy.linalg.block_diag(Q_bar_be, F)
    R_bar = np.kron(np.eye(N), R)

    E = np.matmul(np.matmul(C.transpose(),Q_bar),M)
    L = - np.matmul(C.transpose(), Q_bar)
    H = np.matmul(np.matmul(C.transpose(),Q_bar),C) + R_bar
    M = H
    C = E
    M = matrix(M)

    ref = np.vstack([np.zeros([dim_z, N]), ref_tra])
    ref = ref.transpose().reshape([-1, 1])
    ref = np.vstack([x_kshort, ref])
    T = np.dot(C, x_kshort) + np.dot(L, ref)
    T = matrix(T)
    G = matrix(np.vstack([np.eye(dim_u * N), - np.eye(dim_u * N)]))
    H = matrix(np.ones([dim_u * N * 2, 1]) * 10000)
    try:
        sol = solvers.qp(M, T, G, H, kktsolver='ldl', options={'show_progress': False})
    except:
        u_k = np.random.randn(dim_u)
    else:
        U_thk = np.array(sol["x"])
        u_k = U_thk[:dim_u, :]
    return u_k

def get_trajectory(length):
    z = np.random.randn(dim_z, 1)
    trajectory = np.zeros([dim_u + dim_y, length])
    for i in range(length):
        u = np.random.randn(dim_u, 1)
        z = A_ @ z + B_ @ u
        y = C_ @ z
        trajectory[: dim_u, i] = u.reshape(-1)
        trajectory[dim_u:, i] = y.reshape(-1)
    return trajectory

def evaluate_model():
    z = np.zeros([dim_z, 1])
    u = np.zeros([dim_u, 1])
    y = np.zeros([dim_y, 1])
    trajectory = np.zeros([dim_u + dim_y, len(ref_theta)])
    for i in range(1, num_points):
        s = np.vstack([z, y]).reshape([-1, 1])
        u = cal_matrices(s, ref_trajectory[:, i: i + N])
        z = A_ @ z + B_ @ u
        y = C_ @ z
        trajectory[: dim_u, i] = u.reshape(-1)
        trajectory[dim_u:, i] = y.reshape(-1)
    return trajectory

L = 3
num_T = (dim_u + 1) * L + dim_z - 1
base_trajectory = get_trajectory(num_T)
wd = base_trajectory.transpose().reshape([-1, 1])
H = np.zeros([L * (dim_u + dim_y), num_T - L + 1])
for i in range(num_T - L + 1):
    H[:, i] = wd[i * (dim_u + dim_y) : (i + L) * (dim_u + dim_y)].reshape(-1)

noise_level = 0.1
attack_level = 3
trajectory = evaluate_model()
noise_trajectory = np.copy(trajectory)
noise_trajectory += noise_level * np.random.randn(noise_trajectory.shape[0], noise_trajectory.shape[1])
#noise_trajectory[dim_u, :] = np.copy(trajectory[dim_u, :])
#noise_trajectory[dim_u + 1, :] += noise_level * np.random.randn(len(noise_trajectory[dim_u + 1, :]))
attack_trajectory = np.copy(noise_trajectory)
for i in range(num_points):
    attack_trajectory[dim_u + 1, i] += attack_level * np.random.randn()

recover_trajectory = np.copy(attack_trajectory)
recover_trajectory12 = np.copy(attack_trajectory)
for i in range(num_points - L + 1):
    true_w = trajectory[:, i: i + L].transpose().reshape([-1, 1])
    attack_w = attack_trajectory[:, i: i + L].transpose().reshape([-1, 1])
    g = cp.Variable([H.shape[1], 1])
    est_err = attack_w - H @ g
    res = 0.
    for j in range(dim_u + dim_y):
        idx = [j + k * (dim_u + dim_y) for k in range(L)]
        res += cp.norm2(est_err[idx, 0], 0)
    prob = cp.Problem(cp.Minimize(res))
    prob.solve(verbose=True)
    g = g.value
    recover_w = H @ g
    if i == 0:
        recover_trajectory[:, 0] = recover_w[:(dim_u + dim_y)].reshape(-1)
        recover_trajectory[:, 1] = recover_w[(dim_u + dim_y) : (dim_u + dim_y) * 2].reshape(-1)
    else:
        recover_trajectory[:, i + L - 1] = recover_w[-(dim_u + dim_y):].reshape(-1)
    idx = -1
    tar_dis = 0.
    for j in range(dim_u + dim_y):
        idxs = [j + k * (dim_u + dim_y) for k in range(L)]
        dis = np.sum((recover_w[idxs, :] - attack_w[idxs, :]) ** 2)
        if idx == -1 or dis > tar_dis:
            idx = j
            tar_dis = dis

    tem_H = np.copy(H)
    idxs = [idx + k * (dim_u + dim_y) for k in range(L)]
    tem_H[idxs, :] = 0
    attack_w[idxs] = 0
    g_rec = np.linalg.inv(tem_H.transpose() @ tem_H) @ tem_H.transpose() @ attack_w
    rec_w = H @ g_rec
    if i == 0:
        recover_trajectory12[:, 0] = rec_w[:(dim_u + dim_y)].reshape(-1)
        recover_trajectory12[:, 1] = rec_w[(dim_u + dim_y) : (dim_u + dim_y) * 2].reshape(-1)
    else:
        recover_trajectory12[:, i + L - 1] = rec_w[-(dim_u + dim_y):].reshape(-1)


plt.plot(ref_trajectory[0, 3:num_points], ref_trajectory[1, 3: num_points], label='ref_trajectory')
plt.scatter(trajectory[dim_u, 3:num_points], trajectory[dim_u + 1, 3:num_points], label='true_trajectory')
plt.scatter(attack_trajectory[dim_u, 3:num_points], attack_trajectory[dim_u + 1, 3:num_points], label='attack_trajectory')
plt.scatter(noise_trajectory[dim_u, 3:num_points], noise_trajectory[dim_u + 1, 3:num_points], label='noise_trajectory')
plt.plot(recover_trajectory12[dim_u, 3:num_points], recover_trajectory12[dim_u + 1, 3:num_points], label='recover_trajectory')
plt.legend()
plt.show()

loss1 = np.mean((recover_trajectory[:, 3: num_points] - trajectory[:, 3: num_points]) ** 2)
loss12 = np.mean((recover_trajectory12[:, 3: num_points] - trajectory[:, 3: num_points]) ** 2)
print('loss1: {}, loss12: {}'.format(loss1, loss12))