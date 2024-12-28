import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import scipy.linalg
from cvxopt import solvers, matrix
import cvxpy as cp
import control as ct
from utils import get_trajectory, evaluate_model

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

A = np.hstack([np.vstack([A_, C_ @ A_]), np.zeros([dim_z + dim_y, dim_y])])
B = np.vstack([B_, C_ @ B_])

L = 3
num_T = (dim_u + 1) * L + dim_z - 1
base_trajectory = get_trajectory(num_T, dim_z, dim_u, dim_y, A_, B_, C_)
wd = base_trajectory.transpose().reshape([-1, 1])
H = np.zeros([L * (dim_u + dim_y), num_T - L + 1])
for i in range(num_T - L + 1):
    H[:, i] = wd[i * (dim_u + dim_y) : (i + L) * (dim_u + dim_y)].reshape(-1)

noise_level = 0.1
attack_level = 3
trajectory = evaluate_model(dim_z, dim_u, dim_y, ref_theta, num_points, ref_trajectory, N, A, B, A_, B_, C_)
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



plt.scatter(attack_trajectory[dim_u, 3:num_points], attack_trajectory[dim_u + 1, 3:num_points], label='attack_trajectory')
plt.scatter(noise_trajectory[dim_u, 3:num_points], noise_trajectory[dim_u + 1, 3:num_points], label='noise_trajectory')
plt.plot(recover_trajectory12[dim_u, 3:num_points], recover_trajectory12[dim_u + 1, 3:num_points], label='recover_trajectory')
plt.legend()
plt.show()

loss1 = np.mean((recover_trajectory[:, 3: num_points] - trajectory[:, 3: num_points]) ** 2)
loss12 = np.mean((recover_trajectory12[:, 3: num_points] - trajectory[:, 3: num_points]) ** 2)
print('loss1: {}, loss12: {}'.format(loss1, loss12))
