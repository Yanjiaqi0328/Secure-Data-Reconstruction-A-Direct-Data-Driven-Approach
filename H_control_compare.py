import numpy as np
import matplotlib
from numpy.distutils.core import numpy_cmdclass
from scipy.stats import ks_1samp

matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import scipy.linalg
from cvxopt import solvers, matrix
import copy
import cvxpy as cp
import control as ct
import time
from itertools import combinations
from utils import get_trajectory, evaluate_model


seed = 1
np.random.seed(seed + 123)
dim_y = 3
dim_u = 1
dim_z = 6

N = 10
num_points = 50
delta_x = 4. / 50
solvers.options['show_progress'] = False
ref_theta = np.arange(0, 5., delta_x) * np.pi * 2
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

L = 6
num_T = (dim_u + 1) * L + dim_z - 1
base_trajectory = get_trajectory(num_T, dim_z, dim_u, dim_y, A_, B_, C_)
wd = base_trajectory.transpose().reshape([-1, 1])
H = np.zeros([L * (dim_u + dim_y), num_T - L + 1])
for i in range(num_T - L + 1):
    H[:, i] = wd[i * (dim_u + dim_y) : (i + L) * (dim_u + dim_y)].reshape(-1)

noise_level = 0
attack_level = 3.
trajectory = evaluate_model(dim_z, dim_u, dim_y, ref_theta, num_points, ref_trajectory, N, A, B, A_, B_, C_)
noise_trajectory = np.copy(trajectory)
noise_trajectory += noise_level * np.random.randn(noise_trajectory.shape[0], noise_trajectory.shape[1])
attack_trajectory = np.copy(noise_trajectory)
for i in range(num_points):
    if i % L == 2:
        attack_trajectory[dim_u + 1, i] += attack_level * np.random.randn()

recover_trajectory = np.copy(attack_trajectory)
brute_trajectory = np.copy(attack_trajectory)
model_trajectory = np.copy(attack_trajectory)

M1, M2 = 0, 0
M1 = np.zeros([dim_y * L, dim_z])
M = C_ @ A_
for i in range(L):
    M1[dim_y * i : dim_y * (i + 1), :] = M
    M = M @ A_
M2 = np.zeros([dim_y * L, dim_u * L])
for i in range(L):
    for j in range(i + 1):
        In = np.eye(dim_z)
        for k in range(i - j):
            In = In @ A_
        M2[dim_y * i: dim_y * (i + 1), dim_u * j: dim_u * (j + 1)] = C_ @ In @ B_


t = 0
t_brute = 0
for i in range(num_points - L + 1):
    true_y = trajectory[dim_u:, i: i + L].transpose().reshape([-1, 1])
    attack_y = attack_trajectory[dim_u:, i: i + L].transpose().reshape([-1, 1])
    attack_Y = attack_y - M2 @ trajectory[:dim_u, i: i + L].transpose().reshape([-1, 1])
    true_Y = true_y - M2 @ trajectory[:dim_u, i: i + L].transpose().reshape([-1, 1])

    # model-based algorithm [Mao2022]
    for choose_set in combinations(range(dim_y * L), 1):
        tem_M1 = np.copy(M1)
        tem_M1[choose_set, :] = 0.
        tem_y = np.copy(attack_Y)
        tem_y[choose_set, :] = 0.
        if np.linalg.matrix_rank(tem_M1) == np.linalg.matrix_rank(np.concatenate([tem_M1, tem_y], 1)):
            g_model = np.linalg.pinv(tem_M1.transpose() @ tem_M1) @ tem_M1.transpose() @ tem_y
            model_y = M1 @ g_model

    if i == 0:
        for j in range(L):
            model_trajectory[dim_u:, i] = (model_y + M2 @ trajectory[:dim_u, i: i + L].transpose().reshape([-1, 1]))[
                                          dim_y * i: dim_y * (i + 1)].reshape(-1)
    else:
        model_trajectory[dim_u:, i + L - 1] = (model_y + M2 @ trajectory[:dim_u, i: i + L].transpose().reshape(
            [-1, 1]))[-dim_y:].reshape(-1)

    true_w = trajectory[:, i: i + L].transpose().reshape([-1, 1])
    attack_w = attack_trajectory[:, i: i + L].transpose().reshape([-1, 1])
    g = cp.Variable([H.shape[1], 1])
    tic = time.time()
    prob = cp.Problem(cp.Minimize(cp.norm1(attack_w - H @ g)))
    prob.solve()
    g = g.value
    recover_w = H @ g
    toc = time.time()
    t = t + toc - tic
    if i == 0:
        for j in range(L):
            recover_trajectory[:, j] = recover_w[(dim_u + dim_y) * j:(dim_u + dim_y) * (j + 1)].reshape(-1)
    else:
        recover_trajectory[:, i + L - 1] = recover_w[-(dim_u + dim_y):].reshape(-1)

    # brute force algorithm
    tic_brute = time.time()
    for choose_set in combinations(range((dim_u + dim_y) * L), 3):
        tem_H = np.copy(H)
        tem_H[choose_set, :] = 0.
        tem_w = np.copy(attack_w)
        tem_w[choose_set, :] = 0.
        if np.linalg.matrix_rank(tem_H) == np.linalg.matrix_rank(np.concatenate([tem_H, tem_w], 1)):
            g_brute = np.linalg.inv(tem_H.transpose() @ tem_H) @ tem_H.transpose() @ tem_w
            brute_w = H @ g_brute

    if i == 0:
        for j in range(L):
            brute_trajectory[:, j] = brute_w[(dim_u + dim_y) * j:(dim_u + dim_y) * (j + 1)].reshape(-1)
    else:
        brute_trajectory[:, i + L - 1] = brute_w[-(dim_u + dim_y):].reshape(-1)
    toc_brute = time.time()
    t_brute = t_brute + toc_brute - tic_brute


print('average computation time', t/(num_points - L + 1))
print('average computation time: brute force', t_brute/(num_points - L + 1))


plt.rcParams.update({'font.size': 18})
plt.figure(1)
plt.plot(recover_trajectory[dim_u, L:num_points], recover_trajectory[dim_u + 1, L:num_points], label='Recovered trajectory')
plt.scatter(attack_trajectory[dim_u, L:num_points], attack_trajectory[dim_u + 1, L:num_points], label='Attacked trajectory')
plt.scatter(trajectory[dim_u, L:num_points], trajectory[dim_u + 1, L:num_points], label='True trajectory')
ax = plt.gca()
ax.set_xlabel('$d_1$')
ax.set_ylabel('$d_2$')
plt.legend(fontsize=18)
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.savefig('l1.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(2)
plt.plot(brute_trajectory[dim_u, L:num_points], brute_trajectory[dim_u + 1, L:num_points], label='Recovered trajectory')
plt.scatter(attack_trajectory[dim_u, L:num_points], attack_trajectory[dim_u + 1, L:num_points], label='Attacked trajectory')
plt.scatter(trajectory[dim_u, L:num_points], trajectory[dim_u + 1, L:num_points], label='True trajectory')
ax = plt.gca()
ax.set_xlabel('$d_1$')
ax.set_ylabel('$d_2$')
plt.legend(fontsize=18)
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.savefig('brute.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(3)
plt.plot(model_trajectory[dim_u, L:num_points], model_trajectory[dim_u + 1, L:num_points], label='Recovered trajectory')
plt.scatter(attack_trajectory[dim_u, L:num_points], attack_trajectory[dim_u + 1, L:num_points], label='Attacked trajectory')
plt.scatter(trajectory[dim_u, L:num_points], trajectory[dim_u + 1, L:num_points], label='True trajectory')
ax = plt.gca()
ax.set_xlabel('$d_1$')
ax.set_ylabel('$d_2$')
plt.legend(fontsize=18)
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.savefig('model_based.png', dpi=300, bbox_inches='tight')
plt.show()

loss1 = np.mean((recover_trajectory[:, 3: num_points] - trajectory[:, 3: num_points]) ** 2)
loss_brute = np.mean((brute_trajectory[:, 3: num_points] - trajectory[:, 3: num_points]) ** 2)
print('loss1: {}, loss_brute: {}'.format(loss1, loss_brute))
