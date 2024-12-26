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

L = 6
num_T = (dim_u + 1) * L + dim_z - 1
base_trajectory = get_trajectory(num_T)
wd = base_trajectory.transpose().reshape([-1, 1])
H = np.zeros([L * (dim_u + dim_y), num_T - L + 1])
for i in range(num_T - L + 1):
    H[:, i] = wd[i * (dim_u + dim_y) : (i + L) * (dim_u + dim_y)].reshape(-1)

noise_level = 0
attack_level = 3.
trajectory = evaluate_model()
noise_trajectory = np.copy(trajectory)
noise_trajectory += noise_level * np.random.randn(noise_trajectory.shape[0], noise_trajectory.shape[1])
#noise_trajectory[dim_u, :] = np.copy(trajectory[dim_u, :])
#noise_trajectory[dim_u + 1, :] += noise_level * np.random.randn(len(noise_trajectory[dim_u + 1, :]))
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