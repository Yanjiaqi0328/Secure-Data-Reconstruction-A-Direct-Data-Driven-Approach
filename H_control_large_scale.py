import numpy as np
import matplotlib
from scipy.stats import ks_1samp

matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import scipy.linalg
from cvxopt import solvers, matrix
import copy
import cvxpy as cp
import control as ct
import time

num_mass = 10 # size of the system
seed = 1
np.random.seed(seed + 123)
dim_y = num_mass
dim_u = 1
dim_z = num_mass * 2

N = 10
num_points = 50
delta_x = 4. / 50
solvers.options['show_progress'] = False
ref_theta = np.arange(0, 5., delta_x) * np.pi * 2
#ref_trajectory = np.vstack([ref_theta, np.sin(ref_theta) * (4 * np.pi - ref_theta) / (4 * np.pi), np.zeros([dim_y - 2, len(ref_theta)])])
ref_trajectory = np.vstack([ref_theta, ref_theta + 3 * np.sin(ref_theta / 2), np.zeros([dim_y - 2, len(ref_theta)])])

#plt.plot(ref_trajectory[0, :], ref_trajectory[1, :])
#plt.show()



k = 1.
b = 2.
m = 1.

A = np.zeros((num_mass * 2, num_mass * 2))
for i in range(num_mass):
    A[i * 2, i * 2 + 1] = 1.
    A[i * 2 + 1, i * 2] = - k / m
    A[i * 2 + 1, i * 2 + 1] = - b / m
    if i * 2 + 3 < num_mass * 2:
        A[i * 2 + 3, i * 2] = k / m
# A[0,1] = 1
# A[1,0] = -k/m
# A[1,1] = -b/m
# A[2,3] = 1
# A[3,0] = k/m
# A[3,2] = -k/m
# A[3,3] = -b/m
# A[4,5] = 1
# A[5,2] = k/m
# A[5,4] = -k/m
# A[5,5] = -b/m

B = np.zeros((num_mass * 2, 1))
B[1] = 1/m

C = np.zeros([num_mass, num_mass * 2])
for i in range(num_mass - 1):
    C[i, i * 2 + 2] = 1.
C[-1, -1] = 1.

D = np.zeros((num_mass, 1))
# A = [[0., 1, 0, 0, 0, 0],
#      [-k/m, -b/m, 0, 0, 0, 0],
#      [0, 0, 0, 1, 0, 0],
#      [k/m, 0, -k/m, -b/m, 0, 0],
#      [0, 0, 0, 0, 0, 1],
#      [0, 0, k/m, 0, -k/m, -b/m]]
# B = [[0.],
#      [1/m],
#      [0],
#      [0],
#      [0],
#      [0]]
# C = [[0., 0, 1, 0, 0, 0],
#      [0, 0, 0, 0, 1., 0],
#      [0, 0, 0, 0, 0, 1]]
# D = [[0.],
#      [0],
#      [0]]

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
attack_level = 3.
trajectory = evaluate_model()
noise_trajectory = np.copy(trajectory)
#noise_trajectory += noise_level * np.random.randn(noise_trajectory.shape[0], noise_trajectory.shape[1]) # Gaussian noise
noise_trajectory += np.random.uniform(-0.2, 0.2, size=(noise_trajectory.shape[0], noise_trajectory.shape[1])) # Uniform noise
attack_trajectory = np.copy(noise_trajectory)
for i in range(num_points):
    if i % L == 3:
        attack_trajectory[0, i] += attack_level * np.random.randn() # attack the first input channel
        attack_trajectory[dim_u + 1, i] += attack_level * np.random.randn() # attack the second output channel

recover_trajectory = np.copy(attack_trajectory)
recover_trajectory12 = np.copy(attack_trajectory)
t = 0
worst_time = 0
for i in range(num_points - L + 1):
    true_w = trajectory[:, i: i + L].transpose().reshape([-1, 1])
    attack_w = attack_trajectory[:, i: i + L].transpose().reshape([-1, 1])
    g = cp.Variable([H.shape[1], 1])
    tic = time.time()
    prob = cp.Problem(cp.Minimize(cp.norm1(attack_w - H @ g)))
    prob.solve(solver='CPLEX')
    g = g.value
    recover_w = H @ g
    toc = time.time()
    t = t + toc - tic
    worst_time = max(worst_time, toc - tic)
    if i == 0:
        for j in range(L):
            recover_trajectory[:, j] = recover_w[(dim_u + dim_y) * j:(dim_u + dim_y) * (j + 1)].reshape(-1)
    else:
        recover_trajectory[:, i + L - 1] = recover_w[-(dim_u + dim_y):].reshape(-1)
    # idx = np.argmax(np.abs(H @ g - attack_w))
    # tem_H = np.copy(H)
    # tem_H[idx, :] = 0
    # attack_w[idx] = 0
    # g_rec = np.linalg.pinv(tem_H.transpose() @ tem_H) @ tem_H.transpose() @ attack_w
    # rec_w = H @ g_rec
    # if i == 0:
    #     recover_trajectory12[:, 0] = rec_w[:(dim_u + dim_y)].reshape(-1)
    #     recover_trajectory12[:, 1] = rec_w[(dim_u + dim_y) : (dim_u + dim_y) * 2].reshape(-1)
    #     recover_trajectory12[:, 2] = rec_w[(dim_u + dim_y) * 2: (dim_u + dim_y) * 3].reshape(-1)
    # else:
    #     recover_trajectory12[:, i + L - 1] = rec_w[-(dim_u + dim_y):].reshape(-1)


print('average computation time', t/(num_points - L + 1))
print('worst computation time', worst_time)

plt.figure(1)
plt.plot(ref_trajectory[0, L:num_points], ref_trajectory[1, L: num_points], label='ref_trajectory')
plt.scatter(trajectory[dim_u, L:num_points], trajectory[dim_u + 1, L:num_points], label='true_trajectory')
plt.scatter(attack_trajectory[dim_u, L:num_points], attack_trajectory[dim_u + 1, L:num_points], label='attack_trajectory')
plt.scatter(noise_trajectory[dim_u, L:num_points], noise_trajectory[dim_u + 1, L:num_points], label='noise_trajectory')
plt.plot(recover_trajectory[dim_u, L:num_points], recover_trajectory[dim_u + 1, L:num_points], label='recover_trajectory')
plt.legend()
plt.show()



# loss for each channel
plt.figure(2)
loss_channel = np.zeros(dim_u + dim_y)
for i in range(dim_u + dim_y):
    loss_channel[i] = np.mean((recover_trajectory[i, L: num_points] - trajectory[i, L: num_points]) ** 2)
plt.bar(range(1, dim_u + dim_y + 1), loss_channel)
ax = plt.gca()

ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
my_x_ticks = np.arange(1, dim_u + dim_y + 1, 1)
ax.set_xlabel(r'# channel')
ax.set_ylabel('Average error')
plt.xticks(my_x_ticks)
plt.savefig('error_all_channels.png',bbox_inches = 'tight')


plt.show()

loss1 = np.mean((recover_trajectory[:, L: num_points] - trajectory[:, L: num_points]) ** 2)
loss12 = np.mean((recover_trajectory12[:, L: num_points] - trajectory[:, L: num_points]) ** 2)
print('loss1: {}, loss12: {}'.format(loss1, loss12))