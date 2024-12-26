import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import scipy.linalg
from cvxopt import solvers, matrix
import copy
import cvxpy as cp

seed = 3
np.random.seed(seed + 123)
dim_y = 4
dim_u = 2
dim_z = 4

N = 10
num_points = 50
delta_x = 4. / 50
solvers.options['show_progress'] = False
ref_theta = np.arange(0, 5., delta_x) * np.pi * 2
#ref_trajectory = np.vstack([ref_theta, np.sin(ref_theta) * (4 * np.pi - ref_theta) / (4 * np.pi), np.zeros([dim_y - 2, len(ref_theta)])])
ref_trajectory = np.vstack([ref_theta, ref_theta + 3 * np.sin(ref_theta / 2), np.zeros([dim_y - 2, len(ref_theta)])])

#plt.plot(ref_trajectory[0, :], ref_trajectory[1, :])
#plt.show()
A_ = [[0.921, 0., 0.041, 0.],
      [0., 0.918, 0., 0.033],
      [0., 0., 0.924, 0.],
      [0., 0., 0., 0.937]
      ]
B_ = [[0.017, 0.001], [0.001, 0.023], [0., 0.061], [0.072, 0.]]
C_ = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],[0., 0., 0., 1.]]
A_ = np.array(A_)
B_ = np.array(B_)
C_ = np.array(C_)*15

A = np.hstack([np.vstack([A_, C_ @ A_]), np.zeros([dim_z + dim_y, dim_y])])
B = np.vstack([B_, C_ @ B_])

def cal_matrices(x_kshort, ref_tra):
    Q = np.eye(dim_z + dim_y)
    Q[:dim_z, :] = 0
    Q[dim_z + 2:, :] = 0
    F = np.eye(dim_z + dim_y)
    R = np.eye(dim_u) * 0.01

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

L = 5
num_T = (dim_u + 1) * L + dim_z - 1
base_trajectory = get_trajectory(num_T)
wd = base_trajectory.transpose().reshape([-1, 1])
H = np.zeros([L * (dim_u + dim_y), num_T - L + 1])
for i in range(num_T - L + 1):
    H[:, i] = wd[i * (dim_u + dim_y) : (i + L) * (dim_u + dim_y)].reshape(-1)

attack_level = 1.
trajectory = evaluate_model()
noise_trajectory = np.copy(trajectory)
#noise_trajectory += noise_level * np.random.randn(noise_trajectory.shape[0], noise_trajectory.shape[1]) # Gaussian noise
noise_trajectory += np.random.uniform(-0.1, 0.1, size=(noise_trajectory.shape[0], noise_trajectory.shape[1])) # Uniform noise
attack_trajectory = np.copy(noise_trajectory)
for i in range(num_points):
    if i % 5 == 0:
        attack_trajectory[0, i] += attack_level * np.random.randn()

recover_trajectory = np.copy(attack_trajectory)
recover_trajectory12 = np.copy(attack_trajectory)
for i in range(num_points - L + 1):
    true_w = trajectory[:, i: i + L].transpose().reshape([-1, 1])
    attack_w = attack_trajectory[:, i: i + L].transpose().reshape([-1, 1])
    g = cp.Variable([H.shape[1], 1])
    prob = cp.Problem(cp.Minimize(cp.norm1(attack_w - H @ g)))
    prob.solve()
    g = g.value
    recover_w = H @ g
    if i == 0:
        for j in range(L):
            recover_trajectory[:, j] = recover_w[(dim_u + dim_y) * j:(dim_u + dim_y) * (j + 1)].reshape(-1)
    else:
        recover_trajectory[:, i + L - 1] = recover_w[-(dim_u + dim_y):].reshape(-1)
    idx = np.argmax(np.abs(H @ g - attack_w))
    tem_H = np.copy(H)
    tem_H[idx, :] = 0
    attack_w[idx] = 0
    g_rec = np.linalg.inv(tem_H.transpose() @ tem_H) @ tem_H.transpose() @ attack_w
    rec_w = H @ g_rec
    if i == 0:
        for j in range(L):
            recover_trajectory[:, j] = recover_w[(dim_u + dim_y) * j:(dim_u + dim_y) * (j + 1)].reshape(-1)
    else:
        recover_trajectory12[:, i + L - 1] = rec_w[-(dim_u + dim_y):].reshape(-1)

plt.plot(recover_trajectory12[0, L:num_points], recover_trajectory12[dim_u - 1, L:num_points], label='Recovered trajectory')
plt.scatter(attack_trajectory[0, L:num_points], attack_trajectory[dim_u -1, L:num_points], label='Attacked trajectory')
plt.scatter(trajectory[0, L:num_points], trajectory[dim_u - 1, L:num_points], label='True trajectory')
plt.legend(loc = "upper left")
ax = plt.gca()
ax.set_xlabel(r'$u_1$')
ax.set_ylabel(r'$u_2$')
plt.savefig('tank.png')
plt.show()

loss1 = np.mean((recover_trajectory[:, : num_points] - trajectory[:, : num_points]) ** 2)
loss12 = np.mean((recover_trajectory12[:, : num_points] - trajectory[:, : num_points]) ** 2)
print('loss1: {}, loss12: {}'.format(loss1, loss12))