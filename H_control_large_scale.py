import numpy as np
import matplotlib
from scipy.stats import ks_1samp
from utils import get_trajectory, evaluate_model

matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import scipy.linalg
from cvxopt import solvers, matrix
import copy
import cvxpy as cp
import control as ct
import time


seed = 1
np.random.seed(seed + 123)
num_mass = 10 # size of the system
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


B = np.zeros((num_mass * 2, 1))
B[1] = 1/m

C = np.zeros([num_mass, num_mass * 2])
for i in range(num_mass - 1):
    C[i, i * 2 + 2] = 1.
C[-1, -1] = 1.

D = np.zeros((num_mass, 1))


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
attack_level = 3.
trajectory = evaluate_model(dim_z, dim_u, dim_y, ref_theta, num_points, ref_trajectory, N, A, B, A_, B_, C_)
noise_trajectory = np.copy(trajectory)
noise_trajectory += noise_level * np.random.randn(noise_trajectory.shape[0], noise_trajectory.shape[1]) # Gaussian noise
#noise_trajectory += np.random.uniform(-0.2, 0.2, size=(noise_trajectory.shape[0], noise_trajectory.shape[1])) # Uniform noise
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
    #      for j in range(L):
    #             recover_trajectory[:, j] = recover_w[(dim_u + dim_y) * j:(dim_u + dim_y) * (j + 1)].reshape(-1)
    # else:
    #     recover_trajectory12[:, i + L - 1] = rec_w[-(dim_u + dim_y):].reshape(-1)


print('average computation time', t/(num_points - L + 1))
print('worst computation time', worst_time)

plt.figure(1)
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
