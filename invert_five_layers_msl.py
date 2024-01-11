# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:45:12 2024

@author: UWTUCANMag
"""

import numpy as np
from scipy.linalg import lu, solve

# Define the dimension n of the matrix and the signum s (for LU decomposition)
capm = 5  # number of shields.
rank = 2 * capm  # matrix dimension

n = 1  # multipole order
# 1 = uniform field

# Define geometry
r = np.zeros(rank)
t = np.zeros(capm)
capr1 = np.zeros(capm)

# Define mu
mur = np.zeros(capm)

# Define all the used matrices
a = np.zeros((rank, rank))
inverse = np.zeros((rank, rank))

# Define parameters of the problem
capr1[0] = 2.26 / 2  # m
capr1[1] = 2.4 / 2  # m
capr1[2] = 2.6 / 2  # m
capr1[3] = 3.0 / 2  # m
capr1[4] = 3.5 / 2  # m

t[0] = 0.002  # m
t[1] = 0.002  # m
t[2] = 0.003  # m
t[3] = 0.003  # m
t[4] = 0.004  # m

mur.fill(20000)
mur[0] = 50000
mur[1] = 50000

for i in range(capm):
    print(i, capr1[i], t[i], mur[i])

# Eq. (17)
j = 0
for i in range(capm):
    r[j] = capr1[i]
    print(f"r{j} {r[j]}")
    j += 1
    r[j] = capr1[i] + t[i]
    print(f"r{j} {r[j]}")
    j += 1

print("Fill the Matrix")

# Fill the matrix m
for i in range(rank):
    for j in range(rank):
        if j < i:
            element = (r[j] / r[i]) ** (2 * n)
        elif j > i:
            element = -1
        else:  # j == i
            if (i + 1) % 2 == 1:
                m = ((i + 1) + 1) // 2 - 1
                element = -(mur[m] + 1) / (mur[m] - 1)
            else:
                m = (i + 1) // 2 - 1
                element = (mur[m] + 1) / (mur[m] - 1)
        a[i, j] = element
        print(f"{element} ", end="")
    print()

print("Print the matrix")

# print the matrix
for i in range(rank):
    for j in range(rank):
        print(f"{a[i, j]} ", end="")
    print()

print("Inverse")

# Make LU decomposition of matrix m
p, l, u = lu(a)

# Invert the matrix m
inverse = solve(u, solve(l, p.T))

for i in range(rank):
    for j in range(rank):
        print(f"{inverse[i, j]} ", end="")
    print()

# Act on vector of 1's
# i.e., add up along a row.
# then we add up all the rows anyway.
# just sum the whole matrix, then.
summat = 0
for i in range(rank):
    sumrow = np.sum(inverse[i, :])
    print(f"{i} sumrow {sumrow}")
    summat += sumrow

print(f"summat {summat}")

sfact = 1.0 / (1.0 + summat)

print(f"sfact {sfact}")
