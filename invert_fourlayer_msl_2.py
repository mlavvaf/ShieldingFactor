# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:26:54 2024

@author: UWTUCANMag
"""

import numpy as np
from scipy.linalg import lu, solve


class MatrixSolver:
    def __init__(self, capm, n, capr1, t, mur):
        self.capm = capm
        self.rank = 2 * capm
        self.n = n
        self.capr1 = capr1
        self.t = t
        self.mur = mur
        self.r = np.zeros(self.rank)
        self.a = np.zeros((self.rank, self.rank))
        self.inverse = np.zeros((self.rank, self.rank))
        self.sfact = None

    def define_geometry(self):
        j = 0
        for i in range(self.capm):
            self.r[j] = self.capr1[i]
            j += 1
            self.r[j] = self.capr1[i] + self.t[i]
            j += 1

    def fill_matrix(self):
        for i in range(self.rank):
            for j in range(self.rank):
                if j < i:
                    element = (self.r[j] / self.r[i]) ** (2 * self.n)
                elif j > i:
                    element = -1
                else:
                    if (i + 1) % 2 == 1:
                        m = ((i + 1) + 1) // 2 - 1
                        element = -(self.mur[m] + 1) / (self.mur[m] - 1)
                    else:
                        m = (i + 1) // 2 - 1
                        element = (self.mur[m] + 1) / (self.mur[m] - 1)
                self.a[i, j] = element

    def calculate_inverse(self):
        p, l, u = lu(self.a)
        self.inverse = solve(u, solve(l, p.T))

    def calculate_sfact(self):
        summat = np.sum(self.inverse, axis=1).sum()
        self.sfact = 1.0 / (1.0 + summat)

    def solve(self):
        self.define_geometry()
        self.fill_matrix()
        self.calculate_inverse()
        self.calculate_sfact()
        return self.sfact


if __name__ == "__main__":
    capm = 4
    n = 1
    capr1 = np.array([2.4 / 2, 2.6 / 2, 3.0 / 2, 3.5 / 2])
    t = np.array([0.002, 0.003, 0.003, 0.004])
    mur = np.array([20000] * capm)

    matrix_solver = MatrixSolver(capm, n, capr1, t, mur)
    sfact_result = matrix_solver.solve()

    print(f"The total shielding factor: {sfact_result}")
