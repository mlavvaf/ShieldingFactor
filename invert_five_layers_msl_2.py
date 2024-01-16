# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:28:58 2024

@author: UWTUCANMag
"""

import numpy as np
from scipy.linalg import lu, solve
import matplotlib.pyplot as plt
import os

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
    capm = 5
    n = 1
    capr1 = np.array([2.26 / 2, 2.4 / 2, 2.6 / 2, 3.0 / 2, 3.5 / 2])
    t = np.array([0.002, 0.002, 0.003, 0.003, 0.004])

    # Different values of mur
    mur_values = np.linspace(20000, 50000, 300)

    # Plotting setup
    plt.figure(figsize=(8, 6))

    for mur_value in mur_values:
        mur = np.array([mur_value] * capm)

        matrix_solver = MatrixSolver(capm, n, capr1, t, mur)
        sfact_result = matrix_solver.solve()

        # print(f"For mur = {mur_value}, The total shielding factor: {sfact_result}")

        # Plotting results
        plt.plot(mur_value, sfact_result, 'o', markersize=2, color='navy')
    legend_labels = ['mur']
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='navy', markersize=5)]
    plt.xlabel('mur')
    plt.ylabel('Total Shielding Factor')
    plt.legend(legend_handles, legend_labels)
    plt.grid()
    plt.title('Total Shielding Factor vs mur - 5 layers')
    plt.show()
    
    output_folder = "Figures"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, "DifferentMurCyl5layer.png")
    plt.savefig(output_file_path, dpi=360)
