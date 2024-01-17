# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:15:06 2024

@author: UWTUCANMag
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:26:54 2024

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
    capm_i_4 = 4
    capr1_i_4 = np.array([2.4 / 2, 2.6 / 2, 3.0 / 2, 3.5 / 2])
    t_4 = np.array([0.002, 0.003, 0.003, 0.004])
    # mur_i_4 = np.array([40000] * capm_i_4)

    capm_i_5 = 5
    capr1_i_5 = np.array([2.26 / 2, 2.4 / 2, 2.6 / 2, 3.0 / 2, 3.5 / 2])
    t_5 = np.array([0.002, 0.002, 0.003, 0.003, 0.004])
    # mur_i_5 = np.array([40000] * capm_i_5)
    
    capm_o_4 = 4
    capr1_o_4 = np.sqrt(2) * np.array([2.4 / 2, 2.6 / 2, 3.0 / 2, 3.5 / 2])
    t_4 = np.array([0.002, 0.003, 0.003, 0.004])
    # mur_o_4 = np.array([40000] * capm_o_4)

    capm_o_5 = 5
    capr1_o_5 = np.sqrt(2) * np.array([2.26 / 2, 2.4 / 2, 2.6 / 2, 3.0 / 2, 3.5 / 2])
    t_5 = np.array([0.002, 0.002, 0.003, 0.003, 0.004])
    # mur_o_5 = np.array([40000] * capm_o_5)

    # Plotting setup
    plt.figure(figsize=(8, 6))

    # Plot for capm=4 (using navy color)
    mur_values_4 = np.linspace(20000, 45000, 100)
    plt.plot(mur_values_4, [MatrixSolver(capm_i_4, 1, capr1_i_4, t_4, np.array(
        [mur_value] * capm_i_4)).solve() for mur_value in mur_values_4], 'o', markersize=4, color='steelblue')

    # Plot for capm=5 (using orange color)
    mur_values_5 = np.linspace(20000, 45000, 100)
    plt.plot(mur_values_5, [MatrixSolver(capm_i_5, 1, capr1_i_5, t_5, np.array(
        [mur_value] * capm_i_5)).solve() for mur_value in mur_values_5], 'o', markersize=4, color='peru')
    
    # Plot for capm=4 (using navy color)
    plt.plot(mur_values_4, [MatrixSolver(capm_o_4, 1, capr1_o_4, t_4, np.array(
        [mur_value] * capm_o_4)).solve() for mur_value in mur_values_4], 'o', markersize=4, color='darkorchid')

    # Plot for capm=5 (using orange color)
    plt.plot(mur_values_5, [MatrixSolver(capm_o_5, 1, capr1_o_5, t_5, np.array(
        [mur_value] * capm_o_5)).solve() for mur_value in mur_values_5], 'o', markersize=4, color='olivedrab')


    plt.xlabel('mur')
    plt.ylabel('Total Shielding Factor')
    plt.yscale('log')
    plt.legend(['4 layers - inscribed', '5 layers - inscribed', '4 layers - circumscribed', '5 layers - circumscribed'])
    plt.grid()
    plt.title(
        'Total Shielding Factor vs mur of a cylindrical MSR')
    plt.show()
    output_folder = "Figures"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, "SF_vs_mur_cylinder.png")
    plt.savefig(output_file_path, dpi=360)
