# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:33:23 2024

@author: UWTUCANMag
"""

import numpy as np
from scipy.linalg import lu, solve
import matplotlib.pyplot as plt
import os

class MatrixSolver:
    def __init__(self, capm, n, k_values):
        self.capm = capm
        self.rank = 2 * capm
        self.n = n
        self.k_value = k_value
        self.sfact_values = []
        self.r = np.zeros(self.rank)
        self.t = np.zeros(capm)
        self.capr1 = np.zeros(capm)
        self.mur = np.zeros(capm)
        self.a = np.zeros((self.rank, self.rank))
        self.inverse = np.zeros((self.rank, self.rank))

    def define_geometry(self):
        self.capr1[0] = 0.5
        for i in range(self.capm):
            self.capr1[i] = self.capr1[0] * (1 + self.k_value) ** i

        for i in range(self.capm):
            self.t[i] = 1.0 / 16 * 0.0254
            self.mur[i] = 20000
            print(f"{i} {self.capr1[i]} {self.t[i]} {self.mur[i]}")

         # Initialize the r array based on capr1 and t
        j = 0
        for i in range(self.capm):
            self.r[j] = self.capr1[i]
            print(f"r{j} {self.r[j]}")
            j += 1
            self.r[j] = self.capr1[i] + self.t[i]
            print(f"r{j} {self.r[j]}")
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
                print(f"{element} ", end="")
            print()

    def print_matrix(self):
        for i in range(self.rank):
            for j in range(self.rank):
                print(f"{self.a[i, j]} ", end="")
            print()

    def calculate_inverse(self):
        p, l, u = lu(self.a)
        self.inverse = solve(u, solve(l, p.T))

        for i in range(self.rank):
            for j in range(self.rank):
                print(f"{self.inverse[i, j]} ", end="")
            print()

    def calculate_sfact(self):
        summat = 0
        for i in range(self.rank):
            sumrow = np.sum(self.inverse[i, :])
            summat += sumrow

        sfact = 1.0 / (1.0 + summat)
        self.sfact_values.append(sfact)


if __name__ == "__main__":
    capm = 4
    k_values = np.linspace(0, 1, 50)  # Vary k from 0 to 1
    n_values = [1, 2, 3]
    colors = ['navy', 'purple', 'darkkhaki']

    for n, color in zip(n_values, colors):
        sfact_values = []

        for k_value in k_values:
            matrix_solver = MatrixSolver(capm, n, k_value)
            matrix_solver.define_geometry()
            matrix_solver.fill_matrix()
            matrix_solver.calculate_inverse()
            matrix_solver.calculate_sfact()

            # Plot the result for each iteration
            plt.plot(k_value, matrix_solver.sfact_values[-1], 'o', color=color)
    # Set the common labels, legend, and log scale for y-axis outside the loop
    plt.xlabel('Scale Factor (k)')
    plt.ylabel('Total Shielding Factor')
    plt.yscale('log')  # Set y-axis to log scale
    plt.ylim(100, 1e8)   # Set the y-axis limit from 1 to 10^8
    plt.grid()
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=color, markersize=10) for color in colors]
    legend_labels = [f'n = {n}' for n in n_values]

    plt.legend(legend_handles, legend_labels)
    plt.title(
        'The total shielding factor of four concentric cylindrical shells')
    plt.savefig('CylindricalShells.png', bbox_inches='tight', dpi=360)
    plt.show()
    output_folder = "Figures"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, "CylindricalShells.png")
    plt.savefig(output_file_path, dpi=360)
