import numpy as np
from scipy.linalg import lu, solve
import matplotlib.pyplot as plt
import os


class MatrixSolver:
    def __init__(self, capm, n, capr1, t, m1, m2):
        self.capm = capm
        self.rank = 2 * capm
        self.n = n
        self.m1 = m1
        self.m2 = m2
        self.r = np.zeros(self.rank)
        self.a = np.zeros((self.rank, self.rank))
        self.inverse = np.zeros((self.rank, self.rank))
        self.sfact = None
        self.capr1 = capr1
        self.t = t

    def define_geometry(self):
        
        if self.capm == 4:
            self.mur = [self.m1] + [self.m2] * (self.capm - 1)
        else:
            self.mur = self.mur = [self.m1] + [self.m1] + [self.m2] * (self.capm - 2)

        j = 0
        for i in range(self.capm):
            self.r[j] = self.capr1[i]
            j += 1
            self.r[j] = self.capr1[i] + self.t[i]
            j += 1
            print(f"{i}, {self.capr1[i]}, {self.t[i]}, {self.mur[i]}")

    def fill_matrix(self):
        for i in range(self.rank):
            for j in range(self.rank):
                if j < i:
                    element = (self.n / (self.n + 1)) * \
                        (self.r[j] / self.r[i]) ** (2 * self.n + 1)
                elif j > i:
                    element = -1
                else:  # j == i
                    if (i + 1) % 2 == 1:
                        m = ((i + 1) + 1) // 2 - 1
                        element = - \
                            ((self.n + 1) * self.mur[m] + self.n) / \
                            ((self.n + 1) * (self.mur[m] - 1))
                    else:
                        m = (i + 1) // 2 - 1
                        element = (
                            self.n * self.mur[m] + (self.n + 1)) / ((self.n + 1) * (self.mur[m] - 1))
                self.a[i, j] = element
                print(f"{element} ", end="")
            print()

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
    capm_o_4 = 4
    capm_i_5 = 5
    capm_o_5 = 5

    capr1_i_4 = np.array([2.4 / 2, 2.6 / 2, 3.0 / 2, 3.5 / 2])
    t_4 = np.array([0.002, 0.003, 0.003, 0.004])

    capr1_i_5 = np.array([2.26 / 2, 2.4 / 2, 2.6 / 2, 3.0 / 2, 3.5 / 2])
    t_5 = np.array([0.002, 0.002, 0.003, 0.003, 0.004])

    capr1_o_4 = np.sqrt(2) * np.array([2.4 / 2, 2.6 / 2, 3.0 / 2, 3.5 / 2])
    t_4 = np.array([0.002, 0.003, 0.003, 0.004])

    capr1_o_5 = np.sqrt(
        2) * np.array([2.26 / 2, 2.4 / 2, 2.6 / 2, 3.0 / 2, 3.5 / 2])
    t_5 = np.array([0.002, 0.002, 0.003, 0.003, 0.004])

    start_0 = 50000
    stop_0 = 70000

    start_1 = 20000
    stop_1 = 40000

    # Create linspace arrays for m1 and m2
    m1_range = np.linspace(start_0, stop_0, 101)  # Example: 5 points
    m2_range = np.linspace(start_1, stop_1, 101)  # Example: 5 points

    # Plotting setup
    plt.figure(figsize=(8, 6))

    plt.plot(m2_range, [MatrixSolver(capm_i_4, 1, capr1_i_4, t_4, m1_range, m2_range).solve() for m1_range, m2_range in zip(m1_range, m2_range)],
             'o', markersize=4, color='steelblue')
    
    plt.plot(m2_range, [MatrixSolver(capm_i_5, 1, capr1_i_5, t_5, m1_range, m2_range).solve() for m1_range, m2_range in zip(m1_range, m2_range)],
             'o', markersize=4, color='peru')

    plt.plot(m2_range, [MatrixSolver(capm_o_4, 1, capr1_o_4, t_4, m1_range, m2_range).solve() for m1_range, m2_range in zip(m1_range, m2_range)],
             'o', markersize=4, color='darkorchid')

    plt.plot(m2_range, [MatrixSolver(capm_o_5, 1, capr1_o_5, t_5, m1_range, m2_range).solve() for m1_range, m2_range in zip(m1_range, m2_range)],
             'o', markersize=4, color='olivedrab')

    plt.xlabel('m1')
    plt.ylabel('Total Shielding Factor')
    plt.yscale('log')
    plt.legend(['4 layers - inscribed', '5 layers - inscribed',
               '4 layers - circumscribed', '5 layers - circumscribed'])
    plt.grid()
    plt.title('Total Shielding Factor vs mur1 of a spherical MSR')
    plt.show()

    output_folder = "Figures"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, "SF_vs_m1_sphere_2m_2l.png")
    plt.savefig(output_file_path, dpi=360)
