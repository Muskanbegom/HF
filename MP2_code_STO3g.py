import numpy as np


nao = 7

Coeff_matrix = np.loadtxt("final_coefficients_matrix.dat")
print("Coeff:\n", Coeff_matrix)
eri_matrix = np.zeros((nao, nao, nao, nao))
eri_data = np.loadtxt("eri.dat")
for (i, j, k, l, value) in eri_data:
    i, j, k, l = int(i) - 1, int(j) - 1, int(k) - 1, int(l) - 1
    eri_matrix[i, j, k, l] = eri_matrix[j, i, k, l] = eri_matrix[i, j, l, k] = eri_matrix[j, i, l, k] = value
    eri_matrix[k, l, i, j] = eri_matrix[l, k, i, j] = eri_matrix[k, l, j, i] = eri_matrix[l, k, j, i] = value


matrix1 = np.einsum('mnlo, os -> mnls', eri_matrix, Coeff_matrix)
matrix2 = np.einsum('mnls, lr -> mnsr', matrix1, Coeff_matrix)
matrix3 = np.einsum('mnsr, nq -> msrq', matrix2, Coeff_matrix)
matrix4 = np.einsum('msrq, mp -> srqp', matrix3, Coeff_matrix)
n_occ = 5
nao = 7

mo_energies = np.loadtxt("orbital_energies.dat")

# mo_energies = np.diag(orbital_energies)
print(mo_energies)
E_mp2 = 0.0
for i in range(n_occ):
    for j in range(n_occ):
        for a in range(n_occ, nao):
            for b in range(n_occ, nao):
                iajb = matrix4[i, a, j, b]
                ibja = matrix4[i, b, j, a]
                denomi = mo_energies[i] + mo_energies[j] - mo_energies[a] - mo_energies[b]
                # print(mo_energies[i])
                # print(denomi)
                E_mp2 += (iajb * (2 * iajb - ibja)) / denomi
                print("E_mp2:\n", E_mp2)
                Emp2 = -0.049149636120
