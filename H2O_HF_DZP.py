import numpy as np
from ase.data import atomic_numbers, atomic_masses, covalent_radii, chemical_symbols

NOE = atomic_numbers
n_occ = (NOE["H"] + NOE["H"] + NOE["O"])/2 # Number of occupied orbitals
convergence_threshold = 1e-6
max_iterations = 100


# Load the first column of t.dat
t_data = np.loadtxt("t.dat", usecols=[0])


# Get the maximum integer value from the first column
nao = int(np.max(t_data))  # ✅ Correct way to determine `nao`

print(f"Detected nao = {nao}")  # Should now correctly print 26

# Initialize matrices
t_matrix = np.zeros((nao, nao))
v_matrix = np.zeros((nao, nao))
s_matrix = np.zeros((nao, nao))
eri_matrix = np.zeros((nao, nao, nao, nao))

# Load data
t_data = np.loadtxt("t.dat")
v_data = np.loadtxt("v.dat")
s_data = np.loadtxt("s.dat")
eri_data = np.loadtxt("eri.dat")
enuc = np.loadtxt("enuc.dat")

# Fill matrices (fixing indentation issues)
for (i, j, value) in t_data:
    i, j = int(i) - 1, int(j) - 1
    t_matrix[i, j] = t_matrix[j, i] = value

for (i, j, value) in v_data:
    i, j = int(i) - 1, int(j) - 1
    v_matrix[i, j] = v_matrix[j, i] = value

for (i, j, value) in s_data:
    i, j = int(i) - 1, int(j) - 1
    s_matrix[i, j] = s_matrix[j, i] = value

for (i, j, k, l, value) in eri_data:
    i, j, k, l = int(i) - 1, int(j) - 1, int(k) - 1, int(l) - 1
    eri_matrix[i, j, k, l] = eri_matrix[j, i, k, l] = eri_matrix[i, j, l, k] = eri_matrix[j, i, l, k] = value
    eri_matrix[k, l, i, j] = eri_matrix[l, k, i, j] = eri_matrix[k, l, j, i] = eri_matrix[l, k, j, i] = value

# Core Hamiltonian
H_core_matrix = t_matrix + v_matrix

eigenvalues, eigenvectors = np.linalg.eigh(s_matrix)
print("eigenvectors:\n", eigenvectors)
print("eigenvalues:\n", eigenvalues)
L = eigenvectors
D = np.diag(eigenvalues) #diagonalizing the eigenvalues
print("D:\n", D)

#D_inv_sqrt = np.diag(1 / np.sqrt(eigenvalues))
D_inv = np.linalg.inv(D)
D_inv_sqrt = np.sqrt(D_inv)
Sinv_sqrt= L@D_inv_sqrt@L.T      # '@' denotes the matrix multiplication using numpy, 'L.T' means transpose of L
print("Sinv_sqrt:\n", Sinv_sqrt)

np.savetxt("Sinv_sqrt.dat", Sinv_sqrt, fmt="%f")

F01 = Sinv_sqrt.T@H_core_matrix@Sinv_sqrt
print("F01:\n", F01)
np.savetxt("F01", F01, fmt="%f")

eigenvalues2, eigenvectors2 = np.linalg.eigh(F01)
print("eigenvectors2:\n", eigenvectors2)
C01 = eigenvectors2
print("C01:\n", C01)
print("eigenvalues2:\n", eigenvalues2)
C0 = Sinv_sqrt@C01
np.savetxt("C0", C0, fmt="%f")
print("C0:\n", C0)

n_occ = 5
C02 = C0[:, :n_occ]

D0 = C02@C02.T

np.savetxt("D_prev", D0, fmt = "%f")
# Iterative Hartree-Fock procedure

E_prev = 0.0
for iteration in range(max_iterations):
    # Build the new Fock matrix
    F_new = np.zeros((nao, nao))
    for i in range(nao):
        for j in range(nao):
            F_new[i, j] = H_core_matrix[i, j]
            for k in range(nao):
                for l in range(nao):
                    F_new[i, j] += D0[k, l] * (2 * eri_matrix[i, j, k, l] - eri_matrix[i, k, j, l])

                    # print("F_new:\n", F_new)
                    # print("eri_matrix:\n", eri_matrix[i, j, k, l])

    # Transform Fock matrix to orthogonal basis and diagonalize
    F_ortho = Sinv_sqrt.T @ F_new @ Sinv_sqrt
    _, C_ortho = np.linalg.eigh(F_ortho)
    C = Sinv_sqrt @ C_ortho
    C_occ = C[:, :n_occ]
    D_new = C_occ @ C_occ.T

    # Calculate electronic energy
    E_electronic = np.sum(D_new * (H_core_matrix + F_new))

    # Total energy
    Total_E = E_electronic + enuc

    # Check for convergence
    energy_diff = np.abs(Total_E - E_prev)
    print(f"Iteration {iteration + 1}: Total Energy = {Total_E:.12f}, Energy Diff = {energy_diff:.12f}")
    if energy_diff < convergence_threshold:
        orbital_energies, _ = np.linalg.eigh(F_ortho)
        print("Orbital energies at convergence:\n", orbital_energies)
        np.savetxt("orbital_energies.dat", orbital_energies, fmt="%f")
    if energy_diff < convergence_threshold:
        print("Convergence achieved!")
        break

    # Updating for next iteration
    D0 = D_new
    E_prev = Total_E
else:
    print("Maximum iterations reached without convergence.")

# Save final results
np.savetxt("final_density_matrix.dat", D_new, fmt="%f")
np.savetxt("final_fock_matrix.dat", F_new, fmt="%f")
np.savetxt("final_coefficients_matrix.dat", C, fmt="%f")
print(f"Final Total Energy: {Total_E:.12f}")