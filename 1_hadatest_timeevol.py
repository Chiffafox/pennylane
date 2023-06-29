import pennylane as pl

# get HF state
from pennylane import qchem

# create wire arrays 
ancilla_wires = [0]
n_ancilla_wires = len(ancilla_wires)
n_sysqubits = 4
mol_wires = list(range(n_ancilla_wires, n_ancilla_wires + n_sysqubits))
mol_wires2 = list(range(n_ancilla_wires + n_sysqubits, n_ancilla_wires + 2*n_sysqubits))

# use them to create a Hamiltonian into the right corner
symbols, coordinates = qchem.read_structure("h2.xyz")
H, n_molqubits = qchem.molecular_hamiltonian(symbols, \
                        coordinates, basis = 'sto-3g', \
                            wires=mol_wires, method='pyscf')
H2, n_molqubits = qchem.molecular_hamiltonian(symbols, \
                        coordinates, basis = 'sto-3g', \
                            wires=mol_wires2, method='pyscf')
assert n_molqubits == n_sysqubits

# create the full device
n_shots = 1
dev = pl.device("lightning.qubit", shots = n_shots, wires = n_ancilla_wires + 2*n_sysqubits)

electrons = 4
hf = qchem.hf_state(electrons, n_molqubits)

@pl.qnode(dev)
def lintong(time):    
    pl.Hadamard(wires=ancilla_wires) # apply Hadamard to ancilla qubit
    pl.ctrl( pl.ApproxTimeEvolution(H, time, 1), \
                control=ancilla_wires)
    pl.ctrl( pl.ApproxTimeEvolution(H2, time, 1), \
                control=ancilla_wires)
    pl.Hadamard(wires=ancilla_wires)
    return pl.sample(pl.PauliZ(ancilla_wires))

######### plot without wire grouping ########################
# choose between matplotlib or tape_text
# print(pl.draw(lintong,wire_groups=wire_groups)(0))
fig, ax = pl.draw_mpl(lintong)(0)
import matplotlib.pyplot as plt
plt.show()
# fig.savefig('1_hadatest_timeevol.png', format='png')


######### plot with wire grouping ########################
wire_groups = { 0: [0], r"$\psi_a$": [1,2,3,4], r"$\psi_b$": [5,6,7,8] }
# choose between matplotlib or tape_text
# print(pl.draw(lintong,wire_groups=wire_groups)(0))
fig, ax = pl.draw_mpl(lintong, wire_groups=wire_groups)(0)
import matplotlib.pyplot as plt
plt.show()
# fig.savefig('1_hadatest_timeevol.png', format='png')
