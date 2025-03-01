import numpy as np
import matplotlib.pyplot as plt

# Constants (all MKS, except energy which is in eV)
hbar = 1.055e-34
q = 1.602e-19
I0 = q * q / hbar

# Parameters
U0 = 0.025
kT = 0.025
mu = 0
ep = 0.2
g1 = 0.005
g2 = 0.005
g = g1 + g2
alphag = 0
alphad = 0.5

# Energy grid
NE = 501
E = np.linspace(-1, 1, NE)
dE = E[1] - E[0]
D = (g / (2 * np.pi)) / (E**2 + (g / 2)**2)  # Lorentzian Density of states per eV
D = D / (dE * np.sum(D))  # Normalizing to one

# Bias
IV = 101
VV = np.linspace(0, 1, IV)
I = np.zeros(IV)
N = np.zeros(IV)

for iV in range(IV):
    Vg = 0
    Vd = VV[iV]
    mu1 = mu
    mu2 = mu1 - Vd
    UL = -(alphag * Vg) - (alphad * Vd)

    # Self-consistent field
    U = 0
    dU = 1
    while dU > 1e-6:
        f1 = 1 / (1 + np.exp((E + ep + UL + U - mu1) / kT))
        f2 = 1 / (1 + np.exp((E + ep + UL + U - mu2) / kT))
        N[iV] = dE * np.sum(D * ((f1 * g1 / g) + (f2 * g2 / g)))
        Unew = U0 * N[iV]
        dU = np.abs(U - Unew)
        U = U + 0.1 * (Unew - U)

    I[iV] = dE * I0 * (np.sum(D * (f1 - f2))) * (g1 * g2 / g)

# Plotting
plt.figure()
plt.plot(VV, I, 'b', linewidth=2.0)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.grid(True)
plt.show()