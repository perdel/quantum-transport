import numpy as np
import matplotlib.pyplot as plt

# Constants (all MKS, except energy which is in eV)
hbar = 1.055e-34
q = 1.602e-19
eps0 = 8.854e-12
epsr = 4
m = 0.25 * 9.11e-31  # Effective mass
I0 = q * q / hbar

# Parameters
W = 1e-6
L = 10e-9
t = 1.5e-9  # W = Width, L = Length of active region, t = oxide thickness
Cg = epsr * eps0 * W * L / t
Cs = 0.05 * Cg
Cd = 0.05 * Cg
CE = Cg + Cs + Cd
U0 = q / CE
alphag = Cg / CE
alphad = Cd / CE

# Constants related to temperature and material properties
kT = 0.025
mu = 0
ep = 0.2
v = 1e5  # Escape velocity
g1 = hbar * v / (q * L)
g2 = g1
g = g1 + g2

# Energy grid
NE = 501
E = np.linspace(-1, 1, NE)
dE = E[1] - E[0]
D0 = m * q * W * L / (np.pi * hbar * hbar)  # Step Density of states per eV
D = D0 * np.concatenate([np.zeros(251), np.ones(250)])

# Reference number of electrons
f0 = 1 / (1 + np.exp((E + ep - mu) / kT))
N0 = 2 * dE * np.sum(D * f0)
ns = N0 / (L * W * 1e4)  # /cm^2

# Bias
IV = 61
VV = np.linspace(0, 0.6, IV)
I = np.zeros(IV)

for iV in range(IV):
    Vg = 0.5
    Vd = VV[iV]
    mu1 = mu
    mu2 = mu1 - Vd
    UL = -(alphag * Vg) - (alphad * Vd)

    U = 0  # Self-consistent field
    dU = 1
    while dU > 1e-6:
        f1 = 1 / (1 + np.exp((E + UL + U + ep - mu1) / kT))
        f2 = 1 / (1 + np.exp((E + UL + U + ep - mu2) / kT))
        N = dE * np.sum(D * ((f1 * g1 / g) + (f2 * g2 / g)))
        Unew = U0 * (N - N0)
        dU = abs(U - Unew)
        U = U + 0.1 * (Unew - U)

    I[iV] = dE * I0 * np.sum(D * (f1 - f2)) * g1 * g2 / g

# Plotting
plt.figure()
plt.plot(VV, I, "b", linewidth=2.0)
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.grid(True)
plt.show()
