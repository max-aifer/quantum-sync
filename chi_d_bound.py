import scipy.stats as stats
import scipy.optimize as optimize
import numpy as np
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt

"""Parameters"""
N = 10**2 # size of Hilbert space, should be perfect square
d_S = int(np.sqrt(N)) # dimension of subsystem
om1 = 1 # frequency of first oscillator
om2 = 1 # frequency of second oscillator

"""Operators"""
n = np.diag(np.arange(d_S)) # subsytem number operator
H1 = om1 * (n + 0.5*np.eye(d_S)) # subsystem 1 Hamiltonian
H2 = om2 * (n + 0.5*np.eye(d_S)) # subsystem 2 Hamiltonian
H0 = np.kron(H1, np.eye(d_S)) + np.kron(np.eye(d_S), H2) # uncoupled Hamiltonian

a = np.diag(
    np.sqrt(np.arange(1,d_S)),1) # subsystem annihilation operator
a_dag = np.diag(
    np.sqrt(np.arange(1,d_S)),-1) # subsystem creation operator

x = (a + a_dag)/np.sqrt(2) # subsystem quadrature operators
p = -1j * (a - a_dag)/np.sqrt(2)

x1 = np.kron(x, np.eye(d_S)) # N x N single-mode quadrature operators
x2 = np.kron(np.eye(d_S),x)
p1 = np.kron(p, np.eye(d_S))
p2 = np.kron(np.eye(d_S),p)

# r^2 and d^2 operators
r_squared = x1@x1 + x2@x2 + p1@p1 + p2@p2
d_squared = (x2 - x1)@(x2 - x1) + (p2 - p1)@(p2 - p1)

R = [x1, p1, x2, p2] # list of quadrature operators

"""Symplectic forms"""
Omega1 = np.array([[0, 1],
                  [-1, 0]])
Omega = np.block([[Omega1, np.zeros((2,2))],
                 [np.zeros((2,2)),Omega1]])

"""Math helper functions"""
def ct(x):
    """Complex conjugate transpose"""
    return np.conj(x).T

def coth(x):
    """Hyperbolic cotangent"""
    return 1.0/np.tanh(x)

def arccoth(x):
    """Inverse hyperbolic cotangent"""
    return np.arctanh(1.0/x)

def cothm(A):
    """Matrix hyperbolic cotangent"""
    # get eigenvalues and eigenvectors
    w, v = np.linalg.eig(A)
    # take coth of eigenvalues
    w = coth(w)
    # return matrix
    return v @ np.diag(w) @ np.linalg.inv(v)

def arccothm(A):
    """Matrix hyperbolic cotangent"""
    # get eigenvalues and eigenvectors
    w, v = np.linalg.eig(A)
    # take coth of eigenvalues
    w = arccoth(w)
    # return matrix
    return v @ np.diag(w) @ np.linalg.inv(v)



"""Gaussian QI helper functions"""
def get_displacement_operator(r_d):
    """Get the displacement operator for a given displacement vector"""
    A = r_d[0] * p1 - r_d[1] * x1 + r_d[2] * p2 - r_d[3] * x2
    D = expm(1j * A)
    return D

def covariance_from_hamiltonian(Hm):
    """Get the covariance matrix from a Hamiltonian matrix"""
    sigma = 1j * cothm(1j * Omega @ Hm/2) @ Omega
    return sigma

def hamiltonian_from_covariance(sigma):
    """Get the covariance matrix from a Hamiltonian matrix"""
    Hm = 2j * arccothm(1j * Omega @ sigma) @ Omega
    return Hm

def hamiltonian_gaussian_state(Hm, mu=np.zeros(4)):
    """Generate a Gaussian state from 4 x 4 Hamiltonian matrix Hm"""
    # get the Hamiltonian operator
    H = np.zeros((N, N), dtype=complex)
    for i in range(4):
        for j in range(4):
            H +=( Hm[i,j] * R[i] @ R[j])/2
    rho_unnorm = expm(-H)
    rho = rho_unnorm/np.trace(rho_unnorm)
    return rho


def get_moments(rho):
    """Get the first and second moments of an arbitrary state"""
    # get mean
    mu = np.zeros(4)
    for i in range(4):
        mu[i] = np.trace(rho @ R[i])

    # get covariance matrix
    C = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            C[i,j] =  np.trace((R[i] @ R[j] + R[j] @ R[i]) @ rho) - \
            2*np.trace(R[i] @ rho) * np.trace(R[j] @ rho)
    return mu, C



def get_E(rho):
  """Expectation value E of uncoupled Hamiltonian"""
  E = np.real(np.trace(H0 @ rho))
  return E


def quantum_relative_entropy(rho, sigma):
  """Quantum relative entropy"""
  rel_ent = np.real(np.trace(
      rho@ (logm(rho) - logm(sigma))
       ))
  return max(rel_ent,0.)

def get_uncoupled_gibbs_state(beta):
  """Gibbs state of the uncoupled Hamiltonian at inverse temperature beta"""
  rho_unnormalized = np.exp(-beta * H0)
  rho_unnormalized = np.diag(np.diag(rho_unnormalized))
  rho = rho_unnormalized/np.trace(rho_unnormalized)
  return rho

def get_uncoupled_gibbs_energy(beta):
  """Analytic expression for energy of the Gibbs state of the
  uncoupled Hamiltonian."""
  E1 =  0.5 * om1 * coth(0.5 * beta * om1) 
  E2 =  0.5 * om2 * coth(0.5 * beta * om2) 
  E = E1 + E2
  return E

def get_effective_inverse_temperature(E):
  """Get the inverse temperature beta such that the Gibbs state of the
  uncoupled Hamiltonian with inverse temperature beta has energy E."""
  # get an initial guess using the classical expression
  T_classical = E/2
  # define a function whose root to find
  f = lambda T: get_uncoupled_gibbs_energy(1/T) - E
  # quantum energy is always higher than classical energy for the same temp,
  # so the quantum temp is lower than the classical temp. Therefore we can
  # bracket the root in [0, T_classical]
  res = optimize.root_scalar(f, x0=T_classical, bracket=[1.e-10,T_classical])
  T_eff = res.root
  beta_eff = 1/T_eff
  return beta_eff

def get_chi(rho):
  """Evaluate the measure of departure from equilibrium chi"""
  # first find the energy and effective temperature
  E = get_E(rho)
  beta_eff = get_effective_inverse_temperature(E)
  # get the uncoupled gibbs state at this temperature
  rho_G = get_uncoupled_gibbs_state(beta_eff)
  # evaluate the quantum relative entropy from rho to rho_G
  chi = quantum_relative_entropy(rho, rho_G)
  return chi

def get_D(rho):
  """Evaluate the distance measure D"""
  # get averages of r^2 and d^2 operators 
  r_squared_avg = np.trace(r_squared @ rho)
  d_squared_avg = np.trace(d_squared @ rho)
  # D is the square root of <d^2>/<r^2>
  D = np.real(np.sqrt(d_squared_avg/r_squared_avg))
  return D


# generate random states and look at chi-D curve
"""Look at chi-D curve for random eccentric states"""
NS = 100 # number of states to generate
D = np.zeros(NS)
CHI = np.zeros(NS)

for i in range(NS):
    # generate a hamiltonian matrix
    Hm = np.random.randn(4,4)
    Hm = Hm.T @ Hm
    # get the corresponding gaussian state
    rho = hamiltonian_gaussian_state(Hm)
    D[i] = get_D(rho)
    CHI[i] = get_chi(rho)
    if i%10==0: print(i)


# plot chi-D curve
plt.figure()
plt.scatter(D, CHI)
plt.show()





