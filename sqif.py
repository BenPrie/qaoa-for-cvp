# Imports, as always...
import numpy as np
from math import log2
from fpylll import IntegerMatrix, LLL, GSO
from copy import deepcopy
import cirq
import sympy
import qsimcirq
from scipy.optimize import minimize

# ---

# Some hard-coded primes to use (they're immutable, so why bother calculating them every time).
primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987,1993,1997,1999,2003,2011,2017,2027,2029,2039,2053,2063,2069,2081,2083,2087,2089,2099,2111,2113,2129,2131,2137,2141,2143,2153,2161,2179,2203,2207,2213,2221,2237,2239,2243,2251,2267,2269,2273,2281,2287,2293,2297,2309,2311,2333,2339,2341,2347,2351,2357,2371,2377,2381,2383,2389,2393,2399,2411,2417,2423,2437,2441,2447,2459,2467,2473,2477,2503,2521,2531,2539,2543,2549,2551,2557,2579,2591,2593,2609,2617,2621,2633,2647,2657,2659,2663,2671,2677,2683,2687,2689,2693,2699,2707,2711,2713,2719,2729,2731,2741,2749,2753,2767,2777,2789,2791,2797,2801,2803,2819,2833,2837,2843,2851,2857,2861,2879,2887,2897,2903,2909,2917,2927,2939,2953,2957,2963,2969,2971,2999,3001,3011,3019,3023,3037,3041,3049,3061,3067,3079,3083,3089,3109,3119,3121,3137,3163,3167,3169,3181,3187,3191,3203,3209,3217,3221,3229,3251,3253,3257,3259,3271,3299,3301,3307,3313,3319,3323,3329,3331,3343,3347,3359,3361,3371,3373,3389,3391,3407,3413,3433,3449,3457,3461,3463,3467,3469,3491,3499,3511,3517,3527,3529,3533,3539,3541,3547,3557,3559,3571,3581,3583,3593,3607,3613,3617,3623,3631,3637,3643,3659,3671,3673,3677,3691,3697,3701,3709,3719,3727,3733,3739,3761,3767,3769,3779,3793,3797,3803,3821,3823,3833,3847,3851,3853,3863,3877,3881,3889,3907,3911,3917,3919,3923,3929,3931,3943,3947,3967,3989,4001,4003,4007,4013,4019,4021,4027,4049,4051,4057,4073,4079,4091,4093,4099,4111,4127,4129,4133,4139,4153,4157,4159,4177,4201,4211,4217,4219,4229,4231,4241,4243,4253,4259,4261,4271,4273,4283,4289,4297,4327,4337,4339,4349,4357,4363,4373,4391,4397,4409,4421,4423,4441,4447,4451,4457,4463,4481,4483,4493,4507,4513,4517,4519,4523,4547,4549,4561,4567,4583,4591,4597,4603,4621,4637,4639,4643,4649,4651,4657,4663,4673,4679,4691,4703,4721,4723,4729,4733,4751,4759,4783,4787,4789,4793,4799,4801,4813,4817,4831,4861,4871,4877,4889,4903,4909,4919,4931,4933,4937,4943,4951,4957,4967,4969,4973,4987,4993,4999,5003,5009,5011,5021,5023,5039,5051,5059,5077,5081,5087,5099,5101,5107,5113,5119,5147,5153,5167,5171,5179,5189,5197,5209,5227,5231,5233,5237,5261,5273,5279,5281,5297,5303,5309,5323,5333,5347,5351,5381,5387,5393,5399,5407,5413,5417,5419,5431,5437,5441,5443,5449,5471,5477,5479,5483,5501,5503,5507,5519,5521,5527,5531,5557,5563,5569,5573,5581,5591,5623,5639,5641,5647,5651,5653,5657,5659,5669,5683,5689,5693,5701,5711,5717,5737,5741,5743,5749,5779,5783,5791,5801,5807,5813,5821,5827,5839,5843,5849,5851,5857,5861,5867,5869,5879,5881,5897,5903,5923,5927,5939,5953,5981,5987,6007,6011,6029,6037,6043,6047,6053,6067,6073,6079,6089,6091,6101,6113,6121,6131,6133,6143,6151,6163,6173,6197,6199,6203,6211,6217,6221,6229,6247,6257,6263,6269,6271,6277,6287,6299,6301,6311,6317,6323,6329,6337,6343,6353,6359,6361,6367,6373,6379,6389,6397,6421,6427,6449,6451,6469,6473,6481,6491,6521,6529,6547,6551,6553,6563,6569,6571,6577,6581,6599,6607,6619,6637,6653,6659,6661,6673,6679,6689,6691,6701,6703,6709,6719,6733,6737,6761,6763,6779,6781,6791,6793,6803,6823,6827,6829,6833,6841,6857,6863,6869,6871,6883,6899,6907,6911,6917,6947,6949,6959,6961,6967,6971,6977,6983,6991,6997,7001,7013,7019,7027,7039,7043,7057,7069,7079,7103,7109,7121,7127,7129,7151,7159,7177,7187,7193,7207,7211,7213,7219,7229,7237,7243,7247,7253,7283,7297,7307,7309,7321,7331,7333,7349,7351,7369,7393,7411,7417,7433,7451,7457,7459,7477,7481,7487,7489,7499,7507,7517,7523,7529,7537,7541,7547,7549,7559,7561,7573,7577,7583,7589,7591,7603,7607,7621,7639,7643,7649,7669,7673,7681,7687,7691,7699,7703,7717,7723,7727,7741,7753,7757,7759,7789,7793,7817,7823,7829,7841,7853,7867,7873,7877,7879,7883,7901,7907,7919]

# ---

# Some helper functions...

def integer_matrix_to_numpy(M):
    """
    Convert an fyplll.IntegerMatrix object to a numpy.ndarray object.

    :param M: fpylll.IntegerMatrix.
    :return: The same matrix (M) cast to a numpy ndarray.
    """

    m, n = M.nrows, M.ncols
    D = np.zeros((m, n), dtype=int)
    M.to_matrix(D)
    return D


def generate_gamma_layer(H, i):
    """
    Generate the i-th gamma layer executing the unitary exp(-i * gamma * H).

    :param H: The Hamiltonian.
    :param i: Layer index.
    :return: [] of cirq.DensePauliString object corresponding to the i-th gamma layer in the QAOA circuit.
    """

    # Gamma symbol placeholder.
    gamma = sympy.Symbol(f'gamma_{i}')

    # Instantiate the DensePauliString operators.
    dense_I = cirq.DensePauliString('')
    dense_Z = cirq.DensePauliString('Z')
    dense_ZZ = cirq.DensePauliString('ZZ')

    # Consider the terms in the Hamiltonian.
    for term in H:
        # Split the term into its coefficient and operator.
        coefficient = term.coefficient
        operator = term.with_coefficient(1).gate

        # Map to the appropriate circuit element on the basis of the operator, parameterised by gamma.
        if operator == dense_ZZ:
            yield cirq.ZZ(*term.qubits) ** (gamma * coefficient)
        elif operator == dense_Z:
            yield cirq.Z(*term.qubits) ** (gamma * coefficient)
        elif operator == dense_I:
            yield []
        else:
            raise Exception(f'Unrecognised Pauli string term {term} in the Hamiltonian.')


def generate_beta_layer(qubits, i):
    """
    Generate the i-th beta layer, executing a Pauli-X raised to beta across the given qubits.

    :param qubits: The qubits in the circuit.
    :param i: Layer index.
    :return: [] of cirq.DensePauliString object corresponding to the i-th beta layer in the QAOA circuit.
    """

    # Beta symbol placeholder.
    beta = sympy.Symbol(f'beta_{i}')

    # The layer is trivially defined by NOT gates parameterised by beta.
    return [cirq.X(q) ** beta for q in qubits]

# ---

class CVP:
    def __init__(self) -> None:
        # All of the variables defining the CVP instance held by the object.
        # (Initially, nothing).
        self.N = None
        self.l = None
        self.c = None
        self.n, self.m = None, None
        self.B = None
        self.t = None


    def generate_cvp(self, N, l=1, c=1, seed=42) -> None:
        """
        Reduce a the problem of factorising an integer semi-prime to a CVP on the prime lattice. 

        :param N: Integer semi-prime (to be factored).
        :param l: Lattice parameter (recommend in {1, 2}).
        :param c: Precision parameter.
        :param seed: Seed for random number generation.
        
        No return. The CVP object is populated with an instance of a CVP.
        """

        # Set the seed for random number generation (for reproducability).
        np.random.seed(seed)

        self.N = N
        self.l = l
        self.c = c

        # Determine the claimed lattice dimension.
        self.n = log2(self.N)
        self.m = (self.l * self.n) / log2(self.n)

        # Round them, after the fact.
        self.n, self.m = int(round(self.n)), int(round(self.m))

        # Produce the random permutation for the diagonal.
        f = np.random.permutation([(i + 1) // 2 for i in range(1, self.m + 1)])

        # Create a zero matrix and add in the diagonal permutation.
        self.B = np.zeros(shape=(self.m, self.m))
        np.fill_diagonal(self.B, f)

        # Create the extra final row and stick it on.
        final_row = np.round(10 ** self.c * np.log(np.array(primes[:self.m])))
        self.B = np.vstack((self.B, final_row))

        # fpylll doesn't like numyp arrays, so convert it to a stnadard array.
        self.B = [[int(b) for b in bs] for bs in self.B]

        # Convert B to a matrix of integers (in fpylll's own type).
        self.B = IntegerMatrix.from_matrix(self.B)

        # Define the target vector.
        self.t = np.zeros(self.m + 1)
        self.t[-1] = np.round(10 ** self.c * log2(self.N))
        self.t = tuple(self.t.astype(int))

# ---

def find_b_op(B, t, delta=.75):
    """
    Find an approximate solution to the CVP classically. We do this via Babai's algorithm with LLL-reduction.

    :param B: Prime basis (fpylll.IntegerMatrix object).
    :param t: Target vector (tuple of integers).
    :param delta: Hyperparameter for LLL-reduction (wikipedia recommends .75, as do Yan et al. (2022)).
    :return: The given B and t, the reduced basis (D) and weights (w), the approximate solution (b_op = D*w), the residual vector (b_op - t), and the step signs (for use in the Hamiltonian later on).
    """

    # Create a copy of the prime basis and reduce it by LLL-reduction.
    D = deepcopy(B).transpose()
    LLL.reduction(D, delta)

    # Use a Gram-Schmidt orthogonalisation matrix object to run Babai's algorithm.
    M = GSO.Mat(D, update=True)
    w = M.babai(t)

    # We want to make a note of rounding directions (by comparing coefficients).
    A = IntegerMatrix(2 * D.nrows, D.ncols)
    for i in range(D.nrows):
        for j in range(D.ncols):
            A[i, j] = D[i, j]

    b = np.array(t)
    for i in reversed(range(D.nrows)):
        for j in range(D.ncols):
            A[i + D.nrows, j] = int(b[j])
        b -= w[i] * np.array(D[i])

    # Go through and make the comparisons to track which way each Gram-Schmidt coefficient (mu) was rounded.
    M = GSO.Mat(A, update=True)
    rounding_direction = []
    for i in range(D.nrows):
        mu = M.get_mu(i + D.nrows, i)
        rounding_direction.append(w[i] > mu)

    b_op = np.array(D.multiply_left(w))
    residual_vector = b_op - np.array(t)

    # Reformat the basis and reduced basis (we won't need them as IntegerMatrix objects anymore).
    B = integer_matrix_to_numpy(B)
    D = integer_matrix_to_numpy(D.transpose())
    t = np.array(t)

    # Convert the notes of rounding directions into "step signs", establishing the sign for each operator.
    step_signs = - (np.array(rounding_direction).astype(int) * 2 - 1)

    return D, w, b_op, residual_vector, step_signs


def define_hamiltonian(D, residual_vector, step_signs):
    """
    Define the Hamiltonian using the unit hypercube search as outlined in Yan et al. (2022).

    :param D: The reduced (prime) basis.
    :param residual_vector: The discrepancy between the approximate solution and the target vector (b_op - t).
    :param step_signs: The signs for each operator.
    :param verbose: Whether to display messages during the computation.
    :return: The Hamiltonian (H).
    """

    # Define the circuit.
    circuit = cirq.LineQubit.range(D.shape[0])

    # Add the appropriate operator to each qubit.
    operators = []
    for i, sign in zip(circuit, step_signs):
        operator = sign * ((cirq.I(i) + -cirq.Z(i)) / 2)
        operators.append(operator)

    # Define the Hamiltonian.
    H = cirq.PauliSum()
    for j in range(D.shape[0]):
        h = residual_vector[j]
        for i in range(D.shape[1]):
            h += operators[i] * D[j, i]
        H += h ** 2

    return H


def generate_qaoa_circuit(H, p=1):
    """
    Generate a QAOA circuit for the given Hamiltonian with a given depth.

    :param H: The Hamiltonian.
    :param p: Depth of the circuit (should be kept relatively small -- say 1 to 5).
    :param verbose: Whether to display messages during the computation.
    :return: cirq.Circuit object performing QAOA with the given Hamiltonian p times (depth p).
    """

    # Number of qubits.
    qubits = H.qubits

    # Define the circuit.
    return cirq.Circuit(
        # Hadamard over all qubits first to open uniform superposition.
        cirq.H.on_each(*qubits),

        # p layer of QAOA-ness.
        [
            (
                # Gamma layer.
                generate_gamma_layer(H, i),

                # Beta layer.
                generate_beta_layer(qubits, i)
            )
            for i in range(p)
        ]
    )


def find_optimal_parameters(circuit, H, x0=None, min_method='Nelder-Mead'):
    """
    Find the optimal assignments over the parameters in the given circuit.
    
    :param circuit: The QAOA circuit.
    :param H: The Hamiltonian.
    :param x0: Initial parameter values.

    :return: cirq.ParamResolver object assigning values to each placeholder parameter in the given circuit.
    """
    
    # Define the parameters and observables.
    parameters = sorted(cirq.parameter_names(circuit))
    observables = [term.with_coefficient(1) for term in H]
    
    # Define the function to be minimised -- the expectation value with given assignments.
    def func_to_minimise(x):
        # Assign the parameters their given values.
        parameter_assignments = cirq.ParamResolver({param : val for param, val in zip(parameters, x)})
        
        # Simulator object.
        simulator = qsimcirq.QSimSimulator(
            qsimcirq.QSimOptions(cpu_threads=8, verbosity=0)
        )
        
        # Simulate the expectation value.
        result = simulator.simulate_expectation_values(
            program=circuit, observables=observables, param_resolver=parameter_assignments
        )
        
        # Compute the return, ignoring the imaginary component.
        return sum(term.coefficient * val for term, val in zip(H, result)).real
    
    # Initialise the assignments (with no prior knowledge, all zeros is fine).
    if x0 is None:
        x0 = np.asarray([0.0] * len(parameters))
    
    # Minimise the function over the parameters.
    result = minimize(func_to_minimise, x0, method=min_method)
    return cirq.ParamResolver({param: optimal_val for param, optimal_val in zip(parameters, result.x)})


def sample_bitstring_from_parameters(circuit, parameter_assignments, H, repetitions):
    """
    Sample the states obtained via measurements made on the circuit with the given parameter assignments, given as a histogram of repeated runs.
    
    :param circuit: The circuit to measure.
    :param parameter_assignments: The assignments to the parameters (as a cirq.ParamResolver object).
    :param H: The Hamiltonian (whose qubits are operated on).
    :param repetitions: Number of times to repeat the sampling.
    :return: Histogram of states sampled by the measurements.
    """
    
    # Simulator object.
    simulator = qsimcirq.QSimSimulator(
        qsimcirq.QSimOptions(cpu_threads=8, verbosity=0)
    )
    
    # Add a set of measurement operators to the circuit.
    measurement_circuit = circuit + cirq.Circuit(cirq.measure(H.qubits, key='m'))
    
    # Run the simulation.
    result = simulator.run(measurement_circuit, param_resolver=parameter_assignments, repetitions=repetitions)
    
    # Let's have a histogram.
    return result.histogram(key='m')


def integer_outcomes_to_lattice_vectors(m, states, w, D, step_signs):
    """
    Convert states of integer outcomes to lattice vectors on the given prime lattice.
    """

    # Convert the integer state to a binary state -- telling us WHETHER to step in each basis direction.
    binary_states = (((states[:, None] & (1 << np.arange(m)[::-1]))) > 0).astype(int)
    
    # Pairwise-multiply the binary states and step signs -- telling us HOW to step in each basis.
    steps = np.multiply(binary_states, step_signs)
    
    # Add the steps to the Babai weights.
    w_new = w + steps
    
    # Left-multiply to yield the new vector v_new on the lattice corresponding to the states.
    return w_new @ D.T

# ---

def solve_cvp(cvp, n_samples, delta=.75, p=1, x0=None, min_method='Nelder-Mead', optimal_parameters=None, verbose=True):
    """
    Given a CVP, perform the SQIF algorithm's subroutine to solve it.

    :param n_samples: No. times shots for the circuit.
    :param delta: LLL-reduction hyperparameter.
    :param p: Number of layers in the QAOA circuit.
    :param x0: Initial 'guess' for parameter assignment.
    :param min_method: Method to use in minimisation for parameter optimisation.
    :param optimal_parameters: Optionally pass in the parameter assignment. Leave as None to find them automatically.
    :param verbose: Whether to print messages during the run.

    :return: All 2^n solutions around the approximate solution, sorted according to probability, and corresponding probabilities.
    """

    # Approximate the solution by Babai's algorithm.
    D, w, b_op, residual_vector, step_signs = find_b_op(cvp.B, cvp.t, delta)

    # Define a Hamiltonian.
    H = define_hamiltonian(D, residual_vector, step_signs)

    if verbose:
        # Pretty print the Hamiltonian (because it gets very messy very quickly).
        string_H = str(H)
        string_H = string_H.replace('+', '\n+')
        string_H = string_H.replace('-', '\n-')
        string_H = string_H.replace('*', ' * ')
        print(string_H, '\n')  

    # Generate a circuit for QAOA.
    circuit = generate_qaoa_circuit(H, p)

    if verbose:
        # Print the circuit.
        print(circuit)
        print()

    # Find the optimal set of betas and gammas for the circuit.
    if optimal_parameters is None:
        optimal_parameters = find_optimal_parameters(circuit, H, x0, min_method)

    if verbose:
        # Print the (found) optimal parameter assignments.
        for param, val in optimal_parameters.param_dict.items():
            print(f'{param}: {val}')

    # Sample measurements from the circuit.
    states_histogram = sample_bitstring_from_parameters(circuit, optimal_parameters, H, n_samples)
    outcomes, frequencies = zip(*states_histogram.most_common(len(states_histogram)))

    # Convert the remaining integer outcomes to lattice vectors.
    lattice_vectors = integer_outcomes_to_lattice_vectors(cvp.m, np.array(outcomes), w, D, step_signs)

    return lattice_vectors, np.array(frequencies) / n_samples, b_op, optimal_parameters
