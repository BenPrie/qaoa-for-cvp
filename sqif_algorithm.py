# Imports, as always...
import logging
import numpy as np
from fpylll import IntegerMatrix, LLL, GSO
from copy import deepcopy
import cirq
import sympy
import qsimcirq
from scipy.optimize import minimize


class SQIF:

    def __init__(self, N, log_filename='log', l=1, seed=42):
        """
        Instantiate an instance of a sublinear quantum integer factorisation (SQIF) algorithm.

        :param N: The integer to be factored (assumed to be semi-prime).
        :param log_filename: The name of the file to write logging messages to (no suffix).
        :param l: Lattice hyperparameter (Yan et al. (2022) keep to 1 or 2).
        :param seed: Seed for random generation.
        """

        # Integer to be factored.
        self.N = N

        # Set up logging to be written to the specified file.
        log_filename = log_filename.split('.')[0]
        logging.basicConfig(filename=log_filename + '.log', filemode='w', encoding='utf-8', level=logging.DEBUG)

        # The claimed lattice dimension to factor this integer is l * log N / log log N.
        n = np.log2(N)
        m = (l * n) / np.log2(n)

        # Round them, after the fact.
        self.n, self.m = int(np.floor(n)), int(np.floor(m))

        # Logging.
        logging.info(f'Integer (N) to be factored: {N}.')
        logging.info(f'Bit-length of N: {n}.')
        logging.info(f'The claim is that we need only a lattice of dimension {m}.\n')

        # A bunch of primes to use -- hard-coding this is fine.
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

        # Seed for random generation.
        np.random.seed(seed)

    def generate_CVP(self, c):
        """
        Generate the CVP problem by defining the prime lattice (encoding p_n-sr-pairs) and the target vector (encoding the integer to be factored).

        :param c: "Precision parameter" for the lattice -- conjectured to be positively correlated with the probability of finding close vectors.
        :return: The prime lattice basis (B) and the target vector (t).
        """

        # Produce the random permutation for the diagonal.
        f = np.random.permutation([(i + 1) // 2 for i in range(1, self.m + 1)])

        # Create a zero matrix and add in the diagonal permutation.
        B = np.zeros(shape=(self.m, self.m))
        np.fill_diagonal(B, f)

        # Create the extra final row and stick it on.
        final_row = np.round(10 ** c * np.log(np.array(self.primes[:self.m])))
        B = np.vstack((B, final_row))

        # fpylll doesn't like numpy arrays, so convert it to a standard array.
        B = [[int(b) for b in bs] for bs in B]

        # Convert B to a matrix of integers (in fpylll's own type).
        B = IntegerMatrix.from_matrix(B)
        logging.info(f'B = \n{B}\n')

        # Define the target vector.
        t = np.zeros(self.m + 1)
        t[-1] = np.round(10 ** c * np.log(self.N))
        t = tuple(t.astype(int))
        logging.info(f't = \n{t}\n')

        return B, t

    def integer_matrix_to_numpy(self, M):
        """
        Convert an fyplll.IntegerMatrix object to a numpy.ndarray object.

        :param M: fpylll.IntegerMatrix.
        :return: The same matrix (M) cast to a numpy ndarray.
        """

        m, n = M.nrows, M.ncols
        D = np.zeros((m, n), dtype=int)
        M.to_matrix(D)
        return D

    def find_b_op(self, B, t, delta=.75):
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
        logging.info(f'D (transposed) = \n{D}\n')

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
        logging.info(f'b_op = \n{b_op}\n')
        logging.info(f'Hence, the residual vector is \n{residual_vector}\n')
        logging.info(f'This has a distance of {round(np.linalg.norm(residual_vector), 3)} to the target vector.\n')

        # Reformat the basis and reduced basis (we won't need them as IntegerMatrix objects anymore).
        B = self.integer_matrix_to_numpy(B)
        D = self.integer_matrix_to_numpy(D.transpose())
        t = np.array(t)

        # Convert the notes of rounding directions into "step signs", establishing the sign for each operator.
        step_signs = - (np.array(rounding_direction).astype(int) * 2 - 1)
        logging.info(f'Due to the roundings, our operators will have signs:\n{step_signs}\n')

        return B, t, D, w, b_op, residual_vector, step_signs

    def define_hamiltonian(self, D, residual_vector, step_signs):
        """
        Define the Hamiltonian using the unit hypercube search as outlined in Yan et al. (2022).

        :param D: The reduced (prime) basis.
        :param residual_vector: The discrepancy between the approximate solution and the target vector (b_op - t).
        :param step_signs: The signs for each operator.
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

        # Pretty printing the Hamiltonian (because it can get pretty ugly).
        string_H = str(H)
        string_H = string_H.replace('+', '\n+')
        string_H = string_H.replace('-', '\n-')
        string_H = string_H.replace('*', ' * ')
        logging.info(f'H = \n{string_H}\n')

        return H

    def generate_gamma_layer(self, H, i):
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

    def generate_beta_layer(self, qubits, i):
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

    def generate_qaoa_circuit(self, H, p=1):
        """
        Generate a QAOA circuit for the given Hamiltonian with a given depth.

        :param H: The Hamiltonian.
        :param p: Depth of the circuit (should be kept relatively small -- say 1 to 5).
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
                    self.generate_gamma_layer(H, i),

                    # Beta layer.
                    self.generate_beta_layer(qubits, i)
                )
                for i in range(p)
            ]
        )