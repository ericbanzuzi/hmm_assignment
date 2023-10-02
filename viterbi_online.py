import numpy as np


def viterbi(A, C, B, O):
    """Viterbi algorithm for solving the uncovering problem

    Notebook: C5/C5S3_Viterbi.ipynb

    Args:
        A (np.ndarray): State transition probability matrix of dimension I x I
        C (np.ndarray): Initial state distribution  of dimension I
        B (np.ndarray): Output probability matrix of dimension I x K
        O (np.ndarray): Observation sequence of length N

    Returns:
        S_opt (np.ndarray): Optimal state sequence of length N
        D (np.ndarray): Accumulated probability matrix
        E (np.ndarray): Backtracking matrix
    """
    I = A.shape[0]    # Number of states
    N = len(O)  # Length of observation sequence

    # Initialize D and E matrices
    D = np.zeros((I, N))
    E = np.zeros((I, N-1)).astype(np.int32)
    D[:, 0] = np.multiply(C, B[:, O[0]])

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_product = np.multiply(A[:, i], D[:, n-1])
            D[i, n] = np.max(temp_product) * B[i, O[n]]
            E[i, n-1] = np.argmax(temp_product)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D[:, -1])
    for n in range(N-2, -1, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    return S_opt, D, E

def viterbi_new(A, B, pi, observations):
    N = len(A)
    K = len(observations)

    deltas = []
    deltas_idx = []




# Define model parameters
A = np.array([[0.3, 0.2, 0.2, 0.3],
         [0.5, 0.1, 0.2, 0.2],
         [0.6, 0.1, 0.2, 0.1],
         [0, 0.1, 0.6, 0.3]])

C = np.array([0.4, 0.1, 0, 0.5])

B = np.array([[0.3, 0.2, 0.1, 0.4],
         [0.3, 0.4, 0.2, 0.1],
         [0.1, 0.1, 0, 0.8],
         [0, 0.7, 0.1, 0.2]])

a = [[0.6, 0.1, 0.1, 0.2], [0, 0.3, 0.2, 0.5], [0.8, 0.1, 0, 0.1], [0.2, 0, 0.1, 0.7]]
b = [[0.6, 0.2, 0.1, 0.1], [0.1, 0.4, 0.1, 0.4], [0,0,0.7, 0.3], [0, 0, 0.1, 0.9]]
pi = [[0.5, 0, 0, 0.5]]
O = np.array([2, 0, 3, 1, 2, 1])
# O = np.array([0, 2, 0, 2, 2, 1]).astype(np.int32)
#O = np.array([1]).astype(np.int32)
#O = np.array([1, 2, 0, 2, 2, 1]).astype(np.int32)

# Apply Viterbi algorithm
S_opt, D, E = viterbi(A, C, B, O)
#
print('Observation sequence:   O = ', O)
print('Optimal state sequence: S = ', S_opt)
np.set_printoptions(formatter={'float': "{: 7.4f}".format})
print('D =', D, sep='\n')
np.set_printoptions(formatter={'float': "{: 7.0f}".format})
print('E =', E, sep='\n')