import fileinput
import math
import sys


def create_matrix(input_string):
    """
    Creates a matrix as python list from the input format used in these assignments
    input: input_string (a string in Kattis format)
    """
    input_list = [float(x) for x in input_string.split()]
    rows, cols = int(input_list[0]), int(input_list[1])
    matrix = []
    input_list = input_list[2:]
    col_id = 0
    for _ in range(rows):
        row = input_list[col_id:col_id+cols]
        matrix.append(row)
        col_id += cols
    return matrix


def create_obs_seq(input_string):
    """
    Creates a list of sequences from an input string
    input: input_string (a string in Kattis format)
    """
    input_list = [int(x) for x in input_string.split()[1:]]
    return input_list


def get_column(observation, B):
    """
    Returns a column from a matrix
    input: observation (column id)
    input: B (observation matrix)
    """
    result = []
    for i in range(len(B)):
        result.append(B[i][observation])
    return [result]


# STAMP TUTORIAL: A Revealing Introduction to Hidden Markov Models by Mark Stamp (2004).
def alpha_pass(A, B, pi, observations):
    obs0 = observations[0]
    N, T = len(A), len(observations)
    alphas = []
    cs = []
    alpha0 = []
    for i in range(N):
        b_temp = get_column(obs0, B)
        alpha0.append(pi[0][i]*b_temp[0][i])

    c0 = 1/sum(alpha0)
    cs.append(c0)
    for i in range(N):
        alpha0[i] = c0*alpha0[i]

    alpha0 = [alpha0]
    alphas.append(alpha0)

    for t in range(1, T):
        obs_t = observations[t]
        ct = 0
        alpha_prev = alphas[t-1]
        alpha_t = []
        for i in range(N):
            total = 0
            for j in range(N):
                total += alpha_prev[0][j]*A[j][i]
            b_temp = get_column(obs_t, B)
            alpha_t.append(total*b_temp[0][i])
            ct += alpha_t[i]

        ct = 1/ct
        cs.append(ct)
        for i in range(N):
            alpha_t[i] = ct*alpha_t[i]

        alpha_t = [alpha_t]
        alphas.append(alpha_t)

    return alphas, cs


# STAMP TUTORIAL: A Revealing Introduction to Hidden Markov Models by Mark Stamp (2004).
def beta_pass(A, B, observations, cs):

    N, T = len(A), len(observations)
    beta_last = [[cs[T-1] for _ in range(N)]]
    betas = [beta_last]
    for t in range(T-2, -1, -1):
        beta_future = betas[0]
        ct = cs[t]
        beta_t = []
        obs_t = observations[t+1]
        for i in range(N):
            total = 0
            b_temp = get_column(obs_t, B)
            for j in range(N):
                total += A[i][j]*b_temp[0][j]*beta_future[0][j]
            beta_t.append(total*ct)
        betas.insert(0, [beta_t])

    return betas


# STAMP TUTORIAL: A Revealing Introduction to Hidden Markov Models by Mark Stamp (2004).
def gamma_calculations(A, B, observations, alphas, betas):
    N, T = len(A), len(observations)
    gammas = []
    di_gammas = []
    for t in range(T-1):
        obs_future = observations[t + 1]
        gamma = []
        di_gamma = []
        for i in range(N):
            total = 0
            b_temp = get_column(obs_future, B)
            row = []
            for j in range(N):
                gamma_ij = (alphas[t][0][i]*A[i][j]*b_temp[0][j]*betas[t+1][0][j])
                total += gamma_ij
                row.append(gamma_ij)
            gamma.append(total)
            di_gamma.append(row)
        gammas.append([gamma])
        di_gammas.append(di_gamma)

    gammas.append(alphas[-1])
    return gammas, di_gammas


# STAMP TUTORIAL: A Revealing Introduction to Hidden Markov Models by Mark Stamp (2004).
def baum_welch(A, B, pi, observations, num_iter=math.inf):
    N, T = len(A), len(observations)
    alphas, cs = alpha_pass(A, B, pi, observations)
    betas = beta_pass(A, B, observations, cs)
    gammas, di_gammas = gamma_calculations(A, B, observations, alphas, betas)
    A, B, pi = estimate_model(A, B, pi, observations, gammas, di_gammas)

    oldLogProb, logProb = -float('inf'), 0
    for i in range(T):
        logProb += math.log(cs[i])
    logProb = -logProb

    iter = 0
    while iter < num_iter and logProb > oldLogProb:
        oldLogProb = logProb
        alphas, cs = alpha_pass(A, B, pi, observations)
        betas = beta_pass(A, B, observations, cs)
        gammas, di_gammas = gamma_calculations(A, B, observations, alphas, betas)
        A, B, pi = estimate_model(A, B, pi, observations, gammas, di_gammas)

        logProb = 0
        for i in range(T):
            logProb += math.log(cs[i])
        logProb = -logProb
        iter += 1

    return A, B, pi, iter


# STAMP TUTORIAL: A Revealing Introduction to Hidden Markov Models by Mark Stamp (2004).
def estimate_model(A, B, pi, observations, gammas, di_gammas):
    N, T, M = len(A), len(observations), len(B[0])
    pi = gammas[0]

    # (re-)estimate A
    for i in range(N):
        denom = 0
        for t in range(T-1):
            denom += gammas[t][0][i]

        for j in range(N):
            numer = 0
            for t in range(T - 1):
                numer += di_gammas[t][i][j]

            if denom == 0:
                denom = sys.float_info.epsilon
            A[i][j] = numer / denom

    # (re-)estimate B
    for i in range(N):
        denom = 0
        for t in range(T):
            denom += gammas[t][0][i]

        for j in range(M):
            numer = 0
            for t in range(T):
                if observations[t] == j:
                    numer += gammas[t][0][i]

            if denom == 0:
                denom = sys.float_info.epsilon
            B[i][j] = numer / denom
    return A, B, pi


def get_result(M):
    """
    Formats the result to a correctly fomratted string
    """
    rows = len(M)
    cols = len(M[0])
    result = f'{rows} {cols}'
    for row in M:
        result = result + ' ' + ' '.join(map(str, row)) + ' '
    return result


import numpy as np

def perturb_and_normalize(matrix, std_dev):
    # Add small random perturbations to the matrix
    perturbed_matrix = matrix + np.random.normal(0, std_dev, matrix.shape)

    # Ensure no negative values
    perturbed_matrix = np.abs(perturbed_matrix)

    # Renormalize each row to make the matrix row stochastic
    row_sums = perturbed_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = perturbed_matrix / row_sums

    return normalized_matrix


from scipy.spatial import distance
import numpy as np

if __name__ == '__main__':
    lines = []
    for line in fileinput.input():
        lines.append(line)

    A = [[0.4, 0.6],
         [0.55, 0.45]]

    B = [[0.5, 0.2, 0.11, 0.19],
         [0.22, 0.28, 0.23, 0.27]]

    pi = [[0.4, 0.6]]

    A_prime = [[0.7, 0.05, 0.25],
               [0.1, 0.8, 0.1],
               [0.2, 0.3, 0.5]]
    B_prime = [[0.7, 0.2, 0.1, 0],
               [0.1, 0.4, 0.3, 0.2],
               [0, 0.1, 0.2, 0.7]]
    pi_prime = [[1, 0, 0]]

    # A_prime = perturb_and_normalize(np.array(A_prime), 0.01)
    # B_prime = perturb_and_normalize(np.array(B_prime), 0.01)
    # pi_prime = perturb_and_normalize(np.array(pi_prime), 0.01)
    #
    # print(A_prime)
    # print(B_prime)
    # print(pi_prime)

    # print(np.linalg.norm(np.array(A) - np.array(A_prime), 'fro'))
    # print(np.linalg.norm(np.array(B) - np.array(B_prime), 'fro'))
    # print(np.linalg.norm(np.array(pi) - np.array(pi_prime), 'fro'))

    observations = create_obs_seq(lines[0])

    A, B, pi, iter = baum_welch(A, B, pi, observations)

    result_string_A = get_result(A)
    result_string_B = get_result(B)
    print('A:', A)
    print('B:', B)
    print('Pi:', pi)
    print('ITERATIONS:', iter)
