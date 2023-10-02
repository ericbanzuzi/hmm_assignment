import fileinput
import math
import sys


def create_matrix(input_string):
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


def matrix_mul(a, b):
    """
    Returns the multiplication of two 2D matrices
    input: X (n x m array)
    input: Y (k x l array)
    output: result (n x l multiplied array)
    """
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            temp = 0
            for k in range(len(a[0])):
                temp += a[i][k]*b[k][j]
            row.append(temp)
        result.append(row)
    return result

def create_obs_seq(input_string):
    input_list = [int(x) for x in input_string.split()[1:]]
    return input_list


def get_column(observation, B):
    result = []
    for i in range(len(B)):
        result.append(B[i][observation])
    return [result]


def dot_product(a, b):
    result = []
    for i in range(len(a[0])):
        result.append([a[0][i]*b[0][i]])
    return result


def matrix_dot(a, b):
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(a[0])):
            row.append(a[i][j]*b[0][i])
        result.append(row)
    return result


def matrix_log(a):
    result = []
    for row in range(len(a)):
        row_res = []
        for col in range(len(a[row])):
            row_res.append(math.log(a[row][col] if a[row][col] != 0 else sys.float_info.epsilon))
        result.append(row_res)
    return result


def matrix_log_a(a):
    result = []
    for row in range(len(a)):
        row_res = []
        for col in range(len(a[row])):
            row_res.append(math.log(a[row][col] if a[row][col] != 0 else sys.float_info.epsilon))
        result.append(row_res)
    return result


def reshape_vector(v):
    return [[x[0] for x in v]]


# https://sparkbyexamples.com/python/get-index-of-max-of-list-in-python/
def find_max_id(a):
    return a.index(max(a))


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


def baum_welch(A, B, pi, observations, num_iter=1000):
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

    return A, B, pi


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
    rows = len(M)
    cols = len(M[0])
    result = f'{rows} {cols}'
    for row in M:
        result = result + ' ' + ' '.join(map(str, row)) + ' '
    return result


if __name__ == '__main__':
    lines = []
    for line in fileinput.input():
        lines.append(line)

    A = create_matrix(lines[0])
    B = create_matrix(lines[1])
    pi = create_matrix(lines[2])
    observations = create_obs_seq(lines[3])

    A, B, pi = baum_welch(A, B, pi, observations)

    result_string_A = get_result(A)
    result_string_B = get_result(B)
    print(result_string_A)
    print(result_string_B)
