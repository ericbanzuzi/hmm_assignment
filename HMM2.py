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


def viterbi_matrix(A, delta):
    result = []
    for i in range(len(A)):
        col = get_column(i, A)
        product = dot_product(col, reshape_vector(delta))
        result.append([x[0] for x in product])
    return result


def viterbi(A, B, pi, observations):
    N = len(A)
    K = len(observations)

    deltas = []
    deltas_idx = []

    delta = dot_product(pi, get_column(observations[0], B))
    deltas.append(delta)

    for k in range(1, K):
        row_delta = []
        row_idx = []
        for i in range(N):
            temp_product = viterbi_matrix(A, deltas[-1])
            b_temp = get_column(observations[k], B)
            row_delta.append([max(temp_product[i])*b_temp[0][i]])
            row_idx.append(find_max_id(temp_product[i]))

        deltas.append(row_delta)
        deltas_idx.append(row_idx)

    # Backtracking
    seq = []
    last_delta = reshape_vector(deltas[-1])[0]
    seq.append(find_max_id(last_delta))

    for j in range(K-2, -1, -1):
        likely_best_state = seq[0]
        state = deltas_idx[j][likely_best_state]
        seq.insert(0, state)
    return seq


if __name__ == '__main__':
    lines = []
    for line in fileinput.input():
        lines.append(line)

    A = create_matrix(lines[0])
    B = create_matrix(lines[1])
    pi = create_matrix(lines[2])
    observations = create_obs_seq(lines[3])

    sequence = viterbi(A, B, pi, observations)
    print(' '.join(map(str, sequence)))
