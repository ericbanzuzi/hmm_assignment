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
            print(a[row][col])
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


def viterbi_matrix_revised(A, delta):
    result = []
    temp_delta = reshape_vector(delta)
    for i in range(len(A)):
        row = []
        col = get_column(i, A)
        for j in range(len(col[0])):
            row.append(col[0][j]+temp_delta[0][j])
        result.append(row)
    return result


def viterbi_matrix_add_obs(result_matrix, obs):
    result = []
    for i in range(len(result_matrix)):
        row = result_matrix[i]
        # print('ROW', row)
        row_result = []
        for j in range(len(row)):
            row_result.append(row[j]+obs[0][j])
        result.append(row_result)
    return result


def viterbi_old(A, B, pi, observations):
    delta = dot_product(pi, get_column(observations[0], B))
    deltas = [delta]
    deltas_idx = [None]
    for obs in observations[1:]:

        temp = viterbi_matrix(A, delta)
        temp = matrix_dot(temp, get_column(obs, B))
        delta = []
        delta_idx = []
        for row in temp:
            delta.append([max(row)])
            delta_idx.append(find_max_id(row) if max(row) != 0 else None)
        deltas.append(delta)
        deltas_idx.append(delta_idx)
    return deltas, deltas_idx


def viterbi(A, B, pi, observations):
    delta = matrix_log(dot_product(pi, get_column(observations[0], B)))
    deltas = [delta]
    deltas_idx = [None]
    for obs in observations[1:]:
        b_log = matrix_log(get_column(obs, B))
        #print('BEFORE', A)
        a_log = matrix_log(A)
        #print('PASSED a_log', A)
        temp = viterbi_matrix_revised(a_log, delta)

        temp2 = viterbi_matrix_add_obs(temp, b_log)
        delta = []
        delta_idx = []
        for row in temp2:
            delta.append([max(row)])
            delta_idx.append(find_max_id(row) if max(row) != 0 else None)
        deltas.append(delta)
        deltas_idx.append(delta_idx)
    return deltas, deltas_idx


def backtrack_result(deltas, deltas_idx):
    result = [find_max_id(deltas[-1])]
    for i in range(len(deltas_idx)-1, 0, -1):
        result.insert(0, deltas_idx[i][find_max_id(reshape_vector(deltas[i])[0])])
    return result


def forward_algorithm(A, B, pi, observations):
    alpha = dot_product(pi, get_column(observations[0], B))
    for obs in observations[1:]:
        alpha = dot_product(matrix_mul(alpha, A), get_column(obs, B))
    return sum(alpha[0])


if __name__ == '__main__':
    lines = []
    for line in fileinput.input():
        lines.append(line)

    # A = create_matrix(lines[0])
    # B = create_matrix(lines[1])
    # pi = create_matrix(lines[2])
    # observations = create_obs_seq(lines[3])
    # deltas, deltas_idx = viterbi(A, B, pi, observations)

    a = [[0.6, 0.1, 0.1, 0.2], [0, 0.3, 0.2, 0.5], [0.8, 0.1, 0, 0.1], [0.2, 0, 0.1, 0.7]]
    b = [[0.6, 0.2, 0.1, 0.1], [0.1, 0.4, 0.1, 0.4], [0, 0, 0.7, 0.3], [0, 0, 0.1, 0.9]]
    pi = [[0.5, 0, 0, 0.5]]
    observations = [2, 0, 3, 1]
    deltas, deltas_idx = viterbi(a, b, pi, observations)
    print(deltas)
    # print(deltas_idx)

    result = backtrack_result(deltas, deltas_idx)
    print(' '.join(map(str, result)))
    print()
