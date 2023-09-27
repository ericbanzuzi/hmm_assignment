import fileinput


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
        result.append(a[0][i]*b[0][i])
    return [result]


def forward_algorithm(A, B, pi, observations):
    alpha = dot_product(pi, get_column(observations[0], B))

    for obs in observations[1:]:
        alpha = dot_product(matrix_mul(alpha, A), get_column(obs, B))
    return sum(alpha[0])


if __name__ == '__main__':
    lines = []
    for line in fileinput.input():
        lines.append(line)

    A = create_matrix(lines[0])
    B = create_matrix(lines[1])
    pi = create_matrix(lines[2])
    observations = create_obs_seq(lines[3])
    solved = forward_algorithm(A, B, pi, observations)
    print(solved)
