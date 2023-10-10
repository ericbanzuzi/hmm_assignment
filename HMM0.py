import fileinput


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


def matrix_mul(a, b):
    """
    Returns the multiplication of two 2D matrices
    input: a (n x m array)
    input: b (k x l array)
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


if __name__ == '__main__':
    lines = []
    for line in fileinput.input():
        lines.append(line)

    A = create_matrix(lines[0])
    B = create_matrix(lines[1])
    pi = create_matrix(lines[2])

    next_state = matrix_mul(pi, A)
    observations = matrix_mul(next_state, B)
    result_shape = f'{len(observations)} {len(observations[0])} '
    print(result_shape + ' '.join(map(str, observations[0])))
