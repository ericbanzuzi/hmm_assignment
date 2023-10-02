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


def create_obs_seq(input_string):
    input_list = [int(x) for x in input_string.split()[1:]]
    return input_list


def get_column(observation, B):
    result = []
    for i in range(len(B)):
        result.append(B[i][observation])
    return [result]


def alpha_pass(A, B, pi, observations):
    obs0 = observations[0]
    N, T = len(A), len(observations)
    alphas = []
    alpha0 = []
    for i in range(N):
        b_temp = get_column(obs0, B)
        alpha0.append(pi[0][i]*b_temp[0][i])

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

        alpha_t = [alpha_t]
        alphas.append(alpha_t)

    return alphas


if __name__ == '__main__':
    lines = []
    for line in fileinput.input():
        lines.append(line)

    A = create_matrix(lines[0])
    B = create_matrix(lines[1])
    pi = create_matrix(lines[2])
    observations = create_obs_seq(lines[3])
    solved = alpha_pass(A, B, pi, observations)
    print(sum(solved[-1][0]))  # print sum from last alpha
    # a = [[0.6, 0.1, 0.1, 0.2], [0, 0.3, 0.2, 0.5], [0.8, 0.1, 0, 0.1], [0.2, 0, 0.1, 0.7]]
    # b = [[0.6, 0.2, 0.1, 0.1], [0.1, 0.4, 0.1, 0.4], [0,0,0.7, 0.3], [0, 0, 0.1, 0.9]]
    # pi = [[0.5, 0, 0, 0.5]]
    # obs = [3]
    # print(get_column(obs[0], b))
    # a1 = dot_product(pi, get_column(obs[0], b))
    # print('alpha 1:', a1)
    # a2 = dot_product(matrix_mul(a1, a), get_column(0, b))
    # print('alpha 2:', a2)



    # next_state = matrix_mul(pi, A)
    # observations = matrix_mul(next_state, B)
    # result_shape = f'{len(observations)} {len(observations[0])} '
    # print(result_shape + ' '.join(map(str, observations[0])))


