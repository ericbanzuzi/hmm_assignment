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

    # create initial alpha by multiplying the probability of being in a state i, by the probability of observing
    # obs0 in state i
    alpha0 = []
    for i in range(N):
        b_temp = get_column(obs0, B)
        # b_temp and pi are 1xn vectors, so to access their values we need to take [0][i] 
        alpha0.append(pi[0][i]*b_temp[0][i]) # because 

    alpha0 = [alpha0]
    alphas.append(alpha0)

    for t in range(1, T):
        obs_t = observations[t]
        alpha_prev = alphas[t-1]
        alpha_t = []
        for i in range(N):
            total = 0
            # marginalization over the states by reusing alpha
            for j in range(N):
                total += alpha_prev[0][j]*A[j][i]
            b_temp = get_column(obs_t, B)
            # store probability observing obs_t in state i
            alpha_t.append(total*b_temp[0][i])

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
    print(sum(solved[-1][0]))  # print sum from last alpha as solution
