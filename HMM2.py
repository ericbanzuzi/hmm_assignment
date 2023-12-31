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


def dot_product(a, b):
    """
    Returns a dot product of two vectors
    input: a (a vector in form 1 x n)
    input: b (a vector in form 1 x n)
    """
    result = []
    for i in range(len(a[0])):
        result.append([a[0][i]*b[0][i]])
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
    T = len(observations)

    deltas = []
    deltas_idx = []

    # create initial delta by multiplying the probability of being in a state i, by the probability of observing
    # obs0 in state i
    delta = dot_product(pi, get_column(observations[0], B))
    deltas.append(delta)

    for t in range(1, T):
        row_delta = []
        row_idx = []
        for i in range(N):
            # compute matrix containing values for a_{j,i} * delta_{t-1}
            temp_product = viterbi_matrix(A, deltas[-1])
            b_temp = get_column(observations[t], B)
            # select max_{j∈[1,..., N]} a_{j,i} * delta_{t-1} * b_i(obs_t)
            # we can reuse the max from already computed a_{j,i} * delta_{t-1} | (b_i(obs_t) is a constant)
            row_delta.append([max(temp_product[i])*b_temp[0][i]])
            row_idx.append(find_max_id(temp_product[i]))  # store index of the most likely state for delta idx

        deltas.append(row_delta)
        deltas_idx.append(row_idx)

    # Backtracking
    seq = []
    last_delta = reshape_vector(deltas[-1])[0]
    seq.append(find_max_id(last_delta))

    # start from T-2 because we already have the state for time T (from initialization), and iterate backwards
    for j in range(T-2, -1, -1): 
        likely_best_state = seq[0]
        state = deltas_idx[j][likely_best_state] # look up the state at time j that led to the likely_best_state at time j+1 
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
