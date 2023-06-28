import numpy as np


def dtw(s, t, window: int) -> float:
    """
    Dynamic time wrapping
    """
    n, m = s.shape[0], t.shape[0]
    w = np.max([window, abs(n - m)])
    dtw_matrix = np.zeros((n + 1, m + 1))

    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(np.max([1, i - w]), np.min([m, i + w]) + 1):
            dtw_matrix[i, j] = 0

    for i in range(1, n + 1):
        for j in range(np.max([1, i - w]), np.min([m, i + w]) + 1):
            cost = np.linalg.norm(s[i - 1] - t[j - 1])
            # take last min from a square box
            last_min = np.min(
                [dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]]
            )
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[-1, -1]


if __name__ == "__main__":
    # Example
    s = np.array([[1, 1], [2, 2], [3, 3]])
    t = np.array([[1.1, 1.1], [2.2, 2.2], [3.3, 3.3]])
    print(dtw(s, t, 1))
