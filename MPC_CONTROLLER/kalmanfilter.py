import numpy as np


def kalman_filter(observations, initial_state=0.0, initial_covariance=1.0, process_noise=1e-5,
                  measurement_noise=0.1 ** 2):
    n_iter = len(observations)
    sz = (n_iter,)  # size of array

    # Allocate space for arrays
    xhat = np.zeros(sz)  # a posteri estimate of x
    P = np.zeros(sz)  # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)  # a priori error estimate
    K = np.zeros(sz)  # gain or blending factor

    R = measurement_noise  # estimate of measurement variance
    Q = process_noise  # process variance

    # Initial guesses
    xhat[0] = initial_state
    P[0] = initial_covariance

    for k in range(1, n_iter):
        # Time update
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q

        # Measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (observations[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat, P

# Example usage
