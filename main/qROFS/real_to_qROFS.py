import math
import numpy as np
from typing import List
from qROFS import qROFN

# ==========================================
# Conversion Function
# ==========================================
def real_to_qROFS(data_matrix: np.ndarray, q: float, lambda_val: float) -> List[List[qROFN]]:
    """
    Converts a 2D numpy array of real numbers into a 2D array of qROFN objects.

    :param data_matrix: 2D numpy array of real numbers.
    :param q: The rung parameter for the q-ROFS (must be >= 1).
    :param lambda_val: The lambda parameter for fuzzy mapping (must be > 0).
    :return: A 2D list containing qROFN instances.
    """
    if lambda_val <= 0:
        raise ValueError("Parameter lambda_val must be strictly > 0.")
    if q < 1.0:
        raise ValueError("Parameter q must be >= 1.")

    data = np.asarray(data_matrix, dtype=float)
    if data.ndim != 2:
        raise ValueError("Input data_matrix must be a 2D numpy array.")

    rows, cols = data.shape
    qrofs_matrix = []

    # Step 1: Column-wise Min-Max Normalization
    col_min = np.min(data, axis=0)
    col_max = np.max(data, axis=0)
    col_range = col_max - col_min

    # Pre-calculate exponential constants for mapping formulas
    e_lambda = math.exp(lambda_val)
    e_lambda_plus_1 = math.exp(lambda_val + 1)

    for i in range(rows):
        row_qrofn = []
        for j in range(cols):
            # Calculate normalized value (mu_ij)
            if col_range[j] == 0:
                # Edge case: If min == max, all values in column are identical.
                # Normalizing to 1.0 indicates full membership property (could also default to 0.0 or 0.5)
                mu_ij = 1.0
            else:
                mu_ij = (data[i, j] - col_min[j]) / col_range[j]

            # Step 2: Mapping to q-ROFS properties
            # Denominators are naturally >= 1 because e^lambda > 1, so (1 - e^lambda) < 0.
            # Adding epsilon (1e-12) strictly as defensive programming to prevent ZeroDivisionError.
            denom_mu = 1.0 - (1.0 - e_lambda) * mu_ij
            if abs(denom_mu) < 1e-12:
                denom_mu = 1e-12

            mu_B = 1.0 - (1.0 - mu_ij) / denom_mu

            denom_nu = 1.0 - (1.0 - e_lambda_plus_1) * mu_B
            if abs(denom_nu) < 1e-12:
                denom_nu = 1e-12

            nu_B = (1.0 - mu_B) / denom_nu

            # Clamp boundaries tightly to [0, 1] to filter out micro floating-point inaccuracies
            mu_B = max(0.0, min(1.0, mu_B))
            nu_B = max(0.0, min(1.0, nu_B))

            # Step 3: Object Instantiation
            # Because mathematically mu_B + nu_B <= 1 through this transformation,
            # and since q >= 1, mu_B^q + nu_B^q <= 1 naturally satisfies the q-ROFN constraint.
            obj = qROFN(mu=mu_B, nu=nu_B, q=q)
            row_qrofn.append(obj)

        qrofs_matrix.append(row_qrofn)

    return qrofs_matrix


# ==========================================
# 3. Test Block
# ==========================================
if __name__ == "__main__":
    # Define a sample 3x3 matrix of real numbers
    sample_data = np.array([
        [1.2, 3.4, 5.6],
        [2.1, 1.1, 4.4],
        [3.5, 2.8, 6.2]
    ])

    q_param = 3.0
    lambda_param = 2.5

    print("--- Input Matrix ---")
    print(sample_data)
    print(f"\nParameters: q = {q_param}, lambda = {lambda_param}\n")

    # Perform Conversion
    qrofs_result = real_to_qROFS(sample_data, q=q_param, lambda_val=lambda_param)

    print("--- Resulting qROFS Matrix (mu, nu, q) ---")
    for row in qrofs_result:
        print(row)

    # Extract the top-left element
    top_left_qrofn = qrofs_result[0][0]

    print("\n--- Properties of the Top-Left Element ---")
    print(f"Original Value: {sample_data[0, 0]}")
    # We expect mu=0.0 and nu=1.0 because 1.2 is the absolute minimum in the first column!
    print(f"Membership (mu)    : {top_left_qrofn.mu:.6f}")
    print(f"Non-membership (nu): {top_left_qrofn.nu:.6f}")
    print(f"Hesitancy (pi)     : {top_left_qrofn.hesitancy:.6f}")
    print(f"Score (S)          : {top_left_qrofn.score():.6f}")
    print(f"Accuracy (A)       : {top_left_qrofn.accuracy():.6f}")