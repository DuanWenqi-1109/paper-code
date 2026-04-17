import math
from typing import List
from main.qROFS.qROFS import qROFN


# ==========================================
# q-ROFWA Aggregation Function
# ==========================================
def q_ROFWA(qrofn_list: List[qROFN]) -> qROFN:
    """
    q-Rung Orthopair Fuzzy Aggregation (with equal weights).
    Aggregates a list of qROFN objects into a single qROFN.

    :param qrofn_list: List of qROFN instances.
    :return: A single aggregated qROFN object.
    """
    if not qrofn_list:
        raise ValueError("qROFN list cannot be empty.")
        
    n = len(qrofn_list)
    weights = [1.0 / n] * n

    # Extract q and validate homogeneity
    q = qrofn_list[0].q
    for obj in qrofn_list:
        if not math.isclose(obj.q, q, abs_tol=1e-9):
            raise ValueError("All qROFN elements must share the same 'q' parameter.")

    # Variables for the product terms
    prod_mu_term = 1.0
    prod_nu_term = 1.0

    # Calculate the product terms using the q-ROFWA formulas
    for obj, w in zip(qrofn_list, weights):
        # term: (1 - mu_k^q)^w_k
        prod_mu_term *= (1.0 - (obj.mu ** q)) ** w

        # term: (nu_k)^w_k
        prod_nu_term *= (obj.nu) ** w

    # Final calculations for the aggregated membership and non-membership
    # max() and min() clamps are added defensively against float arithmetic precision limits
    mu_final_q = 1.0 - prod_mu_term
    mu_final_q = max(0.0, min(1.0, mu_final_q))

    mu_final = mu_final_q ** (1.0 / q)
    nu_final = max(0.0, min(1.0, prod_nu_term))

    # The closure property guarantees that mu_final^q + nu_final^q <= 1
    return qROFN(mu=mu_final, nu=nu_final, q=q)


# ==========================================
# Normalized Euclidean Distance Function
# ==========================================
def normalized_euclidean_distance(A_list: List[List[qROFN]], B_list: List[List[qROFN]]) -> float:
    """
    Calculates the normalized Euclidean distance between two matrices of qROFNs.

    :param A_list: First matrix of qROFN instances (m x n).
    :param B_list: Second matrix of qROFN instances (m x n).
    :return: The calculated distance as a float.
    """
    m = len(A_list)
    if m == 0:
        raise ValueError("Matrix cannot be empty.")
    
    n = len(A_list[0])
    if n == 0:
        raise ValueError("Matrix rows cannot be empty.")

    if len(B_list) != m or len(B_list[0]) != n:
        raise ValueError("Dimensions of A_list and B_list must match.")

    distance_sum = 0.0

    for i in range(m):
        for j in range(n):
            a = A_list[i][j]
            b = B_list[i][j]
            
            # Validate that the comparing objects have the same q parameter
            if not math.isclose(a.q, b.q, abs_tol=1e-9):
                raise ValueError("Compared qROFN objects must share the same 'q' parameter.")

            q = a.q

            # Calculate differences raised to q
            mu_diff = (a.mu ** q) - (b.mu ** q)
            nu_diff = (a.nu ** q) - (b.nu ** q)
            pi_diff = (a.hesitancy ** q) - (b.hesitancy ** q)

            # Add the squared differences to the running sum
            distance_sum += (mu_diff**2 + nu_diff**2 + pi_diff**2)

    # Calculate the normalized Euclidean distance as defined by the updated formula
    return math.sqrt(distance_sum / (2.0 * m * n))


# ==========================================
# 4. Test Block
# ==========================================
if __name__ == "__main__":
    q_val = 3.0
    weights_vector = [0.4, 0.3, 0.3]

    print(f"--- Initialization ---")
    print(f"Using q = {q_val}")
    print(f"Weights vector = {weights_vector}\n")

    # Constructing list A
    A1 = qROFN(mu=0.8, nu=0.4, q=q_val)
    A2 = qROFN(mu=0.7, nu=0.5, q=q_val)
    A3 = qROFN(mu=0.6, nu=0.6, q=q_val)
    list_A = [A1, A2, A3]

    # Constructing list B
    B1 = qROFN(mu=0.9, nu=0.3, q=q_val)
    B2 = qROFN(mu=0.6, nu=0.4, q=q_val)
    B3 = qROFN(mu=0.5, nu=0.7, q=q_val)
    list_B = [B1, B2, B3]

    print("List A elements:")
    for a in list_A: print(f"  {a}")

    print("\nList B elements:")
    for b in list_B: print(f"  {b}")

    # Test 1: Aggregation (q-ROFWA)
    print("\n--- Testing q-ROFWA Aggregation ---")
    agg_A = q_ROFWA(list_A)
    agg_B = q_ROFWA(list_B)

    print(f"Aggregated List A: {agg_A}")
    print(f"  => Score (S) = {agg_A.score():.4f}, Accuracy (A) = {agg_A.accuracy():.4f}")

    print(f"Aggregated List B: {agg_B}")
    print(f"  => Score (S) = {agg_B.score():.4f}, Accuracy (A) = {agg_B.accuracy():.4f}")

    print(f"Is Aggregated A > Aggregated B? {agg_A > agg_B}")

    # Test 2: Normalized Euclidean Distance
    print("\n--- Testing Normalized Euclidean Distance ---")
    matrix_A = [list_A] # Wrapping in list to create 1x3 matrix
    matrix_B = [list_B]
    dist_AB = normalized_euclidean_distance(matrix_A, matrix_B)
    print(f"Distance D(A, B) = {dist_AB:.6f}")

    # Distance between identical lists should naturally evaluate to 0
    dist_AA = normalized_euclidean_distance(matrix_A, matrix_A)
    print(f"Distance D(A, A) (Should be 0.0) = {dist_AA:.6f}")