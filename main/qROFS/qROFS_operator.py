import math
from typing import List
from main.qROFS.qROFS import qROFN


# ==========================================
# q-ROFWA Aggregation Function
# ==========================================
def q_ROFWA(qrofn_list: List[qROFN], weights: List[float]) -> qROFN:
    """
    q-Rung Orthopair Fuzzy Weighted Aggregation (q-ROFWA).
    Aggregates a list of qROFN objects into a single qROFN.

    :param qrofn_list: List of qROFN instances.
    :param weights: List of corresponding weights.
    :return: A single aggregated qROFN object.
    """
    if not qrofn_list or not weights:
        raise ValueError("qROFN list and weights list cannot be empty.")
    if len(qrofn_list) != len(weights):
        raise ValueError("The length of qrofn_list and weights must be identical.")

    # Validate weights sum to 1 (using float tolerance)
    # if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
    #     raise ValueError(f"Weights must sum to 1. Current sum: {sum(weights)}")

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
# Weighted Generalized Distance Function
# ==========================================
def weighted_generalized_distance(A_list: List[qROFN], B_list: List[qROFN], weights: List[float]) -> float:
    """
    Calculates the weighted generalized distance between two lists of qROFNs.

    :param A_list: First list of qROFN instances.
    :param B_list: Second list of qROFN instances.
    :param weights: List of attribute weights for each pair.
    :return: The calculated distance as a float.
    """
    if not (len(A_list) == len(B_list) == len(weights)):
        raise ValueError("Lengths of A_list, B_list, and weights must all match.")

    if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
        raise ValueError(f"Weights must sum to 1. Current sum: {sum(weights)}")

    distance_sum = 0.0

    for a, b, w in zip(A_list, B_list, weights):
        # Validate that the comparing objects have the same q parameter
        if not math.isclose(a.q, b.q, abs_tol=1e-9):
            raise ValueError("Compared qROFN objects must share the same 'q' parameter.")

        q = a.q

        # Calculate absolute differences raised to q
        mu_diff = abs((a.mu ** q) - (b.mu ** q))
        nu_diff = abs((a.nu ** q) - (b.nu ** q))
        pi_diff = abs((a.hesitancy ** q) - (b.hesitancy ** q))

        # Add the weighted differences to the running sum
        distance_sum += w * (mu_diff + nu_diff + pi_diff)

    # Multiply by 1/2 as defined by the generalized distance formula
    return 0.5 * distance_sum


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
    agg_A = q_ROFWA(list_A, weights_vector)
    agg_B = q_ROFWA(list_B, weights_vector)

    print(f"Aggregated List A: {agg_A}")
    print(f"  => Score (S) = {agg_A.score():.4f}, Accuracy (A) = {agg_A.accuracy():.4f}")

    print(f"Aggregated List B: {agg_B}")
    print(f"  => Score (S) = {agg_B.score():.4f}, Accuracy (A) = {agg_B.accuracy():.4f}")

    print(f"Is Aggregated A > Aggregated B? {agg_A > agg_B}")

    # Test 2: Weighted Generalized Distance
    print("\n--- Testing Weighted Generalized Distance ---")
    dist_AB = weighted_generalized_distance(list_A, list_B, weights_vector)
    print(f"Distance D(A, B) = {dist_AB:.6f}")

    # Distance between identical lists should naturally evaluate to 0
    dist_AA = weighted_generalized_distance(list_A, list_A, weights_vector)
    print(f"Distance D(A, A) (Should be 0.0) = {dist_AA:.6f}")