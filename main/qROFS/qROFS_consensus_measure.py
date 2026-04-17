import math
from typing import List, Dict, Any
from main.qROFS.qROFS import qROFN
from main.qROFS.qROFS_operator import q_ROFWA,normalized_euclidean_distance


# ==========================================
# Consensus Measurement Framework
# ==========================================
def calculate_collective_opinion_matrix(
        preferences: List[List[List[qROFN]]],
        dm_weights: List[float]
) -> List[List[qROFN]]:
    """
    Calculates the collective opinion matrix.

    :param preferences: 3D list (e x m x n) -> DM k -> Alternative i -> Attribute j
    :param dm_weights: 1D list of expert influence weights (length e)
    :return: 2D collective preference matrix (m x n).
    """
    e = len(preferences)
    if e == 0: raise ValueError("No experts provided.")
    m = len(preferences[0])
    if m == 0: raise ValueError("No alternatives provided.")
    n = len(preferences[0][0])
    if n == 0: raise ValueError("No attributes provided.")

    if not math.isclose(sum(dm_weights), 1.0, abs_tol=1e-6):
        raise ValueError(f"Expert weights must sum to 1. Got: {sum(dm_weights)}")

    collective_prefs = [[None for _ in range(n)] for _ in range(m)]

    for i in range(m):
        for j in range(n):
            expert_evals = [preferences[k][i][j] for k in range(e)]
            # Aggregate them using equal weights to form the group's collective opinion
            collective_prefs[i][j] = q_ROFWA(expert_evals)

    return collective_prefs


def calculate_consensus_degree(
        preferences: List[List[List[qROFN]]],
        collective_prefs: List[List[qROFN]],
        dm_weights: List[float]
) -> Dict[str, Any]:
    """
    Calculates the expert-level and group-level consensus degrees.

    :param preferences: 3D list (e x m x n) -> DM k -> Alternative i -> Attribute j
    :param collective_prefs: 2D collective preference matrix (m x n)
    :param dm_weights: 1D list of expert influence weights (length e)
    :return: Dictionary containing consensus levels.
    """
    e = len(preferences)
    if e == 0: raise ValueError("No experts provided.")
    
    if not math.isclose(sum(dm_weights), 1.0, abs_tol=1e-6):
        raise ValueError(f"Expert weights must sum to 1. Got: {sum(dm_weights)}")

    # Calculate Expert Level Consensus (CD_u = 1 - D(D_u, D_C))
    expert_cd = [0.0 for _ in range(e)]
    for k in range(e):
        # D_u is preferences[k], D_C is collective_prefs
        dm_matrix = preferences[k]
        dist = normalized_euclidean_distance(dm_matrix, collective_prefs)
        expert_cd[k] = 1.0 - dist

    # Calculate Group Level Consensus (CD_C = Sum(w_u * CD_u))
    group_cd = sum(dm_weights[k] * expert_cd[k] for k in range(e))

    return {
        "expert_level": expert_cd,
        "group_level": group_cd
    }


# ==========================================
# 4. Mock Dataset & Execution Test
# ==========================================
if __name__ == "__main__":
    # Settings
    q_val = 3.0

    # Let's assume e=2 (Experts), m=2 (Alternatives), n=2 (Attributes)
    experts_weights = [0.6, 0.4]  # omega_k
    attribute_weights = [0.7, 0.3]  # w_j

    # Building a 3D Mock Preference Matrix: preferences[Expert][Alternative][Attribute]
    prefs = [
        # Expert 1 (k=0)
        [
            [qROFN(0.8, 0.2, q_val), qROFN(0.7, 0.4, q_val)],  # Alternative 1 (i=0)
            [qROFN(0.6, 0.5, q_val), qROFN(0.9, 0.3, q_val)]  # Alternative 2 (i=1)
        ],
        # Expert 2 (k=1)
        [
            [qROFN(0.7, 0.3, q_val), qROFN(0.6, 0.5, q_val)],  # Alternative 1 (i=0)
            [qROFN(0.5, 0.6, q_val), qROFN(0.8, 0.4, q_val)]  # Alternative 2 (i=1)
        ]
    ]

    print("--- Executing Refactored Consensus Measurement Framework ---")
    
    # 1. Calculate collective opinion matrix
    collective_matrix = calculate_collective_opinion_matrix(prefs, experts_weights)
    
    # Output collective matrix
    print("\n1. Collective Preferences Matrix (D_C):")
    for i, row in enumerate(collective_matrix):
        print(f"  Alternative {i + 1}: {row}")

    # 2. Calculate consensus degrees
    consensus_results = calculate_consensus_degree(prefs, collective_matrix, experts_weights)

    # Output consensus degrees
    print("\n2. Expert Level Consensus (CD_u):")
    for k, cd in enumerate(consensus_results["expert_level"]):
        print(f"  Expert {k + 1} Consensus = {cd:.4f}")

    print("\n3. Group Level Consensus (CD_C):")
    print(f"  Overall Group Consensus = {consensus_results['group_level']:.4f}")