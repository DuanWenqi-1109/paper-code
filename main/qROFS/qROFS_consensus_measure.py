import math
from typing import List, Dict, Any
from main.qROFS.qROFS import qROFN
from main.qROFS.qROFS_operator import q_ROFWA,weighted_generalized_distance


# ==========================================
# Consensus Measurement Framework
# ==========================================
def calculate_consensus(
        preferences: List[List[List[qROFN]]],
        attr_weights: List[float],
        dm_weights: List[float]
) -> Dict[str, Any]:
    """
    Calculates the multi-level consensus degrees.

    :param preferences: 3D list (e x m x n) -> DM k -> Alternative i -> Attribute j
    :param attr_weights: 1D list of attribute weights (length n)
    :param dm_weights: 1D list of expert influence weights (length e)
    :return: Dictionary containing collective prefs, and consensus levels.
    """
    # Base dimensions
    e = len(preferences)  # Number of Experts
    if e == 0: raise ValueError("No experts provided.")

    m = len(preferences[0])  # Number of Alternatives
    if m == 0: raise ValueError("No alternatives provided.")

    n = len(preferences[0][0])  # Number of Attributes
    if n == 0: raise ValueError("No attributes provided.")

    # Validate weight sums
    if not math.isclose(sum(dm_weights), 1.0, abs_tol=1e-6):
        raise ValueError(f"Expert weights must sum to 1. Got: {sum(dm_weights)}")
    if not math.isclose(sum(attr_weights), 1.0, abs_tol=1e-6):
        raise ValueError(f"Attribute weights must sum to 1. Got: {sum(attr_weights)}")

    # ---------------------------------------------------------
    # Step 1: Calculate Collective Preference (d_ij^c)
    # ---------------------------------------------------------
    # Initialize a 2D list for collective preference matrix (m x n)
    collective_prefs = [[None for _ in range(n)] for _ in range(m)]

    for i in range(m):
        for j in range(n):
            # Extract evaluation from all 'e' experts for alternative 'i', attribute 'j'
            expert_evals = [preferences[k][i][j] for k in range(e)]
            # Aggregate them using expert weights to form the group's collective opinion
            collective_prefs[i][j] = q_ROFWA(expert_evals, dm_weights)

    # ---------------------------------------------------------
    # Step 2: Alternative Level Consensus (CD_i^k)
    # ---------------------------------------------------------
    # Initialize a 2D list for Alternative-Level Consensus (e x m)
    alternative_cd = [[0.0 for _ in range(m)] for _ in range(e)]

    for k in range(e):
        for i in range(m):
            # The preference vector of DM k on alternative i across attributes
            dm_vector = preferences[k][i]
            # The collective preference vector on alternative i across attributes
            col_vector = collective_prefs[i]

            # Calculate Distance and inherently determine consensus (1 - Distance)
            dist = weighted_generalized_distance(dm_vector, col_vector, attr_weights)
            alternative_cd[k][i] = 1.0 - dist

    # ---------------------------------------------------------
    # Step 3: Expert Level Consensus (CD_k)
    # ---------------------------------------------------------
    # Average the alternative-level consensus across all 'm' alternatives for each expert
    expert_cd = [sum(alternative_cd[k]) / m for k in range(e)]

    # ---------------------------------------------------------
    # Step 4: Group Level Consensus (CD_C)
    # ---------------------------------------------------------
    # Aggregate expert-level consensus using the expert influence weights (omega_k)
    group_cd = sum(dm_weights[k] * expert_cd[k] for k in range(e))

    return {
        "collective_preferences": collective_prefs,
        "alternative_level": alternative_cd,
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

    print("--- Executing Consensus Measurement Framework ---")
    results = calculate_consensus(prefs, attribute_weights, experts_weights)

    # Output formatting
    print("\n1. Collective Preferences Matrix (d_ij^c):")
    for i, row in enumerate(results["collective_preferences"]):
        print(f"  Alternative {i + 1}: {row}")

    print("\n2. Alternative Level Consensus (CD_i^k):")
    for k, dms_cds in enumerate(results["alternative_level"]):
        print(f"  Expert {k + 1}:")
        for i, cd in enumerate(dms_cds):
            print(f"    Alternative {i + 1} Consensus = {cd:.4f}")

    print("\n3. Expert Level Consensus (CD_k):")
    for k, cd in enumerate(results["expert_level"]):
        print(f"  Expert {k + 1} Consensus = {cd:.4f}")

    print("\n4. Group Level Consensus (CD_C):")
    print(f"  Overall Group Consensus = {results['group_level']:.4f}")