from typing import List
from main.qROFS.qROFS import qROFN
from main.qROFS.qROFS_consensus_measure import calculate_collective_opinion_matrix

def selection_process(preferences: List[List[List[qROFN]]], dm_weights: List[float]) -> List[float]:
    """
    Ranks the alternatives through the selection process based on the collective opinion matrix.
    
    :param preferences: 3D list (K x m x n) -> DM k -> Alternative i -> Attribute j
    :param dm_weights: 1D list of expert influence weights (length K)
    :return: A list of length m containing the overall score for each alternative.
    """
    # 1. Calculate the collective opinion matrix D_C^t (size m x n)
    collective_matrix = calculate_collective_opinion_matrix(preferences, dm_weights)
    
    if not collective_matrix:
        return []
        
    m = len(collective_matrix)
    n = len(collective_matrix[0])
    
    overall_scores = []
    
    # 2. Calculate the overall score for each alternative
    for i in range(m):
        # S(d_i^{C,t}) = 1/n * sum(S(d_{ij}^{C,t}))
        score_sum = sum(collective_matrix[i][j].score() for j in range(n))
        overall_score = score_sum / n
        overall_scores.append(overall_score)
        
    return overall_scores

if __name__ == "__main__":
    # Example usage / test
    q_val = 3.0
    experts_weights = [0.6, 0.4]
    prefs = [
        # Expert 1
        [
            [qROFN(0.8, 0.2, q_val), qROFN(0.7, 0.4, q_val)],
            [qROFN(0.6, 0.5, q_val), qROFN(0.9, 0.3, q_val)]
        ],
        # Expert 2
        [
            [qROFN(0.7, 0.3, q_val), qROFN(0.6, 0.5, q_val)],
            [qROFN(0.5, 0.6, q_val), qROFN(0.8, 0.4, q_val)]
        ]
    ]
    
    scores = selection_process(prefs, experts_weights)
    print("Alternative Scores:")
    for i, s in enumerate(scores):
        print(f"Alternative {i + 1}: {s:.4f}")
