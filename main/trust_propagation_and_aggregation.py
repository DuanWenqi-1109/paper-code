import numpy as np
import networkx as nx
from typing import List, Tuple, Set, Union


def build_trust_network(matrices: List[np.ndarray], threshold: float = 0.5) -> nx.DiGraph:
    """
    辅助函数：将 K 个专家的偏好/信任矩阵集合转化为全局初始直接信任网络。
    这里使用矩阵间的相似度（如基于欧氏距离的相似度）来模拟直接信任值。

    Parameters
    ----------
    matrices : List[np.ndarray]
        K 个专家的偏好矩阵列表，每个矩阵形状相同 (N, N)。
    threshold : float, optional
        信任阈值，低于此值的信任边将被截断，以模拟不连通的情况, by default 0.5

    Returns
    -------
    nx.DiGraph
        包含直接信任值 'trust' 和寻路权重 'distance' 的有向图。
    """
    K = len(matrices)
    G = nx.DiGraph()

    # 添加专家节点
    for i in range(K):
        G.add_node(i, label=f"e_{i}")

    # 计算两两专家矩阵之间的相似度作为直接信任值 t_ij
    for i in range(K):
        for j in range(K):
            if i != j:
                # 使用归一化的绝对误差来构造相似度 (仅为示例逻辑)
                diff = np.abs(matrices[i] - matrices[j])
                max_diff = np.max(diff) if np.max(diff) != 0 else 1
                similarity = 1.0 - np.mean(diff) / max_diff

                # 仅保留大于阈值的边，模拟稀疏信任网络
                if similarity >= threshold:
                    # distance 用于 Dijkstra 寻路，信任度越高，距离越短
                    G.add_edge(i, j, trust=similarity, distance=1.0 - similarity)

    return G


def safe_normalize(values: np.ndarray) -> np.ndarray:
    """
    安全归一化函数（步骤 4：处理边界情况 Zero-Division Fallback）。

    Parameters
    ----------
    values : np.ndarray
        需要被归一化的一维数组。

    Returns
    -------
    np.ndarray
        归一化后的数组。如果分母为 0，则平分权重 (1/V)。
    """
    total = np.sum(values)
    V = len(values)
    if total == 0 or np.isnan(total):
        return np.ones(V) / V
    return values / total


def calculate_path_trust(path: List[int], G: nx.DiGraph) -> float:
    """
    步骤 2：衰减信任传播计算 (针对单条路径)。

    Parameters
    ----------
    path : List[int]
        节点索引列表，表示一条从源节点到目标节点的路径。
    G : nx.DiGraph
        信任网络图。

    Returns
    -------
    float
        该路径的传播信任值 s_hk_v。
    """
    L = len(path)
    if L < 2:
        return 0.0

    # 提取路径上的直接信任值 t_i
    t_values = np.array([G[path[i]][path[i + 1]]['trust'] for i in range(L - 1)])

    # 计算衰减权重 w(i)
    # 注意：公式中 i 从 1 到 L-1。在 Python 中 index 从 0 开始，所以 i_math = i_py + 1
    w_values = np.zeros(L - 1)
    for i_py in range(L - 1):
        i_math = i_py + 1
        w_values[i_py] = 1.0 - (2.0 * (i_math - 1)) / (L * (L - 1))

    # 计算 w_v(i) * t_i
    wt_product_array = w_values * t_values

    # 计算分子与分母
    prod_wt = np.prod(wt_product_array)
    prod_2_minus_wt = np.prod(2.0 - wt_product_array)

    numerator = 2.0 * prod_wt
    denominator = prod_2_minus_wt + prod_wt

    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_jaccard_dissimilarity(path_u: List[int], path_v: List[int]) -> float:
    """
    计算两条路径中间节点集合的 Jaccard 不相似度。
    """
    # 提取中间节点集合 N_v (不包含源节点和目标节点)
    N_u = set(path_u[1:-1])
    N_v = set(path_v[1:-1])

    union_set = N_u.union(N_v)
    if len(union_set) == 0:
        return 0.0  # 如果都没有中间节点，认为不相似度为 0 (完全相同)

    intersection_set = N_u.intersection(N_v)
    return 1.0 - len(intersection_set) / len(union_set)


def calculate_indirect_trust(
        source: int,
        target: int,
        G: nx.DiGraph,
        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
) -> float:
    """
    带有衰减效应和综合聚合的信任传播算法核心函数。
    计算某两个未连接的专家节点之间的间接信任值。

    Parameters
    ----------
    source : int
        起始专家节点 e_h 的索引。
    target : int
        目标专家节点 e_k 的索引。
    G : nx.DiGraph
        全局的初始直接信任网络。
    weights : Tuple[float, float, float], optional
        三大维度聚合权重 (alpha_T, alpha_RS, alpha_DS), 默认 (0.4, 0.3, 0.3)。

    Returns
    -------
    float
        专家 e_h 和 e_k 之间的综合间接信任值 s_hk。
    """
    alpha_T, alpha_RS, alpha_DS = weights

    # 步骤 1：寻路与外层判断
    try:
        # 使用 Dijkstra 算法基于 distance (1 - trust) 寻找所有最短路径
        paths = list(nx.all_shortest_paths(G, source, target, weight='distance'))
    except nx.NetworkXNoPath:
        paths = []

    V = len(paths)

    # 判断连通性
    if V == 0:
        print(f"Warning: 节点 {source} 到节点 {target} 不连通。")
        return 0.0

    # 如果只有一条路径，直接返回该路径的传播信任值
    if V == 1:
        return calculate_path_trust(paths[0], G)

    # 步骤 2：针对多条路径，计算衰减信任传播值
    s_hk_v_list = np.array([calculate_path_trust(p, G) for p in paths])

    # 步骤 3 & 4：计算三大维度指标并安全归一化
    # 1. 信任强度 (T_v)
    T_v = safe_normalize(s_hk_v_list)

    # 2. 可靠性 (RS_v)
    RS_v_list = np.zeros(V)
    for v, p in enumerate(paths):
        # 提取该路径上所有直接信任值的最小值
        t_values = [G[p[i]][p[i + 1]]['trust'] for i in range(len(p) - 1)]
        RS_v_list[v] = np.min(t_values)
    RS_v_norm = safe_normalize(RS_v_list)

    # 3. 多样性 (DS_v)
    DS_v_list = np.zeros(V)
    for v in range(V):
        jaccard_sum = 0.0
        for u in range(V):
            if u != v:
                jaccard_sum += calculate_jaccard_dissimilarity(paths[u], paths[v])
        DS_v_list[v] = (1.0 / (V - 1)) * jaccard_sum
    DS_v_norm = safe_normalize(DS_v_list)

    # 步骤 5：最终聚合
    Lambda_v_star = alpha_T * T_v + alpha_RS * RS_v_norm + alpha_DS * DS_v_norm
    Lambda_v = safe_normalize(Lambda_v_star)

    s_hk = np.sum(Lambda_v * s_hk_v_list)

    return float(s_hk)


# ==========================================
# Example Usage & Mock Data Testing
# ==========================================
if __name__ == "__main__":
    # 模拟 6 个专家 (K=6)，每个专家提供 4x4 的偏好矩阵
    np.random.seed(42)
    K, N = 6, 4
    mock_matrices = [np.random.rand(N, N) for _ in range(K)]

    # 构建初始直接信任网络，设置阈值 0.6 使得网络稀疏，产生间接路径
    trust_network = build_trust_network(mock_matrices, threshold=0.55)

    print("--- 初始信任网络边 ---")
    for u, v, data in trust_network.edges(data=True):
        print(f"e_{u} -> e_{v} : 直接信任值 t = {data['trust']:.4f}")

    # 寻找两个没有直接连接的节点
    source_node, target_node = None, None
    for i in range(K):
        for j in range(K):
            if i != j and not trust_network.has_edge(i, j) and nx.has_path(trust_network, i, j):
                source_node, target_node = i, j
                break
        if source_node is not None:
            break

    if source_node is not None and target_node is not None:
        print(f"\n--- 信任传播计算 ---")
        print(f"目标：计算专家 e_{source_node} 到 e_{target_node} 之间的间接信任值")

        # 运行核心算法
        indirect_trust = calculate_indirect_trust(
            source=source_node,
            target=target_node,
            G=trust_network,
            weights=(0.4, 0.3, 0.3)
        )

        print(f"最终综合间接信任值 s_{source_node}{target_node} = {indirect_trust:.6f}")
    else:
        print("\n未找到合适的未连接且可达的节点对，请调整阈值或随机种子。")