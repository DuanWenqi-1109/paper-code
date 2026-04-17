import numpy as np
import networkx as nx


def complete_trust_matrix(
    initial_trust_matrix: np.ndarray,
    alpha_T: float = 0.4,
    alpha_RS: float = 0.3,
    alpha_DS: float = 0.3
) -> np.ndarray:
    """
    基于带信任衰减和多维路径分析聚合模型，补全初始不完整信任矩阵。
    将任意不连接（值为0或NaN）的专家对，通过最短路径搜索与信任传播聚合进行估计。

    Parameters
    ----------
    initial_trust_matrix : np.ndarray
        形状为 (K, K) 的不完整直接信任矩阵。缺失的连接应以 0.0（或 NaN）表示。
        对角线元素不参与传播计算，算法专注于补充非对角线的缺失元素。
    alpha_T : float, optional
        原始传播信任强度的权重因子，默认为 0.4。
    alpha_RS : float, optional
        路径可靠性（最薄弱环节）的权重因子，默认为 0.3。
    alpha_DS : float, optional
        结构多样性（路径独立性）的权重因子，默认为 0.3。
        （三个权重相加应为 1.0）

    Returns
    -------
    np.ndarray
        形状与输入一致的完整聚合信任矩阵。原矩阵中非零/非NaN的值将保持不变。
    """
    K = initial_trust_matrix.shape[0]
    final_matrix = np.copy(initial_trust_matrix)
    
    # 将 NaN 视作 0 处理以简化后续逻辑
    np.nan_to_num(final_matrix, copy=False, nan=0.0)

    # 1. 构建有向图以查找最短路径
    G = nx.DiGraph()
    for i in range(K):
        # 确保每个节点都在图中，即使它是孤立点
        G.add_node(i)
        for j in range(K):
            if i != j and final_matrix[i, j] > 0:
                # 距离权重定义为 1 - trust 用于寻找最高信任路径
                G.add_edge(i, j, trust=final_matrix[i, j], distance=1.0 - final_matrix[i, j])

    # 内部辅助函数：计算单条路径传播信任值 (公式 9, 10)
    def get_path_trust(path: list) -> float:
        L = len(path)
        if L < 2:
            return 0.0
        # 提取路径上的直接信任值 t_i
        t_values = np.array([G[path[idx]][path[idx + 1]]['trust'] for idx in range(L - 1)])
        
        # w_h(i) = 1 - 2(i-1) / (P# * (P# - 1)), 其中 i从1到L-1
        i_array = np.arange(1, L)
        w_values = 1.0 - (2.0 * (i_array - 1)) / (L * (L - 1))
        
        wt = w_values * t_values
        prod_wt = np.prod(wt)
        prod_2_minus_wt = np.prod(2.0 - wt)
        denominator = prod_2_minus_wt + prod_wt
        
        if denominator == 0:
            return 0.0
        return (2.0 * prod_wt) / denominator

    # 内部辅助函数：安全归一化
    def safe_normalize(arr: np.ndarray) -> np.ndarray:
        s = np.sum(arr)
        V = len(arr)
        if s == 0 or np.isnan(s):
            return np.ones(V) / V
        return arr / s

    # 2. 遍历所有未连接的边，计算间接信任
    for u in range(K):
        for v in range(K):
            if u != v and final_matrix[u, v] <= 0:
                try:
                    # 使用 Dijkstra 基于 distance (1 - trust) 寻找所有最短路径
                    paths = list(nx.all_shortest_paths(G, u, v, weight='distance'))
                except nx.NetworkXNoPath:
                    paths = []

                H = len(paths)
                
                # 如果不连通，间接信任为 0
                if H == 0:
                    final_matrix[u, v] = 0.0
                
                # 仅有一条最短路径
                elif H == 1:
                    final_matrix[u, v] = get_path_trust(paths[0])
                
                # 多条最短路径，应用多维聚合机制
                else:
                    # 获取各条路径的传播信任值 s_uv^h
                    s_hk_list = np.array([get_path_trust(p) for p in paths])
                    
                    # 维度一：信任强度归一化 T_h  (公式 13)
                    T_h = safe_normalize(s_hk_list)
                    
                    # 维度二：路径可靠性 RS_h (木桶原理) (公式 11)
                    rs_list = np.array([
                        np.min([G[p[idx]][p[idx + 1]]['trust'] for idx in range(len(p) - 1)]) 
                        for p in paths
                    ])
                    RS_norm = safe_normalize(rs_list)
                    
                    # 维度三：结构多样性 DS_h (Jaccard不相似度) (公式 12)
                    ds_list = np.zeros(H)
                    intermediate_sets = [set(p[1:-1]) for p in paths]
                    for h in range(H):
                        jaccard_sum = 0.0
                        for k in range(H):
                            if h != k:
                                intersection = len(intermediate_sets[h] & intermediate_sets[k])
                                union = len(intermediate_sets[h] | intermediate_sets[k])
                                if union == 0:
                                    jaccard_sum += 0.0
                                else:
                                    jaccard_sum += 1.0 - (intersection / union)
                        ds_list[h] = jaccard_sum / (H - 1)
                    DS_norm = safe_normalize(ds_list)
                    
                    # 综合所有维度产生统一权重 Lambda_h (公式 14)
                    Lambda_h = alpha_T * T_h + alpha_RS * RS_norm + alpha_DS * DS_norm
                    Lambda_h = safe_normalize(Lambda_h)
                    
                    # 最终间接信任估计 (公式 15)
                    final_matrix[u, v] = np.sum(Lambda_h * s_hk_list)

    return final_matrix


if __name__ == "__main__":
    # 简单的测试用例
    K = 5
    # 构建一个不完整的初始信任矩阵
    initial_trust = np.zeros((K, K))
    
    # 预设几条边
    initial_trust[0, 1] = 0.8
    initial_trust[1, 4] = 0.7
    initial_trust[0, 2] = 0.9
    initial_trust[2, 3] = 0.6
    initial_trust[3, 4] = 0.8
    
    print("--- 初始的信任矩阵 ---")
    print(initial_trust)

    # 运行函数完成矩阵
    aggregated_matrix = complete_trust_matrix(initial_trust)
    
    print("\n--- 补全并聚合后的信任矩阵 ---")
    np.set_printoptions(precision=4, suppress=True)
    print(aggregated_matrix)