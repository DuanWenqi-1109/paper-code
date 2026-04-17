import numpy as np
from typing import List, Dict, Any, Optional
from main.qROFS.qROFS import qROFN
from main.qROFS.qROFS_operator import weighted_generalized_distance
from main.qROFS.qROFS_consensus_measure import calculate_consensus


# -----------------------------------------------------------------

class BehaviorManagementMechanism:
    """
    非合作与操纵行为的动态识别与管理模块 (基于 q-ROFS)
    """

    def __init__(self):
        # 模块主要为单次执行的逻辑函数，如果未来需要保存运行日志，可在此扩展
        pass

    def run_mechanism(
            self,
            D_current: List[List[List[qROFN]]],
            D_prev: List[List[List[qROFN]]],
            AD_prev: List[List[List[qROFN]]],
            trust_matrix: np.ndarray,
            credit_array: np.ndarray,
            cost_array: np.ndarray,
            params: Dict[str, float],
            attr_weights: Optional[List[float]] = None,
            dm_weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        在第 t+1 轮共识度未达标时触发的核心判定及管理机制。

        Args:
            D_current: 当前轮 (t+1) 的专家意见矩阵 K * m * n
            D_prev: 上一轮 (t) 的专家意见矩阵 K * m * n
            AD_prev: 上一轮 (t) 的系统调整建议矩阵 K * m * n
            trust_matrix: 上一轮的信任网络矩阵 (K, K)
            credit_array: 上一轮的专家历史信用值 C_u^t
            cost_array: 上一轮的调整成本系数 Psi_u^t
            params: 包含各种阈值和惩罚系数的超参数字典
            attr_weights: 属性权重 (传给 calculate_consensus)
            dm_weights: 专家初始权重 (传给 calculate_consensus)

        Returns:
            Dict: 包含 labels, new_trust_matrix, new_credit_array, new_weights, new_cost_array
        """
        K = len(D_current)  # 专家数量
        m = len(D_current[0])  # 方案数量
        n = len(D_current[0][0])  # 属性数量

        # --- 提取超参数 ---
        rho = params['rho']
        beta = params['beta']
        epsilon = params['epsilon']
        tau = params['tau']
        delta = params['delta']
        xi = params['xi']
        gamma_s = params['gamma_s']
        gamma_h = params['gamma_h']
        gamma_m = params.get('gamma_m', gamma_h * 1.5)  # Label 3的加速衰减系数，如果未传则默认比gamma_h大
        eta_s = params['eta_s']
        eta_h = params['eta_h']
        nu = params['nu']
        kappa = params['kappa']
        psi_min = params['Psi_min']

        # --- Step 1: 共识度计算与初始化 ---
        # 调用外部的共识度计算函数获取 t+1 轮每个专家的个体共识度
        consensus_result = calculate_consensus(D_current, attr_weights, dm_weights)
        CD = consensus_result['expert_consensus']  # 长度为 K 的共识度列表
        labels = np.zeros(K, dtype=int)  # 初始化所有专家的行为标签为0（正常合作）

        # 预计算入度中心性 IC_u^{t+1} (基于上一轮信任网络)
        IC = np.zeros(K)
        for u in range(K):
            # 不含对角线的列均值
            col_sum = np.sum(trust_matrix[:, u]) - trust_matrix[u, u]
            IC[u] = col_sum / (K - 1) if K > 1 else 0.0
        AIC = np.mean(IC)  # 群体平均入度

        PFD_array = np.zeros(K)  # 缓存每个专家的建议遵循度 (PFD)，后续信用更新需要使用

        # --- Step 2: 行为识别及标签判定 ---
        for u in range(K):
            # 2.1 计算建议遵循度 RF_u^{t+1} (基于距离函数等效实现余弦定理)
            # 展平矩阵为一维列表，方便送入 distance 函数计算
            A_u_flat = [item for row in D_current[u] for item in row]
            B_u_flat = [item for row in D_prev[u] for item in row]
            C_u_flat = [item for row in AD_prev[u] for item in row]

            # 默认传入等权重列表 (所有元素权重相同并和为1)
            flat_weights = [1.0 / (m * n)] * (m * n)

            dist_AB = weighted_generalized_distance(A_u_flat, B_u_flat, flat_weights)
            dist_CB = weighted_generalized_distance(C_u_flat, B_u_flat, flat_weights)
            dist_AC = weighted_generalized_distance(A_u_flat, C_u_flat, flat_weights)

            # Measurement in metric space mapping to Eq. 29
            numerator = dist_AB ** 2 + dist_CB ** 2 - dist_AC ** 2
            denominator = 2.0 * (dist_CB ** 2) + xi
            PFD_u = max(0.0, min(1.0, numerator / denominator))
            PFD_array[u] = PFD_u

            # 2.2 计算犹豫度 HD_u^{t+1}
            hd_sum = 0.0
            for i in range(m):
                for j in range(n):
                    val = D_current[u][i][j]
                    # 加入 max(0.0, ...) 防止浮点数精度极小负误差导致开方报错
                    inner_val = max(0.0, 1.0 - (val.mu ** val.q) - (val.nu ** val.q))
                    hd_sum += inner_val ** (1.0 / val.q)
            HD_u = hd_sum / (m * n)

            # 2.3 计算方案最高支持度 Phi_u^{t+1}
            phi_max = -float('inf')
            for i in range(m):
                row_score_avg = sum(D_current[u][i][j].score() for j in range(n)) / n
                if row_score_avg > phi_max:
                    phi_max = row_score_avg
            Phi_u = phi_max

            # 2.4 多条件标签判定树
            if CD[u] < epsilon:
                if (HD_u <= tau) and (IC[u] >= AIC) and (Phi_u > delta):
                    labels[u] = 3  # 判定为操纵者 (Label 3)
                else:
                    if PFD_u <= rho:
                        if credit_array[u] >= beta:
                            labels[u] = 1  # 判定为短期非合作
                        else:
                            labels[u] = 2  # 判定为顽固非合作
                    # 注意：如果 CD_u < epsilon 但是 PFD_u > rho 且不满足操纵者条件，标签保持0不变

        # --- Step 3: Trust Redistribution (信任流重分配) ---
        new_trust_matrix = trust_matrix.copy()

        for v in range(K):
            if labels[v] in [1, 2, 3]:  # 仅处理受到惩罚的节点
                if labels[v] == 1:
                    gamma = gamma_s
                elif labels[v] == 2:
                    gamma = gamma_h
                else:
                    gamma = gamma_m

                for i in range(K):
                    if i == v:
                        continue  # 忽略自我信任

                    s_iv_old = trust_matrix[i, v]
                    # 惩罚后的信任值衰减
                    if labels[v] == 3:
                        s_iv_new = s_iv_old * np.exp(-gamma * (1.0 - credit_array[v]) * (1.0 + IC[v]))
                    else:
                        s_iv_new = s_iv_old * np.exp(-gamma * (1.0 - credit_array[v]))
                    
                    delta_s = s_iv_old - s_iv_new  # 专家i对惩罚对象v流失的信任量

                    # 更新对v的惩罚信任值
                    new_trust_matrix[i, v] = s_iv_new

                    # 重新分配 delta_s 给除 v 和自身 i 以外的其他所有节点（依据原有比例）
                    # 1. 统计 i 的其他原有出度总和
                    others_sum = 0.0
                    for j in range(K):
                        if j != v and j != i:
                            others_sum += trust_matrix[i, j]

                    # 2. 按比例累加分配
                    if others_sum > 0:  # 避免出度全为 0 的孤岛出现除零异常
                        for j in range(K):
                            if j != v and j != i:
                                proportion = trust_matrix[i, j] / others_sum
                                new_trust_matrix[i, j] += delta_s * proportion

        # --- Step 4: 信用、权重及成本更新 ---
        new_credit_array = np.zeros(K)
        for u in range(K):
            # 信用更新机制
            if labels[u] == 1:
                new_credit_array[u] = credit_array[u] - eta_s * (1.0 - PFD_array[u])
            elif labels[u] == 2:
                new_credit_array[u] = credit_array[u] - eta_h * (1.0 - PFD_array[u])
            elif labels[u] == 3:
                new_credit_array[u] = credit_array[u] - eta_h
            else:
                new_credit_array[u] = credit_array[u] + (CD[u] - epsilon)

        # 强制性安全约束: 确保信用值落在 [0.0, 1.0] 区间 (与论文 Eq. 36 保持恒正约束一致)
        new_credit_array = np.clip(new_credit_array, 0.0, 1.0)

        # 权重更新机制 w_u^{t+1}
        new_IC = np.zeros(K)
        for u in range(K):
            col_sum = np.sum(new_trust_matrix[:, u]) - new_trust_matrix[u, u]
            new_IC[u] = col_sum / (K - 1) if K > 1 else 0.0

        weight_numerators = np.zeros(K)
        for u in range(K):
            weight_numerators[u] = new_IC[u] * (nu + new_credit_array[u])

        sum_numerators = np.sum(weight_numerators)
        if sum_numerators > 0:
            new_weights = weight_numerators / sum_numerators
        else:
            # 极端情况回退：如果所有分子均为 0，采用平均权重
            new_weights = np.ones(K) / K

        # 成本更新机制 Psi_u^{t+1}
        new_cost_array = np.zeros(K)
        for u in range(K):
            updated_cost = cost_array[u] * (credit_array[u] ** kappa)
            new_cost_array[u] = max(psi_min, updated_cost)

        # 最终返回打包状态字典
        return {
            'labels': labels,
            'new_trust_matrix': new_trust_matrix,
            'new_credit_array': new_credit_array,
            'new_weights': new_weights,
            'new_cost_array': new_cost_array
        }