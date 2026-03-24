import numpy as np
from pymoo.core.problem import Problem
from main.qROFS.qROFS_operator import q_ROFWA,weighted_generalized_distance
from main.qROFS.qROFS_consensus_measure import calculate_consensus


class TOCM_Problem(Problem):
    """
    【M3: 目标函数与约束封装模块】
    基于 pymoo 框架的三目标共识优化模型 (TOCM)
    目标 1: 最小化调整成本 (f1)
    目标 2: 最大化群体共识度 (转化为最小化 -f2)
    目标 3: 最大化调整公平度 (转化为最小化 -f3)
    约束: 群体共识度 >= epsilon (转化为 epsilon - CD_C <= 0)
    """

    def __init__(self, opinions, reference_opinions, weights, costs, hat_theta_min, epsilon, **kwargs):
        self.opinions = opinions
        self.reference_opinions = reference_opinions
        self.weights = weights
        self.costs = costs
        self.epsilon = epsilon

        self.K = len(opinions)
        self.m = len(opinions[0])
        self.n = len(opinions[0][0])

        # 初始化 pymoo Problem 核心参数
        super().__init__(
            n_var=self.K,  # 决策变量数量: K 个专家的保留系数
            n_obj=3,  # 目标数量: 成本, 共识度, 公平度
            n_ieq_constr=1,  # 不等式约束数量: 1 个共识度底线约束
            xl=np.array(hat_theta_min),  # 变量下界: 阶段一输出的安全下界
            xu=np.ones(self.K),  # 变量上界: 1.0 (完全保留原始意见)
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        pop_size = x.shape[0]

        # 初始化目标矩阵 F 和约束矩阵 G
        F = np.zeros((pop_size, self.n_obj))
        G = np.zeros((pop_size, self.n_ieq_constr))

        # 遍历种群中的每一个候选解
        for p in range(pop_size):
            theta_vec = x[p]
            AD_list = []

            # 1. 生成临时调整意见矩阵 AD_list
            for u in range(self.K):
                theta_u = theta_vec[u]
                adj_matrix_u = []
                for i in range(self.m):
                    row = []
                    for j in range(self.n):
                        op_val = self.opinions[u][i][j]
                        ref_val = self.reference_opinions[u][i][j]
                        # 凸组合: AD_u = theta_u * D_u + (1 - theta_u) * D_Ru
                        adj_val = q_ROFWA([op_val, ref_val], [theta_u, 1.0 - theta_u])
                        row.append(adj_val)
                    adj_matrix_u.append(row)
                AD_list.append(adj_matrix_u)

            # 2. 计算个人成本 r_u
            r = np.zeros(self.K)
            for u in range(self.K):
                # 调用底层距离函数 D(D_u, AD_u)
                dist = weighted_generalized_distance(self.opinions[u], AD_list[u], self.weights)
                r[u] = self.costs[u] * dist

            # 3. 计算目标 1: 最小化总成本 f1
            f1 = np.sum(r)

            # 4. 计算目标 2: 最大化共识度 f2 (转换为最小化 -CD_C)
            consensus_result = calculate_consensus(AD_list, self.weights)
            CD_C = consensus_result["group_level"]
            f2 = -CD_C

            # 5. 计算目标 3: 最大化公平度 f3 (转换为最小化 -f3)
            r_mean = np.mean(r)
            if r_mean < 1e-9:
                # 防零保护: 若平均成本极小，视为完全公平
                f3_val = 1.0
            else:
                # 利用 NumPy 广播机制高效计算所有专家成本的两两绝对差值之和
                sum_diff = np.sum(np.abs(r[:, None] - r[None, :]))
                f3_val = 1.0 - (sum_diff / (2 * (self.K ** 2) * r_mean))
            f3 = -f3_val

            # 6. 计算约束 g: epsilon - CD_C <= 0
            g = self.epsilon - CD_C

            # 整合当前个体的评估结果
            F[p, :] = [f1, f2, f3]
            G[p, :] = [g]

        # 统一输出至 pymoo 的 out 字典
        out["F"] = F
        out["G"] = G