import copy
from main.qROFS.qROFS_operator import q_ROFWA
from main.qROFS.qROFS_consensus_measure import calculate_consensus

def adaptive_boundary_detection(opinions, reference_opinions, weights, theta_min, epsilon, alpha, max_iter=1000,
                                min_theta_threshold=0.01):
    """
    【M2: 阶段一 - 自适应边界检测模块】
    通过底线放松机制，寻找满足共识度阈值的安全保留系数下界。

    参数:
    opinions : list
        K 个专家的原始意见矩阵列表，维度为 (K, m, n)。
    reference_opinions : list
        K 个专家的参考意见矩阵列表，维度为 (K, m, n)。
    weights : list
        专家权重列表。
    theta_min : list
        长度为 K 的列表，代表每一位专家的初始保留系数下界 (0 < theta <= 1)。
    epsilon : float
        目标共识度阈值。
    alpha : float
        衰减因子，用于降低未达标专家的 theta_min。
    max_iter : int
        最大迭代次数，防止死循环。
    min_theta_threshold : float
        保留系数的最小容忍值，防止衰减至 0 或负数导致程序卡死。

    返回:
    hat_theta_min : list
        更新后的安全保留系数下界列表。
    """
    K = len(opinions)
    m = len(opinions[0])
    n = len(opinions[0][0])

    # 初始化 hat_theta_min，深拷贝以防污染原始输入
    hat_theta_min = copy.deepcopy(theta_min)

    iteration = 0
    while iteration < max_iter:
        adjusted_opinions = []

        # a & b. 假设所有专家都取当前的极限妥协度，计算临时调整意见矩阵
        for u in range(K):
            theta_u = hat_theta_min[u]
            adj_matrix_u = []

            # 标准双重循环遍历 m*n 矩阵
            for i in range(m):
                row = []
                for j in range(n):
                    op_val = opinions[u][i][j]
                    ref_val = reference_opinions[u][i][j]

                    # 利用 q_ROFWA 实现凸组合: AD_u = theta_u * op + (1 - theta_u) * ref
                    # 权重列表为 [保留系数, 妥协系数]
                    adj_val = q_ROFWA([op_val, ref_val], [theta_u, 1.0 - theta_u])
                    row.append(adj_val)

                adj_matrix_u.append(row)
            adjusted_opinions.append(adj_matrix_u)

        # c. 调用 calculate_consensus 获取当前极限状态下的共识度结果
        consensus_result = calculate_consensus(adjusted_opinions, weights)

        # 提取群体共识度 (CD_max) 和 专家维度共识度
        CD_max = consensus_result["group_level"]
        expert_cd = consensus_result["expert_level"]

        # d. 如果理论最大群体共识度已达到阈值 epsilon，说明当前底线可行，跳出循环
        if CD_max >= epsilon:
            break

        # e. 如果未达到，触发反馈机制：对未达标的专家降低其保留系数下界
        updated = False
        for u in range(K):
            if expert_cd[u] < epsilon:
                # 衰减更新
                hat_theta_min[u] = alpha * hat_theta_min[u]

                # 防死循环机制：限制最小阈值
                if hat_theta_min[u] < min_theta_threshold:
                    hat_theta_min[u] = min_theta_threshold
                else:
                    updated = True  # 标记本轮确实发生了有效的衰减

        # 如果所有未达标的专家都已经降到了最低阈值(未发生有效衰减)，则强制跳出，防止无限死循环
        if not updated:
            print(f"警告: 无法达到目标共识度 epsilon={epsilon}，已触底极小值保护阈值。")
            break

        iteration += 1

    if iteration == max_iter:
        print(f"警告: 达到最大迭代次数 {max_iter}，自适应边界检测强制终止。")

    return hat_theta_min