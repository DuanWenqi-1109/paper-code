import numpy as np

from main.qROFS.qROFS import qROFN
from main.qROFS.qROFS_operator import q_ROFWA


def precompute_reference_opinions(opinions, weights):
    """
    【M1: 底层数据与前置计算模块】
    预计算所有专家的参考意见矩阵 (Reference Opinions)
    公式: D_{Ru}^t = \sum_{v \neq u}^K w_v^t D_v^t

    参数:
    opinions : list
        长度为 K 的列表，代表 K 个专家的原始意见。每个元素是一个 m*n 的嵌套列表（由 qROFN 对象构成）。
    weights : list or np.ndarray
        长度为 K 的一维浮点数组/列表，代表专家权重，且 sum(weights) == 1。

    返回:
    reference_opinions : list
        长度为 K 的列表，其中 reference_opinions[u] 代表专家 u 的参考意见矩阵（m*n 的嵌套列表）。
    """
    # 将嵌套列表转换为 numpy 的对象数组，维度为 (K, m, n)
    ops_array = np.array(opinions, dtype=object)
    K, m, n = ops_array.shape
    weights_array = np.array(weights)

    reference_opinions = []

    # 遍历每个专家 u，计算其对应的参考意见矩阵
    for u in range(K):
        # 1. 构造布尔掩码，排除当前专家 u
        mask = np.arange(K) != u

        # 2. 提取除 u 以外其他专家的权重，转换为列表。
        # 【注意】按照业务逻辑：排除了专家 u，剩余的权重直接使用原始权重，不进行归一化
        w_other = weights_array[mask].tolist()

        # 3. 提取其他专家的意见张量，维度为 (K-1, m, n)
        other_ops = ops_array[mask]

        # 4. 维度重排：将 K-1 移到最后，变为 (m, n, K-1)
        # 这样一来，矩阵中的每一个 (i, j) 位置都对应一个包含 K-1 个 qROFN 对象的数组
        other_ops_transposed = np.transpose(other_ops, axes=(1, 2, 0))

        # 5. 展平为二维数组 (m*n, K-1)
        # 这一步是消除低效嵌套 for i, for j 循环的关键所在
        flat_ops = other_ops_transposed.reshape(-1, K - 1)

        # 6. 批量计算聚合结果
        # 对展平后的一维数组进行列表推导式遍历，每次取出对应 (i, j) 的 K-1 个 qROFN 对象进行聚合
        flat_ref = [q_ROFWA(row.tolist(), w_other) for row in flat_ops]

        # 7. 将展平的一维聚合结果重新还原为 (m, n) 的矩阵结构，并转回原生的嵌套列表形式
        ref_matrix_u = np.array(flat_ref, dtype=object).reshape(m, n).tolist()

        # 保存当前专家 u 的参考意见矩阵
        reference_opinions.append(ref_matrix_u)

    return reference_opinions






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

a = precompute_reference_opinions(prefs, attribute_weights)
print(a)