import random

from main.optimization_model.basic_information import precompute_reference_opinions
from main.optimization_model.phase_I import adaptive_boundary_detection
from main.optimization_model.phase_II_solve_and_analyze import solve_and_analyze_TOCM
from main.qROFS.qROFS import qROFN

# 假设你已经定义并导入了 qROFN 类以及 M1~M4 的函数
# from your_module import qROFN, precompute_reference_opinions, ...

# ==========================================
# 第一部分：生成 Mock 测试数据
# ==========================================
print(">>> 正在生成 Mock 测试数据...")
random.seed(42)  # 固定随机种子，保证每次运行生成的数据一致，方便调试

q_val = 3.0
K, m, n = 5, 6, 4  # 5个专家, 6个方案(Alternatives), 4个属性(Attributes)

experts_weights = [0.25, 0.20, 0.15, 0.25, 0.15]  # sum = 1.0
attribute_weights = [0.25, 0.25, 0.25, 0.25]  # 权重相同, sum = 1.0

# 随机生成 5 个专家的 6x4 意见矩阵
opinions = []
for k in range(K):
    expert_matrix = []
    for i in range(m):
        row = []
        for j in range(n):
            # 随机生成隶属度 mu，范围 0.3~0.8
            mu = round(random.uniform(0.3, 0.8), 2)
            # 计算合法的非隶属度 nu 的上限 (mu^q + nu^q <= 1)
            nu_max = (1 - mu ** q_val) ** (1 / q_val)
            # 随机生成 nu
            nu = round(random.uniform(0.1, nu_max), 2)

            row.append(qROFN(mu, nu, q_val))
        expert_matrix.append(row)
    opinions.append(expert_matrix)

# 初始化模型所需超参数
# 1. 专家单位调整成本系数 (Psi_u) - 假设不同的专家修改意见索要的代价不同
costs = [1.0, 1.2, 0.9, 1.1, 1.0]
# 2. 初始保留系数下界 (theta_min) - 初始设为 0.8 (即最多只接受妥协 20%)
theta_min_initial = [0.8, 0.8, 0.8, 0.8, 0.8]
# 3. 目标共识度阈值
epsilon = 0.85
# 4. 边界放松的衰减因子
alpha = 0.9

print(">>> 数据生成完毕！")

# ==========================================
# 第二部分：运行 TOCM 核心流水线
# ==========================================
print("\n================ 开始运行 TOCM 模型 ================")

# [步骤 1] M1: 预计算参考意见 (Reference Opinions)
print("\n[Step 1] 执行 M1: 预计算参考意见...")
reference_opinions = precompute_reference_opinions(opinions, experts_weights)
print("M1 完成！")

# [步骤 2] M2: 自适应边界检测 (Adaptive Boundary Detection)
# 这一步会测试初始的 0.8 是否能达到 0.85 的共识度，如果不行就会自动乘以 0.9 降维
print("\n[Step 2] 执行 M2: 启动自适应边界检测机制...")

hat_theta_min = adaptive_boundary_detection(
    opinions=opinions,
    reference_opinions=reference_opinions,
    weights=experts_weights,
    theta_min=theta_min_initial,
    epsilon=epsilon,
    alpha=alpha
)
# 打印最终计算出的安全下界，如果原数据分歧很大，这里的数值会低于 0.8
print(f"M2 完成！找到的安全保留系数下界 (hat_theta_min): \n{hat_theta_min}")

# [步骤 3 & 4] M3 & M4: NSGA-II 求解与帕累托前沿分析
print("\n[Step 3 & 4] 执行 M4 (包含M3): 启动 NSGA-II 多目标优化...")
# 测试环境下，为了看结果快一点，我们可以把 pop_size 和 n_gen 设小一点，比如 50 和 100
F_real, X_optimal = solve_and_analyze_TOCM(
    opinions=opinions,
    reference_opinions=reference_opinions,
    weights=experts_weights,
    costs=costs,
    hat_theta_min=hat_theta_min,
    epsilon=epsilon,
    pop_size=50,
    n_gen=100
)

print("\n================ 优化流水线结束 ================")

# ==========================================
# 第三部分：结果查看
# ==========================================
if X_optimal is not None and len(X_optimal) > 0:
    print(f"\n🎉 成功找到 {len(X_optimal)} 个帕累托最优解 (非支配解)！")

    # 打印前 3 个解供参考
    print("\n展示前 3 个最优决策方案：")
    for idx in range(min(3, len(X_optimal))):
        print(f"--- 方案 {idx + 1} ---")
        # 格式化打印：保留 4 位小数
        theta_formatted = [round(t, 4) for t in X_optimal[idx]]
        f_formatted = [round(f, 4) for f in F_real[idx]]

        print(f"决策变量 (专家的保留系数): {theta_formatted}")
        print(f"目标值 -> [成本(f1): {f_formatted[0]}, 共识度(f2): {f_formatted[1]}, 公平度(f3): {f_formatted[2]}]")
else:
    print(
        "\n⚠️ 优化结束，但未找到满足约束的帕累托解。请检查共识度约束 epsilon 是否设置过高，或检查适应度函数中的约束逻辑。")

# 注意：如果你的 solve_and_analyze_TOCM 中已经写了 plt.show()，这里会自动弹出一个 3D 散点图窗口。