import numpy as np
import matplotlib.pyplot as plt
from main.optimization_model.phase_II_prepare import TOCM_Problem

# 兼容不同版本的 pymoo 导入方式
try:
    # 针对 pymoo < 0.6.0 的经典导入方式
    from pymoo.factory import get_crossover, get_mutation, get_termination
except ImportError:
    # 针对 pymoo >= 0.6.0 的现代导入方式
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.termination import get_termination
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM


    def get_crossover(name, **kwargs):
        return SBX(**kwargs)


    def get_mutation(name, **kwargs):
        return PM(**kwargs)

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


def solve_and_analyze_TOCM(opinions, reference_opinions, weights, costs, hat_theta_min, epsilon, pop_size=100,
                           n_gen=200):
    """
    【M4: 阶段二 - NSGA-II 求解与分析模块】
    配置并运行 NSGA-II 算法求解三目标共识优化模型，提取帕累托前沿并进行 3D 可视化。

    参数:
    opinions : list
        原始意见矩阵列表。
    reference_opinions : list
        M1 阶段预计算的参考意见矩阵列表。
    weights : list
        专家权重列表。
    costs : list
        专家单位调整成本系数列表。
    hat_theta_min : list
        M2 阶段输出的安全保留系数下界列表。
    epsilon : float
        目标共识度阈值。
    pop_size : int
        种群大小 (默认 100)。
    n_gen : int
        最大迭代代数 (默认 200)。

    返回:
    F_real : np.ndarray
        还原符号后的真实目标值矩阵 (帕累托前沿)，维度为 (N_solutions, 3)。
    X_optimal : np.ndarray
        对应的最优决策变量矩阵 (保留系数 theta)，维度为 (N_solutions, K)。
    """

    # 1. 实例化自定义的三目标优化问题
    problem = TOCM_Problem(
        opinions=opinions,
        reference_opinions=reference_opinions,
        weights=weights,
        costs=costs,
        hat_theta_min=hat_theta_min,
        epsilon=epsilon
    )

    # 2. 配置 NSGA-II 算法算子
    # 使用模拟二进制交叉 (SBX) 和 多项式变异 (PM)，专为实数编码设计
    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )

    # 设置终止条件：按最大迭代代数终止
    termination = get_termination("n_gen", n_gen)

    print(f"开始执行 NSGA-II 优化... (种群大小: {pop_size}, 迭代代数: {n_gen})")

    # 3. 执行优化求解
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,  # 固定随机种子以保证结果可复现
        save_history=False,  # 节省内存，不保存每一代的历史
        verbose=True  # 打印迭代过程
    )

    # 4. 异常捕获与结果提取
    try:
        if res.F is None or len(res.F) == 0:
            raise ValueError("未找到满足共识度约束 (CD_C >= epsilon) 的可行解。")
    except ValueError as e:
        print(f"\n[优化失败] {e}")
        print("建议：检查 epsilon 设置是否过高，或检查阶段一输出的安全下界是否合理。")
        return None, None

    # 提取决策变量 (帕累托解集) 和 目标值 (帕累托前沿)
    X_optimal = res.X
    F_pymoo = res.F

    # 5. 符号回调 (极度重要)
    # pymoo 中 f2 和 f3 是作为最小化问题求解的 (-CD_C 和 -f3)
    # 这里必须乘以 -1 还原为真实的共识度和公平度 (极大化指标)
    F_real = F_pymoo.copy()
    F_real[:, 1] = -1.0 * F_pymoo[:, 1]  # 还原 f2 (共识度)
    F_real[:, 2] = -1.0 * F_pymoo[:, 2]  # 还原 f3 (公平度)

    num_solutions = len(F_real)
    print(f"\n优化完成！共找到 {num_solutions} 个非支配解 (Pareto Solutions)。")

    # 6. 3D 可视化帕累托前沿
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 提取三维坐标
    f1_cost = F_real[:, 0]
    f2_consensus = F_real[:, 1]
    f3_fairness = F_real[:, 2]

    # 绘制 3D 散点图，颜色映射根据成本 f1 渐变
    scatter = ax.scatter(f1_cost, f2_consensus, f3_fairness,
                         c=f1_cost, cmap='viridis', marker='o', s=50, alpha=0.8)

    # 设置坐标轴标签
    ax.set_xlabel('Total Cost ($f_1$) - Minimize', fontweight='bold')
    ax.set_ylabel('Group Consensus ($f_2$) - Maximize', fontweight='bold')
    ax.set_zlabel('Fairness ($f_3$) - Maximize', fontweight='bold')
    ax.set_title('3D Pareto Front of TOCM', fontsize=14, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Cost Intensity', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

    return F_real, X_optimal