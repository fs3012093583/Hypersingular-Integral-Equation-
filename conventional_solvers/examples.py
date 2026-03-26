"""
超奇异积分方程求解器的使用示例
演示如何求解 ∫_{-1}^{1} |x-y|^{-1} g(x) dx = -1
"""

import numpy as np
import matplotlib.pyplot as plt
from hypersingular_solver import HypersingularIntegralSolver


def basic_usage_example():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建求解器，N=128表示网格细度
    solver = HypersingularIntegralSolver(N=128)
    
    # 求解方程
    y_points, u_values = solver.solve_equation()
    
    # 计算产能
    Q_h = solver.calculate_productivity(u_values)
    
    print(f"网格参数: N = {solver.N}, h = {solver.h:.6f}")
    print(f"数值产能 Q_h = {Q_h:.8f}")
    
    # 绘制结果
    solver.plot_solution(y_points, u_values)


def convergence_study_example():
    """收敛性研究示例"""
    print("\n=== 收敛性研究示例 ===")
    
    # 创建求解器
    solver = HypersingularIntegralSolver(N=256)
    
    # 进行收敛性研究
    results = solver.convergence_study([32, 64, 128, 256, 512])
    
    print("收敛性结果:")
    print("N\th\t\tQ_h\t\t相对误差")
    print("-" * 60)
    for i in range(len(results['N'])):
        print(f"{results['N'][i]}\t{results['h'][i]:.6f}\t{results['Q_h'][i]:.8f}\t{results['error'][i]:.2e}")
    
    # 绘制收敛性图表
    solver.plot_convergence(results)


def high_precision_example():
    """高精度求解示例"""
    print("\n=== 高精度求解示例 ===")
    
    # 使用更细的网格
    solver = HypersingularIntegralSolver(N=1024)
    
    print(f"使用高精度网格: N = {solver.N}, h = {solver.h:.8f}")
    
    # 求解
    y_points, u_values = solver.solve_equation()
    Q_h = solver.calculate_productivity(u_values)
    
    print(f"高精度产能 Q_h = {Q_h:.10f}")
    
    # 与理论极限值比较
    theoretical_limit = 4.15875
    error = abs(Q_h - theoretical_limit)
    print(f"理论极限值: {theoretical_limit}")
    print(f"绝对误差: {error:.2e}")
    print(f"相对误差: {error/theoretical_limit:.2e}")


def compare_different_methods():
    """比较不同网格尺寸的结果"""
    print("\n=== 不同网格尺寸比较 ===")
    
    N_values = [64, 128, 256, 512, 1024]
    results = []
    
    for N in N_values:
        solver = HypersingularIntegralSolver(N=N)
        _, u_values = solver.solve_equation()
        Q_h = solver.calculate_productivity(u_values)
        
        results.append({
            'N': N,
            'h': 1.0/N,
            'Q_h': Q_h,
            'error_estimate': solver.compute_error_estimate(u_values)
        })
    
    print("不同网格尺寸的结果比较:")
    print("N\th\t\tQ_h\t\t误差估计")
    print("-" * 60)
    for result in results:
        print(f"{result['N']}\t{result['h']:.6f}\t{result['Q_h']:.8f}\t{result['error_estimate']:.2e}")


def custom_analysis_example():
    """自定义分析示例"""
    print("\n=== 自定义分析示例 ===")
    
    # 创建求解器
    solver = HypersingularIntegralSolver(N=256)
    
    # 求解
    y_points, u_values = solver.solve_equation()
    Q_h = solver.calculate_productivity(u_values)
    
    # 自定义分析：计算解的统计特性
    print("解的统计特性:")
    print(f"均值: {np.mean(u_values):.6f}")
    print(f"标准差: {np.std(u_values):.6f}")
    print(f"最大值: {np.max(u_values):.6f} (位置: {y_points[np.argmax(u_values)]:.4f})")
    print(f"最小值: {np.min(u_values):.6f} (位置: {y_points[np.argmin(u_values)]:.4f})")
    
    # 分析解的对称性
    left_half = u_values[:len(u_values)//2]
    right_half = u_values[len(u_values)//2:][::-1]  # 反转右半部分
    symmetry_error = np.linalg.norm(left_half - right_half)
    print(f"解的对称性误差: {symmetry_error:.2e}")
    
    # 绘制详细的分析图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 解的分布
    axes[0].plot(y_points, u_values, 'b-', linewidth=2)
    axes[0].set_xlabel('y')
    axes[0].set_ylabel('g(y)')
    axes[0].set_title('数值解')
    axes[0].grid(True, alpha=0.3)
    
    # 解的直方图
    axes[1].hist(u_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1].set_xlabel('g(y)')
    axes[1].set_ylabel('频次')
    axes[1].set_title('解的分布直方图')
    axes[1].grid(True, alpha=0.3)
    
    # 累积分布
    sorted_values = np.sort(u_values)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    axes[2].plot(sorted_values, cumulative, 'r-', linewidth=2)
    axes[2].set_xlabel('g(y)')
    axes[2].set_ylabel('累积概率')
    axes[2].set_title('累积分布函数')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('custom_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def batch_processing_example():
    """批处理示例"""
    print("\n=== 批处理示例 ===")
    
    # 一系列网格尺寸
    N_values = [32, 64, 128, 256, 512, 1024]
    
    print("批处理计算进行中...")
    results = []
    
    for N in N_values:
        solver = HypersingularIntegralSolver(N=N)
        _, u_values = solver.solve_equation()
        Q_h = solver.calculate_productivity(u_values)
        
        results.append({
            'N': N,
            'h': 1.0/N,
            'Q_h': Q_h,
            'u_values': u_values,
            'y_points': solver.generate_grid_points()
        })
        
        print(f"完成 N={N}, Q_h={Q_h:.8f}")
    
    # 保存结果到文件
    np.savez('batch_results.npz', 
             N_values=N_values,
             results=results)
    
    print("批处理结果已保存到 'batch_results.npz'")
    
    # 绘制收敛趋势
    plt.figure(figsize=(10, 6))
    Q_values = [r['Q_h'] for r in results]
    plt.semilogx(N_values, Q_values, 'bo-', markersize=8, linewidth=2, label='数值解')
    plt.axhline(y=4.15875, color='r', linestyle='--', linewidth=2, label='理论极限')
    plt.xlabel('N')
    plt.ylabel('Q_h')
    plt.title('批处理收敛趋势')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('batch_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数：运行所有示例"""
    print("超奇异积分方程求解器 - 使用示例")
    print("=" * 60)
    
    try:
        # 运行基本示例
        basic_usage_example()
        
        # 收敛性研究
        convergence_study_example()
        
        # 高精度示例
        high_precision_example()
        
        # 比较不同方法
        compare_different_methods()
        
        # 自定义分析
        custom_analysis_example()
        
        # 批处理
        batch_processing_example()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()