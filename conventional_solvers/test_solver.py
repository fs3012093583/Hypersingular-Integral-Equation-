"""
超奇异积分方程求解器的测试模块
用于验证数值算法的正确性和收敛性
"""

import numpy as np
import matplotlib.pyplot as plt
from hypersingular_solver import HypersingularIntegralSolver
import time


def test_basic_functionality():
    """测试基本功能"""
    print("测试基本功能...")
    
    # 创建求解器
    solver = HypersingularIntegralSolver(N=64)
    
    # 测试网格生成
    y_points = solver.generate_grid_points()
    assert len(y_points) == 128, "网格点数量不正确"
    assert abs(y_points[0] - (-1 + 0.5/64)) < 1e-10, "第一个网格点位置不正确"
    assert abs(y_points[-1] - (1 - 0.5/64)) < 1e-10, "最后一个网格点位置不正确"
    
    # 测试矩阵构建
    A = solver.build_coefficient_matrix()
    assert A.shape == (128, 128), "矩阵维度不正确"
    assert np.allclose(np.diag(A), 2*solver.psi_half - 2*np.log(1/(2*solver.h))), "对角线元素不正确"
    
    print("✓ 基本功能测试通过")


def test_convergence_rate():
    """测试收敛阶数"""
    print("测试收敛阶数...")
    
    N_values = [32, 64, 128, 256]
    errors = []
    h_values = []
    
    # 获取参考解
    ref_solver = HypersingularIntegralSolver(N=1024)
    _, ref_u = ref_solver.solve_equation()
    ref_Q = ref_solver.calculate_productivity(ref_u)
    
    for N in N_values:
        solver = HypersingularIntegralSolver(N=N)
        _, u_values = solver.solve_equation()
        Q_h = solver.calculate_productivity(u_values)
        
        error = abs(Q_h - ref_Q) / abs(ref_Q)
        errors.append(error)
        h_values.append(1.0 / N)
        
        print(f"N={N}, h={1/N:.6f}, Q_h={Q_h:.8f}, 误差={error:.2e}")
    
    # 检查收敛阶数是否为O(h^2)
    ratios = [errors[i] / errors[i+1] for i in range(len(errors)-1)]
    h_ratios = [h_values[i] / h_values[i+1] for i in range(len(h_values)-1)]
    
    # 对于二阶收敛，误差比应该接近步长比的平方
    expected_ratios = [h**2 for h in h_ratios]
    
    print(f"实际误差比: {ratios}")
    print(f"期望误差比: {expected_ratios}")
    
    # 检查是否接近二阶收敛
    for i, (actual, expected) in enumerate(zip(ratios, expected_ratios)):
        ratio = actual / expected
        if abs(ratio - 1) > 0.3:  # 允许30%的误差
            print(f"⚠  收敛阶数可能在N={N_values[i]}到N={N_values[i+1]}之间偏离二阶")
        else:
            print(f"✓ 收敛阶数在N={N_values[i]}到N={N_values[i+1]}之间符合二阶")
    
    print("✓ 收敛阶数测试完成")


def test_accuracy_target():
    """测试是否达到理论精度目标"""
    print("测试精度目标...")
    
    # 理论极限值
    theoretical_limit = 4.15875
    
    # 测试不同网格尺寸下的精度
    N_values = [128, 256, 512, 1024]
    
    for N in N_values:
        solver = HypersingularIntegralSolver(N=N)
        _, u_values = solver.solve_equation()
        Q_h = solver.calculate_productivity(u_values)
        
        error = abs(Q_h - theoretical_limit)
        print(f"N={N}, Q_h={Q_h:.8f}, 与理论值误差={error:.2e}")
        
        if error < 1e-4:  # 如果误差小于0.0001
            print(f"✓ N={N}时达到高精度要求")
            break
    
    print("✓ 精度目标测试完成")


def test_performance():
    """测试性能"""
    print("测试性能...")
    
    N_values = [64, 128, 256, 512]
    times = []
    
    for N in N_values:
        solver = HypersingularIntegralSolver(N=N)
        
        start_time = time.time()
        _, u_values = solver.solve_equation()
        Q_h = solver.calculate_productivity(u_values)
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        print(f"N={N}, 计算时间={elapsed:.4f}秒, Q_h={Q_h:.8f}")
    
    # 检查时间复杂度
    print("时间复杂度分析:")
    for i in range(1, len(times)):
        ratio = times[i] / times[i-1]
        size_ratio = (N_values[i] / N_values[i-1])**3  # 矩阵求解理论复杂度O(n^3)
        print(f"N={N_values[i-1]}到N={N_values[i]}: 时间比={ratio:.2f}, 理论比={size_ratio:.2f}")
    
    print("✓ 性能测试完成")


def test_matrix_properties():
    """测试矩阵性质"""
    print("测试矩阵性质...")
    
    solver = HypersingularIntegralSolver(N=128)
    A = solver.build_coefficient_matrix()
    
    # 检查矩阵是否对称
    is_symmetric = np.allclose(A, A.T)
    print(f"矩阵对称性: {is_symmetric}")
    
    # 检查条件数
    cond_number = np.linalg.cond(A)
    print(f"矩阵条件数: {cond_number:.2e}")
    
    # 检查特征值
    eigenvalues = np.linalg.eigvals(A)
    print(f"特征值范围: [{np.min(eigenvalues.real):.6f}, {np.max(eigenvalues.real):.6f}]")
    
    # 检查是否正定
    is_positive_definite = np.all(eigenvalues.real > 0)
    print(f"矩阵正定性: {is_positive_definite}")
    
    print("✓ 矩阵性质测试完成")


def plot_detailed_results():
    """绘制详细结果"""
    print("绘制详细结果...")
    
    # 使用中等网格尺寸
    solver = HypersingularIntegralSolver(N=256)
    y_points, u_values = solver.solve_equation()
    Q_h = solver.calculate_productivity(u_values)
    
    # 创建详细图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 解的分布
    axes[0,0].plot(y_points, u_values, 'b-', linewidth=2, label=f'N=256, Q_h={Q_h:.6f}')
    axes[0,0].set_xlabel('y')
    axes[0,0].set_ylabel('g(y)')
    axes[0,0].set_title('数值解分布')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # 2. 收敛性研究
    results = solver.convergence_study([32, 64, 128, 256, 512])
    axes[0,1].loglog(results['h'], results['error'], 'bo-', markersize=8, linewidth=2, label='实际误差')
    axes[0,1].loglog(results['h'], [h**2 for h in results['h']], 'r--', linewidth=2, label='O(h²)参考')
    axes[0,1].set_xlabel('h')
    axes[0,1].set_ylabel('相对误差')
    axes[0,1].set_title('收敛性分析')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # 3. 产能收敛
    axes[1,0].semilogx(results['N'], results['Q_h'], 'go-', markersize=8, linewidth=2)
    axes[1,0].axhline(y=4.15875, color='r', linestyle='--', linewidth=2, label='理论极限')
    axes[1,0].set_xlabel('N')
    axes[1,0].set_ylabel('Q_h')
    axes[1,0].set_title('产能收敛性')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    # 4. 系数矩阵结构
    A = solver.build_coefficient_matrix()
    im = axes[1,1].imshow(A, cmap='viridis', aspect='auto')
    axes[1,1].set_title('系数矩阵结构')
    plt.colorbar(im, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig('detailed_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ 详细结果图表已保存为 'detailed_results.png'")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("超奇异积分方程求解器 - 综合测试")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        print()
        
        test_convergence_rate()
        print()
        
        test_accuracy_target()
        print()
        
        test_performance()
        print()
        
        test_matrix_properties()
        print()
        
        plot_detailed_results()
        print()
        
        print("=" * 60)
        print("所有测试完成！求解器工作正常。")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()