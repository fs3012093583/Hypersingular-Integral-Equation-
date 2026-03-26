import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from typing import Tuple, Optional

class HypersingularIntegralSolver:
    """
    超奇异积分方程求解器
    求解方程：∫_{-1}^{1} |x-y|^{-1} g(x) dx = -1
    基于中点矩形求积逼近公式
    """
    
    def __init__(self, N: int = 128):
        """
        初始化求解器
        
        Args:
            N: 网格参数，步长 h = 1/N
        """
        self.N = N
        self.h = 1.0 / N
        self.gamma = 0.5772156649015329  # 欧拉常数
        self.psi_half = -self.gamma - 2 * np.log(2)  # ψ(1/2) ≈ -1.9635
        
    def generate_grid_points(self) -> np.ndarray:
        """
        生成网格中点
        
        Returns:
            y_i: 网格中点数组，y_i = -1 + (i + 1/2)h
        """
        return np.array([-1 + (i + 0.5) * self.h for i in range(2 * self.N)])
    
    def build_coefficient_matrix(self) -> np.ndarray:
        """
        构建系数矩阵 A_N
        
        Returns:
            A: 2N × 2N 的系数矩阵
        """
        size = 2 * self.N
        A = np.zeros((size, size))
        
        # 对角线元素
        diagonal_term = 2 * self.psi_half - 2 * np.log(1.0 / (2 * self.h))
        np.fill_diagonal(A, diagonal_term)
        
        # 非对角线元素
        for i in range(size):
            for j in range(size):
                if i != j and abs(i - j) % 2 == 1:  # |i-j| 为奇数
                    A[i, j] = 2.0 / abs(i - j)
                    
        return A
    
    def solve_equation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解离散线性方程组
        
        Returns:
            y_points: 网格中点
            u_values: 解向量 u
        """
        # 生成网格点
        y_points = self.generate_grid_points()
        
        # 构建系数矩阵
        A = self.build_coefficient_matrix()
        
        # 构建右端向量
        b = -np.ones(2 * self.N)
        
        # 求解线性方程组
        u_values = solve(A, b)
        
        return y_points, u_values
    
    def calculate_productivity(self, u_values: np.ndarray) -> float:
        """
        计算产能 Q_h
        
        Args:
            u_values: 解向量 u
            
        Returns:
            Q_h: 产能近似值
        """
        return self.h * np.sum(u_values)
    
    def compute_error_estimate(self, u_values: np.ndarray, 
                             reference_N: int = 1024) -> float:
        """
        计算误差估计（通过比较不同网格尺寸的结果）
        
        Args:
            u_values: 当前解
            reference_N: 参考网格尺寸
            
        Returns:
            error: 相对误差估计
        """
        if self.N == reference_N:
            return 0.0
            
        # 创建参考求解器
        ref_solver = HypersingularIntegralSolver(N=reference_N)
        _, ref_u = ref_solver.solve_equation()
        ref_Q = ref_solver.calculate_productivity(ref_u)
        
        current_Q = self.calculate_productivity(u_values)
        
        return abs(current_Q - ref_Q) / abs(ref_Q)
    
    def plot_solution(self, y_points: np.ndarray, u_values: np.ndarray, 
                     save_path: Optional[str] = None):
        """
        绘制解的图像
        
        Args:
            y_points: 网格中点
            u_values: 解向量
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=(10, 6))
        plt.plot(y_points, u_values, 'b-o', markersize=3, linewidth=1.5, 
                label=f'N={self.N}, h={self.h:.4f}')
        plt.xlabel('y')
        plt.ylabel('g(y)')
        plt.title('超奇异积分方程的数值解')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def convergence_study(self, N_values: list = [32, 64, 128, 256, 512]) -> dict:
        """
        收敛性研究
        
        Args:
            N_values: 不同网格尺寸的列表
            
        Returns:
            results: 包含不同N值的结果
        """
        results = {
            'N': [],
            'h': [],
            'Q_h': [],
            'error': []
        }
        
        # 获取参考解
        ref_solver = HypersingularIntegralSolver(N=1024)
        _, ref_u = ref_solver.solve_equation()
        ref_Q = ref_solver.calculate_productivity(ref_u)
        
        for N in N_values:
            solver = HypersingularIntegralSolver(N=N)
            _, u_values = solver.solve_equation()
            Q_h = solver.calculate_productivity(u_values)
            
            results['N'].append(N)
            results['h'].append(1.0 / N)
            results['Q_h'].append(Q_h)
            results['error'].append(abs(Q_h - ref_Q) / abs(ref_Q))
        
        return results
    
    def plot_convergence(self, results: dict, save_path: Optional[str] = None):
        """
        绘制收敛性图像
        
        Args:
            results: 收敛性研究结果
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制误差随h的变化
        plt.subplot(1, 2, 1)
        plt.loglog(results['h'], results['error'], 'bo-', markersize=8, linewidth=2)
        plt.loglog(results['h'], [h**2 for h in results['h']], 'r--', 
                  label='O(h²)参考线', linewidth=2)
        plt.xlabel('h')
        plt.ylabel('相对误差')
        plt.title('收敛性分析：误差 vs 步长')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 绘制产能收敛
        plt.subplot(1, 2, 2)
        plt.semilogx(results['N'], results['Q_h'], 'go-', markersize=8, linewidth=2)
        plt.axhline(y=4.15875, color='r', linestyle='--', 
                   label='理论极限值', linewidth=2)
        plt.xlabel('N')
        plt.ylabel('Q_h')
        plt.title('产能收敛性')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    主函数：演示求解器的使用
    """
    print("超奇异积分方程数值求解器")
    print("=" * 50)
    
    # 创建求解器
    solver = HypersingularIntegralSolver(N=8)
    
    print(f"网格参数: N = {solver.N}, h = {solver.h:.6f}")
    print(f"ψ(1/2) = {solver.psi_half:.6f}")
    
    # 求解方程
    print("正在求解线性方程组...")
    y_points, u_values = solver.solve_equation()
    
    # 计算产能
    Q_h = solver.calculate_productivity(u_values)
    print(f"数值产能 Q_h = {Q_h:.8f}")
    
    # 误差估计
    error = solver.compute_error_estimate(u_values)
    print(f"相对误差估计: {error:.2e}")
    
    # 收敛性研究
    print("正在进行收敛性研究...")
    results = solver.convergence_study()
    
    print("\n收敛性结果:")
    print("N\th\t\tQ_h\t\t相对误差")
    print("-" * 60)
    for i in range(len(results['N'])):
        print(f"{results['N'][i]}\t{results['h'][i]:.6f}\t{results['Q_h'][i]:.8f}\t{results['error'][i]:.2e}")
    
    # 绘制结果
    solver.plot_solution(y_points, u_values)
    solver.plot_convergence(results)
    
    print("\n求解完成！")


if __name__ == "__main__":
    main()