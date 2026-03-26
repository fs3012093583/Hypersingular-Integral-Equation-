import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class DiscreteIntegralNeuralSolver:
    """
    基于离散积分公式的神经网络超奇异积分方程求解器
    严格使用公式：I_h = Σ h·G(a+(j+1/2)h) + 2ψ(1/2) - 2ln(1/h)
    其中 G(x) = |x-y|⁻¹g(x)
    """
    
    def __init__(self, N: int = 128, hidden_dim: int = 64, lr: float = 0.001):
        """
        初始化求解器
        
        Args:
            N: 网格参数，步长 h = 2/N
            hidden_dim: 隐藏层维度
            lr: 学习率
        """
        self.N = N
        self.h = 2.0 / N  # 注意：区间长度为2，从-1到1
        self.hidden_dim = hidden_dim
        self.gamma = 0.5772156649015329  # 欧拉常数
        self.psi_half = -self.gamma - 2 * np.log(2)  # ψ(1/2) ≈ -1.9635
        
        # 构建神经网络
        self.neural_network = self._build_network()
        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=200, factor=0.5)
        self.loss_history = []
        
    def _build_network(self) -> nn.Module:
        """构建神经网络"""
        return nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def generate_grid_points(self) -> np.ndarray:
        """生成网格中点 a + (j + 1/2)h"""
        return np.array([-1 + (j + 0.5) * self.h for j in range(self.N)])
    
    def compute_discrete_integral(self, x_points: torch.Tensor) -> torch.Tensor:
        """
        使用离散积分公式计算超奇异积分
        I_h = Σ_{j=0}^{N-1} h·G(a+(j+1/2)h) + 2ψ(1/2) - 2ln(1/h)
        其中 G(x) = |x-y|⁻¹g(x)
        
        Args:
            x_points: 评估点 [batch_size, 1]
            
        Returns:
            integral_values: 积分值 [batch_size, 1]
        """
        batch_size = x_points.shape[0]
        integral_values = torch.zeros(batch_size, 1, dtype=torch.float32)
        
        # 生成积分网格点
        y_points_np = self.generate_grid_points()  # [N]
        y_points = torch.tensor(y_points_np.reshape(-1, 1), dtype=torch.float32)  # [N, 1]
        
        # 获取所有g值
        g_values = self.neural_network(y_points)  # [N, 1]
        
        # 计算补偿项
        compensation = 2 * self.psi_half - 2 * np.log(1.0 / self.h)
        
        # 对每个评估点计算积分
        for i in range(batch_size):
            x_i = x_points[i, 0].item()
            
            # 计算距离 |x_i - y_j|
            distances = torch.abs(y_points.flatten() - x_i)  # [N]
            
            # 处理奇异性：使用Hadamard有限部分
            # 当 x_i ≈ y_j 时，使用特殊处理
            eps = 1e-10
            
            # 计算 G(y_j) = |x_i - y_j|⁻¹ * g(y_j)
            G_values = torch.zeros_like(distances)
            
            # 对于非零距离，直接计算
            non_singular_mask = distances > eps
            if torch.any(non_singular_mask):
                G_values[non_singular_mask] = g_values.flatten()[non_singular_mask] / distances[non_singular_mask]
            
            # 对于零距离（奇异情况），使用相邻点的线性近似
            singular_mask = distances <= eps
            if torch.any(singular_mask):
                # 找到奇异点索引
                singular_idx = torch.where(singular_mask)[0]
                
                # 使用左右相邻点进行线性近似
                for idx in singular_idx:
                    left_idx = max(0, idx - 1)
                    right_idx = min(self.N - 1, idx + 1)
                    
                    if left_idx != right_idx:
                        # 线性插值估计奇异点处的G值
                        y_left, y_right = y_points_np[left_idx], y_points_np[right_idx]
                        g_left, g_right = g_values[left_idx, 0], g_values[right_idx, 0]
                        
                        # 估计导数
                        slope = (g_right - g_left) / (y_right - y_left + 1e-10)
                        
                        # 在奇异点处的估计（使用有限部分积分思想）
                        # 这里使用对称平均的思想
                        G_estimated = 0.5 * (g_left / (abs(x_i - y_left) + eps) + 
                                           g_right / (abs(x_i - y_right) + eps))
                        G_values[idx] = G_estimated
                    else:
                        # 边界情况，使用简单近似
                        G_values[idx] = g_values[idx, 0] / (self.h + eps)
            
            # 计算离散求和：Σ h·G(y_j)
            quadrature_sum = torch.sum(G_values) * self.h
            
            # 添加补偿项
            integral_values[i, 0] = quadrature_sum + compensation * g_values[torch.argmin(distances), 0]
        
        return integral_values
    
    def train_step(self, x_points: torch.Tensor, target_values: torch.Tensor) -> float:
        """
        训练一步
        
        Args:
            x_points: 训练点 [batch_size, 1]
            target_values: 目标值 [batch_size, 1]
            
        Returns:
            loss: 损失值
        """
        self.optimizer.zero_grad()
        
        # 使用离散积分公式计算积分值
        integral_values = self.compute_discrete_integral(x_points)
        
        # 计算损失（目标方程：积分 = -1）
        loss = torch.mean((integral_values - target_values) ** 2)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, epochs: int = 2000, batch_size: int = 64, 
             verbose: bool = True) -> dict:
        """
        训练神经网络
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否显示训练过程
            
        Returns:
            training_history: 训练历史
        """
        # 生成训练点
        y_points = self.generate_grid_points()
        n_points = len(y_points)
        
        # 目标值（方程右边 = -1）
        target_values = torch.ones(batch_size, 1) * (-1.0)
        
        training_history = {
            'loss': [],
            'epochs': []
        }
        
        for epoch in range(epochs):
            # 随机选择批次
            indices = np.random.choice(n_points, batch_size, replace=True)
            batch_points = torch.tensor(y_points[indices].reshape(-1, 1), 
                                      dtype=torch.float32)
            
            # 训练一步
            loss = self.train_step(batch_points, target_values)
            
            # 学习率调度
            if epoch % 50 == 0 and epoch > 0:
                self.scheduler.step(loss)
            
            # 记录历史
            if epoch % 10 == 0:
                training_history['loss'].append(loss)
                training_history['epochs'].append(epoch)
                
                if verbose and epoch % 200 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch}: Loss = {loss:.6f}, LR = {current_lr:.6f}")
        
        return training_history
    
    def predict(self, x_points: np.ndarray) -> np.ndarray:
        """
        预测g(x)值
        
        Args:
            x_points: 输入点 [n_points]
            
        Returns:
            g_values: 预测值 [n_points]
        """
        x_tensor = torch.tensor(x_points.reshape(-1, 1), dtype=torch.float32)
        with torch.no_grad():
            g_tensor = self.neural_network(x_tensor)
        return g_tensor.numpy().flatten()
    
    def compute_productivity(self) -> float:
        """
        计算产能Q_h
        
        Returns:
            Q_h: 产能值
        """
        y_points = self.generate_grid_points()
        g_values = self.predict(y_points)
        return self.h * np.sum(g_values)
    
    def validate_solution(self) -> dict:
        """
        验证求解精度
        
        Returns:
            validation_results: 验证结果
        """
        y_points = self.generate_grid_points()
        y_tensor = torch.tensor(y_points.reshape(-1, 1), dtype=torch.float32)
        
        # 计算积分值
        integral_values = self.compute_discrete_integral(y_tensor)
        target_values = torch.ones_like(integral_values) * (-1.0)
        
        # 计算误差
        errors = torch.abs(integral_values - target_values).numpy().flatten()
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        
        # 计算产能
        Q_h = self.compute_productivity()
        
        return {
            'max_integral_error': max_error,
            'mean_integral_error': mean_error,
            'productivity': Q_h,
            'integral_errors': errors,
            'y_points': y_points
        }
    
    def plot_training_history(self, training_history: dict, 
                            save_path: Optional[str] = None, show_plot: bool = True):
        """
        绘制训练历史
        
        Args:
            training_history: 训练历史
            save_path: 保存路径
            show_plot: 是否显示图像
        """
        plt.figure(figsize=(10, 6))
        plt.semilogy(training_history['epochs'], training_history['loss'], 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training History (Discrete Integral Formula)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_solution(self, save_path: Optional[str] = None, show_plot: bool = True):
        """
        绘制神经网络求解结果
        
        Args:
            save_path: 保存路径
            show_plot: 是否显示图像
        """
        y_points = self.generate_grid_points()
        g_values = self.predict(y_points)
        Q_h = self.compute_productivity()
        
        plt.figure(figsize=(10, 6))
        plt.plot(y_points, g_values, 'b-', linewidth=2, label=f'Discrete Integral Solution (Q_h={Q_h:.6f})')
        plt.xlabel('y')
        plt.ylabel('g(y)')
        plt.title('Neural Network Solution (Discrete Integral Formula)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()


def compare_with_analytical():
    """
    与解析解（传统数值解）比较
    """
    from hypersingular_solver import HypersingularIntegralSolver as AnalyticalSolver
    
    print("Comparing Discrete Integral Neural Solution with Analytical Solution")
    print("=" * 70)
    
    # 传统数值解
    analytical_solver = AnalyticalSolver(N=256)
    y_points, u_values = analytical_solver.solve_equation()
    Q_h_analytical = analytical_solver.calculate_productivity(u_values)
    
    print(f"Analytical solution: Q_h = {Q_h_analytical:.8f}")
    
    # 离散积分神经网络解
    neural_solver = DiscreteIntegralNeuralSolver(N=256, hidden_dim=128, lr=0.001)
    
    print("Training discrete integral neural network...")
    training_history = neural_solver.train(epochs=3000, batch_size=64, verbose=True)
    
    # 验证
    validation_results = neural_solver.validate_solution()
    Q_h_neural = validation_results['productivity']
    
    print(f"Discrete integral neural solution: Q_h = {Q_h_neural:.8f}")
    print(f"Difference: {abs(Q_h_neural - Q_h_analytical):.2e}")
    print(f"Relative error: {abs(Q_h_neural - Q_h_analytical) / abs(Q_h_analytical):.2e}")
    print(f"Max integral error: {validation_results['max_integral_error']:.2e}")
    print(f"Mean integral error: {validation_results['mean_integral_error']:.2e}")
    
    # 绘制结果
    neural_solver.plot_training_history(training_history, 'discrete_integral_training.png', show_plot=False)
    neural_solver.plot_solution('discrete_integral_solution.png', show_plot=False)
    
    # 比较解的分布
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(y_points, u_values, 'r-', linewidth=2, label=f'Analytical (Q_h={Q_h_analytical:.6f})')
    neural_g = neural_solver.predict(y_points)
    plt.plot(y_points, neural_g, 'b--', linewidth=2, label=f'Discrete Integral (Q_h={Q_h_neural:.6f})')
    plt.xlabel('y')
    plt.ylabel('g(y)')
    plt.title('Solution Comparison (Discrete Integral Formula)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(y_points, abs(neural_g - u_values), 'g-', linewidth=2)
    plt.xlabel('y')
    plt.ylabel('Difference')
    plt.title('Solution Difference')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('discrete_integral_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Results saved to: discrete_integral_training.png, discrete_integral_solution.png, discrete_integral_comparison.png")


def main():
    """主函数"""
    print("Discrete Integral Neural Network Based Hypersingular Integral Equation Solver")
    print("=" * 80)
    print("Using formula: I_h = Σ h·G(a+(j+1/2)h) + 2ψ(1/2) - 2ln(1/h)")
    print("where G(x) = |x-y|⁻¹g(x)")
    print()
    
    # 创建离散积分神经网络求解器
    solver = DiscreteIntegralNeuralSolver(N=256, hidden_dim=128, lr=0.001)
    
    # 训练
    print("Training discrete integral neural network...")
    training_history = solver.train(epochs=3000, batch_size=64, verbose=True)
    
    # 验证
    print("Validating solution...")
    validation_results = solver.validate_solution()
    
    print(f"Training completed!")
    print(f"Productivity Q_h = {validation_results['productivity']:.8f}")
    print(f"Max integral error = {validation_results['max_integral_error']:.2e}")
    print(f"Mean integral error = {validation_results['mean_integral_error']:.2e}")
    
    # 保存结果
    solver.plot_training_history(training_history, 'discrete_integral_training_final.png', show_plot=False)
    solver.plot_solution('discrete_integral_solution_final.png', show_plot=False)
    
    print("Results saved to: discrete_integral_training_final.png, discrete_integral_solution_final.png")
    
    # 与解析解比较
    try:
        compare_with_analytical()
    except ImportError:
        print("Analytical solver not available for detailed comparison")


if __name__ == "__main__":
    main()