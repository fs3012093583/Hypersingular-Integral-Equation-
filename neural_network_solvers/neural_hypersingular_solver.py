import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class NeuralHypersingularSolver:
    """
    基于神经网络的超奇异积分方程求解器
    使用神经网络逼近g(x)，通过超奇异积分公式训练
    """
    
    def __init__(self, N: int = 128, hidden_dim: int = 64, lr: float = 0.001):
        """
        初始化神经网络求解器
        
        Args:
            N: 网格参数，步长 h = 1/N
            hidden_dim: 隐藏层维度
            lr: 学习率
        """
        self.N = N
        self.h = 1.0 / N
        self.hidden_dim = hidden_dim
        self.gamma = 0.5772156649015329  # 欧拉常数
        self.psi_half = -self.gamma - 2 * np.log(2)  # ψ(1/2) ≈ -1.9635
        
        # 构建神经网络
        self.neural_network = self._build_network()
        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=lr)
        self.loss_history = []
        
    def _build_network(self) -> nn.Module:
        """构建神经网络"""
        return nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def generate_grid_points(self) -> np.ndarray:
        """生成网格中点"""
        return np.array([-1 + (i + 0.5) * self.h for i in range(2 * self.N)])
    
    def compute_hypersingular_integral(self, x_points: torch.Tensor, y_points: torch.Tensor,
                                     g_values: torch.Tensor) -> torch.Tensor:
        """
        计算超奇异积分
        
        Args:
            x_points: 积分点 [batch_size, 1]
            g_values: g(x)值 [batch_size, 1]
            
        Returns:
            integral_values: 积分值 [batch_size, 1]
        """
        batch_size = x_points.shape[0]
        integral_values = torch.zeros(batch_size, 1, dtype=torch.float32)
        
        # 生成积分网格点
      
        
        for i in range(batch_size):
            x_i = x_points[i, 0].item()
            
            # 计算G(y) = |y-x_i|^{-1} * g(y)
            g_y = self.neural_network(y_points)
            
            # 计算距离
            y_np = y_points_np.reshape(-1)
            distances = np.abs(y_np - x_i)
            
            # 使用Hadamard有限部分积分的离散形式
            # 避免奇异性，跳过当前点附近的网格点s
            eps = 1e-11
            mask = distances > 0
            
            if np.sum(mask) > 0:
                # 正常计算非奇异部分
                g_y_masked = g_y[mask]
                distances_torch = torch.tensor(distances[mask].reshape(-1, 1), 
                                              dtype=torch.float32)
                G_values = g_y_masked / distances_torch
                quadrature_sum = torch.sum(G_values) * self.h
            else:
                quadrature_sum = torch.tensor(0.0, dtype=torch.float32)
            
            # 添加奇异性补偿项
            compensation = 2 * self.psi_half - 2 * np.log(1.0 / self.h)
            
            integral_values[i, 0] = quadrature_sum + compensation * g_values[i, 0]
            
        return integral_values
    
    def train_step(self, x_points: torch.Tensor,y_points: torch.Tensor, target_values: torch.Tensor) -> float:
        """
        训练一步
        
        Args:
            这里传入的应该是y的值
            x_points: 训练点 [batch_size, 1]
            target_values: 目标值 [batch_size, 1]
            
        Returns:
            loss: 损失值
        """
        self.optimizer.zero_grad()
        
        # 前向传播得到g(x)
        g_values = self.neural_network(x_points)
        
        # 计算超奇异积分
        integral_values = self.compute_hypersingular_integral(x_points,y_points, g_values)
        
        # 计算损失（目标方程：积分 = -1）
        loss = torch.mean((integral_values - target_values) ** 2)
        
        # 添加正则化项防止过拟合
        # L2正则化
        l2_reg = 0
        # for param in self.neural_network.parameters():
        #     l2_reg += torch.norm(param)**2
        
        # 总损失 = 主损失 + 正则化
        total_loss = loss + 1e-6 * l2_reg
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, epochs: int = 1000, batch_size: int = 16, 
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
        target_values = torch.ones(n_points, 1) * (-1.0)
        
        training_history = {
            'loss': [],
            'epochs': []
        }
        
        for epoch in range(epochs):
            # 随机选择批次
            indices = np.random.choice(n_points, batch_size, replace=True)
            y_batch_points = torch.tensor(y_points.reshape(-1, 1)
            , 
                                      dtype=torch.float32)
            
            # 训练一步
            loss = self.train_step(x_points,y_batch_points, target_values)
            
            # 记录历史
            if epoch % 10 == 0:
                training_history['loss'].append(loss)
                training_history['epochs'].append(epoch)
                
                if verbose and epoch % 100 == 0:
                    print(f"Epoch {epoch}: Losss = {loss:.6f}")
        
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
        
        # 计算积分值
        integral_errors = []
        target_integral = -1.0
        
        for i, y_i in enumerate(y_points):
            y_tensor = torch.tensor([[y_i]], dtype=torch.float32)
            g_value = self.neural_network(y_tensor)
            
            # 计算该点的积分值
            integral_value = self.compute_hypersingular_integral(y_tensor, g_value)
            error = abs(integral_value.item() - target_integral)
            integral_errors.append(error)
        
        # 计算统计量
        max_error = max(integral_errors)
        mean_error = np.mean(integral_errors)
        
        # 计算产能
        Q_h = self.compute_productivity()
        
        return {
            'max_integral_error': max_error,
            'mean_integral_error': mean_error,
            'productivity': Q_h,
            'integral_errors': integral_errors,
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
        plt.title('Training History')
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
        plt.plot(y_points, g_values, 'b-', linewidth=2, label=f'Neural Solution (Q_h={Q_h:.6f})')
        plt.xlabel('y')
        plt.ylabel('g(y)')
        plt.title('Neural Network Solution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_validation(self, validation_results: dict, 
                       save_path: Optional[str] = None, show_plot: bool = True):
        """
        绘制验证结果
        
        Args:
            validation_results: 验证结果
            save_path: 保存路径
            show_plot: 是否显示图像
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 积分误差分布
        axes[0].plot(validation_results['y_points'], validation_results['integral_errors'], 
                    'r-', linewidth=2)
        axes[0].set_xlabel('y')
        axes[0].set_ylabel('Integral Error')
        axes[0].set_title('Integral Error Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # 解的分布
        axes[1].plot(validation_results['y_points'], 
                    validation_results['integral_errors'], 'b-', linewidth=2)
        axes[1].set_xlabel('y')
        axes[1].set_ylabel('g(y)')
        axes[1].set_title(f'Solution Distribution (Q_h={validation_results["productivity"]:.6f})')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
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
    
    print("Comparing Neural Network Solution with Analytical Solution")
    print("=" * 60)
    
    # 传统数值解
    analytical_solver = AnalyticalSolver(N=256)
    y_points, u_values = analytical_solver.solve_equation()
    Q_h_analytical = analytical_solver.calculate_productivity(u_values)
    
    print(f"Analytical solution: Q_h = {Q_h_analytical:.8f}")
    
    # 神经网络解
    neural_solver = NeuralHypersingularSolver(N=256, hidden_dim=128, lr=0.001)
    
    print("Training neural network...")
    training_history = neural_solver.train(epochs=2000, batch_size=64, verbose=True)
    
    # 验证
    validation_results = neural_solver.validate_solution()
    Q_h_neural = validation_results['productivity']
    
    print(f"Neural network solution: Q_h = {Q_h_neural:.8f}")
    print(f"Difference: {abs(Q_h_neural - Q_h_analytical):.2e}")
    print(f"Max integral error: {validation_results['max_integral_error']:.2e}")
    print(f"Mean integral error: {validation_results['mean_integral_error']:.2e}")
    
    # 绘制结果
    neural_solver.plot_training_history(training_history, 'neural_training.png', show_plot=False)
    neural_solver.plot_solution('neural_solution.png', show_plot=False)
    
    # 比较解的分布
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(y_points, u_values, 'r-', linewidth=2, label=f'Analytical (Q_h={Q_h_analytical:.6f})')
    neural_g = neural_solver.predict(y_points)
    plt.plot(y_points, neural_g, 'b--', linewidth=2, label=f'Neural (Q_h={Q_h_neural:.6f})')
    plt.xlabel('y')
    plt.ylabel('g(y)')
    plt.title('Solution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(y_points, abs(neural_g - u_values), 'g-', linewidth=2)
    plt.xlabel('y')
    plt.ylabel('Difference')
    plt.title('Solution Difference')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('solution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Results saved to: neural_training.png, neural_solution.png, solution_comparison.png")


def main():
    """主函数"""
    print("Neural Network Based Hypersingular Integral Equation Solver")
    print("=" * 60)
    
    # 创建神经网络求解器
    solver = NeuralHypersingularSolver(N=128, hidden_dim=64, lr=0.0001)
    
    # 训练
    print("Training neural network...")
    training_history = solver.train(epochs=1000, batch_size=128, verbose=True)
    
    # 验证
    print("Validating solution...")
    validation_results = solver.validate_solution()
    
    print(f"Training completed!")
    print(f"Productivity Q_h = {validation_results['productivity']:.8f}")
    print(f"Max integral error = {validation_results['max_integral_error']:.2e}")
    print(f"Mean integral error = {validation_results['mean_integral_error']:.2e}")
    
    # 保存结果
    solver.plot_training_history(training_history, 'training_history.png', show_plot=False)
    solver.plot_solution('neural_solution.png', show_plot=False)
    
    print("Results saved to: training_history.png, neural_solution.png")
    
    # # 与解析解比较（可选）
    # try:
    #     compare_with_analytical()
    # except ImportError:
    #     print("Analytical solver not available for comparison")


if __name__ == "__main__":
    main()