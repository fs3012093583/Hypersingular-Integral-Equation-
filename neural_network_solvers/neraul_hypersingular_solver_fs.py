'''
Author: Fan Shun
Email: fs3012093583@gmail.com
Date: 2026-03-13 14:51:18
LastEditTime: 2026-03-14 02:10:52
LastEditors: Fan Shun
Description: 
'''


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
# matplotlib.rcParams['axes.unicode_minus'] = False

class HypersingularIntegralSolver:
    """
    超奇异积分方程求解器
    求解方程：∫_{-1}^{1} |x-y|^{-1} g(x) dx = -1
    基于中点矩形求积逼近公式
    """
    
    def __init__(self, N: int = 16, hidden_dim: int = 64, neural_network: Optional[nn.Sequential] = None):
        """
        初始化求解器
    
        Args:
            N: 网格参数，步长 h = 1/N 分段数2N 
            hidden_dim: 隐藏层维度
            neural_network: 自定义神经网络（可选）
        """
        self.N = N
        self.h = 1.0 / N
        self.gamma = 0.5772156649015329  # 欧拉常数
        self.psi_half = -self.gamma - 2 * np.log(2)  # ψ(1/2) ≈ -1.9635
        
        if neural_network is None:
            self.neural_network = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.neural_network = neural_network
        # self.a = -1
        # self.b = 1

        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=0.001)
        self.loss_history = []

    def generate_grids_points_y(self) -> np.ndarray :
         """
         生成网格中点(y)
         return_shape :2N
         """

         return np.array([-1 + (i + 0.5) * self.h for i in range(2 * self.N)])
    

    def generate_grids_points_x(self) -> np.ndarray :
         """生成网格点(x) or (t)
         return_shape :2N"""
         return np.array([-1 + i * self.h for i in range(2 * self.N)])
    

    def compute_hypersingular_integral(self, x_points: torch.Tensor, y_points: torch.Tensor) -> torch.Tensor:
        """
        计算超奇异积分 
        Args:
            x_points: 积分点 [2N, 1]
            g_values: g(x)值 [2N, 1]
            
        Returns:
            integral_values: 积分值 [batch_size, 1]
        """
        
        # 积分值向量
        integral_values = torch.zeros(self.N*2, 1, dtype=torch.float32)
        
        for i in range(2*self.N):

            # 每一次t = x[i]
            x_i = x_points[i, 0].item()
            # 计算G(y) = |y-x_i|^{-1} * g(y)
            g_y = self.neural_network(y_points)
            # 计算距离
            distances = np.abs(y_points - x_i)
            
            # 使用Hadamard有限部分积分的离散形式
            G_values = g_y/distances.reshape(-1, 1)

            integral_values[i, 0] = torch.sum(G_values)* self.h-2 * np.log(1.0 / self.h)  + 2 * self.psi_half  
           
            
            # 添加奇异性补偿项
        
            
        return integral_values
   
    def compute_loss(self, x_points: torch.Tensor, y_points: torch.Tensor) -> torch.Tensor:
        # 对每个y_i计算积分值
        integral_values = self.compute_hypersingular_integral(x_points, y_points)
        # 目标方程：积分值 = -1
        loss = torch.mean((integral_values + 1.0) ** 2)
        return loss


    def train(self, epochs=1000):
        
        x_points = self.generate_grids_points_x()
        x_tensor = torch.tensor(x_points.reshape(-1, 1), dtype=torch.float32)
        y_points = self.generate_grids_points_y()
        y_tensor = torch.tensor(y_points.reshape(-1, 1), dtype=torch.float32)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = self.compute_loss(x_tensor, y_tensor)
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())
            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    def get_solution(self, x_points: np.ndarray) -> np.ndarray:
        """获取神经网络预测的解g(x)"""
        self.neural_network.eval()
        x_tensor = torch.tensor(x_points.reshape(-1, 1), dtype=torch.float32)
        with torch.no_grad():
            g_values = self.neural_network(x_tensor)
        return g_values.numpy().reshape(-1)
    
    def plot_solution(self, x_points: np.ndarray, y_points: np.ndarray):
        """
        绘制神经网络解的可视化图
        """
        self.neural_network.eval()
        
        # 将numpy数组转换为tensor
        y_tensor = torch.tensor(y_points.reshape(-1, 1), dtype=torch.float32)
        
        with torch.no_grad():
            u_pred = self.neural_network(y_tensor).numpy()
        
        plt.figure(figsize=(10, 6))
        plt.plot(y_points, u_pred, label='神经网络解')
        plt.plot(y_points, -1.0 * np.ones_like(u_pred), '--', label='目标解')
        plt.xlabel('y')
        plt.ylabel('u(y)')
        plt.title('超奇异积分方程解的可视化')
        plt.legend()
        plt.show()




if __name__ == "__main__":
    
    hidden_dim = 64
    fc_model = nn.Sequential(
        nn.Linear(1, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )

    solver = HypersingularIntegralSolver(N=32, neural_network=fc_model)
    solver.train(epochs=1000)
    # plt.plot(solver.loss_history)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.show()
    
    # 绘制解的可视化图
    x_points = solver.generate_grids_points_x()
    y_points = solver.generate_grids_points_y()
    solver.plot_solution(x_points, y_points)