import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class NeuralSolverLN2Approx(nn.Module):
    """
    基于ln(2)近似的神经网络超奇异积分方程求解器
    近似：ln((1+x)/(1-x)) ≈ ln(2)
    """
    def __init__(self, hidden_dim=64):
        super(NeuralSolverLN2Approx, self).__init__()
        # 神经网络架构
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # 预计算ln(2)
        self.ln2 = torch.log(torch.tensor(2.0))
    
    def forward(self, x):
        """前向传播"""
        return self.net(x)
    
    def compute_integral_loss(self, x_collo, nodes, weights, eps=1e-4):
        """
        计算积分损失函数
        使用ln(2)近似ln((1+x)/(1-x))
        """
        x_collo.requires_grad_(True)
        u_x = self.forward(x_collo)
        
        # 计算u'(x)用于奇点逼近
        u_prime = torch.autograd.grad(
            u_x, x_collo, 
            grad_outputs=torch.ones_like(u_x), 
            create_graph=True
        )[0]
        
        # 建立积分矩阵
        t = nodes  # 形状 (M, 1)
        w = weights  # 形状 (M, 1)
        u_t = self.forward(t)  # 形状 (M, 1)
        
        # 计算被积函数: (u(t) - u(x)) / |x - t|
        diff_u = u_t - x_collo.T  # 形状 (M, N)
        diff_dist = torch.abs(t - x_collo.T)  # 形状 (M, N)
        
        # 正则化处理：在|x-t| < eps时使用导数项
        # 计算符号函数 sign(t - x)
        sign_term = torch.sign(t - x_collo.T)  # 形状 (M, N)
        approx_term = u_prime.T * sign_term  # 形状 (1, N) -> 广播到 (M, N)
        
        integrand = torch.where(
            diff_dist > eps, 
            diff_u / diff_dist, 
            approx_term
        )
        
        # 数值积分部分 I1
        I1 = torch.sum(integrand * w, dim=0, keepdim=True).T  # 形状 (N, 1)
        
        # 只在x接近±1时使用ln(2)近似
        # 正常情况下使用完整的对数计算
        log_term = torch.log((1 + x_collo + 1e-7) / (1 - x_collo + 1e-7))
        
        # 当x接近±1时使用ln(2)近似
        # 定义接近边界的阈值
        threshold = 0.95
        is_near_boundary = (torch.abs(x_collo) > threshold)
        
        # 混合计算：边界附近使用ln(2)，其他区域使用正常对数
        analytical_term = torch.where(
            is_near_boundary,
            u_x * self.ln2,
            u_x * log_term
        )
        
        # 总算子 L[u](x)
        L_u = I1 + analytical_term
        
        # 目标值 f(x) = -1
        target = -torch.ones_like(L_u)
        return torch.mean((L_u - target)**2)

def get_gauss_quadrature(n=150, device='cpu'):
    """获取高斯-勒让德积分点和权重"""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes_t = torch.tensor(nodes, dtype=torch.float32).to(device).view(-1, 1)
    weights_t = torch.tensor(weights, dtype=torch.float32).to(device).view(-1, 1)
    return nodes_t, weights_t

def train_model():
    """训练模型"""
    # 设备选择
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"使用设备: {device}")
    
    # 初始化模型和优化器
    model = NeuralSolverLN2Approx(hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 预准备高斯点
    nodes, weights = get_gauss_quadrature(n=150, device=device)
    
    print("开始训练...")
    print("使用ln(2)近似ln((1+x)/(1-x))")
    
    # 训练循环
    for epoch in range(2001):
        optimizer.zero_grad()
        
        # 随机采样训练点
        x_train = (torch.rand(200, 1, device=device) * 1.98) - 0.99
        
        # 计算损失
        loss = model.compute_integral_loss(x_train, nodes, weights)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 打印训练进度
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return model, device

def visualize_results(model, device):
    """可视化结果"""
    with torch.no_grad():
        x_test = torch.linspace(-0.99, 0.99, 100).view(-1, 1).to(device)
        y_pred = model(x_test).cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_test.cpu().numpy(), y_pred, 'b-', label='神经网络解 (ln(2)近似)')
        plt.title('超奇异积分方程解 (使用ln(2)近似)')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # 训练模型
    model, device = train_model()
    
    # 可视化结果
    visualize_results(model, device)
