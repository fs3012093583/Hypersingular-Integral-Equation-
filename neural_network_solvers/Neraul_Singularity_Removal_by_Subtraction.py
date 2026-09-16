from tkinter import Y
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 定义神经网络 (PINN)
class PINNSolver(nn.Module):
    def __init__(self):
        super(PINNSolver, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # 1. 强制对称性: 使用 x^2   
        # x_input = x**2 
        # raw_u = self.net(x_input)       
        
        # 2. 注入物理先验: 假设解在边界处有根号奇异性 (电化学中常见)
        # 这种预设能抵消解析项中的 log 爆炸，让 NN 只负责平滑部分
        # weight = torch.sqrt(torch.clamp(1 - x**2, min=1e-4))
        y = self.net(x)
        return y

# 2. 生成高斯积分点和权重
def get_gauss_points(n=128, device='cpu'):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes_t = torch.tensor(nodes, dtype=torch.float32).to(device).view(-1, 1)   # (M, 1)
    weights_t = torch.tensor(weights, dtype=torch.float32).to(device).view(-1, 1) # (M, 1)
    return nodes_t, weights_t

# 3. 核心损失函数：显式极限值填充版
def singular_loss(model, x_collo, nodes, weights, eps=1e-1):
    """ 
    x_collo: (N, 1), nodes: (M, 1)
    """
    x_collo.requires_grad_(True)
    u_x = model(x_collo)  # (N, 1)
    
    # --- A. 获取 u'(x) 用于奇异区填充 ---
    u_prime = torch.autograd.grad(
        u_x, x_collo, 
        grad_outputs=torch.ones_like(u_x), 
        create_graph=True,   # 必须为 True，否则无法反向传播给网络参数
        retain_graph=True
    )[0]
    
    # --- B. 准备广播矩阵 ---
    u_t = model(nodes).view(1, -1)      # (1, M)
    t_row = nodes.view(1, -1)            # (1, M)
    
    diff_u = u_t - u_x                  # (N, M)
    diff_dist = torch.abs(x_collo - t_row) # (N, M)
    
    # --- C. 构造被积函数 ---
    # 常规计算区
    regular_val = diff_u / (diff_dist )
    # 奇异填充区: t > x 取 u'(x), t < x 取 -u'(x)
    # sign(t - x) 自动处理了左右极限的符号
    limit_val = u_prime * torch.sign(t_row - x_collo)
    
    # 核心逻辑：小于 eps 用极限值，大于 eps 用常规值
    integrand = torch.where(diff_dist > eps, regular_val, limit_val)

    # --- D. 计算积分 I1 ---
    I1 = torch.matmul(integrand, weights)  # (N, 1)
    
    # --- E. 解析项处理 (增加数值稳定性保护) ---
    dist_to_1 = torch.clamp(1 - x_collo, min=1e-2)
    dist_to_neg1 = torch.clamp(1 + x_collo, min=1e-2)
    analytical_term = u_x * torch.log(dist_to_neg1 / dist_to_1)
    
    # --- F. 总 Loss ---
    L_u = I1 + analytical_term
    target = -1.0
    
    # 额外增加边界约束防止 u(x) 在两端爆炸导致 nan
    boundary_pts = torch.tensor([[-1.0], [1.0]], device=x_collo.device)
    loss_boundary = torch.mean(model(boundary_pts)**2)
    
    loss_pde = torch.mean((L_u - target)**10)
    return loss_pde 
    # + 0.1 * loss_boundary

# 4. 主训练流程
def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PINNSolver().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    nodes, weights = get_gauss_points(n=18, device=device)
    
    print("Training starts...")
    for epoch in range(2001):
        model.train()
        optimizer.zero_grad()
        
        # 采样训练点 x: 稍微避开最极端的端点 ([-0.99, 0.99])
        x_train = (torch.rand(128, 1, device=device) * 1.8) - 0.9
        
        loss = singular_loss(model, x_train, nodes, weights)
        
        if torch.isnan(loss):
            print(f"NaN detected at epoch {epoch}, stopping...")
            break
            
        loss.backward()
        
        # 梯度裁剪：防止梯度爆炸导致 nan
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.8e}")

    # 5. 可视化
    model.eval()
    with torch.no_grad():
        x_plot = torch.linspace(-0.99, 0.99, 200).view(-1, 1).to(device)
        u_plot = model(x_plot).cpu().numpy()
        x_axis = x_plot.cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, u_plot, label='PINN Solution (Limit Fill)', color='darkred', lw=2)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.title("Neural Solution for Singular Integral Equation\nWith $\pm u'(x)$ Limit Filling")
        plt.xlabel("t")
        plt.ylabel("u(t)")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    train()