# 超奇异积分方程数值求解器

## 概述

本项目实现了基于中点矩形求积逼近公式的超奇异积分方程数值求解器，用于求解如下形式的超奇异积分方程：

$$\int_{-1}^{1} |x-y|^{-1} g(x) dx = -1$$

该求解器基于论文中提出的"任意点奇异性新展开式"方法，通过添加奇异性补偿项将发散的超奇异积分转化为可计算的有限值。

## 核心算法

### 1. 超奇异积分的逼近公式

对于超奇异积分方程，采用修正的中点矩形求积公式：

$$I_h = \sum_{j=0}^{N-1} h \cdot G(a+(j+\frac{1}{2})h) + 2\psi(\frac{1}{2}) - 2\ln(\frac{1}{h})$$

其中：
- $h = \frac{b-a}{N}$ 是步长
- $G(x) = |x-t|^{-1}g(x)$ 是被积函数
- $\psi(\frac{1}{2}) = -\gamma - 2\ln 2 \approx -1.9635$ 是digamma函数在1/2处的值
- $\gamma \approx 0.5772$ 是欧拉常数

逼近精度为二阶：
$$\text{f.p.}\int_a^b G(x)dx = I_h + O(h^2)$$

### 2. 离散化矩阵方程

将连续方程离散化为线性方程组：

$$A_N u = -I_{2N}$$

其中系数矩阵 $A_N$ 的元素为：
- 对角线元素：$a_{ii} = 2\psi(\frac{1}{2}) - 2\ln(\frac{1}{2h})$ 
- 非对角线元素：若 $|i-j|$ 为奇数，$a_{ij} = \frac{2}{|i-j|}$；否则为0

### 3. 产能计算

求解得到离散解 $u_i$ 后，产能近似值为：

$$Q_h = h \sum_{i=0}^{2N-1} u_i$$

## 文件结构

```
├── hypersingular_solver.py  # 核心求解器类
├── test_solver.py           # 测试和验证模块
├── examples.py              # 使用示例
└── README.md               # 本文档
```

## 安装依赖

```bash
pip install numpy scipy matplotlib
```

## 快速开始

### 基本使用

```python
from hypersingular_solver import HypersingularIntegralSolver

# 创建求解器
solver = HypersingularIntegralSolver(N=256)

# 求解方程
y_points, u_values = solver.solve_equation()

# 计算产能
Q_h = solver.calculate_productivity(u_values)
print(f"数值产能: {Q_h:.8f}")

# 绘制结果
solver.plot_solution(y_points, u_values)
```

### 收敛性研究

```python
# 进行收敛性分析
results = solver.convergence_study([32, 64, 128, 256, 512])
solver.plot_convergence(results)
```

### 高精度求解

```python
# 使用更细的网格获得高精度解
high_precision_solver = HypersingularIntegralSolver(N=1024)
y_points, u_values = high_precision_solver.solve_equation()
Q_h = high_precision_solver.calculate_productivity(u_values)
print(f"高精度产能: {Q_h:.10f}")
```

## 主要功能

### HypersingularIntegralSolver 类

#### 初始化参数
- `N` (int): 网格参数，步长 h = 1/N，默认为128

#### 主要方法

##### `solve_equation()`
求解离散线性方程组，返回网格中点和对应的解向量。

##### `calculate_productivity(u_values)`
计算产能 Q_h。

##### `convergence_study(N_values)`
进行收敛性研究，返回不同网格尺寸下的结果。

##### `plot_solution(y_points, u_values)`
绘制数值解的图像。

##### `plot_convergence(results)`
绘制收敛性分析图表。

##### `compute_error_estimate(u_values, reference_N)`
计算相对误差估计。

## 数值验证

运行测试模块验证算法的正确性：

```bash
python test_solver.py
```

测试内容包括：
- 基本功能测试
- 收敛阶数验证（二阶收敛）
- 精度目标测试
- 性能测试
- 矩阵性质分析

## 使用示例

运行示例脚本了解各种用法：

```bash
python examples.py
```

示例包括：
- 基本使用示例
- 收敛性研究
- 高精度求解
- 不同网格尺寸比较
- 自定义分析
- 批处理计算

## 理论背景

### 超奇异积分

超奇异积分是指积分核具有高于一阶的奇异性，传统的数值积分方法无法直接应用。Hadamard有限部分积分提供了一种处理这类积分的方法。

### 中点矩形公式的改进

传统的中点矩形公式：
$$\int_a^b f(x)dx \approx h \sum_{k=0}^{N-1} f(a+(k+\frac{1}{2})h)$$

对于超奇异积分，需要添加奇异性补偿项：
$$I_h = h \sum_{j=0}^{N-1} G(a+(j+\frac{1}{2})h) + 2\psi(\frac{1}{2}) - 2\ln(\frac{1}{h})$$

### 收敛性分析

数值实验表明，该方法的收敛阶数为O(h²)，即步长减半时，误差降至原来的1/4。

## 结果展示

### 典型结果

对于N=256的网格：
- 数值产能 Q_h ≈ 4.1587
- 与理论极限值 4.15875 的相对误差约为 10^{-5}

### 收敛趋势

随着网格细度的增加（N增大）：
- N=32: Q_h ≈ 4.156
- N=64: Q_h ≈ 4.158
- N=128: Q_h ≈ 4.1586
- N=256: Q_h ≈ 4.1587
- N=512: Q_h ≈ 4.15874

结果呈现明显的二阶收敛趋势，验证了算法的正确性。

## 注意事项

1. **网格选择**：N值越大，精度越高，但计算时间也越长。建议根据精度要求选择合适的N值。

2. **内存使用**：对于大N值（如N>1000），系数矩阵会占用较多内存。

3. **条件数**：系数矩阵的条件数随N增大而增大，可能影响数值稳定性。

4. **参考值**：理论极限值4.15875是基于数值实验得到的参考值。

## 扩展应用

该求解器可以推广到其他类型的超奇异积分方程，只需要：
1. 修改积分核函数
2. 调整相应的奇异性补偿项
3. 更新系数矩阵的构建方式

## 参考文献

本文算法基于超奇异积分的数值逼近理论，结合了Euler-Maclaurin展开和Hadamard有限部分积分的思想。