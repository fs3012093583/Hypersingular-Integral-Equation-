# 超奇异积分方程神经求解方法的创新性与可行性论证

## 1. 研究定位

现有工作已经证明，前馈神经网络可以用于求解带 Hadamard 有限部分核的线性和非线性积分方程。因此，论文不能再以“首次使用神经网络求解超奇异积分方程”为创新点。Taylor 减奇异恒等式、分部积分公式以及一般阶有限部分定义也属于经典结果，不能把公式本身作为新理论。

目前最有希望形成独立贡献的方向是：

> 构造一种面向一般整数阶 Hadamard 有限部分算子的相消控制 Taylor 神经配置方法，根据奇异阶数、局部延拓阶数和浮点精度自动确定近对角切换尺度，使高阶 Taylor 余项差商在神经网络训练和高阶求积中保持稳定。

该方向研究的不是 Taylor 公式是否成立，而是经典恒等式进入自动微分和浮点求积以后怎样稳定计算。已有的二阶专用方法不能自然解决高阶差商的严重相消问题，这构成了本文可以进入的技术空白。

## 2. 一般阶正则化算子

考虑

\[
I_m[u](t)=\operatorname{f.p.}\!\int_a^b
\frac{u(\tau)}{(\tau-t)^m}\,\mathrm d\tau,
\qquad m\in\mathbb N^+,
\quad a<t<b.
\]

在 \(t\) 处减去 \(m-1\) 阶 Taylor 多项式：

\[
P_{m-1}(\tau;t)
=
\sum_{k=0}^{m-1}
\frac{u^{(k)}(t)}{k!}(\tau-t)^k.
\]

有限部分积分可表示为

\[
\begin{aligned}
I_m[u](t)
={}&
\sum_{k=0}^{m-2}
\frac{u^{(k)}(t)}{k!}
\frac{(a-t)^{-(m-k-1)}-(b-t)^{-(m-k-1)}}{m-k-1}
\\
&+
\frac{u^{(m-1)}(t)}{(m-1)!}
\log\left|\frac{b-t}{a-t}\right|
\\
&+
\int_a^b
\frac{u(\tau)-P_{m-1}(\tau;t)}{(\tau-t)^m}
\,\mathrm d\tau .
\end{aligned}
\]

正则积分中的差商记为

\[
Q_m[u](t,\tau)
=
\frac{u(\tau)-P_{m-1}(\tau;t)}{(\tau-t)^m}.
\]

连续意义下有

\[
\lim_{\tau\to t}Q_m[u](t,\tau)
=\frac{u^{(m)}(t)}{m!}.
\]

但是，在浮点计算中，分子由多个大小接近的量相减得到。当 \(|\tau-t|\) 较小时，直接计算的舍入误差近似按

\[
E_{\mathrm{round}}
\sim
\frac{\varepsilon_{\mathrm{mach}}}{|\tau-t|^m}
\]

增长。因此，阶数越高，直接 Taylor 差商越容易失去有效数字。

## 3. 相消控制局部延拓

在近对角区域，用下面的局部展开代替直接差商：

\[
Q_{m,p}^{\mathrm{loc}}[u](t,\tau)
=
\sum_{j=0}^{p}
\frac{u^{(m+j)}(t)}{(m+j)!}
(\tau-t)^j.
\]

其中 \(p\) 为局部延拓阶数。相应的截断误差满足

\[
E_{\mathrm{local}}=O\!\left(|\tau-t|^{p+1}\right).
\]

平衡舍入误差和局部截断误差：

\[
\frac{\varepsilon_{\mathrm{mach}}}{\delta^m}
\sim
\delta^{p+1},
\]

得到切换尺度

\[
\delta_*
\sim
L\,\varepsilon_{\mathrm{mach}}^{1/(m+p+1)},
\]

其中 \(L=b-a\) 为区间尺度。因此定义

\[
\widetilde Q_{m,p}[u](t,\tau)
=
\begin{cases}
Q_m[u](t,\tau),
&|\tau-t|>\delta_*,\\[1ex]
Q_{m,p}^{\mathrm{loc}}[u](t,\tau),
&|\tau-t|\leq\delta_*.
\end{cases}
\]

与固定经验阈值相比，\(\delta_*\) 随奇异阶数、局部延拓阶数和浮点精度自动变化。该机制既适用于解析函数，也适用于由平滑神经网络表示的未知函数；Taylor 系数由自动微分生成。

## 4. 高精度算子实验

采用非多项式函数

\[
u(t)=e^t,
\qquad -1\leq t\leq1,
\]

以 80 位高精度计算作为参考，对 \(m=1,\ldots,6\) 进行验证。普通积分采用 2048 点 Gauss--Legendre 求积。稳定方法采用二阶局部延拓 \(p=2\)。

| 奇异阶数 \(m\) | 直接差商绝对误差 | 相消控制绝对误差 | 误差改善倍数 |
|---:|---:|---:|---:|
| 1 | \(4.35\times10^{-14}\) | \(4.35\times10^{-14}\) | \(1.0\) |
| 2 | \(1.14\times10^{-13}\) | \(4.25\times10^{-14}\) | \(2.7\) |
| 3 | \(2.32\times10^{-10}\) | \(4.05\times10^{-13}\) | \(5.7\times10^2\) |
| 4 | \(2.59\times10^{-7}\) | \(4.46\times10^{-12}\) | \(5.8\times10^4\) |
| 5 | \(4.12\times10^{-4}\) | \(5.10\times10^{-11}\) | \(8.1\times10^6\) |
| 6 | \(4.52\times10^{-1}\) | \(2.11\times10^{-11}\) | \(2.1\times10^{10}\) |

结果显示：

1. 当 \(m=1,2\) 时，直接差商尚未出现严重问题；
2. 从 \(m=3\) 开始，直接差商误差随阶数迅速放大；
3. 到 \(m=6\) 时，直接方法已经失去实际计算价值；
4. 相消控制方法在六阶算子上仍保持约 \(10^{-11}\) 的绝对误差；
5. 改进幅度随阶数单调扩大，符合舍入误差模型的预期。

另外，在局部差商实验中，双精度直接计算的最大相对误差从 \(m=2\) 的 \(1.28\times10^{12}\) 増长到 \(m=6\) 的 \(4.63\times10^{70}\)；相消控制方法在相同测试点上的最大相对误差均未超过 \(9.73\times10^{-9}\)。单精度实验也表现出相同趋势。

这些结果初步证明，相消控制不是一般性的数值微调，而是高阶有限部分算子能否可靠计算的关键机制。

## 5. 神经网络求解实验

为了检验稳定算子能否直接用于神经网络训练，构造二类方程

\[
u(t)+\lambda(1-t^2)^{m-1}I_m[u](t)=f_m(t),
\]

取 \(\lambda=0.05\)，并由解析解 \(u(t)=e^t\) 制造右端项。网络采用三层平滑全连接结构；训练使用 64 个配置点和 64 点求积，验证使用独立的 192 点求积。

| 奇异阶数 \(m\) | 解的均方根误差 | 独立方程残差均方根 | 训练时间 |
|---:|---:|---:|---:|
| 2 | \(8.10\times10^{-4}\) | \(1.83\times10^{-4}\) | 6.3 s |
| 3 | \(7.11\times10^{-5}\) | \(2.04\times10^{-4}\) | 14.4 s |
| 4 | \(7.60\times10^{-5}\) | \(1.42\times10^{-4}\) | 37.1 s |

四阶算子需要最高六阶局部导数，但训练仍然保持有限并得到 \(10^{-5}\) 量级的解误差。这说明一般阶正则化算子不仅能准确计算解析函数，也能在反向传播中保持可用。

## 6. 辅助 Cauchy 降阶方案的对照结果

另一种可能的方法是利用

\[
I_m[u](t)
=
\frac{1}{(m-1)!}
\frac{\mathrm d^{m-1}}{\mathrm dt^{m-1}}
\operatorname{PV}\!\int_{-1}^{1}
\frac{u(\tau)}{\tau-t}\,\mathrm d\tau,
\]

引入辅助网络表示一阶 Cauchy 变换。初步实现了两种形式：

1. 一个辅助网络表示 Cauchy 变换，再对它进行高阶自动微分；
2. 显式拆出端点对数项，并以多个辅助输出建立一阶导数链。

两种方法都可以训练，说明数学变形是可行的，但当前解误差约为 \(10^{-2}\) 至 \(10^{-1}\)，显著高于相消控制 Taylor 方法。主要原因是：Cauchy 变换具有端点对数行为；高阶微分会放大辅助函数的逼近误差；多约束联合优化增加了损失平衡难度。

因此，辅助 Cauchy 方法暂不作为论文主方法。它可以保留为后续研究方向或对照实验，但目前没有证据表明它优于直接稳定正则化。

## 7. 可主张的创新内容

在目前证据范围内，论文可以围绕以下贡献展开：

1. 建立适用于任意正整数阶 \(m\) 的可微 Taylor 有限部分算子，将解析奇异矩、自动微分 Taylor 系数和普通求积统一到一个计算框架中；
2. 提出阶数、局部延拓阶数和浮点精度相关的近对角切换尺度，避免使用对所有阶数相同的经验阈值；
3. 通过局部 jet 延拓抑制高阶 Taylor 余项差商的灾难性相消；
4. 分离并报告差商误差、有限部分算子误差、求积误差、训练残差和解误差；
5. 验证方法在一至六阶算子上的数值稳定性，以及在二至四阶方程上的可训练性。

建议把方法称为：

> 相消控制的一般阶 Taylor 有限部分神经配置方法

英文可表述为：

> A cancellation-controlled arbitrary-order Taylor finite-part neural collocation method

## 8. 不能使用的创新表述

以下表述应避免：

- 首次使用神经网络求解超奇异积分方程；
- 提出新的 Hadamard 有限部分公式；
- 首次使用 Taylor 多项式消除超奇异性；
- 神经网络在一维问题上普遍优于经典方法；
- 已经证明任意阶收敛性；
- 已经得到后验误差证书。

较为稳妥的表述是：

> 本文针对一般整数阶 Hadamard 有限部分算子在自动微分和浮点求积中的近对角相消问题，提出阶数相关的局部 Taylor 延拓与误差平衡切换策略，并将其嵌入神经配置求解框架。

## 9. 投稿前必须补充的工作

当前结果证明了创新机制的初步可行性，但还不足以直接投稿。下一阶段至少需要：

1. 对核心算例进行 5 至 10 个随机种子重复；
2. 增加 \(\sin(\pi t)\)、有理函数及带端点非光滑性的解析解；
3. 对配置点、求积点、网络宽度和局部延拓阶数做系统收敛实验；
4. 与专用有限部分求积、Chebyshev 或 Legendre 配置法进行公平比较；
5. 给出近对角截断误差和舍入误差的严格或半严格分析；
6. 研究高阶自动微分的时间和显存复杂度；
7. 检验五阶和六阶神经方程训练，而不仅是算子计算；
8. 增加非线性或参数化方程，以说明神经表示相对于传统单次一维求解的使用价值；
9. 在完整数据库中进一步核对是否已有相同的阶数相关切换策略。

## 10. 初步结论

一般阶 Taylor 恒等式本身不新，但其高阶神经计算存在一个明确且可复现的数值缺口：直接余项差商在高阶情况下会发生随 \(m\) 急剧加重的浮点相消。阶数相关的局部延拓和误差平衡阈值能够把六阶算子的误差从 \(10^{-1}\) 降至 \(10^{-11}\) 量级，并已经支持二至四阶非多项式制造方程的神经训练。

因此，这一方向具备形成论文的可行性。论文的重点应当是“高阶有限部分算子的稳定可微实现”，而不是“神经网络能够拟合某个超奇异积分方程”。

## 参考文献线索

1. S. Zehtabian Rezaei, Y. Mahmoudi and M. Baghmisheh, *Application of feed-forward neural networks for solving linear and nonlinear integral equations with hypersingular kernels*, Physica Scripta 100 (2025) 125215. DOI: <https://doi.org/10.1088/1402-4896/ae239a>.
2. L. Gori, E. Pellegrino and E. Santi, *Numerical evaluation of certain hypersingular integrals using refinable operators*, Mathematics and Computers in Simulation 82 (2011) 132--143. DOI: <https://doi.org/10.1016/j.matcom.2010.07.006>.
3. K.-C. Toh and S. Mukherjee, *Hypersingular and finite part integrals in the boundary element method*, International Journal of Solids and Structures 31 (1994) 2299--2312. DOI: <https://doi.org/10.1016/0020-7683(94)90153-8>.
4. M. Feischl et al., *ReLU Neural Network Galerkin BEM*, Journal of Scientific Computing 95 (2023). DOI: <https://doi.org/10.1007/s10915-023-02120-w>.
5. Y. Yuan, G. Ni, X. Deng and S. Hao, *A-PINN: Auxiliary physics informed neural networks for forward and inverse problems of nonlinear integro-differential equations*, Journal of Computational Physics 462 (2022) 111260. DOI: <https://doi.org/10.1016/j.jcp.2022.111260>.
