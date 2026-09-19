# 术语与符号表

| 规范用语或符号 | 首次定义 | 禁用或需避免的变体 | 采用原则 |
|---|---|---|---|
| Hadamard 有限部分积分 | Hadamard finite-part integral | “普通积分”“超奇异积分”混用而不说明有限部分意义 | 中文首次写全称，后文可简称“有限部分积分” |
| 一般整数阶有限部分算子 \(I_m\) | \(I_m[u](t)=\operatorname{f.p.}\int_a^b u(\tau)(\tau-t)^{-m}\,\mathrm d\tau\) | 把 \(m=1\) 也称为 Hadamard 超奇异积分 | \(m=1\) 为 Cauchy 主值，\(m\ge2\) 为 Hadamard 有限部分；统一记 \(I_m\) |
| Taylor 减奇异表示 | Taylor-subtraction representation | “Taylor 公式”“修正公式” | 强调它是经典恒等式，不把公式本身称为新方法 |
| 近对角差商 \(Q_m\) | Taylor 余项除以 \((\tau-t)^m\) | “积分核误差”与“差商误差”混用 | 差商、算子、方程三个层级分开报告 |
| 局部 Taylor 延拓 | local Taylor continuation | “局部外推”“补丁公式” | 近对角区域用 \(p\) 阶 jet 代替直接差商 |
| 延拓阶数 \(p\) | 局部 Taylor 延拓保留到 \((\tau-t)^p\) | 与奇异阶数 \(m\) 混用 | 固定符号 \(p\) |
| 误差平衡切换尺度 \(\delta_*\) | \(cL\varepsilon_{\rm mach}^{1/(m+p+1)}\) | “最优阈值”“理论最优阈值” | 在未证明最优性前只称平衡尺度或启发式尺度 |
| 比例因子 \(c\) | 吸收函数、导数、求积和实现常数的无量纲系数 | “经验常数”但不说明作用 | 默认 \(c=1\)，用消融说明鲁棒区间 |
| 相消控制 Taylor 算子 | cancellation-controlled Taylor operator | “修正版公式”“正确公式” | 方法简称；强调控制浮点相消而非修改数学恒等式 |
| 神经配置法 | neural collocation method | PINN、ANN、神经求解器无规则交替 | 中文正文统一“神经配置法”；首次说明使用平滑前馈网络和自动微分 |
| 配置点 | collocation points | “训练点”“采样点”随意替换 | 训练方程残差的位置称配置点 |
| 求积点 | quadrature nodes | “配置点” | 积分离散节点单独称求积点 |
| 解 RMSE | root-mean-square error of the solution | “全区间残差” | 明确评价区间；闭区间误差与内点方程残差分开 |
| 方程残差 RMSE | root-mean-square equation residual | “损失”“归一化损失”直接等同 | 必须用独立求积和内点评价，并说明是否归一化 |
| direct | 仅精确对角点使用连续极限，其余直接计算差商 | “原始公式” | 作为故意暴露相消的数值基线 |
| fixed | 使用固定距离阈值切换 | “经验最优” | 每个 dtype 报告所用阈值 |
| balanced | 使用 \(\delta_*\) 自动切换 | “自适应最优” | 不暗示严格最优性 |

## 一句话论点

在一般整数阶 Hadamard 有限部分算子的浮点求积中，本文以阶数、局部延拓阶数和机器精度共同决定的近对角切换尺度抑制 Taylor 余项差商的灾难性相消，并通过算子实验、神经配置制造方程和匹配谱基线界定其精度收益与高阶自动微分成本。
