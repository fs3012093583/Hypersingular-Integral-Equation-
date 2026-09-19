# 论文初稿与复现实验

主文档为 `main.tex`，图表由结构化实验结果自动生成。

## 参考值审计后的算子实验

以下命令从项目根目录运行。参考值按各浮点格式实际存储的坐标计算，在高精度中完成误差相减；旧结果目录保留，不能与 v2 参考链混用。

```bash
PYTHONPATH=. python -m experiments.general_taylor_operator_benchmark \
  --preset full --mp-dps 90 --no-plots \
  --output-dir results/general_taylor_operator_benchmark_full_dtype_ref_v2

python -m experiments.general_taylor_multiplier_sweep \
  --reference-metrics results/general_taylor_operator_benchmark_full_dtype_ref_v2/metrics.json \
  --output-dir results/general_taylor_multiplier_sweep_dtype_ref_v2

python -m experiments.general_taylor_component_ablation \
  --source results/general_taylor_operator_benchmark_full_dtype_ref_v2/metrics.json \
  --output-dir results/general_taylor_component_ablation_dtype_ref_v2

python -m experiments.general_taylor_adaptive_order_benchmark \
  --reference-metrics results/general_taylor_operator_benchmark_full_dtype_ref_v2/metrics.json \
  --output-dir results/general_taylor_adaptive_order_dtype_ref_v2

python -m experiments.operator_budget_accuracy_benchmark \
  --quadrature-points 256 --mp-dps 90 --residual-budget 1e-4 \
  --output-dir results/operator_budget_accuracy_v2_20260919

python -m experiments.neural_high_order_derivative_benchmark \
  --highest-order 10 --decimal-digits 100 \
  --output-dir results/neural_high_order_derivatives_ref_v2

python -m paper.build_operator_revision_assets
python -m paper.build_derivative_audit_figure
```

历史神经主实验、非线性和函数族压力试验的配置保存在各结果记录中。旧 `build_paper_figures` 仅用于历史版本，不用于当前精度图表。

## 固定阶数按需实现的配对复核

以下入口保留原固定展开，另加公式与阈值相同的按需固定展开，并与算子预算比较。所有方法串行运行，固定单 CPU 线程，按种子轮换方法顺序。输出目录必须不存在，以保护历史结果。

```bash
python -m experiments.lazy_fixed_quality_benchmark \
  --task single --orders 4 6 --seeds 20260918 20260919 20260920 \
  --derivative-metrics results/neural_high_order_derivatives_ref_v2/metrics.json \
  --output-dir results/lazy_fixed_quality_single_calibration_v2_20260919
python -m experiments.lazy_fixed_quality_benchmark \
  --task parameterized --orders 4 --seeds 20260918 20260919 20260920 \
  --output-dir results/lazy_fixed_quality_parameterized_calibration_v2_20260919
```

两条命令必须依次运行，测时期间不运行其他数值任务。2026-09-19 的 v2 批次已完成，包含 18 条单函数及 9 条参数化记录；实际源码、校准文件与归档 SHA-256 全部核对一致。旧配对结果及其源码归档保留，但不与新批次混用耗时。正文主对照已全部更新到 v2；历史单种子分支消融仍明确标识早期校准。

`method_setup_and_training_seconds` 包含初始导数包络及训练期更新；参数化实验还包含每次运行的高精度校准。单函数实验复用既有自动微分校准文件，其历史生成成本未包含在内，协议中保留该文件的 SHA-256。`solve_seconds_before_validation` 另包含模型、节点和右端构造，排除检验与单次反向成本探针。保存张量载荷不是硬件峰值内存。

算子误差应区分差商替换、求积离散、解析矩以及累加舍入；命题中的加权差商指标仅控制第一项。三种子结果不能视为显著性证明或跨训练轨迹的误差认证。

## 表格与编译

表格从指定的完整结果生成，不从控制台抄录。生成器校验每组的三个不同种子、配对初值、方法间及种子间除指定变化外的全部配置：

```bash
python -m paper.build_quality_revision_tables \
  --single results/lazy_fixed_quality_single_calibration_v2_20260919/metrics.json \
  --parameterized results/lazy_fixed_quality_parameterized_calibration_v2_20260919/metrics.json \
  --operator results/operator_budget_accuracy_v2_20260919/metrics.json \
  --output-dir paper/tables_quality
```

编译应在 `paper` 目录下执行，使图表相对路径正确：

```bash
cd paper
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=../output/pdf/quality_revision_calibration_v2_20260919 main.tex
```

该目录是独立修订工作稿，不覆盖原 PDF。当前目标为理论、实验与论证质量；期刊格式及投稿操作不在本阶段范围内。下一轮首先检验训练轨迹与梯度可靠性，具体范围和失败判据见 `../reports/training_reliability_audit_plan.md`；该协议不是已完成的实验结果。
