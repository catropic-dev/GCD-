# GenDis 论文—代码对照（中文解释版）

论文：https://aclanthology.org/2026.acl-long.107.pdf
官方仓库：https://github.com/cx9941/GenDis
固定源码提交：6241621ccc0cda250cac5aea1ec26c4c75271fdc

这是作者公开项目的保存与注释版本，不是根据论文重新实现。页码均从 PDF 第一页开始计数；新增中文注释不改变执行逻辑。

| 文件 | 论文出处 | 在方法中的作用 |
|---|---|---|
| [code/main.py](code/main.py) | §4 方法整体，PDF 第4–6页 | `main`：串联模型、数据、训练器及 train/eval/gen 模式；具体阶段由训练器及配置控制。 |
| [code/models/modeling_qwen2.py](code/models/modeling_qwen2.py) | §4.1–4.4，PDF 第4–6页 | `set_class`：在共享 Qwen 主干上增加判别投影、分类层和 CCA 投影；class_mlp 实际为线性层。；`compute_loss`：汇总生成、已知类分类、判别伪标签、生成伪标签和 CCA 损失；是否启用取决于配置权重。 |
| [code/models/cca_model.py](code/models/cca_model.py) | §4.3、附录 B.4，PDF 第5、13页 | `CCAProjection`：将两路表示投影至 k 维空间，并计算跨视图相关性约束。；`forward`：返回两路投影及拼接表示；实际实现使用 concat。；`calculate_loss`：计算正则化协方差及相关性损失；不同 mode 对应不同实现分支。 |
| [code/trainer.py](code/trainer.py) | §4.3–4.4，PDF 第4–6页 | `get_semi_train_dataloader`：提取表示、构造伪标签并更新半监督训练集；包含课程筛选及生成标签过滤。；`adjust_logits`：每个类别按得分选择 top-k 样本；未选中样本伪标签置为 -1。；`generate`：为输入生成多个候选语义标签，供生成伪监督及不确定性评估。；`uncertain_quantify`：收集候选生成序列的 token 概率，计算 PPL 与 Wasserstein 离散度。 |
| [code/utils/utils.py](code/utils/utils.py) | §4.2、§4.4，PDF 第4、5–6页 | `generate_prompt`：把文本及标签组织成分类对话，提供生成训练输入。；`compute_wasserstein_distance`：以 token 概率向量间欧氏距离为运输代价，用 POT 的 emd2 计算最优运输距离。；`compute_uncertainty`：对候选两两计算平均 Wasserstein 距离；计算每个候选 PPL，返回的是 log(PPL) 及平均距离。 |
| [code/utils/metric.py](code/utils/metric.py) | §5 及附录 B.3，PDF 第6页起、第12–13页 | `compute_metrics`：对 class/mlp/com 表示分别执行 K-means 与 Sinkhorn 归一化，计算聚类指标；Hungarian 用于标签对齐。 |
| [code/models/sinkhorn_knopp.py](code/models/sinkhorn_knopp.py) | 附录 B.3，PDF 第12–13页 | 相关模块实现；结合调用位置及配置阅读。 |
| [code/trainer_callback.py](code/trainer_callback.py) | §4.3–4.4 迭代更新，PDF 第4–6页 | 相关模块实现；结合调用位置及配置阅读。 |
| [code/init_parameters.py](code/init_parameters.py) | 实验配置入口；具体值以脚本和参数为准 | 相关模块实现；结合调用位置及配置阅读。 |

## 从输入到输出

1. `data/step0-process.ipynb`、`step1-data_split.ipynb` 预处理数据并划分已知类、未知类和有标注子集。
2. `code/utils/utils.py` 构造对话和模型；`modeling_qwen2.py` 在共享 LLM 上增加判别分支。
3. §4.2：生成目标及已知类判别目标初始化模型。
4. §4.3：训练器构建判别伪标签，通过课程策略选择样本；CCA 损失约束跨视图表示。
5. §4.4：生成多个候选标签，用 PPL 与 Wasserstein 离散度过滤后加入联合训练。
6. `metric.py` 对不同视图表示执行聚类、Sinkhorn 及指标评估。

## 读代码时需要留意

- 上游 README 的论文链接仍是占位符，引用作者也与正式 PDF 不完全一致，正式出处以本文件链接的 PDF 为准。
- `CCAProjection.forward` 实际拼接两路投影；上游 docstring 中的“平均”描述不准确。
- `compute_uncertainty` 最终返回 log(PPL)，变量名不能直接等同论文中的 PPL 数值。
- `trainer.py` 当前生成标签过滤采用 Wasserstein < 0.01 的实现条件；不能把每个配置分支都当作论文默认实验。
- 官方脚本还列出 `ecdt`，当前仓库只提供论文使用的五个数据集；整批脚本运行前需调整该列表。
- 本次检查为 Python 语法与程序结构校验，没有运行模型训练或复现论文分数。运行需要另备模型权重、兼容依赖及 GPU。
- 保留完整 MIT LICENSE 和版权。数据沿用上游文件，使用和再分发时还应遵守各数据集原始条款。
