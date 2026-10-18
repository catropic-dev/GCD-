# GCD 论文代码保存

## GenDis（ACL 2026）

论文：[GenDis: Generative-Discriminative Dual-View Co-Training for Generalized Category Discovery](https://aclanthology.org/2026.acl-long.107/)

官方源码：https://github.com/cx9941/GenDis

保存的上游提交：`6241621ccc0cda250cac5aea1ec26c4c75271fdc`

- [中文注释版](GenDis-annotated/)：完整项目，核心函数添加中文论文出处及用途注释；[论文与代码对照](GenDis-annotated/PAPER_CODE_MAP.md)。
- [无注释版](GenDis-no-comments/)：移除 Python 注释、docstring 及 Notebook 说明单元格，保留配置、数据、许可证和运行所需字符串。
- [校验记录](GenDis-validation.json)：两个版本共 42 个 Python 文件通过语法检查；无注释版校验移除 docstring 后的语法树一致。未运行 GPU 训练或复现论文分数。

两份代码均保留上游 MIT 许可证及版权信息。上传内容不含 `.env` 和数据集文件，保留数据预处理 Notebook；运行时请从官方仓库取得数据，并遵守各数据集原始条款。
