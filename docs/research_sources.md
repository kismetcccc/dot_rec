# 已核验参考证据

核验方式：通过 GitHub 只读读取固定 commit 的目录、文本源码及许可证；未执行第三方代码、下载数据或复现实验。以下不包含未经核验的论文结论，也不代表完整的 2025–2026 文献审查。

## MMRec

- 仓库：https://github.com/enoche/MMRec
- 核验 SHA：`c68ee7c9b0de1ca7c5150258322edb4994be29d9`，master，commit 日期 2026-07-14。
- [接口](https://github.com/enoche/MMRec/blob/c68ee7c9b0de1ca7c5150258322edb4994be29d9/src/common/abstract_recommender.py)：calculate_loss(interaction)、full_sort_predict(interaction)；GeneralRecommender 从 dataloader 取得规模并读取图文特征。dot_rec 仅参考接口职责，不复制基类。
- [划分](https://github.com/enoche/MMRec/blob/c68ee7c9b0de1ca7c5150258322edb4994be29d9/src/utils/dataset.py#L57)：按配置指定的 splitting label 取 0/1/2；overall.yaml 将字段设为 x_label。baby.yaml 配置 image_feat.npy、text_feat.npy。
- [epoch 早停](https://github.com/enoche/MMRec/blob/c68ee7c9b0de1ca7c5150258322edb4994be29d9/src/common/trainer.py#L262)：使用验证指标；同一循环也计算测试指标。
- [外层选参风险](https://github.com/enoche/MMRec/blob/c68ee7c9b0de1ca7c5150258322edb4994be29d9/src/utils/quick_start.py#L85)：best_test_upon_valid 的测试指标决定 best_test_idx；因此不能将整个搜索流程视为纯验证集选择。
- [许可证](https://github.com/enoche/MMRec/blob/c68ee7c9b0de1ca7c5150258322edb4994be29d9/LICENSE)：根目录为 GNU GPL version 3。复制或改编前仍须核对具体文件归属和适用条款，不自行给 dot_rec 指定许可证。

## ChaoRec

- 仓库：https://github.com/Ricardo-Ping/ChaoRec
- 核验 SHA：`a68427250dd2626f6044411ba0c9e36070f57174`，master，commit 日期 2026-04-08。
- [数据加载](https://github.com/Ricardo-Ping/ChaoRec/blob/a68427250dd2626f6044411ba0c9e36070f57174/dataload.py#L21)：独立 train/val/test NPY、user_item_dict、v_feat/t_feat；规模按数据集名硬编码，训练采样的物品 ID 从 num_user 开始。
- [SMORE 接口与编号转换](https://github.com/Ricardo-Ping/ChaoRec/blob/a68427250dd2626f6044411ba0c9e36070f57174/Model/SMORE.py#L407)：loss(users,pos_items,neg_items) 内减去 n_users；gene_ranklist 输出偏移物品 ID。这是已检查模型的接口例子，不宣称所有模型统一。
- [测试集早停](https://github.com/Ricardo-Ping/ChaoRec/blob/a68427250dd2626f6044411ba0c9e36070f57174/train_and_evaluate.py#L673)：test_metrics[max(topk)]['recall'] 输入 early_stopping。
- [外层选参](https://github.com/Ricardo-Ping/ChaoRec/blob/a68427250dd2626f6044411ba0c9e36070f57174/main.py#L428)：用返回的 current_best_metrics[20]['recall'] 选 best_params；结合上述返回值来源，存在测试集参与选择的问题。
- [屏蔽风险](https://github.com/Ricardo-Ping/ChaoRec/blob/a68427250dd2626f6044411ba0c9e36070f57174/Model/SMORE.py#L443)：历史物品分数设 1e-6，可能高于其他候选分数，不能保证排除历史项。
- 许可证状态：已检查该 SHA 的递归文件树，未发现许可证文件；这不等于获得再发布许可。当前不复制其实现。

## 对 dot_rec 的约束

独立实现数据边界和评测；物品 ID 转换仅发生在适配处；所有行为结构只用 TRAIN；早停与超参选择只用验证集；测试在选择冻结后报告；历史候选屏蔽用负无穷并测试。上述是依据代码审查形成的设计决策，不是性能结论。后续作者官方来源、论文机制与创新差异需单独核验。
