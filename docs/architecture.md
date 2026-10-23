# 最小架构草案

状态：薄层目录与接口已经实现；候选机制按 model_design.md 冻结，尚未运行训练或真实数据评测。协作规范唯一入口为根目录 agent.md。

## 框架选择依据

采用独立 PyTorch 实现，保持 MMRec 风格的 `calculate_loss(batch)` 与 `full_sort_predict(users)` 适配能力，不继承或复制第三方框架。核心使用显式张量输入；适配层只有在实际对接需要时才增加。

只读核验的参考快照：

- [MMRec c68ee7c9b0de1ca7c5150258322edb4994be29d9](https://github.com/enoche/MMRec/tree/c68ee7c9b0de1ca7c5150258322edb4994be29d9)：`src/common/abstract_recommender.py` 的批损失/全排序接口；`src/utils/dataset.py` 用 x_label=0/1/2 划分训练/验证/测试；`src/configs/dataset/baby.yaml` 定义交互表及 image_feat.npy、text_feat.npy。
- [ChaoRec a68427250dd2626f6044411ba0c9e36070f57174](https://github.com/Ricardo-Ping/ChaoRec/tree/a68427250dd2626f6044411ba0c9e36070f57174)：`dataload.py` 使用分开的 NPY 数据、偏移物品 ID 与硬编码规模；`Model/SMORE.py` 使用 `loss(users, pos_items, neg_items)` 和 `gene_ranklist()`。不让这些格式假设进入主模型。

不复制评测流程：ChaoRec `train_and_evaluate.py:673-674` 按测试 Recall 早停；MMRec `src/utils/quick_start.py:85-88` 按测试表现选超参组合。MMRec 的 epoch 早停使用验证集，这与外层选参风险应区分。ChaoRec SMORE 将已见物品分数置 1e-6，不能保证排除，独立评测使用负无穷。

MMRec 有 GPLv3 LICENSE；ChaoRec 该快照未发现许可证文件。上述仅为接口和协议审查，不代表已核验其所有模型的作者官方来源。2025–2026 相关官方实现及机制差异由后续研究补充；不把模型集合仓库一概称为作者官方代码。

## Current flat layout

```text
main.py                         # guarded validation-only training entrypoint
evaluate.py                     # guarded final-test entrypoint
model/bast.py                   # single main model
model/modules/{observed,missing}.py
common/{dataset,trainer,evaluator,final_test}.py
utils/{config,incidence,masks,sampling,reproducibility}.py
configs/model/bast.json
configs/dataset/manifest.example.json
data/README.md                  # real data ignored/external
requirement.txt                 # minimum runtime dependencies
pyproject.toml                  # flat-package discovery
agent.md
README.md
tests/
docs/
```

Package directories include __init__.py. All future models belong in singular model/. No src duplicate, img placeholder, registry or automatic training launcher is retained. log/ is ignored but not created as an empty shell. The reference [LGMRec snapshot d66655fc4a2b066f915365b58e806f39cd08f0dd](https://github.com/georgeguo-cn/LGMRec/tree/d66655fc4a2b066f915365b58e806f39cd08f0dd) was inspected read-only: common contains abstract_recommender/trainer; utils contains configurator/data utilities/dataloader/dataset/logger/metrics/quick_start/topk_evaluator; configs separates model/dataset. Its main.py parses model/dataset/GPU arguments and delegates to quick_start. Only directory responsibilities inform this layout; no upstream code, environment mutation, training or evaluation procedure is copied.

common.dataset validates external arrays; utils.incidence consumes TRAIN edges only; model.bast performs the unchanged two-module forward pass; common.trainer selects on validation only; common.final_test verifies one selected run and invokes common.evaluator on test once. Data under root data/ is not a Python source package.

## 数据流与最小接口

1. configs → common.dataset：解析配置并验证外部数据；产生训练/验证/测试划分、独立 ID 映射、特征、固定缺失掩码与 manifest。
2. common.dataset 的 TRAIN 交互 → utils.incidence：产生稀疏关联结构和训练统计。utils.incidence 不接收验证/测试交互；若结构权重使用模态，只能使用当前协议允许的可见特征。
3. 训练结构、可见特征、掩码 → model：同一支持集计算浅层协同基底，模块一输出带局部内容的组条件 LOO 表示，模块二输出缺失模态补偿；系数融合后得到用户/物品表示。
4. common.trainer 从训练样本获得批损失；common.evaluator 从冻结模型取得分数，负责候选过滤、全排序及指标。验证指标用于选择；测试指标不反馈选择过程。

| 契约 | 建议形状/语义 |
|---|---|
| 训练交互 | int64 [E,2]；user ∈ [0,U)，item ∈ [0,I)，与验证/测试分开 |
| 图文特征 | float32 [I,Dv] / [I,Dt]；行号等于内部 item ID，维度从 manifest 验证 |
| 可用性 | bool [I,2]，视觉/文本列顺序固定；天然可用性和实验掩码分离 |
| 稀疏结构 | 节点—锚点关联索引与权重；避免构造完整 I×I 稠密矩阵 |
| 训练 batch | user、positive_item、negative_item 各 [B]，零基 ID |
| encode() | 用户 [U,d]、物品 [I,d]；可选诊断可靠性权重，不强制保留大型中间量 |
| score(users, items) | 对指定配对输出 [B]；与全排序结果一致 |
| full_sort_predict(users) | [B,I]；仅小批次，评测可进一步按物品分块合并 top-k |
| calculate_loss(batch) | 字典 total/bpr/l2；标量总损失与未加权分量，BPR+显式 L2 |

缺失位置的数值在模型使用前屏蔽，禁止以零值自动判断缺失；模块一所有缺失模态权重严格为零，模块二只在单模态缺失时允许补偿，双缺失走行为回退。支持/一致性衰减不等同概率校准；具体公式和消融见 model_design.md。

## Amazon 数据与实验边界

暂支持 Baby/Sports/Clothing 命名接口，不预设 Amazon 发布年份、k-core、随机/时间划分或编码器。MMRec 报告 Baby 19,445 用户/7,050 物品，ChaoRec dataload.py 为 12,351/4,794，证明同名不能直接比较。

manifest 必填：来源/版本、过滤规则、划分方式与哈希、用户/物品映射标识、编码器与特征维度/哈希、天然缺失定义、实验缺失率/种子/掩码哈希。缺失信息应明确报错或标记未核验，不编造默认来源。

建议报告 Recall/NDCG@10/20，并在所有比较中统一候选集、历史屏蔽、空正例处理与固定掩码。训练期缺失、推理期缺失分别建实验配置。测试集只在验证集选定配置后评估；不将文献或第三方仓库表格作为本模型结果。

## 机制冻结与后续验证

- 锚点固定为 TRAIN 用户历史，二值稀疏支持；具体 LOO 传播与浅层协同基底见 model_design.md。
- 缺失补偿采用同一批完整 donor 的支持/一致性衰减，只使用 BPR+L2；不宣称概率校准或因果性。
- Amazon 数据的实际版本、特征及划分；用户外部验证的配置和运行说明。
- 主消融为 fixed-LOO/learned-group × compensation off/on 的 2×2，辅以 q=1、去 support、去 agreement；行为基底可设 alpha=beta=0 检查，但不另建模型动物园。此处是验证计划，不是实验结果。

本机仅静态审查、编译和轻量合成无训练测试；不运行训练、不下载数据/大模型、不改动现有环境。仅经审查的白名单源码/文档允许发布，数据和实验产物禁止发布；不自行设定项目许可证。
