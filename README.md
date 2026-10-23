# dot_rec：BAST 多模态推荐研究候选

BAST（Behavior-Anchored Selective Transport）研究如何在同一组训练交互上利用图像、文本和行为信息，并在模态缺失时选择性补充信息。项目只维护一个主模型和两个独立研究模块。

**当前没有真实数据训练或推荐效果实验，所有 BAST 实验结果均为 NOT RUN。尚未证明性能提升、收敛性或新颖性。禁止在当前开发电脑运行训练；下文训练与最终评测命令仅供用户在另行授权的外部环境执行。** 默认参数仅供调试，不代表最优配置。

## 两个模块

1. **行为锚定的组条件传播**：每个用户的 TRAIN 历史形成一个物品群组。结合行为相似性、可见内容的一致性、方差和支持数量决定消息强度。群组统计和接收消息均排除目标自身，保留原始成员数作为分母，使弱证据保持较弱；物品也保留自身可见内容和协同表示。
2. **缺失模态的选择性补偿**：仅在目标缺失一种模态、另一种可见时，从同一行为群组中当前图文都可见的物品获取信息。支持少、方差大或不一致时补偿衰减或为零。补偿结果不成为供体、不扩展图、不多轮补全。

双模态都缺失时严格回到同一模型的协同表示。TRAIN 零度用户表示为零、评分同分，这不是已解决用户冷启动。传播仍可表达为动态二部图上的两步消息传递；“超图”、留一法、行为锚定或 dropout 本身均不被宣称首次提出。

详见 [模型公式](docs/model_design.md)、[新颖性审查](docs/novelty_review.md) 和 [架构说明](docs/architecture.md)。

## 目录职责

```text
model/                 # 所有模型放在此处，目录名为单数
  bast.py              # 主模型与损失接口
  modules/             # observed.py、missing.py
common/                # 数据、验证选模、排名、独立最终测试
utils/                 # 配置、TRAIN incidence、mask、采样、源码指纹
configs/model/         # 基础配置及 LGMRec preset
configs/dataset/       # 数据 manifest 示例
data/README.md         # 外部数据说明，实际数据不提交
tests/                 # CPU 合成正确性测试
docs/                  # 设计、协议、来源、实验计划
main.py                # 显式训练入口，仅用 validation 选模
evaluate.py            # 显式最终测试入口
requirement.txt        # 最小运行依赖
pyproject.toml         # 包配置
agent.md               # 协作与实现规范
```

布局参考 LGMRec 的职责划分，代码独立实现，没有并行 src 实现、模型注册系统或自动训练 CI。

## 数据与原始划分

参考 [LGMRec 固定提交](https://github.com/georgeguo-cn/LGMRec/tree/d66655fc4a2b066f915365b58e806f39cd08f0dd) 的 Baby、Sports、Clothing。读取原始 .inter 文件中 TAB 分隔的 userID、itemID、x_label：0/1/2 分别为 TRAIN、validation、test。**保留原始标签，不按论文的 8:1:1 描述重新随机切分。**

参考配置使用 image_feat.npy、text_feat.npy。参考 README 声明 VGG-16 图像特征和 Sentence-Transformers 文本特征；历史日志显示维度 **4096/384**。这不表示本项目已核验实际特征文件。Amazon 发布版本、预处理或精确编码器版本不明时必须如实记录未知。

模板：[Baby](configs/dataset/baby.manifest.example.json)、[Sports](configs/dataset/sports.manifest.example.json)、[Clothing](configs/dataset/clothing.manifest.example.json)。

**模板不是即跑数据。** 用户须取得并核验文件、零基 ID、特征行映射、数量与自然缺失情况，填写真实 SHA256。Git blob SHA1 仅标识参考对象，不是文件 SHA256；占位哈希会被拒绝。项目不下载数据、特征或预训练模型。

自然 mask 与实验 mask 分开保存，均为 [物品数, 2]，列顺序为图像、文本。不能以零特征判断缺失。utils.masks.fixed_mask(natural, rate, seed, pattern) 是 Python API，支持 independent/coupled/visual/text，**没有 mask 生成 CLI**。用户显式保存结果及哈希，验证和测试共用固定 mask。

详见 [数据协议](docs/data_protocol.md) 与 [LGMRec 对齐说明](docs/lgmrec_alignment.md)。

## 指标与选模

- 论文对照主四项：Recall@10、Recall@20、NDCG@10、NDCG@20。
- 仓库完整十六项：Recall、NDCG、Precision、MAP × @5、@10、@20、@50。
- 用户宏平均；Precision 分母为 K，AP 分母为 min(K, 正例数)。JSON 保存完整精度，展示表可保留四位小数。
- [LGMRec preset](configs/model/bast_lgmrec.json) 使用十六项指标、recall@20 验证选模、test_mask_valid=false、train_users_only。只过滤评估目标中的冷用户，保留冷物品及完整候选集，记录过滤量。
- [基础配置](configs/model/bast.json) 默认 Recall/NDCG@10/20、人口 all。warm_only 严格拒绝冷用户或冷物品目标，不等价于 train_users_only。

验证和测试默认只屏蔽 TRAIN 历史。负无穷屏蔽、较小物品 ID 优先的稳定同分规则与参考实现存在差异，不宣称逐位复现。

**训练只用 validation 选 checkpoint，不自动测试。** 冻结配置与 run 选择后，再调用独立最终测试入口。不复制参考仓库以 test 选择外层 BEST 的流程。论文表格、仓库 BEST 与单轮日志分别标注，均不是本模型结果。

## 环境与无训练检查

依赖为 Python ≥3.10、numpy>=1.24、torch>=2.6,<3，见 [requirement.txt](requirement.txt) 与 [pyproject.toml](pyproject.toml)。无需 PyG、DGL、SciPy；开发过程中未安装或更改依赖。

实际合成测试环境为 Python 3.12.7、PyTorch 2.6.0；声明范围不代表每个版本或设备都已测试。**GPU 执行与端到端训练未验证**。支持 float32/float64，拒绝 half、bfloat16 与 AMP。

从仓库根目录执行，无需安装本项目包：

```text
python -m unittest discover -s tests -v
python -m compileall -q model common utils tests main.py evaluate.py
python main.py --help
python evaluate.py --help
```

当前 **42 项 CPU 合成测试通过**，包括手算传播、NaN 隔离、缺失回退、指标、人口过滤与最终入口 mock 检查。没有训练循环或 optimizer step；这些是正确性证据，不是性能、GPU 可用性或可扩展性证据。见 [测试报告](docs/test_report.md)。

## 用户在外部授权环境的运行步骤

1. 将对应 manifest 模板放到外部数据旁，核验并填写路径、元信息、mask 与真实 SHA256。相对路径以 manifest 所在目录为基准。
2. 在 JSON 配置中设置设备、种子、协议；device、seed、protocol 等不是 CLI 参数。
3. 显式训练，输出目录必须尚不存在：

```text
python main.py --config configs/model/bast_lgmrec.json --manifest /external/dataset/manifest.json --output /external/new-run --execute-training
```

4. 只根据 validation 冻结选择，再显式最终测试：

```text
python evaluate.py --run /external/selected-run --manifest /external/dataset/manifest.json --execute-final-test
```

/external/... 是待替换的示例路径。没有执行开关时入口拒绝运行，导入模块不会启动训练。上述命令不授权在当前开发电脑执行训练或真实 benchmark。

最终测试核对配置、数据来源与哈希、checkpoint 参数和文件绑定，并比较保存的源码指纹与当前源码。源码变化或指纹缺失时拒绝；仅文档造成的 Git commit 变化不单独触发拒绝。已有 final_test.json 不覆盖；中途失败留下标记时应检查原因，不得静默覆盖后反复选择测试结果。

配置、来源、环境与源码指纹、验证日志和 checkpoint 保存在外部 run 目录，最终测试另写结果。数据、权重和日志不提交。没有自动跨 run 搜索、续训或测试选优功能。

## 缺失协议与消融

training.protocol：

- mtmt：训练基础可见性是自然 mask 与实验 mask 的交集。
- ftmt：训练使用自然可见内容；验证和测试仍使用上述交集，不揭露天然缺失内容。

训练 dropout 只减少对应训练视图的可见内容；两个模块共用视图。评估无随机 dropout，保持 TRAIN 支持不变并重算 mask 下的统计。

已有 JSON 开关：model.learned_group（固定 LOO/学习组权重）、model.compensation（补偿关/开）、model.quality_mode（full/unit/no_support/no_agreement）。eta=0 检查局部内容；alpha=beta=0 检查同一模型的协同表示，但不是独立轻量 CF 耗时基线。匹配常数门控、打乱供体和自动子组分析尚未实现，没有对应 CLI。

每个 minibatch 重算全物品投影与全图传播，L2 是各参数张量均方之和；不能跨 optimizer step 缓存旧表示。稀疏实现不等于已证明可扩展。高缺失时供体不足、多兴趣历史和过度衰减都可能限制效果。按 [实验计划](docs/experiment_plan.md) 分阶段验证，不据默认参数承诺提升。

## 参考与边界

[来源核验](docs/research_sources.md)、[新颖性审查](docs/novelty_review.md)、[LGMRec 对齐](docs/lgmrec_alignment.md) 记录证据范围。项目不复制未获许可的第三方实现，也未代用户新增 LICENSE。当前交付是可审查、待外部实验验证的研究候选。
