# dot_rec 协作与实现规范

本文件是本仓库唯一的协作规范入口，按用户要求使用小写名称 `agent.md`。工具若不自动识别此名称，执行任务前应显式读取；不要再建立内容重复的 AGENTS.md。当前候选机制按 docs/model_design.md 冻结；实现和合成测试仍不代表性能验证。

## 目标与范围

- 只建设一个主模型，服务毕业论文的两项研究工作，不建设模型动物园。
- 模块一：行为锚定多模态超图。仅从 TRAIN 交互构造锚点群组，以可用图文与协同信息确定传播权重，结合局部表示和行为残差。
- 模块二：模态缺失感知与可靠性融合。显式区分缺失和低可靠性，融合可用模态与行为表示；两种模态均缺失时必须有定义明确的行为回退。
- 第一模块允许独立实现，不声称复现用户未提供的原代码。锚点策略、权重公式、可靠性估计和损失项在设计审查后再实现。
- 新机制必须说明解决的问题、输入输出和必要的消融；不因其他论文使用某组件而直接叠加组件。不预设对比学习、扩散、重建或教师网络都是必需项。

## 架构边界

使用独立 PyTorch 核心，参考 MMRec 的批损失/全物品评分接口；不整体搬运 MMRec 或 ChaoRec。精确版本与理由见 docs/architecture.md。

Code belongs in root model/ (all future models), common/ (dataset/trainer/evaluator/final_test), and utils/ (config/incidence/masks/sampling/reproducibility). BAST has two modules under model/modules/. configs/model and configs/dataset hold JSON inputs. main.py and evaluate.py are thin explicitly guarded entrypoints. data/README.md explains external data; real data and log outputs are ignored. tests/ contains CPU-only synthetic checks. No src duplicate, model zoo, plugin registry, frontend, service or database.

## 数据、缺失与评测

- 目标接口覆盖 Amazon Baby/Sports/Clothing；同名数据集不代表同版本。记录发布版本、来源、过滤规则、划分哈希、ID 映射、特征编码器/维度及特征哈希，不虚构已确定的数据版本。
- 用户与物品分别使用零基连续 ID；第三方偏移物品 ID 在 data 边界转换并验证，模型内部不混用编号。
- 所有行为图、锚点、流行度、协同统计与训练负采样历史均仅来自 TRAIN。验证/测试标签不得进入构图、可靠性监督或特征拟合。
- 真实可用性与人为缺失掩码分别保存，最终掩码为二者交集。掩码由显式种子和协议确定、持久化并记录哈希；所有比较模型共享相同掩码。训练增强若需要，另设种子与名称，不能替代固定评测掩码。
- 显式测试图像缺失、文本缺失、两者缺失以及完整模态。缺失位置不得以隐藏的原特征参与相似度、图权重或融合；全缺失不能产生 NaN。
- 区分训练期缺失和仅推理期缺失；缓存必须绑定训练划分、特征、掩码和构图配置，禁止跨设置复用错误缓存。
- 仅验证集用于早停及超参选择；测试集在选择冻结后用于最终报告。采用全物品排名，已见物品设为负无穷；明确验证/测试各自屏蔽集合、候选集合及无正例用户规则，所有基线一致。
- 没有真实数据实验，不得宣称性能提升、创新已被证明、复现成功或达到 SOTA；合成测试只能证明相应实现性质。

## 执行限制与依赖

- 严格禁止在当前电脑训练：不执行优化器训练循环、真实数据 benchmark、自动超参搜索或训练试跑。训练代码可以供用户审查和自行执行，但生成代码不构成执行授权。
- 不下载数据集、大型特征或模型；不运行外部仓库代码，不启用付费 API，不安装/卸载或改动现有环境。
- 可以静态审查、编译和运行轻量无训练单测；只用小型合成张量，默认 CPU，无优化器步骤，避免耗时/大内存计算。
- 最低必要依赖优先 PyTorch、NumPy；配置需要 YAML 时才引入 PyYAML。额外依赖必须说明必要性，不默认依赖 PyG、DGL 或 SciPy。
- 不扫描无关私人文件，不读取 .env 或密钥，不修改、恢复或引用已停止的旧项目作为本项目实现。

## 可追溯性与交付

- 保存解析后的完整配置、随机种子、代码版本、数据/特征/掩码标识、环境版本、选择指标和检查点选择依据。拒绝未知配置键；禁止静默覆盖配置。
- 轻量测试优先覆盖 ID/shape 错误、TRAIN-only 边界、固定掩码复现、缺失特征不影响结果、全缺失有限输出、稀疏传播小例与排名屏蔽；避免只重复实现的测试。
- 第三方参考须记录仓库、精确 commit、文件和用途；复制/改编前检查文件归属及许可证，保留适用声明。公开可读不等于允许再发布。
- MMRec 固定参考版本含 GPLv3；ChaoRec 已审版本未发现许可证文件。不得据此替 dot_rec 自行指定许可证或未经审查复制其实现。
- 仅经审查的白名单源码/文档允许发布，数据和实验产物禁止发布；不包含特征、检查点、日志、缓存或密钥。


## Required flat layout (supersedes earlier directory names)

All future model implementations MUST go in root `model/` (singular). BAST stays in model/bast.py with exactly two readable modules in model/modules/. Do not recreate src/dot_rec or maintain duplicate implementations. common/ owns dataset, validation-only training, ranking and explicit final testing; utils/ owns configuration, incidence, masks, sampling and reproducibility. configs/model and configs/dataset hold JSON configurations. main.py and evaluate.py are thin guarded entrypoints. data/ contains only its tracked README; actual data and log outputs are ignored. No img placeholder, model registry or plugin machinery is needed. Use git moves for existing code. Final testing is a separate explicit operation after validation selection, never automatic at the end of each training run.
