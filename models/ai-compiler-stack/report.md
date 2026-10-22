# AI 编译器生态与异构全栈算力底座前沿技术报告：当算力成为战略资源——从 CUDA 护城河、Triton 生态到 LLM 自动写 Kernel 与 FlagOS「一次开发、多芯运行」

> **选题定位**：北京智源人工智能研究院《2026 十大 AI 技术趋势》第 9 位——「开源编译器生态汇聚众智，异构全栈底座引领算力普惠」。报告日期 2026-10-21。
> **信源说明**：23 篇专题论文（其中 arXiv 论文编号均经 arxiv.org 元数据逐一核实；Triton 原论文、Roller、Welder、Exo 原始论文等未上 arXiv 的工作改引 MAPL/OSDI/MLSys/PLDI 正式出处）+ NVIDIA/Intel SEC 10-K/10-Q/8-K 一手披露 + AMD ROCm、FlagOS、PyTorch、UALink、UEC 官方文档交叉验证。厂商自报数据在正文显式标注「厂商口径」，媒体估算标注「第三方」。

**标签**：AI 编译器 · 异构计算 · CUDA · Triton · MLIR · TVM · ROCm · oneAPI · FlagOS · 昇腾 CANN · 算子语言 · Kernel 自动生成 · UALink · 出口管制

---

## 目录

1. 引言：为什么「算力底座」在 2026 年成为主战场
2. 护城河解剖：CUDA 二十年积累了什么
3. AI 编译器核心原理：多层 IR 与 lowering 流水线
4. 算子语言光谱：从裸 CUDA 到 Tile 级编程
5. 自动调优、图优化与自动并行
6. 2025–2026 新范式：LLM/Agent 自动写 Kernel
7. 全球异构战局：ROCm、oneAPI 与其他厂商栈
8. 中国国产算力软件栈与 FlagOS 深度拆解
9. 开放互联与标准生态：UALink、UEC、UXL 与兼容层
10. 出口管制时间线：软件栈替代的需求动因
11. 技术选型与工程落地建议
12. 开放问题与展望
13. 参考文献

---

## 1. 引言：为什么「算力底座」在 2026 年成为主战场

2026 年 1 月 8 日，智源研究院发布《2026 十大 AI 技术趋势》，在世界模型、具身智能、多智能体等「智能能力」方向之外，罕见地把一个纯基础设施命题列入第 9 位：**「开源编译器生态汇聚众智，异构全栈底座引领算力普惠」**。其原文判断是：「为打破算力垄断与供应风险，构建兼容异构芯片的软件栈至关重要。繁荣的算子语言与趋于收敛的编译器技术正在降低开发门槛。以智源 FlagOS 为代表的平台，致力于构建软硬解耦、开放普惠的 AI 算力底座」（新华网，2026-01-08）。

这个判断背后是三股力量在 2024–2026 年的合流：

- **第一，智能能力的竞争已经下沉为系统软件的竞争。** 当模型架构与训练方法快速趋同，决定一个模型能否在某款芯片上跑得起来、跑得高效、跑得稳定的，不再是芯片峰值算力（TOPS）这一个数字，而是编译器、算子库、通信库、框架适配组成的**全栈软件生态**。
- **第二，地缘政治把「软件栈自主」从技术议题变成采购刚需。** 2025 年 4 月美国对华 H20 出口管制骤然而至，NVIDIA 单季计提 45 亿美元费用；同年 8 月有条件恢复（美方预期抽取 15% 收入），2026 年 2 月 H200 个案放行——供给的不确定性使「多芯片可选、代码可迁移」成为算力采购方的硬需求（详见第 10 章）。
- **第三，编译器与语言技术本身正在成熟，使「绕开 CUDA 重写生态」第一次在工程上可见。** 其标志有三个：Triton 从 OpenAI 的内部 DSL 成长为多厂商算子的事实共同语言；MLIR（Lattner 等，arXiv:2002.11054）提供了可复用的多层 IR 公共底座；2025 年起 LLM/Agent 开始能自动生成高性能 GPU kernel，把生态迁移中最昂贵的「人力写算子」环节部分自动化（第 6 章）。

> ![](diagrams/1-stack-landscape.html)

本报告要回答四个问题：**(1) CUDA 的护城河到底由什么构成、有多深？(2) 现代 AI 编译器的技术原理是什么，为什么「多层 IR + 算子语言 + 自动调优」是收敛方向？(3) LLM 自动写 kernel 这一新范式进展到什么程度、边界在哪？(4) 以 AMD ROCm 和智源 FlagOS 为代表的异构全栈路线，能否以及在什么条件下实现算力普惠？**

---

## 2. 护城河解剖：CUDA 二十年积累了什么

### 2.1 规模：从 590 万到 750 万开发者

讨论「替代 CUDA」之前，先要量化护城河的规模。根据 NVIDIA 向美国证券交易委员会（SEC）提交的年报：

- **FY2025 10-K（财年截至 2025-01-26）原文**：「over 5.9 million developers worldwide using CUDA」；
- **FY2026 10-K（财年截至 2026-01-25）原文**：「over 7.5 million developers worldwide using CUDA」。

也就是说，仅一年时间 CUDA 开发者净增约 **160 万**。CUDA 本身自 2006 年随 G80 架构发布，到 2026 年已积累约 20 年的软件资产；NVIDIA 官方在 GB200/GH200 技术博客（2025-03-18）中称其加速库体系已达 **900+ 个 CUDA-X 库与预训练模型**（厂商口径）。

| 指标 | 数值 | 口径 |
|---|---|---|
| CUDA 全球开发者 | 590 万（FY2025）→ 750 万（FY2026） | 官方，SEC 10-K |
| CUDA-X 库与预训练模型 | 900+ | 厂商口径，NVIDIA 技术博客 |
| CUDA 发布年份 | 2006（G80） | 官方 |
| 首个支持 Blackwell 的 CUDA | 12.8（SM_100/SM_120） | 官方 release notes |
| CUDA Toolkit 最新小版本 | 13.4.3（13.4 新增 Tile/Tile IR 编程模型） | 官方，2026-10-21 核实 |

### 2.2 护城河的四层结构

常被简化为「CUDA 库多」的护城河，实际上是四层相互锁定的结构：

1. **硬件指令层**：PTX 虚拟指令集 + SASS 机器码，与每代 GPU 微架构（Tensor Core、Warp 级调度、TMA、异步拷贝等）深度绑定。新架构能力（如 Hopper 的 TMA、Blackwell 的 Tensor Memory Accelerator）首先且往往只通过 CUDA 工具链暴露。
2. **系统软件层**：nvcc 编译器、驱动、profiler/debugger（Nsight）、CUDA Graphs、多实例 GPU（MIG）等工具链，构成开发者日常工作流。
3. **加速库层**：cuBLAS、cuDNN、NCCL、TensorRT、cuSPARSE、CUTLASS 等数百个高度优化的库，把「为新架构重写算子」的成本由 NVIDIA 的工程师团队承担，应用方只需链接。
4. **人力与惯性层**：750 万开发者、二十年的教程/问答/代码存量、企业招聘与培训体系——这是任何替代栈最难复制的「网络效应」资产。

第三方分析也承认这一层的韧性：Tom's Hardware（2025-07-31）引述 Gavekal 分析师 Tilly Zhang 的判断称，尽管北京对国产芯片能力比以往更有信心，但「许多（甚至大多数）模型的软件栈」仍使中国在关键用途上依赖 NVIDIA 硬件。

### 2.3 NVIDIA 的自我进化：CUDA Tile IR

值得注意的是，NVIDIA 并未坐等 Triton 等高层语言「架空」CUDA。**CUDA 13.4 推出了新一代 Tile 级编程模型：CUDA Tile / Tile IR、Tile C++ 与 cuTile Python**，并加入 Windows on Arm 支持与 Rubin 架构开发者预备；PyTorch 2.14（2026-09-02）的 Inductor 后端已支持 Rubin（sm_107），并将 CuTeDSL 生成的 CUTLASS 内核（NVGEMM，含 NVFP4 GEMM）引入自动调优。这说明「算子语言向 tile 级抽象上移」是全行业共识——连 CUDA 自己也在向更高抽象层迁移（第 4 章详述）。

> 说明：cuDNN/TensorRT 的装机量或市场份额无公开权威数字（NVIDIA 不披露），本报告只做「事实标准」的定性表述，不引用无法核实的市占率测算。

---

## 3. AI 编译器核心原理：多层 IR 与 lowering 流水线

### 3.1 为什么深度学习需要专门的编译器

传统编译器（GCC/LLVM）解决的是「通用高级语言 → 通用 CPU 机器码」；而深度学习负载有三个独特性质，使通用编译器无法直接胜任：

- **计算是规则的张量级嵌套循环**：算子（GEMM、卷积、attention）本质上是多层完美嵌套循环，优化手段（分块、张量化、循环重排、流水线）作用在 tile（瓦片）级而非指令级；
- **优化空间是组合爆炸的**：一个 GEMM 仅分块大小、循环顺序、流水线级数的合法组合就数以万计，且强依赖具体硬件的存储层级与张量核形状；
- **存在两个优化层级**：图级（算子融合、常量折叠、布局优化、内存规划、自动并行）与算子级（循环变换、张量化、向量化），二者必须协同。

AI 编译器的核心任务因此可以概括为：**在框架（PyTorch/JAX）与硬件（GPU/NPU/CPU）之间，建立多层中间表示（IR），逐层把高层计算 lowering（下降）为目标代码，并在每一层施加该层级最擅长的优化。**

### 3.2 MLIR：「摩尔定律终结」时代的编译器基础设施

理解现代 AI 编译器绕不开 MLIR。Chris Lattner（LLVM 之父）等人在论文《MLIR: A Compiler Infrastructure for the End of Moore's Law》（**arXiv:2002.11054**，2020-02-25）中提出：随着 Dennard 缩放终结、专用加速器爆发，需要一个**可复用、可扩展的多层方言（dialect）编译器基础设施**。MLIR 的关键设计是：

- **Dialect（方言）体系**：每个抽象层级定义自己的类型、操作与合法性规则（如 `tosa`/`mhlo` 表达前端算子、`ttir` 表达 Triton 块级程序、`gpu`/`nvvm`/`rocdl` 表达设备级原语、`llvm` 表达 LLVM IR）；
- **Conversion（转换）框架**：在方言之间定义 lowering pattern，逐层下降且每层可验证合法性；
- **可组合的优化 pass**：公共子表达式消除、内存提升、循环变换等基础设施只写一次，各方言复用。

MLIR 已成为事实公共底座：**Triton、OpenXLA、IREE、Mojo、FlagTree、TensorFlow 的新栈都构建在 MLIR 方言体系之上**。这意味着异构竞争不再需要每家从零造编译器，竞争焦点上移到「方言设计、优化 pass、后端代码生成质量」。

### 3.3 两条代表性 lowering 流水线

**（A）Triton 流水线**（据 triton-lang.org 官方文档与仓库源码）：

```
Python @jit 函数
   │  前端：AST → Triton IR
   ▼
TTIR   (Triton IR：块级 SSA，虚拟寄存器为张量/指针，硬件无关)
   │  pass：算子结合、数据布局、流水线规划
   ▼
TTGIR  (Triton GPU IR：引入分布式布局(distributed layout)、
   │     warp/线程映射、共享内存、张量指令)
   │  各硬件后端：nvidia / amd / 以及 FlagTree 的 15+ 国产后端
   ▼
LLVM IR (NVVM / ROCDL / 各目标 dialect)
   ▼
PTX/SASS (NVIDIA) ｜ GFX 机器码 (AMD) ｜ 各 NPU 目标码
```

Triton 的关键抽象是**块（block/tile）**：程序员操作的是 N 维张量块，编译器负责把块在「线程—warp—计算核心」之间做分布式布局，并决定共享内存分配与异步流水线。这把 CUDA 中最易出错的索引计算、bank conflict、warp 归约交给编译器，同时保留接近手写的性能（数据见第 4 章）。

**（B）TVM/Relax 流水线**：

```
框架模型 (PyTorch/ONNX/...)
   ▼
Relax IR   (图级 + 符号化动态形状，跨层可调用外部库)   arXiv:2311.02103
   │  图优化：融合、布局改写、内存规划
   ▼
Tensor IR (TIR：张量级循环 + schedule)
   │  调度原语：tile/reorder/compute_at/tensorize/...
   ▼
自动调优 (搜索 + 学习型代价模型)  →  TVM 初代 arXiv:1802.04799 / Ansor arXiv:2006.06762
   ▼
LLVM / CUDA-C / ROCm / 移动端 GPU / WebGPU ...
```

TVM 是这条路线的开创者：陈天奇等的《TVM: An Automated End-to-End Optimizing Compiler for Deep Learning》（**arXiv:1802.04799**，2018-02-12，即 OSDI'18 论文）首次给出图级 + 算子级统一、并以学习型代价模型指导内核搜索的端到端栈，证明编译器生成的内核可以比肩手工调优库，并能快速新增 FPGA、移动端 GPU 等后端。新一代 IR Relax（**arXiv:2311.02103**，2023-11-01，ASPLOS'25）进一步解决了大模型时代的**动态形状端到端编译**：以符号化形状贯穿图与张量级，并允许与外部库、运行时可组合地混合，支撑了 MLC-LLM 把 LLM 部署到 GPU、手机、嵌入式乃至浏览器（WebLLM，arXiv:2412.15803，同设备保留最高 80% 原生性能）。

> ![](diagrams/2-lowering-pipeline.html)

### 3.4 核心优化技术清单

| 层级 | 优化技术 | 作用 |
|---|---|---|
| 图级 | 算子融合（elementwise/归约/模板） | 消除中间张量落地与 kernel launch 开销 |
| 图级 | 布局优化、常量折叠、公共子表达式消除 | 减少访存与冗余计算 |
| 图级 | 内存规划、重计算（checkpointing） | 降低峰值显存 |
| 算子级 | 分块（tiling）、循环重排 | 匹配存储层级、提高复用 |
| 算子级 | 张量化（tensorize） | 映射到 Tensor Core / 矩阵单元 |
| 算子级 | 共享内存/寄存器提升、双缓冲/流水线 | 掩盖访存延迟 |
| 算子级 | 向量化、并行化、索引简化 | 提升指令吞吐 |
| 跨层 | 动态形状符号化（Relax/BladeDISC） | 避免重新编译、支撑真实推理形状 |
| 系统级 | 自动分片/自动并行（GSPMD/Alpa） | 生成分布式执行计划 |

---

## 4. 算子语言光谱：从裸 CUDA 到 Tile 级编程

### 4.1 核心矛盾：生产力与性能的权衡

写高性能 kernel 长期存在一个根本矛盾：**抽象越高，生产力越高但可控性能越可能损失；抽象越低，性能可控但门槛极高、迁移成本巨大。** 2023 年以来，研究界的共识是在二者之间开辟「**tile 级（块级）**」这一甜点层：程序员以 tile 为基本对象表达数据流，编译器负责 tile 到硬件线程/指令的映射与调度。

> ![](diagrams/3-kernel-languages.html)

### 4.2 代表性 tile 级语言

**ThunderKittens**（Stanford）：Benjamin Spector 等《ThunderKittens: Simple, Fast, and Adorable AI Kernels》（**arXiv:2410.20399**，2024-10-27）。设计哲学是用极少的抽象（warp/block/grid 三级、以 tile 为中心的「小猫」式组合）让简单的线性注意力类 kernel 既好写又快。实测（论文口径）：GEMM 与推理 kernel 对齐 cuBLAS 和 FlashAttention-3；注意力反向传播快 10–40%；SSM kernel 快 8×；线性注意力快 14×。

**TileLang**：Lei Wang 等《TileLang: A Composable Tiled Programming Model for AI Systems》（**arXiv:2504.17577**，2025-04-24）。核心思想是把**调度空间（线程绑定、数据布局、张量化、流水线）与数据流表达解耦**并可组合，在多家厂商的常用设备上让关键 kernel 达到 SOTA。TileLang 也是 FlagOS AKG kernel Agent 支持的跨平台目标语言之一（第 6 章）。

**Exo / Exo 2 / Exo-GPU**（MIT）：Yuka Ikarashi 等提出「exocompilation」思想——把调度（schedule）从编译器内部移到编译器外部，由用户用 Cursors 对程序施加调度，编译器负责保证调度的正确性。续作《Exo 2: Growing a Scheduling Language》（**arXiv:2411.07211**，2024-11-11）用 80+ 个高性能 kernel 证明调度代码量可降低一个数量级、覆盖 3 个平台；最新《Exo-GPU》（**arXiv:2609.16389**，2026-09-14）进一步把并行与同步降为顺序代码上的「注解」，以最小抽象面向 Hopper/Blackwell 的异步 Tensor Core 与多级存储，且可静态验证。Exo 原始论文发表于 PLDI 2022（未上 arXiv）。

**Hidet**：Yaoyao Ding 等《Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs》（**arXiv:2210.09603**，2022-10-18）。以「task mapping」把调度嵌入张量程序并配合调度后融合，比 ONNX Runtime 及 TVM（AutoTVM/Ansor）最高快 1.48×（平均 1.22×），且调优时间比 AutoTVM/Ansor 省 20×/11×。

### 4.3 Triton：从 DSL 到多厂商事实标准

Triton 最初由 Philippe Tillet 等在《Triton: an intermediate language and compiler for tiled neural network computations》（**MAPL 2019 workshop**，ACM DOI 10.1145/3315508.3329973；该论文未上 arXiv）中提出。它的历史意义在于：第一次让研究者无需写 CUDA 就能产出竞争力极强的 GPU kernel——FlashAttention、Hopper 上的大量 MoE/线性注意力 kernel 都以 Triton 写成。

其生态价值在 2024–2026 年集中兑现：

- **Liger Kernel**：Pin-Lun Hsu 等《Liger Kernel: Efficient Triton Kernels for LLM Training》（**arXiv:2410.10989**，2024-10-14），面向 LLM 训练的 Triton 融合算子集，训练吞吐平均 +20%、显存占用降 60%（对比 Hugging Face 实现）；
- **FlashInfer**：Zihao Ye 等《FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving》（**arXiv:2501.01005**，2025-01-02），以块稀疏 KV-cache、可组合格式与 JIT 注意力模板，把 token 间延迟降低 29–69%、长上下文降低 28–30%、并行生成加速 13–17%，已进入 SGLang/vLLM/MLC-Engine；
- **FlagTree**：把 Triton 分叉扩展为 15+ 厂商后端的统一编译器（第 8 章）；
- 项目以独立 `triton-lang` 组织运营，Triton Conference 2026 定于 2026-10-19 圣何塞（注：网络上「Triton 已捐赠 Linux 基金会」的说法在 LF 官网对应页 404，本报告不予采信）。

### 4.4 语言选型速查

| 语言/栈 | 抽象层级 | 最适合的场景 | 硬件范围 |
|---|---|---|---|
| CUDA C/C++（CUTLASS/CuTe） | 线程/指令 | 极致性能、新架构首发 | NVIDIA |
| CUDA Tile / Tile IR（13.4 新） | Tile | NVIDIA 官方高层 tile 编程 | NVIDIA |
| Triton | 块（tile） | 快速写融合算子、注意力/MoE | NVIDIA/AMD/15+ 国产（经 FlagTree） |
| ThunderKittens | tile（warp 中心） | 注意力/SSM/线性注意力 | NVIDIA |
| TileLang | tile（调度可组合） | 跨厂商 SOTA kernel | 多厂商 |
| Exo 2 / Exo-GPU | 外部调度 | 需要可验证自定义调度 | 3 平台，含 Hopper/Blackwell |
| Hidet | task mapping | 快速调优、调度后融合 | GPU/CPU |
| Ascend C / Bang C / MUSA 等 | 厂商私有 | 各自 NPU/GPU 原生开发 | 单一厂商 |

---

## 5. 自动调优、图优化与自动并行

### 5.1 自动张量化：从模板搜索到无模板生成

算子优化空间的组合爆炸，使「人工选 schedule」不可持续，自动调优由此成为 AI 编译器的核心能力，经历了三代演进：

- **AutoTVM（TVM）**：基于模板的搜索，开发者定义调度模板，搜索在参数空间内进行，配合学习型代价模型（XGBoost）预测性能（arXiv:1802.04799）；
- **Ansor（即 Auto-Tensorization）**：Lianmin Zheng 等《Ansor: Generating High-Performance Tensor Programs for Deep Learning》（**arXiv:2006.06762**，2020-06-11，OSDI'20）。**无需手写模板**：分层从计算定义采样生成子图结构与参数，再以进化搜索 + 学习代价模型筛选，在 Intel CPU 上最高 3.8×、ARM 2.6×、NVIDIA GPU 1.7× 优于当时 SOTA；
- **基于 rTile 的快速编译**：代表性工作 **Roller**（《Fast and Efficient Tensor Compilation》，OSDI 2022，未上 arXiv）把算子编译从「搜索数小时」压缩到秒级——通过构造 rTile（封装硬件关键参数的瓦片）直接生成少量高性能候选而非大规模搜索；**Welder**（《Scheduling General Deep Learning Operator Fusion》，MLSys 2023，未上 arXiv）则解决通用算子融合的调度，对内存瓶颈型负载收益显著。

阿里的动态形状编译器 **BladeDISC** 走另一条路：Xiulong Yuan 等《BladeDISC++: Memory Optimizations Based On Symbolic Shape》（**arXiv:2412.16985**，2024-12-22）在符号化形状下做算子调度与重计算，使动态图显存消耗逼近「已知精确形状」的优化效果，解决生产推理中形状频繁变化导致反复编译/无法静态优化的痛点。

### 5.2 图捕获与编译：PyTorch 2.x 的工程现实

PyTorch 2.x 以 `torch.compile` 把编译器能力引入主流框架，其栈为 **Dynamo（字节码分析做图捕获）＋ AOTAutograd（提前求导）＋ Inductor（默认代码生成后端，生成 Triton/C++）**。工程上的关键痛点是 **graph break（图断裂）**：当捕获遇到不支持的 Python 特性/动态控制流时，Dynamo 被迫从图中退回 eager，打断优化。

Savini Kashmira 等《GraphMend: Code Transformations for Fixing Graph Breaks in PyTorch 2》（**arXiv:2509.16248**，2025-09-17）针对真实代码库做源码级自动修复，可消除 147 个 break 中的 107 个（73%），冷启动最高加速 26×（平均 5×）——这说明「让更多模型被编译器完整捕获」仍是当前工程优化的活跃前沿。

### 5.3 自动并行：从手工切分到编译器生成计划

分布式训练的并行策略（数据并行、张量并行、流水并行及其组合）搜索空间同样巨大：

- **GSPMD**：Yuanzhong Xu 等《GSPMD: General and Scalable Parallelization for ML Computation Graphs》（**arXiv:2105.04663**，2021-05-10），XLA 后端的通用自动分片，用户只需给出少量分片注解，编译器推断全图分区；在 2048 个 Cloud TPUv3 核、万亿参数模型上取得 50%–62% 的算力利用率；
- **Alpa**：Lianmin Zheng 等《Alpa: Automating Inter- and Intra-Operator Parallelism for Deep Learning》（**arXiv:2201.12023**，2022-01-28，OSDI'22），把**算子间并行（流水并行）与算子内并行（数据/张量并行）统一建模并自动生成**计划，结果匹配甚至超过手工调优的模型并行系统。

自动并行与 AI 编译器的合流，意味着「给定模型 + 给定集群拓扑，编译器自动决定怎么切、怎么插通信、怎么编排流水」正从研究走向工程——这对国产多芯集群（往往是不同型号/厂商芯片混编）尤其关键。

---

## 6. 2025–2026 新范式：LLM/Agent 自动写 Kernel

### 6.1 为什么这是异构普惠的关键一环

「重写算子」是 CUDA 生态迁移中最昂贵的人力瓶颈：一个新硬件后端要可用，需要数百个算子达到有竞争力的性能，而顶级 kernel 工程师是稀缺资源。2025 年起，一个直接冲击这一瓶颈的方向迅速升温——**让 LLM/Agent 直接生成、测试、迭代优化 GPU kernel**。如果这一路线成立，「为国产 NPU 补齐算子生态」的边际成本将从「招聘大量 kernel 工程师」变为「跑智能体 + 硬件在环验证」，这与智源趋势 9「繁荣的算子语言与趋于收敛的编译器技术正在降低开发门槛」的判断直接呼应。

### 6.2 评测基准：KernelBench 与「开箱能力」的冷水

Anne Ouyang（Stanford）等《KernelBench: Can LLMs Write Efficient GPU Kernels?》（**arXiv:2502.10517**，2025-02-14）提供了这一方向的标准评测：250 个来自 PyTorch 算子的任务，从易到难分 L1/L2/L3，核心指标是 `fast_p`（生成 kernel 比 PyTorch 基线更快的任务比例）。其结论给行业泼了冷水也指明了方向：**即使最强的前沿模型，开箱（无迭代、无编译反馈）也只能在不到 20% 的用例上追平 PyTorch 基线**——单次生成不可靠，必须靠「编译—运行—反馈—迭代」的智能体闭环。

### 6.3 三条技术路线与代表性工作

**路线一：面向更高层抽象生成（而非裸 CUDA）。** 裸 CUDA 涉及海量线程索引与内存层级细节，生成易错；面向 CuTe/Triton/TileLang 等结构化抽象生成，搜索空间更小、正确性更易保证。

- Tara Saba 等《CuTeGen》（**arXiv:2604.01489**，2026-04-01）：面向 NVIDIA CuTe 抽象层（而非裸 CUDA）的「生成—测试—精炼」智能体，在 KernelBench L1/L2 共 209 个任务上平均 1.71× 于 PyTorch，优于直接生成 CUDA 的 CudaForge（0.89×）；
- Jinye Du 等《AKG kernel Agent: A Multi-Agent Framework for Cross-Platform Kernel Synthesis》（**arXiv:2512.23424**，2025-12-29）：多智能体框架支持跨平台 kernel 生成/迁移/调优，目标语言覆盖 **Triton / TileLang / CPP / CUDA-C，后端覆盖 GPU 与 NPU**，在 GPU+NPU 后端的 KernelBench 上平均比 PyTorch Eager 快 1.46×——这是学术界少有的明确面向「GPU+NPU 跨平台」的 kernel 智能体。

**路线二：大规模 Agentic RL 训练专用 kernel 能力。** 把「写 kernel」当作可强化学习的技能，以编译/正确性/性能为奖励大规模训练。

- Weinan Dai 等《CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation》（**arXiv:2602.24286**，2026-02-27）：用大规模智能体强化学习训练 CUDA kernel 优化能力，在 KernelBench L1/L2/L3 上对 `torch.compile` 的「更快率」达到 **100% / 100% / 92%**，L3 上比 Claude Opus 4.5、Gemini 3 Pro 高约 40 个百分点（论文口径）；
- Mao Luo 等《Agentic Kernel Optimization: Generating State-of-the-Art GPU Kernels Without Hand-Written CUDA》（**arXiv:2608.14560**，2026-05-25）：通用代码智能体（Houmao 多智能体框架）在**零手写 CUDA** 条件下产出 SOTA kernel，全过程消耗约 19 亿 token；在 B200 上对比 PyTorch，Fused MoE 达 92.68×、TopK Indexer 达 1101×、Sparse Attention 达 181×（论文口径，注意为特定算子的极端加速比，不代表整体模型平均水平）。

**路线三：用 AI 重写编译器后端本身。** 最激进的尝试是跳过传统编译器后端。

- François Costa、Charly Castes、Thomas Bourgeat、Azalia Mirhoseini（含 MIT/产业研究者）《AI as a Compiler: Compiling Triton kernels without the Triton compiler》（**arXiv:2609.36800**，2026-09-29）：用 LLM 智能体直接把 Triton 程序「AI lowering」成 PTX，取代传统的 TTIR→TTGIR→LLVM 后端流水线；在 Ada/Hopper/Blackwell 三代架构的 22 个 kernel 上达到 autotuned Triton 性能的 **0.83–3.34×**。这篇论文的象征意义大于当下实用性——它暗示「编译器后端的规则代码生成」本身可能成为 LLM 的应用场景，但稳定性、可复现性、形式正确性仍是未解问题。

综述方面，Yang Yu 等《Towards Automated Kernel Generation in the Era of LLMs》（**arXiv:2601.15727**，2026-01-22）系统梳理了 LLM/Agent kernel 生成的方法、数据集与基准分类学。

### 6.4 产业侧：KernelGen 与 CUDA Engineer

学术界之外，产业界已把这类能力产品化：

- **FlagOS KernelGen 2.2.0**：FlagOS 2.2 开发者工具之一，定位「自然语言 → Triton 算子」，配套 **KernelGenBench 0.2.0** 评测与 FlagOS Skills，直接服务于多后端算子补全（官方文档，docs.flagos.io，2026-10-21 核实）；
- **Sakana AI CUDA Engineer / robust-kbench**：2025 年推出的 CUDA kernel 智能体与更鲁棒的 KernelBench 评测变体，是该方向最早引发广泛关注的产业实践之一（第三方报道）。

### 6.5 能力边界与风险（必须冷静的部分）

| 维度 | 当前进展 | 边界/风险 |
|---|---|---|
| 规则算子（elementwise/归约/简单 GEMM） | 智能体闭环已稳定超过 PyTorch | 价值有限，本就是库的强项 |
| 复杂融合算子（MoE/注意力/稀疏） | RL 训练的专用智能体可达 SOTA | 依赖海量在环算力与验证，结果可复现性参差 |
| 正确性保证 | 靠测试用例 + 编译反馈迭代 | 测试覆盖不到的数值 corner case 仍可能静默错误 |
| 跨硬件迁移 | AKG/KernelGen 已支持 GPU+NPU | 目标硬件的编译器/性能计数器成熟度决定上限 |
| 取代编译器 | 「AI as a Compiler」初步可行 | 无法保证确定性与形式正确，尚不能用于生产工具链 |
| 人力替代 | 显著降低初级 kernel 工作量 | 硬件在环验证、内存层级调优仍需资深工程师 |

结论：**LLM 写 kernel 是「降低门槛」而非「消灭专家」**——它把算子生态迁移从纯人力密集型转变为「智能体生成 + 自动化验证 + 专家审计」的新流水线，是异构全栈在 2026 年最值得关注的使能技术之一，但生产环境必须保留正确性门禁与性能回归基线。

---

## 7. 全球异构战局：ROCm、oneAPI 与其他厂商栈

### 7.1 AMD ROCm：最成熟的 CUDA 替代栈

ROCm（Radeon Open Compute）是目前除 CUDA 外最完整的开源 GPU 软件栈。其核心是 **HIP（Heterogeneous-Computing Interface for Portability）**——一个面向 AMD GPU 的 C++ 运行时 API 与内核语言，官方称其为「thin API with little or no performance impact over coding directly in ROCm」；hipBLAS/hipFFT 等库与 NVIDIA 同名库 API 设计对齐，配套 **HIPIFY** 工具做 CUDA 源码级转译。组件完整对标 CUDA 生态：clang/hipcc 编译器、RCCL（对 NCCL）、MIOpen（对 cuDNN）、rocBLAS、MIGraphX（对 TensorRT）、rocprofiler、ROCgdb 等。

2025–2026 年 ROCm 有两个关键进展：

- **节奏加快、版本跃迁**：从 2024-12 的 6.3.0 一路迭代，2025-09 进入 7.0，2026-08-26 发布 **ROCm 10.0**，2026-10-05 更新到 **10.1.0（当前最新）**（官方版本表）；
- **开放构建体系 TheRock**：自 ROCm 7.14（2026-07-15）起，AMD 采用 TheRock 开放构建/发布系统，PyTorch 官方 ROCm wheel 即由 TheRock pip SDK 生产——这是 AMD 对标 CUDA 分发封闭性、拉拢开源社区的关键治理动作。

框架就绪度（ROCm 10.x 官方兼容矩阵，2026-08-26 页内版本）：

| 框架 | 官方验证版本 |
|---|---|
| PyTorch | 2.14 / 2.13 / 2.12 |
| JAX | 0.11.1 / 0.11.0 / 0.10.2 |
| vLLM | 0.29.0（需 PyTorch 2.13） |
| SGLang | 0.5.18 |
| TensorFlow | 2.21 |
| ONNX Runtime | 1.29 |

硬件覆盖 MI355X/MI350X（gfx950）、MI325X/MI300X（gfx942）并下延至 RDNA4 消费级。短板同样由官方自认：MI350X/MI300X 上 vLLM 重启可能触发 OOM（ROCm 10.1 Known Issues）。AMD 未公布 ROCm 下载量/采用率绝对数字，生态规模只能以框架版本矩阵与硬件出货间接衡量。

### 7.2 Intel：oneAPI 开源化与加速器商业失利

Intel 路线在 2025 年遭遇重大挫折，但其软件资产以开源方式存续：

- **oneAPI 治理转移**：oneAPI 已从 Intel 自有品牌转为 **UXL Foundation（Linux Foundation 旗下统一加速基金会）**治理，核心库 oneDNN 托管于 `uxlfoundation/oneDNN`；底层设备抽象 **Level Zero** 规范亦由 UXL 托管，Intel Compute Runtime 开源实现支持 Arc Alchemist/Battlemage 与核显；
- **Gaudi 商业失利被官方承认**：Intel FY2025 10-K（2026-01-23 提交）原文称「Our most recent efforts to compete in the accelerator market with our Gaudi AI accelerators was unsuccessful」，并披露 Gaudi 存货相关减值 **2025 年 3.75 亿美元、2024 年 9.22 亿美元**；
- **Falcon Shores 取消**：2025-01-30 财报电话会宣布 Falcon Shores 不上市、转为内部测试芯片，未来押注机架级 Jaguar Shores；Gaudi 文档（v1.24.2）仍在维护，vLLM 保留 Intel XPU/CPU 后端入口。

Intel 的教训说明：**仅有软件栈不足以支撑加速器生态，硬件竞争力、供给规模与商业投入缺一不可**；而 oneAPI 转入 UXL 则表明，开放治理正在成为失意者联合制衡 CUDA 的共同选择。

### 7.3 其他玩家速览

- **Google TPU + XLA/JAX**：TPU 走编译器路线，XLA 是唯一一等编程入口，JAX/PyTorch-XLA 经 XLA 编译到 TPU，Pallas 提供 TPU 上的 tile 级 kernel 编程；
- **Apple MLX**：面向 Apple silicon 统一内存的数组框架（类 NumPy/PyTorch），vLLM 经社区插件 vLLM-Metal（MLX 后端）支持 Apple Silicon；
- **OpenXLA 社区**：2022 年由 Google 联合 AMD、Apple、Arm、Intel、Meta、NVIDIA 共同发起（NVIDIA 也是成员，凸显 XLA 在 TPU 之外的中立价值）；
- **Qualcomm / 三星端侧栈**：主要依托私有 AI Stack 与 Android NNAPI，公开技术资料有限。

---

## 8. 中国国产算力软件栈与 FlagOS 深度拆解

> ![](diagrams/4-china-ecosystem.html)

### 8.1 国产厂商软件栈盘点

| 厂商 | 软件栈/算子语言 | 现状（2026-10-21 核实） |
|---|---|---|
| 华为昇腾 | **CANN**（文档已至 9.1.0）、算子语言 **Ascend C**、框架 **MindSpore** | 产品线含 Ascend 950PR/950DT、Atlas 950/900 A3 SuperPoD（灵衢互联，384 卡）；ModelZoo 已收录 DeepSeek-V3/R1 及多个蒸馏模型 |
| 寒武纪 | **NeuWare**（CNNL/MagicMind）、算子语言 **Bang C** | 官网本次不可达，行业公认口径，未获一手页面证实 |
| 摩尔线程 | **MUSA**（MUSACODE、KUAE 云原生套件） | MTT AIOS 1.6.0（2026-10-12），硬件 MTT S5000/KUAE 集群 |
| 沐曦 | **MACA**（官网前端证实；网传「GPNNA」无任何来源） | 开发者站在线，2025-06 接入 FlagTree |
| 燧原 | **驭算 TopsRider** | 2026-09-11 登陆上交所科创板（688801.SH），云燧 OGX+ 超节点 |
| 海光 DCU | **DTK**（行业共识为类 ROCm/HIP 路线，海光官方一手表述未取得） | FlagTree 后端标识 hcu，2025-09-16 接入 |
| 昆仑芯 | 软件栈名称「XTK/XCCL」本次未获证实 | 产品 R100/R200/RG800 等，开发者站不可达 |
| 天数智芯 | 自有软件栈 | 2025-03 首批接入 FlagTree |

昇腾是国产软件栈中最完整者：CANN 提供从驱动、编译器、算子库（Ascend C 算子开发）到 MindSpore 框架与 MindSpeed 大模型套件的全栈，DeepSeek-V3/R1 进入官方 ModelZoo 是其大模型适配能力的标志性证据。但需如实指出：2025–2026 年昇腾开发者/适配模型的当季官方总数未在可访问页面取得（官网为 SPA），报告不沿用旧口径。

### 8.2 FlagOS 深度拆解：「一次开发、多芯运行」

**定位**。FlagOS 官方定义（docs.flagos.io，2026-10-03 构建）：「一个完全开源的异构 AI 芯片系统软件栈，允许 AI 模型一次开发即可无缝移植到广泛的 AI 硬件平台，实现最小化的适配成本」，当前版本 **FlagOS 2.2**。与各家私有栈不同，FlagOS 的定位是**中立的公共底座**——由智源牵头、社区共创，目标不是替代某一家的栈，而是让同一套算子/编译/通信/框架适配代码在所有成员芯片上运行。

**架构分层（2.2 官方组件）**：

| 层 | 组件（版本） |
|---|---|
| 算子库 | FlagGems 5.4.0（Triton 通用算子）、FlagGems-vllm 0.2.0、FlagGems-sglang 0.1.0 |
| 多领域算子 | FlagDNN/FlagBLAS/FlagSparse/FlagTensor 0.3.0、FlagFFT/FlagAudio 0.2.0、FlagAttention 0.4.0（dense/paged/sparse KV、recurrent linear attention） |
| 统一编译器 | **FlagTree**（GPU 0.7.0 + Triton 3.6/3.5/3.3；CPU 0.1.0） |
| 全生命周期 | FlagScale 2.1.0（基于 Megatron-LM/vLLM） |
| 统一通信 | **FlagCX 0.14.0**（跨芯片 P2P/集合通信，IPC + RDMA） |
| 框架插件 | **Torch-FL 2.10.0（基于 PyTorch PrivateUse1）**、Megatron-LM-FL、TransformerEngine-FL、verl-FL |
| 引擎插件 | vllm-plugin-FL、sglang-plugin-FL 0.2.0 |
| 开发者工具 | **KernelGen 2.2.0（自然语言→Triton）**、KernelGenBench、FlagOS Skills |
| 平台服务 | FlagPerf 1.2.0（硬件评测）、FlagRelease、FlagCICD |
| 领域项目 | FlagOS-Robo 0.1.0（VLA 具身）、FlagQuantum 0.1.0 |

**FlagTree：多后端统一编译器**。这是 FlagOS 的技术核心。它不是另起炉灶的新语言，而是**向上游推进、向下游分叉扩展 Triton**：扩展包括 TLE（Triton Language Extensions: Lite/Struct/Raw）、FLIR 与 HINTS。后端覆盖（据官方 README，2026-10-21）：

- **main 分支（Triton 3.6）**：NVIDIA（含 NVIDIA TileIR）、AMD、燧原、天数智芯、海光(hcu)、摩尔线程、阿里达摩院(thrive)、辉羲(rpu)、沐曦、曦望芯科、KLX、平头哥(ppu)、进迭时空、清微智能等；
- **独立分支**：昇腾在 `triton_v3.5.x`、寒武纪在 `triton_v3.2.x`、安谋科技(aipu)与 CPU 在 3.3 分支。

接入时间线：2025-03 天数智芯/摩尔线程等首批 → 2025-06 沐曦、昇腾（3.2）、清微 → 2025-09 海光、支持 Paddle → 2025-12 燧原 → 2026 年各厂陆续升 3.5/3.6。

**演进时间线**：

| 时间 | 事件 | 口径 |
|---|---|---|
| 2024-09 | FlagOS 1.0 于中关村论坛发布 | 第三方快照 |
| 2025-09-26 | FlagOS 1.5 开发者大会，18 个共创团队 | 知乎快照 |
| 2026-03-27 | **FlagOS 2.0：32 款 AI 芯片、497 个算子**（中关村论坛） | 知乎快照口径，待官方佐证 |
| 2026-10 | FlagOS 2.2（当前） | 官方 |

**生态有效性的硬证据**比口径数字更有说服力：2026-08-27 **Qwen3.8-Flash-Next 发布首日（Day0）适配 8 款芯片**，2026-08-28 **GLM-5.3-Flash Day0 适配 9 款芯片**（FlagOS CSDN 官方社区）。「新模型发布当天就能在近十款国产芯片上跑」在 FlagOS 出现前是不可想象的，这直接验证了「最小化适配成本」的价值主张。GitHub 组织 `flagos-ai` 有 30 个仓库。

### 8.3 冷静评估：FlagOS 还没解决什么

1. **性能透明度不足**：目前没有官方发布的 FlagOS vs CUDA/ROCm 原生的系统 benchmark，超大规模训练的性能损失、稳定性、长时程可靠性仍需使用方逐模型验证；
2. **统一仓库目标未完全达成**：最重要的昇腾、寒武纪仍在独立分支而非 main，意味着跨后端代码尚未真正完全收敛；
3. **算子深度与广度**：497 个算子（2.0 快照口径）相对 CUDA-X 900+ 库的覆盖面仍有量级差距，冷门/前沿算子仍需自行补齐；
4. **生态治理的可持续性**：依托智源与项目制资金，能否长期维持中立运营、吸引更多商业贡献者，仍待观察。

---

## 9. 开放互联与标准生态：UALink、UEC、UXL 与兼容层

异构全栈不只是「单卡编译器」，多卡/多机集群的互联与通信同样是 CUDA 生态锁定的一环（NVLink/NVSwitch + NCCL）。2024–2026 年，开放阵营在这一层完成了关键的标准布局：

### 9.1 UALink：对标 NVLink 的开放背板/铜缆互联

- **2024-05-30**：AMD、Arm、Astera Labs、Broadcom、Cisco、Eviden、Google、HPE、Intel、Meta、Microsoft 等成立 UALink Promoter Group——阵容几乎是「除 NVIDIA 外的全部算力巨头」；
- **2025-01-14**：阿里巴巴、Apple、Synopsys 加入董事会；
- **2025-04-08**：发布 **UALink 200G 1.0 规范**（单通道 200G，对标同期 NVLink）；
- **2026-04-07**：一次发布四份规范（In-Network Compute、Chiplets、Manageability、200G 性能），并出现 UALink 2.0 支持声明。

### 9.2 Ultra Ethernet Consortium：把扩展寄托在以太网上

UEC（Linux Foundation JDF 项目）走「以太网原生 AI 通信」路线，试图用开放、可大规模扩展的以太网替代专有 Scale-Up 互联：**2025-06-11 发布 1.0**，经 1.0.1–1.0.3 勘误，**2026-10-12 更新至 1.1（当前推荐版本）**。通信库层面，FlagOS 的 FlagCX（跨芯片 IPC + RDMA）与 RCCL、NCCL 替代生态共同构成软件侧的开放通信栈。

### 9.3 UXL Foundation 与 PyTorch 的后端治理

- **UXL Foundation** 托管 oneAPI 规范、oneDNN、Level Zero 等，定位为「跨所有厂商加速器的开放软件层」，是 Intel 失意后联合开放阵营的制度载体；
- **PyTorch 的两个官方扩展点**使第三方硬件接入标准化：其一是 `torch._dynamo.register_backend` 注册编译后端（输入 FX Graph）；其二是 **PrivateUse1 设备 key**——允许注册非 CUDA 自定义设备，FlagOS Torch-FL 即基于此把多后端统一路由到 `flagos` 设备名下。PyTorch 2.14 还把容错（Flight Recorder）扩展为适用于任意后端的一等概念，而非仅 NCCL。

### 9.4 二进制兼容层：ZLUDA 的价值与局限

ZLUDA 是非 NVIDIA GPU 上 CUDA 的「直接替换」（drop-in），允许未修改的 CUDA 程序运行，底层主要依托 ROCm。**2026-06 发布 Version 6**，扩大了 PyTorch trace、Blender、32 位 PhysX（预览）兼容面。但其局限是决定性的：项目已无商业资助、回到作者个人「周末维护」，ML 场景需逐模型验证。结论是——**「可以试，不该押；适合验证，不适合作为正式工具链底座」**。兼容层可以作为迁移期的过渡桥梁，但无法替代原生栈的长期价值。

### 9.5 标准生态的结构性观察

| 层 | NVIDIA 专有 | 开放替代 | 状态（2026-10） |
|---|---|---|---|
| 单卡编程 | CUDA / PTX | Triton/TileLang/MLIR 系 | 开放语言已具竞争力 |
| 编译器底座 | nvcc | MLIR/OpenXLA/FlagTree | 收敛中，公共底座成型 |
| Scale-Up 互联 | NVLink/NVSwitch | UALink | 1.0 已发布，产品落地早期 |
| Scale-Out 网络 | Spectrum/InfiniBand | Ultra Ethernet | 1.1，生态采用起步 |
| 通信库 | NCCL | RCCL/FlagCX/oneCCL | 单芯成熟，跨芯仍早期 |
| 二进制兼容 | — | ZLUDA | 个人维护，仅作过渡 |

---

## 10. 出口管制时间线：软件栈替代的需求动因

理解中国市场对异构全栈的迫切需求，必须回到 2024–2026 的出口管制时间线。以下关键财务数字均来自 NVIDIA 向 SEC 的一手披露。

### 10.1 规则演进

- **2024-12-05**：BIS 发布《Foreign-Produced Direct Product Rule Additions...》（联邦公报 2024-28270），增补外国直接产品规则；
- **2025-01-15**：拜登政府发布《Framework for Artificial Intelligence Diffusion》（联邦公报 2025-00636），按国家分层（Tier 1/2/3）设定芯片出口与算力配额；2025 年 5 月政府宣布撤销该框架并将以替代规则取代；截至本报告，BIS 以《Guidance on Advanced Computing Items》（2026-05-31 版）与 FAQ 实施许可口径。

### 10.2 H20 事件：一场 45 亿美元的「断供演练」

| 时间 | 事件 | 口径 |
|---|---|---|
| 2025-04-09 | 美政府通知 H20 及同等带宽芯片对华出口需许可证；NVIDIA 8-K 预警最高约 **55 亿美元**费用 | 官方 8-K |
| 2025-04-14 | 限制称无限期；Q1 FY26 实际计提 **45 亿美元**（存货减值 19 亿 + 采购承诺 26 亿）；管制前该季度 H20 销售 **46 亿美元** | 官方 10-Q |
| 2025-04-23 | 媒体称中国企业禁令前大规模囤货（第三方报告口径） | 第三方 |
| 2025-07-31 | 中国网信办约谈 NVIDIA，要求说明 H20 是否存在安全漏洞/隐蔽追踪功能 | 第三方 |
| 2025-08 | H20 获批向部分客户出货（美方预期抽取 **15%** 收入）；Q2 对中国客户销售为零、对中国以外客户销售 **6.5 亿美元** | 官方 10-Q |
| FY2026 全年 | 许可证下已确认约 **6000 万美元** H20 收入 | 官方 10-K |
| 2026-02 | 少量 **H200** 个案放行（需在美检验）；媒体称字节、腾讯接收首批 | 官方 10-K + 第三方 |

### 10.3 对软件栈竞争的三重影响

1. **需求侧**：断供的现实风险使「单一 CUDA 依赖」成为不可接受的供应链结构，国产芯片 + 国产/开放软件栈从「备份选项」变为「并行采购的刚需」；
2. **供给侧**：NVIDIA 被迫接受许可制、收入抽成与在美检验等条件，其商业模式在中国市场的确定性下降，客观上为替代栈让出时间窗口；
3. **安全侧**：网信办对 H20 隐蔽追踪/安全漏洞的关切，把「软件栈可信」纳入与「供应链安全」同等的高度——这与趋势 10（AI 安全）形成呼应，软硬两条线的自主可控诉求合流。

第三方对国产硬件的能力估算（须保留「媒体估算」标注）：Tom's Hardware（2025-04-19）称华为 Ascend 910C 推理性能约为 H100 的 60%，下一代 Ascend 920 采用 6nm 工艺、单卡算力 >900 TFLOPS、显存带宽 4TB/s。这些数字非华为官方规格，仅作能力量级参考。

> 政策面补充：公开可核实的方向性事实是，监管层通过政府采购导向、信创目录与智算中心建设推动国产芯片与软件栈适配；但带文号的「国产算力强制适配」中央政策原文本次未取得，本报告只做方向定性，不杜撰文件号。

---

## 11. 技术选型与工程落地建议

### 11.1 不同角色的策略矩阵

| 角色 | 建议策略 | 关键理由 |
|---|---|---|
| 大模型训练方（万卡级） | CUDA 栈为主力，国产栈按 20–30% 比例并行建设；模型设计阶段即做「多芯友好」约束（避免厂商私有算子） | 供应风险对冲；训练稳定性仍以 CUDA 最成熟 |
| 推理服务提供商 | 优先 Triton/TorchCompile 层抽象；用 vLLM/SGLang + 插件接入多后端，按性价比跨芯调度 | 推理负载迁移成本低，跨芯套利空间大 |
| 应用/Agent 开发者 | 坚持框架层（PyTorch/JAX）编程，不直接依赖私有 CUDA 扩展；用 Liger/FlashInfer 等 Triton 生态库 | 保持后端可切换，避免锁定 |
| 国产芯片厂商 | 优先接入 FlagTree/PrivateUse1 与主流引擎插件，而非只维护私有 SDK；补齐 profiler/性能计数器 | 生态可见度决定采购，编译器接入成本最低 |
| Kernel/系统工程师 | 投入 Triton/TileLang/CuTe 与 kernel agent 工作流，把精力转向验证与内存层级调优 | LLM 接管样板代码，专家价值上移 |

### 11.2 迁移到异构后端的工程 Checklist

1. **抽象审计**：盘点代码中所有直接 CUDA 依赖（自定义扩展、`.cu`、NCCL 直接调用、硬编码 dtype/shape），替换为 Triton/框架等价物；
2. **编译路径打通**：在目标后端跑通 `torch.compile`（或 FlagOS Torch-FL），记录 graph break 清单并优先修复高频断裂点（参考 GraphMend 思路）；
3. **算子覆盖核对**：对照目标硬件算子库（FlagGems/厂商库）逐一验证算子存在性与数值一致性（ATOL/RTOL）；
4. **正确性门禁**：建立 CPU/参考实现的数值对拍、边界 shape、长序列与低精度（FP8/FP4）专项测试——kernel agent 生成的代码必须过此门禁；
5. **性能基线**：以 FlagPerf 或自建基准建立逐算子与端到端基线，设置性能回归阈值，避免「能跑但悄悄变慢」；
6. **通信与集群**：验证跨芯片集合通信（FlagCX/RCCL）、并行策略（结合 Alpa/GSPMD 式自动分片）与容错（Flight Recorder 类）；
7. **灰度与双跑**：新后端先承接推理/容错性高的流量，与 CUDA 双跑比对一段时间后再扩大比例；
8. **供应链合规**：关注出口管制与许可状态变化，合同中明确硬件/软件栈的持续供给与升级条款。

### 11.3 一个务实判断

短期内（1–2 年）不存在「CUDA 的全面替代者」，但**「CUDA 唯一栈」的时代已经结束**。更可能的终态是一个分层结构：**NVIDIA 栈在最高端训练保持领先；ROCm 在开放 GPU 市场分食；中国市场形成「昇腾私有全栈 + FlagOS 公共底座」双轨；Triton/MLIR 成为跨栈的共同算子与编译语言；LLM kernel agent 把跨栈迁移成本持续压低。** 对绝大多数团队而言，最优策略不是「押注某一个替代栈」，而是「把代码写在可移植的抽象层，保留随时切换后端的自由」。

---

## 12. 开放问题与展望

1. **性能可比性的公开证据**：替代栈亟需中立、可复现的端到端 benchmark（训练稳定性、长时间运行可靠性、故障恢复），而非单算子加速比——这是采购决策从「敢试」到「敢押」的前提。
2. **形式化正确性**：kernel agent 与「AI as a Compiler」目前靠测试保证正确性；未来需要把形式化验证、符号执行与生成流程结合，覆盖测试触不到的数值 corner case。
3. **真正的统一仓库**：FlagTree 能否把昇腾、寒武纪等主力后端收敛进 main 分支，是「一次开发、多芯运行」从口号变现实的关键指标。
4. **异构混编调度**：当集群由不同厂商/型号芯片组成，编译器与调度器需要联合优化「算子放置—通信—负载均衡」，这比单后端编译难得多，也是 Alpa/GSPMD 路线的下一站。
5. **开放互联的规模化验证**：UALink/UEC 需要经过万卡级真实训练的检验，才能证明可替代 NVLink/InfiniBand；软件通信库（FlagCX 等）的成熟度要同步跟进。
6. **生态治理的长期性**：FlagOS、UXL、OpenXLA 都面临资金、中立性与商业贡献者激励问题；开放治理能否穿越技术周期，尚无定论。
7. **与智能范式的协同**：世界模型、VLA 等新负载带来全新算子模式（仿真、物理求解、动作生成），谁先让这些新算子在多后端高效运行，谁就掌握下一阶段的生态定义权。

**总判断**：智源把「开源编译器生态与异构全栈底座」列入 2026 十大趋势，本质上是在标记一个事实——**算力的竞争已从「造出更快的芯片」扩展为「构建让所有芯片都能高效工作的软件生态」**。Triton/MLIR 解决了语言与编译器的收敛问题，LLM kernel agent 正在解决人力瓶颈，FlagOS/ROCm/UALink/UEC 则在补齐全栈与互联的最后拼图。技术方向已经清晰，剩下的是工程深度、生态耐心与时间。

---

## 13. 参考文献

### 13.1 arXiv 论文（23 篇，编号经 arxiv.org 元数据逐一核实，2026-10-21）

**编译器 IR / 语言底座**

1. Tianqi Chen, Thierry Moreau, Ziheng Jiang, et al. **TVM: An Automated End-to-End Optimizing Compiler for Deep Learning**. arXiv:1802.04799（2018-02-12；OSDI'18）. https://arxiv.org/abs/1802.04799
2. Chris Lattner, Mehdi Amini, Uday Bondhugula, et al. **MLIR: A Compiler Infrastructure for the End of Moore's Law**. arXiv:2002.11054（2020-02-25）. https://arxiv.org/abs/2002.11054
3. Ruihang Lai, Junru Shao, Siyuan Feng, et al. **Relax: Composable Abstractions for End-to-End Dynamic Machine Learning**. arXiv:2311.02103（2023-11-01；ASPLOS'25）. https://arxiv.org/abs/2311.02103
4. Lianmin Zheng, Chengfan Jia, Minmin Sun, et al. **Ansor: Generating High-Performance Tensor Programs for Deep Learning**. arXiv:2006.06762（2020-06-11；OSDI'20）. https://arxiv.org/abs/2006.06762

**Kernel 语言 / 编程模型**

5. Benjamin F. Spector, Simran Arora, Aaryan Singhal, et al. **ThunderKittens: Simple, Fast, and Adorable AI Kernels**. arXiv:2410.20399（2024-10-27）. https://arxiv.org/abs/2410.20399
6. Lei Wang, Yu Cheng, Yining Shi, et al. **TileLang: A Composable Tiled Programming Model for AI Systems**. arXiv:2504.17577（2025-04-24）. https://arxiv.org/abs/2504.17577
7. Yuka Ikarashi, Kevin Qian, Samir Droubi, et al. **Exo 2: Growing a Scheduling Language**. arXiv:2411.07211（2024-11-11）. https://arxiv.org/abs/2411.07211
8. David Zhao Akeley, Yuka Ikarashi, Jonathan Ragan-Kelley. **Exo-GPU: Safe, Imperative, User-schedulable Programming for Tensor Cores**. arXiv:2609.16389（2026-09-14）. https://arxiv.org/abs/2609.16389
9. Yaoyao Ding, Cody Hao Yu, Bojian Zheng, et al. **Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs**. arXiv:2210.09603（2022-10-18）. https://arxiv.org/abs/2210.09603
10. Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, et al. **Liger Kernel: Efficient Triton Kernels for LLM Training**. arXiv:2410.10989（2024-10-14）. https://arxiv.org/abs/2410.10989

**自动并行 / XLA 生态**

11. Lianmin Zheng, Zhuohan Li, Hao Zhang, et al. **Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning**. arXiv:2201.12023（2022-01-28；OSDI'22）. https://arxiv.org/abs/2201.12023
12. Yuanzhong Xu, HyoukJoong Lee, Dehao Chen, et al. **GSPMD: General and Scalable Parallelization for ML Computation Graphs**. arXiv:2105.04663（2021-05-10）. https://arxiv.org/abs/2105.04663

**MLC / 推理引擎 / 动态编译**

13. Charlie F. Ruan, Yucheng Qin, Akaash R. Parthasarathy, et al. **WebLLM: A High-Performance In-Browser LLM Inference Engine**. arXiv:2412.15803（2024-12-20）. https://arxiv.org/abs/2412.15803
14. Zihao Ye, Lequn Chen, Ruihang Lai, et al. **FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving**. arXiv:2501.01005（2025-01-02）. https://arxiv.org/abs/2501.01005
15. Xiulong Yuan, Xu Yan, Wenting Shen, et al. **BladeDISC++: Memory Optimizations Based On Symbolic Shape**. arXiv:2412.16985（2024-12-22）. https://arxiv.org/abs/2412.16985
16. Savini Kashmira, Jayanaka Dantanarayana, Thamirawaran Sathiyalogeswaran, et al. **GraphMend: Code Transformations for Fixing Graph Breaks in PyTorch 2**. arXiv:2509.16248（2025-09-17）. https://arxiv.org/abs/2509.16248

**LLM/Agent 驱动的 Kernel 自动生成（2025–2026 热点）**

17. Anne Ouyang, Simon Guo, Simran Arora, et al. **KernelBench: Can LLMs Write Efficient GPU Kernels?**. arXiv:2502.10517（2025-02-14）. https://arxiv.org/abs/2502.10517
18. François Costa, Charly Castes, Thomas Bourgeat, Azalia Mirhoseini. **AI as a Compiler: Compiling Triton kernels without the Triton compiler**. arXiv:2609.36800（2026-09-29）. https://arxiv.org/abs/2609.36800
19. Tara Saba, Zhiyang Chen, Jikai Jason Li, et al. **CuTeGen: An LLM-Based Agentic Framework for … Kernels using CuTe**. arXiv:2604.01489（2026-04-01）. https://arxiv.org/abs/2604.01489
20. Weinan Dai, Hanlin Wu, Qiying Yu, et al. **CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation**. arXiv:2602.24286（2026-02-27）. https://arxiv.org/abs/2602.24286
21. Jinye Du, Quan Yuan, Zuyao Zhang, et al. **AKG kernel Agent: A Multi-Agent Framework for Cross-Platform Kernel Synthesis**. arXiv:2512.23424（2025-12-29）. https://arxiv.org/abs/2512.23424
22. Mao Luo, Hongbin Li, Feng Lin, et al. **Agentic Kernel Optimization: Generating State-of-the-Art GPU Kernels Without Hand-Written CUDA**. arXiv:2608.14560（2026-05-25）. https://arxiv.org/abs/2608.14560
23. Yang Yu, Peiyu Zang, Chi Hsu Tsai, et al. **Towards Automated Kernel Generation in the Era of LLMs（综述）**. arXiv:2601.15727（2026-01-22）. https://arxiv.org/abs/2601.15727

### 13.2 非 arXiv 的正式论文出处

- Philippe Tillet, H. T. Kung, David Cox. **Triton: an intermediate language and compiler for tiled neural network computations**. MAPL 2019（ACM DOI: 10.1145/3315508.3329973）.
- Hongyu Zhu, Ruofan Wu, et al. **Roller: Fast and Efficient Tensor Compilation for Deep Learning**. OSDI 2022, USENIX.
- Shixiong Zheng, et al. **Welder: Scheduling General Deep Learning Operator Fusion through Adaptive Epilogue Fusion**. MLSys 2023.
- Yuka Ikarashi, et al. **Exocompilation for Productive Programming of Hardware Accelerators**. PLDI 2022.
- Yihua Lai, et al. **HeteroCL: A Multi-Paradigm Programming Infrastructure for Software-Defined Reconfigurable Computing**. FPGA 2019.
- Yuxi Cai, et al. **PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections**. OSDI 2021.

### 13.3 官方与一手信源

- 新华网：《智源发布 2026 十大 AI 技术趋势》（2026-01-08）. http://www.news.cn/tech/20260108/b99127c88a4640bbab49e5c6294264d2/c.html
- NVIDIA FY2025 / FY2026 Form 10-K，SEC EDGAR（CUDA 开发者 590 万 / 750 万）.
- NVIDIA 8-K（2025-04-09）、Q1/Q2 FY26 Form 10-Q、FY2026 10-K，SEC EDGAR（H20 事件财务披露）.
- NVIDIA Developer：CUDA Toolkit 12.8 Release Notes、CUDA Toolkit Archive（13.4.3 / Tile IR）、GB200/GH200 技术博客（900+ CUDA-X）.
- AMD ROCm 官方文档：Release Versions（10.1.0）、General Release Notes（TheRock）、Compatibility Matrix（PyTorch 2.14 / vLLM 0.29 / SGLang 0.5.18）、What is HIP.
- Intel FY2025 Form 10-K，SEC（Gaudi 失利与减值）；Tom's Hardware（Falcon Shores 取消，2025-01-31，引财报电话会原话）.
- UXL Foundation：oneDNN、Level Zero 规范（GitHub uxlfoundation）.
- FlagOS 官方文档 docs.flagos.io（2.2 组件版本，2026-10-03 构建）；GitHub flagos-ai/FlagTree README（后端清单与接入时间线）；FlagOS CSDN 官方社区（Day0 适配）.
- 华为昇腾社区 hiascend.com（CANN 9.1.0、Ascend 950、Atlas SuperPoD、DeepSeek ModelZoo）；MindSpore 官网.
- 摩尔线程官网（MUSA / MTT AIOS 1.6.0）；沐曦官网（MACA）；燧原科技官网（驭算 TopsRider、科创板上市）；海光 DTK 页；昆仑芯/天数智芯官网.
- UALink Consortium 官网新闻室（200G 1.0，2025-04-08；2026-04-07 四份规范）；Ultra Ethernet Consortium Specification History（1.0，2025-06-11；1.1，2026-10-12）.
- PyTorch 2.14 发布博客（2026-09-02）与自定义后端/PrivateUse1 文档；JAX Installation；vLLM/SGLang 官方文档；OpenXLA GitHub.
- 美国联邦公报 2024-28270（2024-12-05）、2025-00636（2025-01-15）；BIS《Guidance on Advanced Computing Items》（2026-05-31）.
- ZLUDA GitHub README 与 Version 6 发布综述（ic.work，2026-06-30）；triton-lang/triton GitHub（Triton Conference 2026）.
- MLIR 官网 mlir.llvm.org；Apple MLX GitHub；Tom's Hardware（H20 安全关切、Ascend 910C/920 第三方估算）.

> **核实与口径声明**：本报告 23 篇 arXiv 论文的编号、标题、作者、日期均经 arxiv.org 摘要页 `citation_*` 元数据逐一核对（主代理另抽查 2609.36800、2602.24286、2608.14560、2504.17577 四篇，全部一致）。SEC/联邦公报/厂商官方文档为一手信源；FlagOS 2.0「32 芯片/497 算子」、Ascend 910C/920 性能等仅见于中文快照或媒体估算者，正文已显式标注，不作为官方数据。知乎全站调研期间返回 403，未作破解，相关社区视角仅以搜索引擎快照标题级线索使用。
