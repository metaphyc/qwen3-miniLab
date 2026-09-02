# qwen3-miniLab

用 Jupyter Notebook 一步一步拆开 Qwen3-0.6B-Base：每一步计算都自己重写，再和官方实现逐个张量对比。

这不是一份讲义，而是一份实验记录。每个结论都尽量用当场跑出来的数据说话；没有实际验证过的内容，就不写。

## 两个实验

| notebook                                               | 回答的问题                              |
| ------------------------------------------------------ | ---------------------------------- |
| [实验 1：外部骨架](notebooks/01_inspect_architecture.ipynb)   | 一个 token 进入模型后，要依次经过哪些主要模块？        |
| [实验 2：完整前向传播](notebooks/02_dissect_forward_pass.ipynb) | 这些计算到底是怎么进行的？能不能自己重新实现一遍，并和官方结果对上？ |

<div align="center">
  <img src="assets/qwen3-backbone-overview.png" alt="Qwen3 顶层结构总览图" width="320">
</div>

实验 2 从 `RMSNorm`、`Linear`、`SiLU` 三块基础积木开始，一路自己实现 embedding、RoPE、causal mask、GQA attention、SwiGLU，最后把这些模块拼成完整的 28 层模型，并完成端到端验证：

**输入一句话后，自己的实现和官方模型得到相同的 argmax。**

## 怎么算「验证过」

统一采用**尺度相对误差 < 1e-5**作为判断标准：用绝对误差除以两边张量的量级，尽量避免因为数值本身较大而把正常的浮点误差误判成实现错误。

实验 2 一共进行了 40 项验证，**全部通过**。

这里有两个刻意的选择：

* **使用 `float32`**，而不是直接使用磁盘上的 `bfloat16`。`bf16` 只有 7 位尾数，一次 1024 维矩阵乘法产生的舍入误差就可以达到 `3.75e-03`，已经明显高于我们的验证阈值。这样很难区分“实现写错了”和“bf16 本身带来的误差”。

* **使用 `eager attention`**，而不是默认的 `sdpa`。`sdpa` 会把整个 attention 融合到一个算子里，这样 attention weights 和 causal mask 都不方便直接拿出来观察，而这两部分正是本实验希望仔细看的。

## 跑起来

```bash
python -m venv .venv && . .venv/bin/activate    # 实测 Python 3.14.6

pip install -r requirements.txt
```

两本 Notebook 全程都在 CPU 上运行 `float32`，不需要 GPU。

在 Linux 上直接执行 `pip install torch` 时，默认可能会安装 CUDA 版本，光附带的 NVIDIA wheel 就会占掉 2.66 GiB。既然这里完全用不到 GPU，可以先安装 CPU 版本：

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt    # torch 已满足，不会被重新安装
```

实测这样安装后的环境大约占 1.3 GB；默认 CUDA 版本约 5.1 GB。整本 Notebook 跑完 40 项验证，结果同样全部通过。

## 下权重

模型权重不放进仓库，请自行下载到：

```text
models/Qwen3-0.6B-Base/
```

大小约 1.2 GB。安装 `huggingface_hub` 后就可以直接使用 `hf` 命令下载：

```bash
hf download Qwen/Qwen3-0.6B-Base --local-dir models/Qwen3-0.6B-Base
```

下载完成后，从 `notebooks/01_inspect_architecture.ipynb` 开始，按顺序运行即可。

## 目录

```text
notebooks/qwen_kit.py    共用工具：权重与中间状态目录、look/check/summary、HTML 表格

notebooks/0*.ipynb       实验本体，按编号顺序阅读

assets/                   结构图（PNG 用于展示，tex/ 下保存 TikZ 源码）

tools/                    协作脚本：setup-git.sh 配置一次，nb_clean.py 用于清理 Notebook

models/                   模型权重存放处，不入库
```

`qwen_kit` 主要是把重复的准备工作集中到一起：它会用 hook 把官方模型每一层的中间状态抓下来，整理成可以直接访问的 `qwen.L[i]` 和 `W.L[i]` 两套目录。

这样 Notebook 里就不用重复写这些准备代码，可以把篇幅留给真正的计算和比对。

## 一起改

**第一次 clone 后运行一次：**

```bash
pip install -r requirements-dev.txt
bash tools/setup-git.sh
```

**保存修改后提交前运行一次：**

```bash
git tidy
```

就这两条。

为什么需要 `git tidy`？

Notebook 跑过一遍以后，即使你一个字都没改，文件里的时间戳、cell 编号等信息也可能发生变化，所以 VS Code 会把它显示成“已修改”。

`git tidy` 会把这些运行产生的噪音清掉，让文件恢复成和仓库里一样的状态，修改标记也就消失了。

真正改过的内容不会被删掉——改了就是改了，仍然会正常显示。

就算忘了运行 `git tidy`，也不用担心把这些垃圾提交进仓库：`tools/setup-git.sh` 配置的过滤器会负责拦住它们。只是 VS Code 里的修改标记可能会一直亮着。

最后还有一条：

**同一本 Notebook 不要两个人同时修改。**

工具可以帮我们处理运行产生的噪音，但处理不了真正的代码冲突。最好一人负责一本，或者开始修改前先说一声。
