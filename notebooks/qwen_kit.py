"""实验 2 的工具箱：取数、显示、比对。

notebook 正文只用这几个名字：

    show(rows, header=, title=, fmt=, align=)   出一张 HTML 表格
    observe(model, input_ids)                   跑一次官方 forward，记下全部中间状态
    params(model)                               把全模型权重整理成可查的目录
    look(x)                                     看任何东西，按类型自动选显示方式
    check(mine, ref, name)                      比对两个张量，判据 max|差| / max|ref| < 1e-5
    record(name, ok, rel, max_abs)              登记一个不是逐元素比对的结论
    summary()                                   汇总本次记录到的全部 check

用之前先调一次 setup(model, tokenizer)：维度常量、字段表都从 model.config 现推，
不在这里写死数字。

搬到外部文件是因为这些都是脚手架，不是实验内容。notebook 里留下的是
"怎么用"和"为什么这么设计"，实现细节在这里。
"""
import html as _html
import inspect
from pathlib import Path

import torch
from IPython.display import HTML, display
from transformers.models.qwen3 import modeling_qwen3 as Q3


# ── 路径 ──────────────────────────────────────────────────────────────

def find_project_root() -> Path:
    for directory in (Path.cwd(), *Path.cwd().parents):
        if (directory / 'models' / 'Qwen3-0.6B-Base').is_dir():
            return directory
    raise FileNotFoundError('找不到 models/Qwen3-0.6B-Base')


PROJECT_ROOT = find_project_root()
MODEL_PATH = PROJECT_ROOT / 'models' / 'Qwen3-0.6B-Base'
SOURCE_PATH = Path(Q3.__file__)


def show_path(path: Path) -> str:
    """把绝对路径收敛成可展示的相对形式，输出里不带机器上的绝对路径。

    项目内的相对 PROJECT_ROOT；第三方库的从 site-packages 起算（层数不写死，
    换个 venv 布局也不会切错）。
    """
    path = Path(path)
    if path.is_relative_to(PROJECT_ROOT):
        return str(path.relative_to(PROJECT_ROOT))
    if 'site-packages' in path.parts:
        cut = path.parts.index('site-packages')
        return '.../' + '/'.join(path.parts[cut:])
    return path.name



# ── 表格 ──────────────────────────────────────────────────────────────

import html as _html

_TABLE_CSS = """<style>
.ktab{border-collapse:collapse;margin:.35em 0;
      font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.ktab caption{text-align:left;padding:0 0 .45em;font-weight:600;white-space:pre-wrap}
.ktab th,.ktab td{padding:.16em .75em;white-space:pre;
                  border-bottom:1px solid rgba(128,128,128,.22)}
.ktab td{text-align:left}
.ktab th{text-align:center;border-bottom:1px solid rgba(128,128,128,.55)}
.ktab tr:last-child td{border-bottom:none}
.ktab .r{text-align:right}
.ktab .c{text-align:center}
.ktab .ok{color:#2e9e4f;font-weight:600}
.ktab .no{color:#d63b3b;font-weight:600}
.ktab .dim{opacity:.6}
</style>"""


def _cell(value, fmt, force=None):
    """一个格子。内容一律左对齐（表头居中），True/False 上色，None 打成灰色短横。

    需要右对齐或居中的列，用 show(..., align='llr') 逐列指定。
    """
    classes = [force] if force else []
    if value is True or value is False:
        classes.append('ok' if value else 'no')
        text = '✓' if value else '✗'
    elif value is None:
        classes.append('dim')
        text = '–'
    elif isinstance(value, float):
        text = fmt.format(value)
    else:
        text = _html.escape(str(value))
    css = f' class="{" ".join(classes)}"' if classes else ''
    return f'<td{css}>{text}</td>'


def table_html(rows, header=None, title=None, fmt='{:.3e}', align=None):
    """拼一张 HTML 表并返回字符串。align 传 'llrc' 之类逐列覆盖默认对齐。"""
    out = [_TABLE_CSS, '<table class="ktab">']
    if title:
        out.append(f'<caption>{_html.escape(title)}</caption>')
    if header:
        out.append('<thead><tr>'
                   + ''.join(f'<th>{_html.escape(str(h))}</th>' for h in header)
                   + '</tr></thead>')
    out.append('<tbody>')
    for row in rows:
        out.append('<tr>' + ''.join(
            _cell(value, fmt, align[n] if align and n < len(align) else None)
            for n, value in enumerate(row)) + '</tr>')
    out.append('</tbody></table>')
    return ''.join(out)


def show(rows, header=None, title=None, fmt='{:.3e}', align=None):
    """把 rows 显示成表格。列宽由浏览器算，中文占多宽都不影响对齐。"""
    display(HTML(table_html(rows, header, title, fmt, align)))



# ── 证据链 ────────────────────────────────────────────────────────────

def source_location(obj) -> str:
    """返回对象在源码中的文件名与起止行号。装饰器包装过的对象会回退到 __wrapped__。"""
    target = inspect.unwrap(obj)
    try:
        lines, start = inspect.getsourcelines(target)
    except (OSError, TypeError) as exc:
        return f'<无法定位: {exc}>'
    return f'{Path(inspect.getfile(target)).name}:{start}-{start + len(lines) - 1}'


EVIDENCE = [
    ('Qwen3RMSNorm',            Q3.Qwen3RMSNorm),
    ('  └ forward',             Q3.Qwen3RMSNorm.forward),
    ('Qwen3MLP',                Q3.Qwen3MLP),
    ('  └ forward',             Q3.Qwen3MLP.forward),
    ('Qwen3RotaryEmbedding',    Q3.Qwen3RotaryEmbedding),
    ('  └ forward',             Q3.Qwen3RotaryEmbedding.forward),
    ('rotate_half',             Q3.rotate_half),
    ('apply_rotary_pos_emb',    Q3.apply_rotary_pos_emb),
    ('repeat_kv',               Q3.repeat_kv),
    ('eager_attention_forward', Q3.eager_attention_forward),
    ('Qwen3Attention',          Q3.Qwen3Attention),
    ('  └ forward',             Q3.Qwen3Attention.forward),
    ('Qwen3DecoderLayer',       Q3.Qwen3DecoderLayer),
    ('  └ forward',             Q3.Qwen3DecoderLayer.forward),
    ('Qwen3Model',              Q3.Qwen3Model),
    ('  └ forward',             Q3.Qwen3Model.forward),
    ('Qwen3ForCausalLM',        Q3.Qwen3ForCausalLM),
    ('  └ forward',             Q3.Qwen3ForCausalLM.forward),
]

def show_evidence():
    """打印本实验涉及的全部源码位置。行号从实际加载的模块动态读取，不是手抄的。"""
    show([(name, source_location(obj)) for name, obj in EVIDENCE],
         header=['对象', '源码位置'])


# ── setup ──────────────────────────────────────────────────────────────

# 下面这些由 setup(model, tokenizer) 填。留成 None 是为了：忘了 setup 时
# 报 "先调 setup" 而不是一句光秃秃的 NameError。
TOKENIZER = None
N_LAYERS = HIDDEN = N_HEADS = N_KV_HEADS = HEAD_DIM = INTERMEDIATE = VOCAB = None
LAYER_FIELDS = _PARAM_NAMES = _NO_WEIGHT = None


def _require_setup():
    if N_LAYERS is None:
        raise RuntimeError('先调一次 qwen_kit.setup(model, tokenizer)，'
                           '维度常量和字段表都在那里建。')


def setup(model, tokenizer):
    """把维度常量和两张字段表建起来。用工具之前调一次。

    七个维度全部从 model.config 现推，不写死数字：换个规格的 Qwen3 也能直接用。
    """
    global TOKENIZER, N_LAYERS, HIDDEN, N_HEADS, N_KV_HEADS, HEAD_DIM
    global INTERMEDIATE, VOCAB, LAYER_FIELDS, _PARAM_NAMES, _NO_WEIGHT
    config = model.config
    TOKENIZER = tokenizer
    N_LAYERS = config.num_hidden_layers
    HIDDEN = config.hidden_size
    N_HEADS = config.num_attention_heads
    N_KV_HEADS = config.num_key_value_heads
    HEAD_DIM = config.head_dim
    INTERMEDIATE = config.intermediate_size
    VOCAB = config.vocab_size

    # 列表顺序 = 真实计算顺序。
    LAYER_FIELDS = [
        ('inp',          '这一层的输入（未归一化，残差记住的就是它）'),
        ('norm1',        'input_layernorm，pre-norm'),
        ('q_proj',       f'{HIDDEN} → {N_HEADS * HEAD_DIM}，{N_HEADS} 头 × {HEAD_DIM}'),
        ('k_proj',       f'{HIDDEN} → {N_KV_HEADS * HEAD_DIM}，{N_KV_HEADS} 头 × {HEAD_DIM}'),
        ('v_proj',       f'{HIDDEN} → {N_KV_HEADS * HEAD_DIM}，{N_KV_HEADS} 头 × {HEAD_DIM}'),
        ('q_norm',       f'在 head_dim={HEAD_DIM} 上归一化'),
        ('k_norm',       '同上。V 没有 norm'),
        ('attn_weights', 'softmax 之后的注意力权重'),
        ('o_proj',       f'{N_HEADS * HEAD_DIM} → {HIDDEN}'),
        ('attn_out',     'Attention 的最终输出（= o_proj 的输出）'),
        ('after_attn',   '第一次残差之后 = inp + attn_out'),
        ('norm2',        'post_attention_layernorm'),
        ('gate',         f'{HIDDEN} → {INTERMEDIATE}'),
        ('up',           f'{HIDDEN} → {INTERMEDIATE}'),
        ('down',         f'{INTERMEDIATE} → {HIDDEN}'),
        ('mlp_out',      'MLP 的最终输出（= down 的输出）'),
        ('out',          '这一层的输出，也是下一层的 inp'),
    ]

    # qwen.L[i] 有 17 个字段，W.L[i] 只有 11 个。差集就是「纯运算的产物，没有权重」
    # 那几个，算出来而不是手抄，这样两张表将来改了也不会对不上。
    _PARAM_NAMES = [field for field, _, _ in PARAM_FIELDS]
    _NO_WEIGHT = [field for field, _ in LAYER_FIELDS if field not in _PARAM_NAMES]




# ── 中间状态 ──────────────────────────────────────────────────────────

class LayerView:
    """第 index 层的全部中间状态。属性名见 LAYER_FIELDS。"""

    def __init__(self, index):
        self.index = index

    def __repr__(self):
        return f'<LayerView layer={self.index}，{len(LAYER_FIELDS)} 个字段>'

    def _repr_html_(self):
        rows = []
        for field, why in LAYER_FIELDS:
            tensor = getattr(self, field, None)
            rows.append((f'.{field}',
                         str(tuple(tensor.shape)) if torch.is_tensor(tensor) else '(未记录)',
                         why))
        return table_html(
            rows, header=['字段（可 Tab 补全）', '形状', '是什么'],
            title=f'Layer {self.index} 的中间状态，按真实计算顺序\n'
                  f'用法： look(qwen.L[{self.index}].q_proj)   '
                  f'check(我的结果, qwen.L[{self.index}].q_proj)   （两者见 §3）')



# 模块路径 → LayerView 字段名。self_attn 和 DecoderLayer 本身单独处理。
_FIELD_OF = {
    'input_layernorm': 'norm1',
    'self_attn.q_proj': 'q_proj',
    'self_attn.k_proj': 'k_proj',
    'self_attn.v_proj': 'v_proj',
    'self_attn.q_norm': 'q_norm',
    'self_attn.k_norm': 'k_norm',
    'self_attn.o_proj': 'o_proj',
    'post_attention_layernorm': 'norm2',
    'mlp.gate_proj': 'gate',
    'mlp.up_proj': 'up',
    'mlp.down_proj': 'down',
    'mlp': 'mlp_out',
}


class Observed:
    """一次官方 forward 的全部内部状态。用 observe(model, input_ids) 得到。"""

    current = None                  # 记住最近一次，look() 解码 token 时要用

    def __init__(self, model, input_ids):
        _require_setup()
        self.input_ids = input_ids
        self.L = [LayerView(i) for i in range(N_LAYERS)]
        handles = self._mount(model)
        try:
            with torch.no_grad():
                self.official = model(input_ids, output_attentions=True,
                                      use_cache=False)
        finally:
            for handle in handles:  # 无论成功失败都摘干净，不会污染后续 forward
                handle.remove()
        self.logits = self.official.logits
        # mask 与 cos/sin 由 Qwen3Model 算一次后传给所有 28 层，取第 0 层的即可（§7.3 验证共享）
        self.mask = self.L[0].mask
        self.cos, self.sin = self.L[0].cos, self.L[0].sin
        self.position_ids = self.L[0].position_ids
        self.n_hooks = len(handles)
        Observed.current = self

    def _mount(self, model):
        named = dict(model.named_modules())
        handles = []

        def watch(module, fn):
            # with_kwargs=True 是必需的：attention_mask / position_embeddings 是关键字参数
            handles.append(module.register_forward_hook(
                lambda mod, args, kwargs, output: fn(args, kwargs, output),
                with_kwargs=True))

        def keep(name):
            return lambda args, kwargs, output: setattr(self, name, output)

        watch(model.model.embed_tokens, keep('embed'))
        watch(model.model.norm, keep('final_norm'))

        for index in range(N_LAYERS):
            prefix = f'model.layers.{index}'
            watch(named[prefix], self._layer_hook(index))
            watch(named[f'{prefix}.self_attn'], self._attn_hook(index))
            for path, field in _FIELD_OF.items():
                watch(named[f'{prefix}.{path}'], self._leaf_hook(index, field))
        return handles

    def _leaf_hook(self, index, field):
        return lambda args, kwargs, output: setattr(self.L[index], field, output)

    def _attn_hook(self, index):
        def fn(args, kwargs, output):
            # Qwen3Attention.forward 返回 (attn_output, attn_weights)，在这里就拆开。
            # 对外只有两个普通张量，不需要记得"哪个节点要加 [0]"。
            self.L[index].attn_out, self.L[index].attn_weights = output
        return fn

    def _layer_hook(self, index):
        def fn(args, kwargs, output):
            view = self.L[index]
            view.inp = args[0] if args else kwargs['hidden_states']
            view.out = output                          # DecoderLayer.forward 返回裸张量
            view.after_attn = view.inp + view.attn_out  # 第一次残差，不是任何模块的输出
            view.mask = kwargs['attention_mask']
            view.position_ids = kwargs['position_ids']
            view.cos, view.sin = kwargs['position_embeddings']
        return fn

    def __repr__(self):
        return f'<Observed：{N_LAYERS} 层内部状态，{self.n_hooks} 个 hook 已摘除>'

    def _repr_html_(self):
        last = N_LAYERS - 1
        rows = [('input_ids', self.input_ids, '输入 token id'),
                ('embed', self.embed, 'embed_tokens 查表，进第 0 层之前'),
                (f'L[0] … L[{last}]', self.L[0].out, f'{N_LAYERS} 层 Decoder Layer，每层保形'),
                ('final_norm', self.final_norm, 'model.norm，最后一次 RMSNorm'),
                ('logits', self.logits, f'lm_head 投到 {VOCAB} 词表')]
        shapes = {tuple(view.out.shape) for view in self.L}
        usage = [('看某一层', 'qwen.L[0]', '直接敲，会打印字段清单'),
                 ('跨层共享件', 'qwen.cos  qwen.sin  qwen.mask  qwen.position_ids', ''),
                 ('看某个张量', 'look(qwen.L[0].q_proj)', '§3 的两个动作之一'),
                 ('比对结果', 'check(我算的, qwen.L[0].q_proj)', '§3 的两个动作之一')]
        return (table_html(
                    [(name, str(tuple(tensor.shape)), why) for name, tensor, why in rows],
                    header=['字段', '形状', '是什么'],
                    title=f'observe() 已记录 {N_LAYERS} 层内部状态'
                          f'（{self.n_hooks} 个 hook，已全部摘除）\n'
                          f'{N_LAYERS} 层输出形状只有 {len(shapes)} 种：{sorted(shapes)}'
                          f'  → 每层都是保形函数')
                + table_html(usage, header=['要做什么', '怎么敲', '说明'])
                )


def observe(model, input_ids):
    """挂 hook → 跑一次官方 forward → 摘 hook，把全部内部状态装进返回值。"""
    return Observed(model, input_ids)



# ── 模型参数 ──────────────────────────────────────────────────────────

# 权重分两类，用法不同。kind 决定清单里「怎么用」那一列。
PARAM_FIELDS = [
    ('norm1',  'input_layernorm.weight',          'vector'),
    ('q_proj', 'self_attn.q_proj.weight',         'matrix'),
    ('k_proj', 'self_attn.k_proj.weight',         'matrix'),
    ('v_proj', 'self_attn.v_proj.weight',         'matrix'),
    ('q_norm', 'self_attn.q_norm.weight',         'vector'),
    ('k_norm', 'self_attn.k_norm.weight',         'vector'),
    ('o_proj', 'self_attn.o_proj.weight',         'matrix'),
    ('norm2',  'post_attention_layernorm.weight', 'vector'),
    ('gate',   'mlp.gate_proj.weight',            'matrix'),
    ('up',     'mlp.up_proj.weight',              'matrix'),
    ('down',   'mlp.down_proj.weight',            'matrix'),
]
_HOWTO = {'matrix': 'my_linear(x, w)', 'vector': 'my_rmsnorm(x, w)'}


class LayerParams:
    """第 index 层的 11 个权重。字段名与 qwen.L[index] 的中间状态一一对应。"""

    def __init__(self, model, index):
        self._index = index
        layer = model.model.layers[index]
        for field, suffix, _ in PARAM_FIELDS:
            obj = layer
            for part in suffix.split('.'):      # 'self_attn.q_proj.weight' 逐段取下去
                obj = getattr(obj, part)
            setattr(self, field, obj)
        missing = [f for f in _PARAM_NAMES if not torch.is_tensor(getattr(self, f, None))]
        assert not missing, f'Layer {index} 少了权重 {missing}，模型结构与本实验的假设不同'

    def __getattr__(self, name):
        # 只有正常属性找不到时才会走到这里。把几种常见手滑变成会自我解释的报错，
        # 而不是一句光秃秃的 AttributeError。
        if name.startswith('_'):
            raise AttributeError(name)
        index = self.__dict__.get('_index', '?')
        if name in ('bias', 'biases'):
            raise AttributeError(f'本模型没有任何 bias（config 里 attention_bias=false），'
                                 f'W.L[{index}] 只有 weight')
        if name == 'weight':
            raise AttributeError(f'W.L[{index}].字段 取到的已经是权重张量本身，不必再 .weight，'
                                 f'直接写 W.L[{index}].q_proj')
        if name in _NO_WEIGHT:
            raise AttributeError(f'{name!r} 是中间状态、不是权重，在 qwen.L[{index}].{name} 里；'
                                 f'W.L[{index}] 只有 {len(_PARAM_NAMES)} 个权重字段')
        raise AttributeError(f'W.L[{index}] 没有字段 {name!r}。'
                             f'敲 W.L[{index}] 看清单，或 W.find({name!r}) 模糊搜')

    def __repr__(self):
        return f'<LayerParams layer={self._index}，{len(PARAM_FIELDS)} 个权重>'

    def _repr_html_(self):
        i = self._index
        rows = [(f'.{field}', str(tuple(getattr(self, field).shape)),
                 _HOWTO[kind], suffix)
                for field, suffix, kind in PARAM_FIELDS]
        return table_html(
            rows, header=['字段', '形状', '怎么用', '裸路径（可直接复制）'],
            title=f'Layer {i} 的 {len(PARAM_FIELDS)} 个权重'
                  f'（裸路径前缀 model.model.layers[{i}].）\n'
                  f'矩阵按 [out, in] 转置存放，交给 my_linear 就不用管 .T。\n'
                  f'qwen.L[{i}] 里另有 {len(_NO_WEIGHT)} 个字段是纯中间状态、没有权重：'
                  + '  '.join(_NO_WEIGHT) + f'\n'
                  f'用法： W.L[{i}].q_proj   W.find("norm")')



class LayerList:
    """W.L 的容器。只认 0..N_LAYERS-1，越界和负数层号都给中文提示。"""

    def __init__(self, items):
        self._items = list(items)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError(f'层号要写整数，收到 {index!r}。合法范围 0..{len(self) - 1}')
        if index < 0:
            # 负下标在这里没有好处，只会把 off-by-one 悄悄兜住。最后一层请写明层号。
            raise IndexError(f'不接受负数层号 {index}。最后一层是 W.L[{len(self) - 1}]')
        if index >= len(self):
            raise IndexError(f'层号 {index} 越界。本模型 num_hidden_layers={len(self)}，'
                             f'合法范围 0..{len(self) - 1}')
        return self._items[index]

    def __repr__(self):
        return f'W.L：{len(self)} 层，每层 {len(PARAM_FIELDS)} 个权重。敲 W.L[0] 看清单'


class Params:
    """全模型权重目录。用 params(model) 得到，敲 W 或 W.L[0] 就是清单。"""

    def __init__(self, model):
        _require_setup()
        self.model = model
        self.L = LayerList(LayerParams(model, i) for i in range(N_LAYERS))
        self.embed = model.model.embed_tokens.weight
        self.final_norm = model.model.norm.weight
        self.lm_head = model.lm_head.weight              # 与 embed 共享同一块内存
        self.inv_freq = model.model.rotary_emb.inv_freq  # buffer，不在 named_parameters() 里

    def _top(self):
        return [('W.embed', self.embed, 'model.model.embed_tokens.weight',
                 '查表取行 → my_embedding'),
                ('W.final_norm', self.final_norm, 'model.model.norm.weight',
                 'my_rmsnorm(x, w)'),
                ('W.lm_head', self.lm_head, 'model.lm_head.weight',
                 'my_linear(x, w)  <- 与 embed 同一块内存'),
                ('W.inv_freq', self.inv_freq, 'model.model.rotary_emb.inv_freq',
                 'buffer，不是 parameter')]

    def __repr__(self):
        return f'<Params：{N_LAYERS} 层 × {len(PARAM_FIELDS)} 个权重 + 4 个顶层>'

    def _repr_html_(self):
        # 三个数都是现算的,不是写死的:换个模型这张表会跟着变,不会撒谎。
        names = [name for name, _ in self.model.named_parameters()]
        n_bias = sum(1 for name in names if name.endswith('.bias'))
        sample = self.L[0].q_proj
        rows = [(name, str(tuple(tensor.shape)), path, how)
                for name, tensor, path, how in self._top()]
        return table_html(
            rows, header=['字段', '形状', '裸路径', '怎么用'],
            title=f'模型权重目录（named_parameters() 共 {len(names)} 项 '
                  f'= {N_LAYERS} 层 × {len(PARAM_FIELDS)} + 2）\n'
                  f'dtype={sample.dtype}  device={sample.device}\n'
                  f'W.L[0] … W.L[{N_LAYERS - 1}] 每层 {len(PARAM_FIELDS)} 个权重，'
                  f'敲 W.L[0] 看清单；忘了字段名用 W.find("norm")\n'
                  f'全模型 bias 数量 = {n_bias}（attention_bias=false），找 .bias 会报错。\n'
                  f'取到的是官方权重本身，不是副本 —— 别原地改。')

    def find(self, pattern):
        """按名字模糊搜。忘了字段叫什么就 W.find('norm')。"""
        pattern = pattern.lower()
        hits = [(name, tuple(t.shape), path) for name, t, path, _ in self._top()
                if pattern in name.lower() or pattern in path.lower()]
        hits += [(f'W.L[i].{field}', tuple(getattr(self.L[0], field).shape),
                  f'...layers[i].{suffix}')
                 for field, suffix, _ in PARAM_FIELDS
                 if pattern in field.lower() or pattern in suffix.lower()]
        if hits:
            show([(n, str(s), p) for n, s, p in hits],
                 header=['字段', '形状', '裸路径'],
                 title=f'匹配 {pattern!r} 的 {len(hits)} 个权重字段')
        # 搜不到权重时，很可能要找的是中间状态。指过去，而不是只说「没有」。
        states = [field for field, _ in LAYER_FIELDS if pattern in field.lower()]
        if states:
            print(f'\n{pattern!r} 还匹配 {len(states)} 个中间状态（在 qwen.L[i] 里，不是权重）：')
            print('  ' + '  '.join(f'qwen.L[i].{s}' for s in states))
        if not hits and not states:
            print(f'没有匹配 {pattern!r} 的字段。敲 W 看权重目录，敲 qwen.L[0] 看中间状态。')



def params(model):
    """把全模型权重整理成一份可查的目录。"""
    return Params(model)



# ── 动作：look ────────────────────────────────────────────────────────

def _tokens():
    """当前输入的逐 token 文本，供 look 画注意力网格时当行列标签。"""
    source = Observed.current
    if source is None:
        return None
    return [TOKENIZER.decode([i]) for i in source.input_ids[0].tolist()]


def _show_grid(weights, head, title):
    """把 [B, heads, S, S] 的注意力权重画成位置 × 位置网格（值 ×100）。"""
    labels = _tokens() or [str(i) for i in range(weights.shape[-1])]
    seq_len = weights.shape[-1]
    # 上三角被 causal mask 挡住，传 None，表格里打成灰色短横。
    rows = [(i, *[weights[0, head, i, j].item() * 100 if j <= i else None
                  for j in range(seq_len)], repr(labels[i]))
            for i in range(seq_len)]
    show(rows, header=['query', *[f'k{j}' for j in range(seq_len)], 'query 的 token'],
         fmt='{:.1f}', title=f'{title}  (行=query，列=key，值 ×100)')


def _stats(tensor):
    """权重的分布摘要。三个数一起看，才发现得了 q_norm / final_norm 的量级异常。"""
    values = tensor.float()
    return (f'mean={values.mean().item():+.4f}  std={values.std().item():.4f}  '
            f'absmax={values.abs().max().item():.4f}')


def _show_matrix(tensor, name, rows, width):
    """打印二维张量的形状与左上角一小块。权重矩阵是 [out_features, in_features]。"""
    out_features, in_features = tensor.shape
    body = [('形状', f'{tuple(tensor.shape)}   矩阵 [out={out_features}, in={in_features}]'),
            ('怎么用', 'my_linear(x, w) 内部做 x @ w.T，不必手动转置')]
    for row in range(min(rows, out_features)):
        values = ', '.join(f'{v:+.4f}' for v in tensor[row, :width].tolist())
        note = f'   ← 第 0 个输出通道，对应前 {width} 个输入维' if row == 0 else ''
        body.append((f'w[{row}, :{width}]', f'[{values}]{note}'))
    body.append(('分布', _stats(tensor)))
    show(body, title=name)


def _show_vector(tensor, name, width):
    """打印一维张量的形状与开头几个值。RMSNorm 的缩放向量走这里。"""
    values = ', '.join(f'{v:+.4f}' for v in tensor[:width].tolist())
    show([('形状', f'{tuple(tensor.shape)}   向量 [dim={tensor.shape[0]}]'),
          ('怎么用', 'my_rmsnorm(x, w) 内部逐元素相乘，不改形状'),
          (f'w[:{width}]', f'[{values}]'),
          ('分布', _stats(tensor))], title=name)


def _show_window(tensor, name, position, width):
    """打印形状 + 一个小窗口的实际数值。窗口位置随维度自动选。"""
    if tensor.dim() == 3:                                   # [B, S, H]
        window, where = tensor[0, position, :width], f'[0, {position}, :{width}]'
    elif tensor.dim() == 4:                                 # [B, heads, S, D]
        window, where = tensor[0, 0, position, :width], f'[0, 0, {position}, :{width}]'
    else:
        window, where = tensor.flatten()[:width], f'flat[:{width}]'
    values = ', '.join(f'{v:+.4f}' for v in window.tolist())
    show([('形状', str(tuple(tensor.shape))),
          (where, f'[{values}]')], title=name)


# 自带清单式 repr 的对象，look() 直接打印它们的 repr —— §2 两份目录里的都算。
_LOOK_REPR = [Observed, LayerView, Params, LayerParams, LayerList]


def look(x, name='', head=0, position=-1, width=6, rows=3):
    """看任何东西。按类型和维度自动选显示方式，正常使用不用传后面几个参数。"""
    if isinstance(x, tuple(_LOOK_REPR)):
        display(x)                  # 这些对象有 _repr_html_，交给 Jupyter 渲染成表格
        return
    if not torch.is_tensor(x):
        print(f'{name or type(x).__name__}: {x!r}')
        return

    if x.dtype in (torch.int32, torch.int64) and x.dim() <= 2:   # token ids
        ids = x.flatten().tolist()
        show([(n, i, repr(TOKENIZER.decode([i]))) for n, i in enumerate(ids)],
             header=['pos', 'id', 'decode'],
             title=f'{name or "token ids"}  {tuple(x.shape)}')
        return

    if x.dim() == 4 and x.shape[-1] == x.shape[-2]:              # 注意力权重
        _show_grid(x, head, name or f'attention weights, head {head}')
        return

    if x.dim() == 2:                                             # 权重矩阵 [out, in]
        _show_matrix(x, name or '2D tensor', rows, width)
        return

    if x.dim() == 1:                                             # RMSNorm 权重等一维张量
        _show_vector(x, name or '1D tensor', width)
        return

    _show_window(x, name or f'{x.dim()}D tensor', position, width)



# ── 动作：check ───────────────────────────────────────────────────────

REL_THRESHOLD = 1e-5        # 判定阈值：max|差| / max|ref|
CHECKS = []                 # (名字, 是否通过, max_abs, max_rel)，按调用顺序


def check(mine, ref, name=None, quiet=False):
    """比对两个张量。判据固定为 max|差| / max|ref| < 1e-5，没有公差参数可调。"""
    name = name or f'check #{len(CHECKS) + 1}'
    mine, ref = mine.float(), ref.float()
    if mine.shape != ref.shape:
        # 先记一条失败再抛。断言在登记之前抛的话，形状写错是唯一不会让汇总变红的错。
        CHECKS.append((f'{name}（形状不一致）', False, None, None))
        raise AssertionError(f'{name}: shape 不一致 {tuple(mine.shape)} vs {tuple(ref.shape)}')
    diff = (mine - ref).abs()
    max_abs, mean_abs = diff.max().item(), diff.mean().item()
    scale = ref.abs().max().item()
    rel = max_abs / scale if scale > 0 else 0.0
    ok = rel < REL_THRESHOLD
    CHECKS.append((name, ok, max_abs, rel))
    if not quiet:
        # 单行输出不用表格：{:.3e} 恒为 9 字符，名字放行尾，没有需要补齐的字段。
        print(f'{"✓" if ok else "✗"}  max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  '
              f'max_rel={rel:.3e}  {name}')
    return ok


def record(name, ok, rel=None, max_abs=None):
    """登记一个不是逐元素比对的结论（例如"28 层全部通过"、"预测一致"）。

    没有实测到的那一栏传 None，汇总里打印成 `-`。不要拿 0.0 顶替：
    0.000e+00 会被读成"测过，差异为零"，而对象相等、预测一致这类结论根本没测差异。
    """
    CHECKS.append((name, ok, max_abs, rel))
    return ok


def summary():
    """汇总本次记录到的全部 check。

    只报"本次记录到什么"，不报"该验证的都验证了"——后者 summary 无从知晓：
    漏跑一格，分子分母一起少一项，比值不变。完整性靠 execution_count 连续来证。
    """
    latest, changed = {}, []
    for name, ok, max_abs, rel in CHECKS:
        if name in latest and latest[name][0] != ok:
            changed.append(name)          # 同名但结论不同 —— 撞名字，不是重跑
        latest[name] = (ok, max_abs, rel)

    failed = [name for name, (ok, _, _) in latest.items() if not ok]
    show([(max_abs, rel, ok, name) for name, (ok, max_abs, rel) in latest.items()],
         header=['max_abs', 'max_rel', 'ok', '检查项'],
         title=f'本次记录到 {len(latest)} 项验证'
               f'（判据：尺度相对误差 < {REL_THRESHOLD:.0e}）\n'
               f'通过 {len(latest) - len(failed)} / {len(latest)}')

    if changed:
        print(f'⚠ 这些名字被两次不同结论的 check 用过，汇总只留最后一次: {changed}')
    return not failed
