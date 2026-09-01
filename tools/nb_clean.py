"""git clean 过滤器：把 notebook 里每跑一次就变、又不带信息的字段抹掉。

抹两样，归并一样：
  cell metadata 的 execution —— 四个纳秒级时间戳，VS Code 和 nbconvert 都会写，
                                 每跑一遍全变，于是整本文件必然显示"已修改"。
  顶层 metadata 的 widgets  —— 进度条 widget 的状态，model_id 是随机 UUID。
  连续的 stream 输出        —— 同一个 cell 的 stdout 被切成几段取决于刷新时机，
                                 渲染出来一模一样，切法却每次不同。合并成一段。

输出一律保留：它们是这个项目的证据，不是噪音。

装法见 README「一起改」。git 从仓库根目录调用本脚本，stdin 进 stdout 出。
"""
import json
import sys


def coalesce(outputs):
    """把相邻的同名 stream 输出并成一条，文字内容不变。"""
    merged = []
    for out in outputs:
        prev = merged[-1] if merged else None
        if (out.get('output_type') == 'stream' and prev is not None
                and prev.get('output_type') == 'stream'
                and prev.get('name') == out.get('name')):
            prev['text'] = _lines(_text(prev) + _text(out))
        else:
            merged.append(out)
    return merged


def _text(output):
    text = output.get('text', '')
    return text if isinstance(text, str) else ''.join(text)


def _lines(text):
    return text.splitlines(keepends=True)


def clean(nb):
    nb.get('metadata', {}).pop('widgets', None)
    for cell in nb.get('cells', []):
        cell.get('metadata', {}).pop('execution', None)
        if cell.get('outputs'):
            cell['outputs'] = coalesce(cell['outputs'])
    return nb


if __name__ == '__main__':
    notebook = clean(json.load(sys.stdin))
    json.dump(notebook, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write('\n')
