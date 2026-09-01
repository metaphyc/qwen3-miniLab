"""git clean 过滤器：把 notebook 里每跑一次就变、又不带信息的字段抹平。

不这么做的话，整本从头跑一遍、一个字都没改，git 也会说它改了。四个来源：

  cell metadata 的 execution   四个纳秒级时间戳，VS Code 和 nbconvert 都写。
  顶层 metadata 的 widgets     进度条 widget 的状态，model_id 是随机 UUID。
  进度条那条 widget 输出       同上，跑完不留任何信息，整条丢掉。
  execution_count 的起点       在开着的 kernel 里重跑，编号从上次的末尾接着走
                               （110、111、112…）。整体减掉偏移量回到 1 开头 ——
                               只抹掉"这是第几个 kernel session"，编号的顺序和
                               连续性都留着：跳号、乱序仍然会显示成改动，
                               因为那确实是改动。

输出本身一律保留：它们是这个项目的证据，不是噪音。

两种用法，装法见 README「一起改」：

    tools/nb_clean.py                       git clean 过滤器，stdin 进 stdout 出
    tools/nb_clean.py -i notebooks/*.ipynb  就地清理

过滤器只管入库那一份，管不到 VS Code 的「已修改」角标 —— git 判断工作区有没有变
是拿文件大小跟 index 里记的比，大小一对不上就直接算改了，根本不看内容。
跑完一遍 notebook 文件大小必然变，所以角标必然亮。要它灭掉，就得把工作区这一份
也清干净：跑 -i，文件就回到和入库完全一样的字节。
"""
import json
import sys

WIDGET = 'application/vnd.jupyter.widget-view+json'


def _text(output):
    text = output.get('text', '')
    return text if isinstance(text, str) else ''.join(text)


def coalesce(outputs):
    """相邻的同名 stream 输出并成一条。切成几段取决于刷新时机，渲染出来一样。"""
    merged = []
    for output in outputs:
        previous = merged[-1] if merged else None
        if (output.get('output_type') == 'stream' and previous is not None
                and previous.get('output_type') == 'stream'
                and previous.get('name') == output.get('name')):
            previous['text'] = (_text(previous) + _text(output)).splitlines(keepends=True)
        else:
            merged.append(output)
    return merged


def renumber(cells):
    """execution_count 整体平移，让最小的那个回到 1。"""
    counts = [c['execution_count'] for c in cells
              if c.get('execution_count') is not None]
    offset = min(counts) - 1 if counts else 0
    if offset <= 0:
        return
    for cell in cells:
        if cell.get('execution_count') is not None:
            cell['execution_count'] -= offset
        for output in cell.get('outputs', []):
            if output.get('execution_count') is not None:
                output['execution_count'] -= offset


def clean(nb):
    nb.get('metadata', {}).pop('widgets', None)
    for cell in nb.get('cells', []):
        cell.get('metadata', {}).pop('execution', None)
        if cell.get('outputs'):
            kept = [o for o in cell['outputs'] if WIDGET not in o.get('data', {})]
            cell['outputs'] = coalesce(kept)
    renumber(nb.get('cells', []))
    return nb


def dumps(nb):
    return json.dumps(nb, ensure_ascii=False, indent=1) + '\n'


def main(argv):
    if argv and argv[0] in ('-i', '--inplace'):
        import pathlib
        for name in argv[1:]:
            path = pathlib.Path(name)
            before = path.read_text()
            after = dumps(clean(json.loads(before)))
            if before != after:
                path.write_text(after)
            print(f'{name}: {"已清理" if before != after else "本来就是干净的"}')
        return
    sys.stdout.write(dumps(clean(json.load(sys.stdin))))


if __name__ == '__main__':
    main(sys.argv[1:])
