"""git clean 过滤器：把 notebook 里每跑一次就变、又不带信息的字段抹掉。

抹两样：
  cell metadata 的 execution —— 四个纳秒级时间戳，VS Code 和 nbconvert 都会写，
                                 每跑一遍全变，于是整本文件必然显示"已修改"。
  顶层 metadata 的 widgets  —— 进度条 widget 的状态，model_id 是随机 UUID。

输出一律保留：它们是这个项目的证据，不是噪音。

装法见 README「一起改」。git 从仓库根目录调用本脚本，stdin 进 stdout 出。
"""
import json
import sys


def clean(nb):
    nb.get('metadata', {}).pop('widgets', None)
    for cell in nb.get('cells', []):
        cell.get('metadata', {}).pop('execution', None)
    return nb


if __name__ == '__main__':
    notebook = clean(json.load(sys.stdin))
    json.dump(notebook, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write('\n')
