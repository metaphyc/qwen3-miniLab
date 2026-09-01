#!/usr/bin/env bash
# clone 完跑一次，配好 notebook 的 diff / merge / 清理。
#
# 这些配置写在 .git/config 里，而 .git/config 不入库 —— 所以每个人 clone 之后
# 都要各自跑一遍，不跑的话 notebook 的改动会显示成几千行 JSON。
#
# 重复跑没有副作用。
set -euo pipefail
cd "$(dirname "$0")/.."

command -v nbdime >/dev/null || {
    echo "缺 nbdime，先跑：pip install -r requirements-dev.txt" >&2; exit 1; }

nbdime config-git --enable >/dev/null
git config diff.jupyternotebook.command 'git-nbdiffdriver diff --ignore-details'
git config filter.nbclean.clean 'python3 tools/nb_clean.py'
git config alias.tidy '!python3 tools/nb_clean.py -i notebooks/*.ipynb'

echo "配好了："
echo "  git diff  按 cell 显示，不再是几千行 JSON"
echo "  git tidy  清掉 notebook 里每跑一次就变的东西，提交前跑一下"
