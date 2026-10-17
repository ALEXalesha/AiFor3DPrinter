#!/usr/bin/env bash
# Публикация на GitHub (адрес - REPO в tools/publish.conf):
#
#   bash tools/publish_github.sh            # только код
#   bash tools/publish_github.sh v1.0.0     # код и тег
#   DRY_RUN=1 bash tools/publish_github.sh  # собрать и проверить, ничего не отправлять
#
# Тонкая обёртка: вся работа - в общем скрипте Python-игр AiGames/PythonAiGames/tools/
# publish_game.sh, он чистит личное по всей истории (почта, адрес Gitea, имя, путь к папке
# Drive в любом написании) и проверяет его до пуша. Особенности проекта - в tools/publish.conf.
# Если общий скрипт лежит не там (копия в wip), путь к нему - в переменной PUBLISH_GAME.
# До 17.10.2026 здесь был свой скрипт: он не чистил путь Drive и проверял через конвейеры
# (SIGPIPE на длинной истории молча пропускал проверку).
set -euo pipefail
GAME="$(cd "$(dirname "$0")/.." && pwd)"
COMMON="${PUBLISH_GAME:-$GAME/../../AiGames/PythonAiGames/tools/publish_game.sh}"
[ -f "$COMMON" ] || { echo "нет общего скрипта $COMMON - задайте PUBLISH_GAME" >&2; exit 1; }
exec bash "$COMMON" "$GAME" "$@"
