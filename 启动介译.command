#!/bin/zsh

set -e
unsetopt BG_NICE

SCRIPT_DIR="${0:A:h}"
WEB_DIR="$SCRIPT_DIR/web"
RUNTIME_FILE="$SCRIPT_DIR/.jieyi.runtime"
LAUNCH_LOCK="$SCRIPT_DIR/.jieyi.launch.lock"
EXPECTED_DB="$SCRIPT_DIR/jieyi.db"
API_PID=""
WEB_PID=""
LOCK_HELD=0

clear 2>/dev/null || true
print ""
print "  介译 · 翻译工作台"
print "  ─────────────────────────"
print ""

runtime_value() {
  local key="$1"
  [[ -f "$RUNTIME_FILE" ]] || return 1
  sed -n "s/^${key}=\([0-9][0-9]*\)$/\1/p" "$RUNTIME_FILE" | head -n 1
}

release_launch_lock() {
  if [[ "$LOCK_HELD" == "1" ]]; then
    command rm -f "$LAUNCH_LOCK/owner" 2>/dev/null || true
    rmdir "$LAUNCH_LOCK" 2>/dev/null || true
    LOCK_HELD=0
  fi
}

acquire_launch_lock() {
  local attempt lock_owner
  for attempt in {1..120}; do
    if mkdir "$LAUNCH_LOCK" 2>/dev/null; then
      print "$$" > "$LAUNCH_LOCK/owner"
      LOCK_HELD=1
      return 0
    fi
    lock_owner="$(cat "$LAUNCH_LOCK/owner" 2>/dev/null || true)"
    if [[ -z "$lock_owner" || "$lock_owner" != <-> ]] || ! kill -0 "$lock_owner" 2>/dev/null; then
      command rm -f "$LAUNCH_LOCK/owner" 2>/dev/null || true
      rmdir "$LAUNCH_LOCK" 2>/dev/null || true
      continue
    fi
    sleep 0.25
  done
  print "  另一个启动过程仍在运行，请稍后再试。"
  return 1
}

api_is_current_project() {
  local port="$1"
  local health
  health="$(curl -fsS --connect-timeout 1 --max-time 3 "http://127.0.0.1:${port}/health" 2>/dev/null || true)"
  [[ "$health" == *'"api_version":2'* && "$health" == *"\"db_path\":\"$EXPECTED_DB\""* ]]
}

build_asset() {
  local key="$1"
  local manifest="$WEB_DIR/dist/client/.vite/manifest.json"
  local python_bin="$SCRIPT_DIR/.venv/bin/python"
  [[ -f "$manifest" && -x "$python_bin" ]] || return 1
  "$python_bin" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]["file"])' "$manifest" "$key" 2>/dev/null
}

web_build_is_fresh() {
  local manifest="$WEB_DIR/dist/client/.vite/manifest.json"
  local input
  [[ -f "$manifest" ]] || return 1
  for input in "$WEB_DIR/app" "$WEB_DIR/public" "$WEB_DIR/worker" \
    "$WEB_DIR/package.json" "$WEB_DIR/package-lock.json" "$WEB_DIR/vite.config.ts" \
    "$WEB_DIR/vinext.config.ts" "$WEB_DIR/next.config.ts" "$WEB_DIR/tsconfig.json"; do
    [[ -e "$input" ]] || continue
    if find "$input" -type f -newer "$manifest" -print -quit 2>/dev/null | grep -q .; then
      return 1
    fi
  done
  return 0
}

running_stack_is_fresh() {
  local runtime_marker="$RUNTIME_FILE"
  local input
  [[ -f "$runtime_marker" ]] || return 1
  web_build_is_fresh || return 1
  for input in "$SCRIPT_DIR/src" "$SCRIPT_DIR/pyproject.toml" "$SCRIPT_DIR/uv.lock" "$SCRIPT_DIR/启动介译.command"; do
    [[ -e "$input" ]] || continue
    if find "$input" -type f -newer "$runtime_marker" -print -quit 2>/dev/null | grep -q .; then
      return 1
    fi
  done
  return 0
}

web_build_targets_api() {
  local api_port="$1"
  local page_asset page_file
  page_asset="$(build_asset 'app/page.tsx' || true)"
  [[ -n "$page_asset" ]] || return 1
  page_file="$WEB_DIR/dist/client/$page_asset"
  [[ -s "$page_file" ]] || return 1
  grep -Fq "http://127.0.0.1:${api_port}" "$page_file"
}

web_matches_current_build() {
  local web_port="$1"
  local api_port="$2"
  local html entry_asset page_asset asset
  local -a referenced_assets
  entry_asset="$(build_asset 'virtual:vinext-app-browser-entry' || true)"
  page_asset="$(build_asset 'app/page.tsx' || true)"
  [[ -n "$entry_asset" && -n "$page_asset" ]] || return 1
  html="$(curl -fsS --connect-timeout 1 --max-time 5 "http://127.0.0.1:${web_port}/?healthcheck=$(date +%s)" 2>/dev/null || true)"
  [[ -n "$html" ]] || return 1
  [[ "$html" == *"/$entry_asset"* && "$html" == *"/$page_asset"* ]] || return 1
  referenced_assets=("${(@f)$(print -r -- "$html" | grep -Eo '/_next/static/[A-Za-z0-9._/-]+' | sort -u)}")
  (( ${#referenced_assets[@]} > 0 )) || return 1
  for asset in "${referenced_assets[@]}"; do
    curl -fsS --connect-timeout 1 --max-time 5 "http://127.0.0.1:${web_port}${asset}" -o /dev/null 2>/dev/null || return 1
  done
  web_build_targets_api "$api_port" || return 1
  curl -fsS --connect-timeout 1 --max-time 5 "http://127.0.0.1:${api_port}/projects" -o /dev/null 2>/dev/null || return 1
}

listener_pid() {
  local port="$1"
  lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -n 1
}

process_cwd() {
  local pid="$1"
  lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1
}

stop_pid() {
  local pid="$1"
  local attempt
  [[ -n "$pid" && "$pid" == <-> ]] || return 0
  kill -TERM "$pid" 2>/dev/null || return 0
  for attempt in {1..40}; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.1
  done
  kill -KILL "$pid" 2>/dev/null || true
}

stop_saved_services() {
  local owner_pid="$1"
  local web_port="$2"
  local api_port="$3"
  local pid cwd command_line

  pid="$(listener_pid "$web_port" || true)"
  if [[ -n "$pid" ]]; then
    cwd="$(process_cwd "$pid" || true)"
    command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if [[ "$cwd" == "$WEB_DIR" && "$command_line" == *"vinext"* ]]; then
      stop_pid "$pid"
    fi
  fi

  if [[ -n "$owner_pid" && "$owner_pid" == <-> ]]; then
    cwd="$(process_cwd "$owner_pid" || true)"
    command_line="$(ps -p "$owner_pid" -o command= 2>/dev/null || true)"
    if [[ "$cwd" == "$SCRIPT_DIR" || "$cwd" == "$WEB_DIR" ]] && [[ "$command_line" == *"启动介译.command"* ]]; then
      stop_pid "$owner_pid"
    fi
  fi

  pid="$(listener_pid "$api_port" || true)"
  if [[ -n "$pid" ]] && api_is_current_project "$api_port"; then
    cwd="$(process_cwd "$pid" || true)"
    command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if [[ "$cwd" == "$SCRIPT_DIR" && "$command_line" == *"uvicorn"* ]]; then
      stop_pid "$pid"
    fi
  fi
}

open_workbench() {
  local url="$1"
  if [[ "${JIEYI_NO_OPEN:-0}" != "1" ]]; then
    open "${url}?launch=$(date +%s)"
  fi
}

acquire_launch_lock
trap release_launch_lock EXIT INT TERM

saved_owner_pid="$(runtime_value OWNER_PID || true)"
saved_web_port="$(runtime_value WEB_PORT || true)"
saved_api_port="$(runtime_value API_PORT || true)"
if [[ -n "$saved_web_port" && -n "$saved_api_port" ]]; then
  if running_stack_is_fresh && api_is_current_project "$saved_api_port" && web_matches_current_build "$saved_web_port" "$saved_api_port"; then
    LOCAL_URL="http://localhost:${saved_web_port}/"
    print "  当前项目的工作台已经通过完整检查，正在打开…"
    release_launch_lock
    open_workbench "$LOCAL_URL"
    print ""
    print "  地址：$LOCAL_URL"
    print ""
    exit 0
  fi
  print "  检测到旧服务或构建文件不一致，正在安全重启…"
  stop_saved_services "$saved_owner_pid" "$saved_web_port" "$saved_api_port"
fi

if [[ ! -d "$WEB_DIR" ]]; then
  print "  找不到网页目录：$WEB_DIR"
  print "  请将启动器保留在项目根目录。"
  print ""
  read "?  按回车键关闭…"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  print "  尚未安装 Node.js / npm。"
  print "  安装后再次双击这个启动器即可。"
  print ""
  read "?  按回车键关闭…"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  print "  尚未安装 uv，无法启动本地 API。"
  print "  安装 uv 后再次双击这个启动器即可。"
  print ""
  read "?  按回车键关闭…"
  exit 1
fi

if [[ ! -d "$SCRIPT_DIR/.venv" ]]; then
  print "  首次启动，正在准备 Python 运行环境…"
  cd "$SCRIPT_DIR"
  uv sync --extra api
  print ""
fi

port_is_busy() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
  else
    nc -z 127.0.0.1 "$port" >/dev/null 2>&1
  fi
}

next_free_port() {
  local port="$1"
  while port_is_busy "$port"; do (( port += 1 )); done
  print "$port"
}

API_PORT="$(next_free_port 8000)"
WEB_PORT="$(next_free_port 3000)"
API_ROOT="http://127.0.0.1:${API_PORT}"
LOCAL_URL="http://localhost:${WEB_PORT}/"

print "  正在启动当前项目的本地 API…"
(
  cd "$SCRIPT_DIR"
  JIEYI_DB="$EXPECTED_DB" \
  JIEYI_CONFIG="$SCRIPT_DIR/jieyi.settings.json" \
  uv run uvicorn jieyi.api.app:create_app --factory --host 127.0.0.1 --port "$API_PORT"
) &
API_PID=$!

print "OWNER_PID=$$" > "$RUNTIME_FILE"
print "WEB_PORT=$WEB_PORT" >> "$RUNTIME_FILE"
print "API_PORT=$API_PORT" >> "$RUNTIME_FILE"

cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$WEB_PID" ]]; then kill "$WEB_PID" >/dev/null 2>&1 || true; fi
  if [[ -n "$API_PID" ]]; then kill "$API_PID" >/dev/null 2>&1 || true; fi
  local owner
  owner="$(runtime_value OWNER_PID || true)"
  if [[ "$owner" == "$$" ]]; then command rm -f "$RUNTIME_FILE"; fi
  release_launch_lock
}
trap cleanup EXIT INT TERM

cd "$WEB_DIR"
if [[ ! -d node_modules ]]; then
  print "  首次启动，正在准备网页运行环境…"
  npm ci --ignore-scripts --no-audit --no-fund
  print ""
fi

if web_build_is_fresh && web_build_targets_api "$API_PORT"; then
  print "  已找到有效工作台构建，直接启动…"
else
  print "  正在构建最新工作台…"
  NEXT_PUBLIC_JIEYI_API="$API_ROOT" npm run build
fi

print ""
print "  正在启动并验证完整工作台…"
WRANGLER_LOG_PATH=.wrangler/wrangler.log ./node_modules/.bin/vinext start --port "$WEB_PORT" &
WEB_PID=$!

ready=0
for attempt in {1..120}; do
  if ! kill -0 "$WEB_PID" 2>/dev/null; then
    print "  网页服务意外退出，请查看上方错误信息。"
    exit 1
  fi
  if api_is_current_project "$API_PORT" && web_matches_current_build "$WEB_PORT" "$API_PORT"; then
    ready=1
    break
  fi
  sleep 0.25
done

if [[ "$ready" != "1" ]]; then
  print "  工作台未能通过资源一致性检查，已停止本次启动。"
  exit 1
fi

release_launch_lock
print ""
print "  工作台已通过完整检查"
print "  地址：$LOCAL_URL"
print "  API： $API_ROOT"
print "  关闭这个终端窗口即可停止本次服务。"
print ""
open_workbench "$LOCAL_URL"

wait "$WEB_PID"
