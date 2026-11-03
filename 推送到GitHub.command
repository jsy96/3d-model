#!/bin/bash
# 双击运行：把本项目的改动提交并推送到 GitHub（origin）
# 也可以在终端执行：./推送到GitHub.command

GREEN='\033[32m'; RED='\033[31m'; YELLOW='\033[33m'; CYAN='\033[36m'; BOLD='\033[1m'; OFF='\033[0m'

pause_exit() {
  echo
  read -p "按回车键关闭窗口…" _
  exit "${1:-0}"
}

# 定位到本脚本所在目录（双击打开时当前目录是家目录，必须先切换）
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { printf "${RED}错误：无法进入项目目录 %s${OFF}\n" "$SCRIPT_DIR"; pause_exit 1; }

printf "${BOLD}===== 推送到 GitHub =====${OFF}\n"
printf "项目目录：%s\n" "$SCRIPT_DIR"

# ---- 基础检查 ----
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf "${RED}错误：当前目录不是 git 仓库。${OFF}\n"
  pause_exit 1
fi

BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse --abbrev-ref HEAD 2>/dev/null)"
[ -z "$BRANCH" ] && BRANCH="main"
REMOTE_URL="$(git remote get-url origin 2>/dev/null)"
if [ -z "$REMOTE_URL" ]; then
  printf "${RED}错误：未配置远程仓库 origin。请先在终端执行：git remote add origin <仓库地址>${OFF}\n"
  pause_exit 1
fi
printf "远程仓库：%s\n" "$REMOTE_URL"
printf "当前分支：%s\n\n" "$BRANCH"

# ---- 查看改动 ----
STATUS="$(git -c core.quotepath=false status --short)"
if [ -z "$STATUS" ]; then
  printf "${YELLOW}本地没有新的改动。${OFF}\n"
  read -p "是否仍执行 git push 同步远端？(Y/n) " ANSWER
  case "$ANSWER" in
    n|N) printf "已取消，未做任何操作。\n"; pause_exit 0 ;;
  esac
else
  printf "${BOLD}以下改动将被提交（git add -A，包含全部文件）：${OFF}\n"
  printf '%s\n' "$STATUS"
  echo
  read -p "确认提交并推送以上全部改动？(Y/n) " ANSWER
  case "$ANSWER" in
    n|N) printf "已取消，未做任何操作。\n"; pause_exit 0 ;;
  esac

  git add -A

  # ---- 提交前安全检查：疑似 API 密钥 / 私钥 ----
  Q="$(printf '\047')"  # 单引号，用于拼接正则
  KEY_PATTERN="sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(api[_-]?key|secret|token)[\"${Q}]?[[:space:]]*[:=][[:space:]]*[\"${Q}][A-Za-z0-9_+/=-]{20,}[\"${Q}]"
  SCAN_TMP="$(mktemp)"
  if git diff --cached | grep -nEi "$KEY_PATTERN" > "$SCAN_TMP" 2>/dev/null; then
    printf "\n${RED}警告：暂存内容中发现疑似 API 密钥 / 私钥：${OFF}\n"
    head -10 "$SCAN_TMP"
    rm -f "$SCAN_TMP"
    read -p "仍要继续提交吗？(y/N) " RISK
    case "$RISK" in
      y|Y) printf "${YELLOW}已按你的选择继续。${OFF}\n" ;;
      *) printf "已取消。文件仍处于暂存状态，可用 git reset 取消暂存。\n"; pause_exit 1 ;;
    esac
  else
    rm -f "$SCAN_TMP"
  fi

  # ---- 提交说明（直接回车使用自动生成的默认说明）----
  TOTAL="$(printf '%s\n' "$STATUS" | grep -c '^')"
  FILES_PREVIEW="$(printf '%s\n' "$STATUS" | sed 's/^...//' | head -3 | awk 'NR>1{printf "、"}{printf "%s",$0}')"
  NOW="$(date '+%Y-%m-%d %H:%M')"
  if [ "$TOTAL" -gt 3 ]; then
    DEFAULT_MSG="更新 ${TOTAL} 个文件：${FILES_PREVIEW} 等（${NOW}）"
  else
    DEFAULT_MSG="更新 ${TOTAL} 个文件：${FILES_PREVIEW}（${NOW}）"
  fi
  echo
  read -p "输入提交说明（直接回车使用默认）：" MSG
  [ -z "$MSG" ] && MSG="$DEFAULT_MSG"

  printf "\n${CYAN}>> 提交中…${OFF}\n"
  if ! git commit -m "$MSG"; then
    printf "${RED}提交失败，请根据上方 git 提示处理。${OFF}\n"
    pause_exit 1
  fi
fi

# ---- 推送 ----
printf "\n${CYAN}>> 推送到 %s（分支 %s）…${OFF}\n" "$REMOTE_URL" "$BRANCH"
if git push origin "$BRANCH" 2>&1; then
  printf "\n${GREEN}推送成功。最新提交：${OFF}\n"
  git log -1 --oneline
else
  printf "\n${RED}推送失败。常见原因：${OFF}\n"
  printf "  1. 网络或代理问题，无法访问 github.com\n"
  printf "  2. 没有该仓库的推送权限（需在本机配置 GitHub 凭证）\n"
  printf "  3. 远端有新提交，可先在终端执行：git pull --rebase origin %s\n" "$BRANCH"
  pause_exit 1
fi
pause_exit 0
