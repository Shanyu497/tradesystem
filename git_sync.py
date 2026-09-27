"""
git_sync.py
============
Phase 5 雲端部署專用：把 data/processed/ 的最新資料 commit + push 回這個 git repo，
讓部署在 Streamlit Community Cloud 上的 Dashboard 能讀到當天最新資料——雲端環境看不到
你本機的檔案系統，只能讀 repo 裡的檔案；Streamlit Cloud 偵測到 repo 有新的 push
會自動重新部署，這是它官方支援的更新機制，不需要另外寫部署腳本。

只有這個目錄本身是「已經設定好遠端（origin）」的 git repo 時才會真的推送成功；
沒有 git、還沒 git init、或還沒設定 remote 的情況下（例如只是想在本機看 Dashboard，
不打算部署雲端），這裡的每一步都會安全地略過並記 log，不會讓資料更新排程失敗
（呼應 line_client.py／base_adapter.py 的既有設計原則）。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("git_sync")

PROJECT_ROOT = Path(__file__).resolve().parent

# scheduler.py 常常是在「PATH 還沒吃到新裝軟體」的 shell 裡被啟動的（例如剛用
# winget 裝完 git、還沒重開終端機），所以不能只靠 shutil.which("git") 找不到
# 就放棄，這裡多一層 Windows 預設安裝路徑的 fallback。
_GIT_FALLBACK_PATHS = [
    r"C:\Program Files\Git\cmd\git.exe",
    r"C:\Program Files (x86)\Git\cmd\git.exe",
]


def _resolve_git() -> str:
    found = shutil.which("git")
    if found:
        return found
    for candidate in _GIT_FALLBACK_PATHS:
        if Path(candidate).exists():
            return candidate
    return "git"  # 交給 subprocess 嘗試，找不到的話 _run_git 會捕捉例外並記 log


def _run_git(*args: str) -> subprocess.CompletedProcess | None:
    """回傳 None 代表連 git 執行檔都找不到／啟動失敗（不是 git 指令本身失敗）"""
    try:
        return subprocess.run(
            [_resolve_git(), *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60,
        )
    except OSError as e:
        logger.warning(f"無法執行 git（可能沒裝或不在 PATH 上）: {e}")
        return None


def _ok(result: subprocess.CompletedProcess | None, step: str) -> bool:
    if result is None:
        return False  # _run_git 已經記過 log 了
    if result.returncode != 0:
        logger.warning(f"git {step} 失敗: {result.stderr.strip()}")
        return False
    return True


def sync_processed_data() -> bool:
    """
    把 data/processed/ 的變更 commit + push。只有真的有變更時才會 commit
    （避免每次排程觸發都產生空 commit）。回傳是否成功推送；
    任何一步失敗都只記 log、不丟例外，呼叫端不需要包 try/except。
    """
    if not (PROJECT_ROOT / ".git").exists():
        logger.debug("尚未初始化 git repo，略過資料同步（未部署雲端時這是預期行為）")
        return False

    status = _run_git("status", "--porcelain", "data/processed")
    if not _ok(status, "status"):
        return False
    if not status.stdout.strip():
        logger.info("data/processed 沒有變更，略過同步")
        return False

    if not _ok(_run_git("add", "data/processed"), "add"):
        return False

    message = f"資料更新 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    if not _ok(_run_git("commit", "-m", message), "commit"):
        return False

    push = _run_git("push")
    if not _ok(push, "push"):
        logger.warning("commit 已保留在本機，下次同步時會一起推送")
        return False

    logger.info("已將最新資料 push 回 git repo，Streamlit Cloud 會自動重新部署")
    return True
