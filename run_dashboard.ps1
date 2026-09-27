# run_dashboard.ps1
# ===================
# 本機啟動 Streamlit Dashboard 的便利腳本。
#
# 為什麼不能直接 `streamlit run dashboard\Home.py`：
# 這台機器預設的 Python 是 3.14，而 streamlit 目前用到的 anyio 版本在 Python 3.14 上
# 有已知的相容性問題（靜態檔案伺服器會丟出 TypeError: cannot create weak reference
# to 'NoneType' object，整個 Dashboard 打不開）。雲端部署（Streamlit Community Cloud）
# 用 .python-version 檔案鎖定 3.11 解決了這個問題，本機也用同樣的做法：
# 開一個獨立的 Python 3.11 虛擬環境（.venv311/，不進版控）專門跑 Dashboard，
# 其他腳本（pipeline.py、scheduler.py、signal_report.py、backtest_report.py）
# 不受影響，繼續用系統預設的 Python 3.14 就好。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File run_dashboard.ps1
#   或在 PowerShell 裡直接 .\run_dashboard.ps1

$ErrorActionPreference = "Stop"
$venvPython = Join-Path $PSScriptRoot ".venv311\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "找不到 .venv311，建立中（第一次執行會比較久）..."
    py -3.11 -m venv (Join-Path $PSScriptRoot ".venv311")
    if (-not $?) {
        Write-Host "建立虛擬環境失敗，請先確認已安裝 Python 3.11（winget install --id Python.Python.3.11）"
        exit 1
    }
    & (Join-Path $PSScriptRoot ".venv311\Scripts\pip.exe") install -r (Join-Path $PSScriptRoot "requirements.txt")
}

& $venvPython -m streamlit run (Join-Path $PSScriptRoot "dashboard\Home.py")
