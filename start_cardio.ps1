# ========================================================
# FedCardio PowerShell Launcher
# ========================================================

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Starting FedCardio Multi-Hospital Clinical AI Platform " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Info
python cardio_cli.py info

# 2. Launch Streamlit
Write-Host "[INFO] Launching Web Portal at http://localhost:8501..." -ForegroundColor Green
python -m streamlit run web/app.py --server.port 8501
