#!/bin/bash
# 학습 진행 실시간 모니터 — 5초마다 refresh
# 사용: ./scripts/monitor_training.sh
# 종료: Ctrl+C (학습 자체는 계속 진행됨)

LOG=/tmp/baseline_ft.log
CSV=runs/detect/runs/baseline_ft/yolov8m_ft_902/results.csv

watch -n 5 -t "
echo '=== ⏱  $(date +%H:%M:%S)  |  Baseline FT (100 epoch) 진행 모니터 ==='
echo ''
echo '── GPU ────────────────────────────────────────────────'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null
echo ''
echo '── Epoch별 metric (results.csv) ───────────────────────'
if [ -f \"$CSV\" ]; then
    awk -F, 'NR==1 || NR>1' \"$CSV\" | awk -F, '{printf \"%-6s %-9s %-9s %-9s %-9s %-9s %-9s\n\", \$1, \$3, \$4, \$5, \$8, \$9, \$10}'
else
    echo '(아직 results.csv 없음)'
fi
echo ''
echo '── 현재 step (live) ───────────────────────────────────'
tr '\r' '\n' < \"$LOG\" 2>/dev/null | grep -E '^\\s+[0-9]+/100' | tail -1 | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g'
echo ''
echo '── 프로세스 상태 ──────────────────────────────────────'
if pgrep -f train_baseline_ft > /dev/null; then
    echo '✅ 학습 진행 중 (PID:' \$(pgrep -f train_baseline_ft | head -1) ')'
else
    echo '🛑 학습 프로세스 없음 (완료되었거나 종료됨)'
fi
"
