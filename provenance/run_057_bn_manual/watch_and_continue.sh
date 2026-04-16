#!/bin/bash
# Watch for WideResNet to finish, then launch TinyImageNet
WRESNET_PID=14993
RUNDIR=~/burn-tokens/research/runs/run_057_bn_manual

echo "[$(date)] Watching PID $WRESNET_PID (WideResNet)..."

while kill -0 $WRESNET_PID 2>/dev/null; do
    sleep 30
done

echo "[$(date)] WideResNet finished!"

# Check if results were saved
if [ -f "$RUNDIR/results_wideresnet.json" ]; then
    echo "[$(date)] WideResNet results saved successfully"
else
    echo "[$(date)] WARNING: No results_wideresnet.json found"
fi

# Launch TinyImageNet
echo "[$(date)] Launching TinyImageNet experiment..."
cd "$RUNDIR"
~/burn-tokens/.venv/bin/python3 -u exp3_tinyimagenet.py 2>&1 | tee tinyimagenet_output.log

echo "[$(date)] TinyImageNet finished!"

# Signal completion
echo "ALL_DONE" > "$RUNDIR/.experiments_complete"
date >> "$RUNDIR/.experiments_complete"
