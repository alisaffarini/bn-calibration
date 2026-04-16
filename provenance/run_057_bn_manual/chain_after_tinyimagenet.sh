#!/bin/bash
# After TinyImageNet finishes, retry WideResNet
TINYIMAGENET_PID=29522
RUNDIR=~/burn-tokens/research/runs/run_057_bn_manual
BB_URL="http://127.0.0.1:1234/api/v1/message/text?password=Ldasak2027!"
CHAT="iMessage;-;ali.saffarini8@gmail.com"

echo "[$(date)] Watching TinyImageNet PID $TINYIMAGENET_PID..."

while kill -0 $TINYIMAGENET_PID 2>/dev/null; do
    sleep 60
done

echo "[$(date)] TinyImageNet finished!"

# Check TinyImageNet results
if [ -f "$RUNDIR/results_tinyimagenet.json" ]; then
    echo "[$(date)] TinyImageNet results saved"
    TINY_STATUS="TinyImageNet: DONE"
else
    echo "[$(date)] WARNING: No TinyImageNet results"
    TINY_STATUS="TinyImageNet: FAILED (no results)"
fi

# Brief pause to free memory
sleep 10

# Retry WideResNet (previous run OOM'd on CPU after 4.5hrs — now should use MPS)
echo "[$(date)] Retrying WideResNet..."
cd "$RUNDIR"
~/burn-tokens/.venv/bin/python3 -u exp2_wideresnet.py > wideresnet_output_v2.log 2>&1
WRN_EXIT=$?

if [ -f "$RUNDIR/results_wideresnet.json" ]; then
    WRN_STATUS="WideResNet: DONE"
else
    WRN_STATUS="WideResNet: FAILED (exit $WRN_EXIT)"
fi

# Signal completion
echo "ALL_DONE" > "$RUNDIR/.experiments_complete"
date >> "$RUNDIR/.experiments_complete"
echo "$TINY_STATUS" >> "$RUNDIR/.experiments_complete"
echo "$WRN_STATUS" >> "$RUNDIR/.experiments_complete"

echo "[$(date)] All done: $TINY_STATUS | $WRN_STATUS"
