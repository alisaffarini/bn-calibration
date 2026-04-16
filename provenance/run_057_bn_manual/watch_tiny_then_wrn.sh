#!/bin/bash
# Watch TinyImageNet, then retry WideResNet with smaller batch
TINY_PID=29522
RUNDIR=~/burn-tokens/research/runs/run_057_bn_manual
PYTHON=~/burn-tokens/.venv/bin/python3
BB_URL="http://127.0.0.1:1234/api/v1/message/text?password=Ldasak2027!"
CHAT="iMessage;-;ali.saffarini8@gmail.com"

echo "[$(date)] Watching TinyImageNet PID $TINY_PID..."

while kill -0 $TINY_PID 2>/dev/null; do
    sleep 30
done

echo "[$(date)] TinyImageNet finished!"

# Check TinyImageNet results
if grep -q "RESULTS" "$RUNDIR/tinyimagenet_output.log" 2>/dev/null; then
    echo "[$(date)] TinyImageNet has results"
    TINY_STATUS="TinyImageNet: DONE"
else
    echo "[$(date)] WARNING: TinyImageNet may have failed"
    TINY_STATUS="TinyImageNet: CHECK RESULTS"
fi

# Now retry WideResNet with smaller batch
echo "[$(date)] Launching WideResNet (batch_size=32)..."
cd "$RUNDIR"
$PYTHON -u exp2_wideresnet_small_batch.py > wideresnet_v2_output.log 2>&1
WRN_EXIT=$?

if [ $WRN_EXIT -eq 0 ] && [ -f "$RUNDIR/results_wideresnet.json" ]; then
    WRN_STATUS="WideResNet: DONE"
else
    WRN_STATUS="WideResNet: FAILED (exit $WRN_EXIT)"
fi

# Signal complete and notify
echo "ALL_DONE" > "$RUNDIR/.experiments_complete"
date >> "$RUNDIR/.experiments_complete"

MSG="All experiments finished. $TINY_STATUS | $WRN_STATUS — ping me to review results and write the paper"
curl -s -X POST "$BB_URL" \
  -H "Content-Type: application/json" \
  -d "{\"chatGuid\":\"$CHAT\",\"message\":\"$MSG\",\"tempGuid\":\"temp-all-done-$(date +%s)\",\"method\":\"apple-script\"}"

echo "[$(date)] All done. Notification sent."
