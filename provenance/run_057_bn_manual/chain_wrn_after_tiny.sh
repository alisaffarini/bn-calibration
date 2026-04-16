#!/bin/bash
# Wait for TinyImageNet to finish, then relaunch WideResNet (on MPS this time)
TINY_PID=29522
RUNDIR=~/burn-tokens/research/runs/run_057_bn_manual
PYTHON=~/burn-tokens/.venv/bin/python3

echo "[$(date)] Watching TinyImageNet PID $TINY_PID..."

while kill -0 $TINY_PID 2>/dev/null; do
    sleep 30
done

echo "[$(date)] TinyImageNet finished!"

# Check TinyImageNet results
if [ -f "$RUNDIR/results_tinyimagenet.json" ]; then
    echo "[$(date)] TinyImageNet results saved"
else
    echo "[$(date)] WARNING: No TinyImageNet results file"
fi

# Relaunch WideResNet (should use MPS now that TinyIN freed the GPU)
echo "[$(date)] Relaunching WideResNet on MPS..."
cd "$RUNDIR"
$PYTHON -u exp2_wideresnet.py > wideresnet_output_v2.log 2>&1
WRN_EXIT=$?

echo "[$(date)] WideResNet exited with code $WRN_EXIT"

if [ -f "$RUNDIR/results_wideresnet.json" ]; then
    echo "[$(date)] WideResNet results saved!"
else
    echo "[$(date)] WARNING: No WideResNet results"
fi

# Signal ALL done
echo "ALL_DONE" > "$RUNDIR/.experiments_complete"
date >> "$RUNDIR/.experiments_complete"

# Alert Ali
RESULTS=""
[ -f "$RUNDIR/results_tinyimagenet.json" ] && RESULTS="TinyImageNet: done" || RESULTS="TinyImageNet: FAILED"
[ -f "$RUNDIR/results_wideresnet.json" ] && RESULTS="$RESULTS | WideResNet: done" || RESULTS="$RESULTS | WideResNet: FAILED"

curl -s -X POST "http://127.0.0.1:1234/api/v1/message/text?password=Ldasak2027!" \
  -H "Content-Type: application/json" \
  -d "{\"chatGuid\":\"iMessage;-;ali.saffarini8@gmail.com\",\"message\":\"All experiments finished. $RESULTS — ready to write the paper\",\"tempGuid\":\"temp-all-done-$(date +%s)\",\"method\":\"apple-script\"}"

echo "[$(date)] All done."
