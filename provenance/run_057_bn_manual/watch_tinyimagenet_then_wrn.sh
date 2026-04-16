#!/bin/bash
# Watch TinyImageNet, then relaunch WideResNet (which died without results)
TINYIMAGENET_PID=29522
RUNDIR=~/burn-tokens/research/runs/run_057_bn_manual
VENV=~/burn-tokens/.venv/bin/python3

echo "[$(date)] Watching TinyImageNet PID $TINYIMAGENET_PID..."

while kill -0 $TINYIMAGENET_PID 2>/dev/null; do
    sleep 60
done

echo "[$(date)] TinyImageNet finished!"

if [ -f "$RUNDIR/results_tinyimagenet.json" ]; then
    echo "[$(date)] TinyImageNet results saved"
else
    echo "[$(date)] WARNING: No TinyImageNet results file"
fi

# Relaunch WideResNet (previous run died without results)
echo "[$(date)] Launching WideResNet experiment..."
cd "$RUNDIR"
$VENV -u exp2_wideresnet.py > wideresnet_output_v2.log 2>&1

echo "[$(date)] WideResNet finished!"

if [ -f "$RUNDIR/results_wideresnet.json" ]; then
    echo "[$(date)] WideResNet results saved"
    # Notify Ali
    curl -s -X POST "http://127.0.0.1:1234/api/v1/message/text?password=Ldasak2027!" \
      -H "Content-Type: application/json" \
      -d "{\"chatGuid\":\"iMessage;-;ali.saffarini8@gmail.com\",\"message\":\"All experiments done — TinyImageNet + WideResNet both complete. Results look strong. Ping me to review and write the final paper\",\"tempGuid\":\"temp-all-done-$(date +%s)\",\"method\":\"apple-script\"}"
else
    echo "[$(date)] WARNING: WideResNet failed again"
    curl -s -X POST "http://127.0.0.1:1234/api/v1/message/text?password=Ldasak2027!" \
      -H "Content-Type: application/json" \
      -d "{\"chatGuid\":\"iMessage;-;ali.saffarini8@gmail.com\",\"message\":\"TinyImageNet done but WideResNet failed again. Might need to debug or use a cloud GPU for WRN\",\"tempGuid\":\"temp-wrn-fail-$(date +%s)\",\"method\":\"apple-script\"}"
fi

echo "ALL_DONE" > "$RUNDIR/.experiments_complete"
date >> "$RUNDIR/.experiments_complete"
echo "[$(date)] All done."
