#!/bin/bash
# Watch for .experiments_complete flag and alert Ali via BB
RUNDIR=~/burn-tokens/research/runs/run_057_bn_manual

echo "[$(date)] Waiting for all experiments to complete..."

while [ ! -f "$RUNDIR/.experiments_complete" ]; do
    sleep 60
done

echo "[$(date)] Experiments complete! Notifying Ali..."

# Check what results exist
RESULTS=""
if [ -f "$RUNDIR/results_wideresnet.json" ]; then
    RESULTS="WideResNet: done"
else
    RESULTS="WideResNet: FAILED (no results file)"
fi

if [ -f "$RUNDIR/tinyimagenet_output.log" ]; then
    RESULTS="$RESULTS | TinyImageNet: done"
else
    RESULTS="$RESULTS | TinyImageNet: FAILED"
fi

# Alert via BB
curl -s -X POST "http://127.0.0.1:1234/api/v1/message/text?password=Ldasak2027!" \
  -H "Content-Type: application/json" \
  -d "{\"chatGuid\":\"iMessage;-;ali.saffarini8@gmail.com\",\"message\":\"Experiments finished. $RESULTS — ping me to review and write the paper\",\"tempGuid\":\"temp-exp-done-$(date +%s)\",\"method\":\"apple-script\"}"

echo "[$(date)] Notification sent. Done."
