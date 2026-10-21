#!/bin/bash

# Duration settings
RUN_TIME=205
PAUSE_TIME=40
N_TRIALS=10

# World to run — one of: exp1_speed12, exp2_speed9, exp3_speed8
WORLD=exp1_speed12

cleanup() {
    echo "Cleaning up..."
    pkill -SIGINT -f "gz sim" 2>/dev/null
    sleep 3
    pkill -SIGKILL -f "gz sim" 2>/dev/null
    pkill -SIGKILL -f "gz" 2>/dev/null
    pkill -SIGKILL -f "ros2" 2>/dev/null
    sleep 3
    echo "Cleanup done."
}

trap 'echo "Script interrupted!"; cleanup; exit 1' SIGINT SIGTERM

echo "=========================================="
echo "Starting world: $WORLD"
echo "=========================================="

for i in $(seq 1 $N_TRIALS); do
    echo "--------------------------------------------------"
    echo "Running world: $WORLD (iteration $i / $N_TRIALS)"
    echo "--------------------------------------------------"

    cmd="ros2 launch informative_planning hybrid_plan_log.launch.py headless:=True world:=$WORLD"

    echo "Executing: $cmd"

    # Run launch in background and capture PID
    $cmd &
    LAUNCH_PID=$!

    # Wait for RUN_TIME seconds then stop
    sleep $RUN_TIME

    echo "Time up, shutting down..."

    # Kill gz sim FIRST — it ignores SIGINT after the first, needs SIGKILL
    pkill -SIGINT -f "gz sim" 2>/dev/null
    sleep 2
    pkill -SIGKILL -f "gz sim" 2>/dev/null
    pkill -SIGKILL -f "gz" 2>/dev/null
    # Wait until gz processes are gone before touching launch
    for j in $(seq 1 10); do
        pgrep -f "gz" > /dev/null 2>&1 || break
        sleep 1
    done

    # Now send SIGINT to the launch process — it should exit cleanly now
    kill -SIGINT $LAUNCH_PID 2>/dev/null
    sleep 5

    # Force kill if still running
    kill -SIGKILL $LAUNCH_PID 2>/dev/null
    cleanup

    echo "Finished: $WORLD (iteration $i)"
    echo "Waiting $PAUSE_TIME seconds..."
    sleep $PAUSE_TIME
done

echo "All iterations complete!"
