#!/bin/bash
# Save as start_worker.sh

# Load the virtual environment
source env/bin/activate

MAX_ATTEMPTS=10
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "Attempt $ATTEMPT to start worker with rank $1..."
    
    # Run the worker script
    python3 rpc_layer_split.py --rank $1 --world-size 3
    
    # Check the exit code
    if [ $? -eq 0 ]; then
        echo "Worker successfully completed"
        exit 0
    fi
    
    echo "Worker attempt $ATTEMPT failed. Waiting 10 seconds before retry..."
    sleep 10
    ATTEMPT=$((ATTEMPT + 1))
done

echo "Failed to start worker after $MAX_ATTEMPTS attempts"
exit 1