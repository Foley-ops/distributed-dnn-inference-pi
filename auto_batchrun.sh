#!/bin/bash

# Create a script file
cat > run_batches.sh << 'EOF'
#!/bin/bash

# Directory to save results
OUTPUT_BASE_DIR="single_results"

# Create the base directory if it doesn't exist
mkdir -p $OUTPUT_BASE_DIR

# Loop through batch sizes 1-16
for batch_size in {1..16}; do
    echo "Running with batch size $batch_size"
    
    # Create a specific directory for this batch size
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/batch_${batch_size}"
    mkdir -p $OUTPUT_DIR
    
    # Run the evaluation script with the current batch size
    python3 scripts/model_evalutation/model_evaluation.py \
        --output-dir $OUTPUT_DIR \
        --warmup-iterations 10 \
        --num-inferences 100 \
        --num-workers 3 \
        --batch-size $batch_size \
        --aggregate \
        --models mobilenetv2 inception resnet18 alexnet vgg16 squeezenet
    
    echo "Completed batch size $batch_size"
done

echo "All batch sizes completed"
EOF

# Make the script executable
chmod +x run_batches.sh

# Run with nohup so it continues after you log out
nohup ./run_batches.sh > batch_run.log 2>&1 &

# Show the process ID so you can check on it later
echo "Process started with PID $!"
echo "You can check progress with: tail -f batch_run.log"