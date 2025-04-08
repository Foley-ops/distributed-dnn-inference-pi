#!/bin/bash
#
# Batch Evaluation Script
# 
# This script runs all model evaluations in sequence and generates
# a combined report with comparative metrics.
#

# Set paths and parameters
SCRIPT_DIR="$(dirname "$0")"
SCRIPT_PATH="${SCRIPT_DIR}/model_evaluation.py"
OUTPUT_DIR="$(eval echo ~/datasets/single_device_results/$(date +%Y%m%d_%H%M%S))"
LOG_DIR="${OUTPUT_DIR}/logs"
REPORT_PATH="${OUTPUT_DIR}/comparative_report.csv"

# Create output and log directories
mkdir -p "${LOG_DIR}"
echo "Created output directory: ${OUTPUT_DIR}"

# Models to evaluate
MODELS=(
    "mobilenetv2"
    "resnet18"
    "alexnet"
    "vgg16"
    "squeezenet"
    "inception"
    "deeplabv3"
)

# Evaluation parameters
BATCH_SIZE=16
NUM_INFERENCES=100
WARMUP_ITERATIONS=10

# Log system information
echo "=== System Information ===" | tee "${OUTPUT_DIR}/system_info.txt"
echo "Date: $(date)" | tee -a "${OUTPUT_DIR}/system_info.txt"
echo "Hostname: $(hostname)" | tee -a "${OUTPUT_DIR}/system_info.txt"

# Log CPU information if available
if command -v lscpu &> /dev/null; then
    echo "=== CPU Information ===" | tee -a "${OUTPUT_DIR}/system_info.txt"
    lscpu | tee -a "${OUTPUT_DIR}/system_info.txt"
fi

# Log memory information if available
if command -v free &> /dev/null; then
    echo "=== Memory Information ===" | tee -a "${OUTPUT_DIR}/system_info.txt"
    free -h | tee -a "${OUTPUT_DIR}/system_info.txt"
fi

# Log GPU information if available
if command -v nvidia-smi &> /dev/null; then
    echo "=== GPU Information ===" | tee -a "${OUTPUT_DIR}/system_info.txt"
    nvidia-smi | tee -a "${OUTPUT_DIR}/system_info.txt"
fi

# Run each model evaluation
echo "Starting batch evaluation of ${#MODELS[@]} models..."

for MODEL in "${MODELS[@]}"; do
    echo "Evaluating ${MODEL}..."
    LOG_FILE="${LOG_DIR}/${MODEL}.log"
    
    # Run the evaluation script
    python "${SCRIPT_PATH}" \
        --models "${MODEL}" \
        --batch-size "${BATCH_SIZE}" \
        --num-inferences "${NUM_INFERENCES}" \
        --warmup-iterations "${WARMUP_ITERATIONS}" \
        --output-dir "${OUTPUT_DIR}/individual" \
        --output-format "json" \
        2>&1 | tee "${LOG_FILE}"
    
    # Check if the evaluation was successful
    if [ $? -eq 0 ]; then
        echo "✓ ${MODEL} evaluation completed successfully"
    else
        echo "✗ ${MODEL} evaluation failed"
    fi
done

# Run comparative evaluation with all models
echo "Running comparative evaluation of all models..."
python "${SCRIPT_PATH}" \
    --models "${MODELS[@]}" \
    --batch-size "${BATCH_SIZE}" \
    --num-inferences "${NUM_INFERENCES}" \
    --warmup-iterations "${WARMUP_ITERATIONS}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-format "csv" \
    --aggregate \
    2>&1 | tee "${LOG_DIR}/comparative.log"

# Generate a simple HTML report if the CSV was created
if [ -f "${OUTPUT_DIR}/aggregate_results_"*".csv" ]; then
    LATEST_CSV=$(ls -t "${OUTPUT_DIR}/aggregate_results_"*".csv" | head -n 1)
    
    # Create HTML report
    HTML_REPORT="${OUTPUT_DIR}/report.html"
    echo "Generating HTML report: ${HTML_REPORT}"
    
    cat > "${HTML_REPORT}" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        .metadata { margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>Model Evaluation Results</h1>
    <div class="metadata">
        <p><strong>Date:</strong> $(date)</p>
        <p><strong>System:</strong> $(hostname)</p>
    </div>
    
    <h2>Comparative Results</h2>
    <table id="results">
        <thead>
            <tr>
EOF

    # Add table headers
    HEADERS=$(head -n 1 "${LATEST_CSV}")
    IFS=',' read -ra HEADER_ARRAY <<< "${HEADERS}"
    for HEADER in "${HEADER_ARRAY[@]}"; do
        echo "<th>${HEADER}</th>" >> "${HTML_REPORT}"
    done
    
    echo "</tr></thead><tbody>" >> "${HTML_REPORT}"
    
    # Add table data
    tail -n +2 "${LATEST_CSV}" | while IFS= read -r LINE; do
        echo "<tr>" >> "${HTML_REPORT}"
        IFS=',' read -ra VALUES <<< "${LINE}"
        for VALUE in "${VALUES[@]}"; do
            echo "<td>${VALUE}</td>" >> "${HTML_REPORT}"
        done
        echo "</tr>" >> "${HTML_REPORT}"
    done
    
    cat >> "${HTML_REPORT}" << EOF
        </tbody>
    </table>
    
    <h2>Key Metrics Comparison</h2>
    <div id="charts">
        <p>Charts would be displayed here with a JavaScript charting library.</p>
        <p>For a more interactive experience, import the CSV data into a spreadsheet application.</p>
    </div>
</body>
</html>
EOF

    echo "HTML report generated: ${HTML_REPORT}"
    echo "To view the report, open the file in a web browser."
fi

echo "Batch evaluation completed. Results saved to ${OUTPUT_DIR}"