# 1. Export environment variables
export STEPBASE=$PWD
export PYTHONPATH="$STEPBASE:$PYTHONPATH"
TAG="single-ws"
EXEC="python $STEPBASE/experiments/$TAG/main.py"
MODEL_NAME="DeepSeek-V3"
TEMPERATURE=1.0
BASE_PATH="$STEPBASE/result/$TAG"
EXAMPLE_PATH="$STEPBASE/prompts/proposer_base_ws.yaml"
NUM_SAMPLES=1
INPUT_CSV="$STEPBASE/experiments/benchcard_fullpath.csv"
OUTPUT_CSV="$STEPBASE/experiments/$TAG/result.csv"
$EXEC --model_name $MODEL_NAME --base_path $BASE_PATH --input_csv $INPUT_CSV --output_csv $OUTPUT_CSV --example_path $EXAMPLE_PATH --num_samples $NUM_SAMPLES --temperature $TEMPERATURE