# 1. Export environment variables
export STEPBASE=$PWD
export PYTHONPATH="$STEPBASE:$PYTHONPATH"
TAG="agent"
EXEC="python $STEPBASE/experiments/$TAG/main.py"
MODEL_NAME="DeepSeek-V3"
TEMPERATURE=0.5
BASE_PATH="$STEPBASE/result/$TAG"
EXAMPLE_0_PATH="$STEPBASE/prompts/proposer_base.yaml"
EXAMPLE_1_PATH="$STEPBASE/prompts/guardian_base.yaml"
NUM_SAMPLES=1
INPUT_CSV="$STEPBASE/experiments/benchcard_fullpath.csv"
OUTPUT_CSV="$STEPBASE/experiments/$TAG/result.csv"
$EXEC --model_name $MODEL_NAME --base_path $BASE_PATH --input_csv $INPUT_CSV --output_csv $OUTPUT_CSV --example_0_path $EXAMPLE_0_PATH --example_1_path $EXAMPLE_1_PATH --num_samples $NUM_SAMPLES --temperature $TEMPERATURE