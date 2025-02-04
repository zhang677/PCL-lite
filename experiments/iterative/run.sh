# 1. Export environment variables
export STEPBASE=$PWD
export PYTHONPATH="$STEPBASE:$PYTHONPATH"
TAG="iterative"
EXEC="python $STEPBASE/experiments/$TAG/main.py"
MODEL_NAME="DeepSeek-V3"
MODEL_NICKNAME="ds"
BASE_PATH="$STEPBASE/result/$TAG"
EXAMPLE_1_PATH="$STEPBASE/prompts/guardian_base.yaml"
NUM_SAMPLES=1
NUM_GROUPS=1
ITER=0
cp $STEPBASE/experiments/agent/result.csv $STEPBASE/experiments/$TAG/result_${MODEL_NICKNAME}_merged_${ITER}.csv
$EXEC --model_name $MODEL_NAME --model_nickname $MODEL_NICKNAME --base_path $BASE_PATH --example_1_path $EXAMPLE_1_PATH --num_samples $NUM_SAMPLES --base_date $TAG --num_groups $NUM_GROUPS --start_iter $ITER