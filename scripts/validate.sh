# 1. Export environment variables
export STEPBASE=$PWD
export PYTHONPATH="$STEPBASE:$PYTHONPATH"
mkdir -p $STEPBASE/validation
EXEC="python"

# 2. Generate pytestable scripts from yaml files
$EXEC $STEPBASE/tools/yaml_to_code.py --mode decode --yaml $STEPBASE/prompts/proposer_base.yaml --output $STEPBASE/validation
for file in $STEPBASE/benchmark/bmm/*.yaml; do
    filepy=$(basename $file .yaml).py
    if echo "$filepy" | grep -q "test"; then
        $EXEC $STEPBASE/tools/yaml_to_code.py --mode plan --yaml $file --output $STEPBASE/validation/$filepy
    else
        $EXEC $STEPBASE/tools/yaml_to_code.py --mode single --yaml $file --output $STEPBASE/validation/$filepy
    fi
done
for file in $STEPBASE/validation/*.py; do
    $EXEC $STEPBASE/scripts/check_validation.py --file $file
done
