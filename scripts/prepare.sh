# 1. Export environment variables
export STEPBASE=$PWD
export PYTHONPATH="$STEPBASE:$PYTHONPATH"
EXEC="python"

$EXEC $STEPBASE/scripts/add_fullpath.py