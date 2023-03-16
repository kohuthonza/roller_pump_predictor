source /home/ikohut/python_environments/roller_pump_predictor/bin/activate 
PROJECT_PATH="/home/ikohut/Projects/roller_pump_predictor/"
BASE_EXPERIMENTS_PATH="/home/ikohut/experiments/"
export PYTHONPATH=$PROJECT_PATH

DATE="2023-03-16"
NET="conv_attention_net"

BASE="/home/ikohut/data/roller_pump_data.2022-06-17/"

TRN_NUM_WORKERS=4
TST_NUM_WORKERS=4
TRN_PREFETCH_FACTOR=1
TST_PREFETCH_FACTOR=1

CHECKPOINT=$1
TST_WAVE_DIRECTORY_PATH=$2
TEST_NAME=$3

BATCH_SIZE=64

EXPERIMENT_DIR=$BASE_EXPERIMENTS_PATH"roller_pump_data.2022-06-17/testing/"$TEST_NAME"."$NET"."$DATE"/"
mkdir -p $EXPERIMENT_DIR
SHOW_DIR=$EXPERIMENT_DIR"show"
mkdir -p $SHOW_DIR
EXPORT_DIR=$EXPERIMENT_DIR"export"
mkdir -p $EXPORT_DIR

LOG=$EXPERIMENT_DIR$NET"."$DATE"_"$START_ITERATION".log"

sync

python -u $PROJECT_PATH"train.py" \
  --tst-wave-directory-path $TST_WAVE_DIRECTORY_PATH \
  --tst-num-workers $TST_NUM_WORKERS \
  --tst-prefetch-factor $TST_PREFETCH_FACTOR \
  --net $NET \
  --batch-size $BATCH_SIZE \
  --test \
  --in-checkpoint $CHECKPOINT \
  --show-dir $SHOW_DIR \
  --export-dir $EXPORT_DIR > $LOG 2>&1

echo DONE
