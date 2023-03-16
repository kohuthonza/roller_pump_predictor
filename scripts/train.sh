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

TRN_WAVE_DIRECTORY_PATH=$BASE"trn"
TST_WAVE_DIRECTORY_PATH=$BASE"tst"

OPTIMIZER="Adam"
LEARNING_RATE=0.0003
BATCH_SIZE=64
DROPOUT_RATE=0
START_ITERATION=0
MAX_ITERATIONS=20000

TEST_STEP=1000
SAVE_STEP=1000

EXPERIMENT_DIR=$BASE_EXPERIMENTS_PATH"roller_pump_data.2022-06-17/training/"$NET"."$DATE"/"
mkdir -p $EXPERIMENT_DIR
CHECKPOINT_DIR=$EXPERIMENT_DIR"checkpoints"
mkdir -p $CHECKPOINT_DIR
SHOW_DIR=$EXPERIMENT_DIR"show"
mkdir -p $SHOW_DIR
EXPORT_DIR=$EXPERIMENT_DIR"export"
mkdir -p $EXPORT_DIR

LOG=$EXPERIMENT_DIR$NET"."$DATE"_"$START_ITERATION".log"

sync

python -u $PROJECT_PATH"train.py" \
  --trn-wave-directory-path $TRN_WAVE_DIRECTORY_PATH \
  --tst-wave-directory-path $TST_WAVE_DIRECTORY_PATH \
  --trn-num-workers $TRN_NUM_WORKERS \
  --tst-num-workers $TST_NUM_WORKERS \
  --trn-prefetch-factor $TRN_PREFETCH_FACTOR \
  --tst-prefetch-factor $TST_PREFETCH_FACTOR \
  --net $NET \
  --optimizer $OPTIMIZER \
  --learning-rate $LEARNING_RATE \
  --batch-size $BATCH_SIZE \
  --dropout-rate $DROPOUT_RATE \
  --start-iteration $START_ITERATION \
  --max-iterations $MAX_ITERATIONS \
  --test \
  --checkpoint-dir $CHECKPOINT_DIR \
  --save-step $SAVE_STEP \
  --test-step $TEST_STEP \
  --show-dir $SHOW_DIR \
  --export-dir $EXPORT_DIR > $LOG 2>&1

echo DONE
