#!/bin/bash
# Run experiments for different prediction tasks
# Usage: ./run_task_experiments.sh <task_name> [experiment_number]
#
# Tasks: subgoal_label, helper_aggregate, target_pos
#
# If experiment_number is not provided, runs all experiments for the task

set -e

CMD="python train_node_classifier.py"

# Check if task name is provided
if [ -z "$1" ]; then
    echo "Error: Task name required"
    echo ""
    echo "Usage: $0 <task_name> [experiment_number]"
    echo ""
    echo "Available tasks:"
    echo "  subgoal_label     - Predict subgoal labels"
    echo "  helper_aggregate  - Predict helper aggregate goal positions"
    echo "  target_pos        - Predict target goal positions"
    echo ""
    echo "Examples:"
    echo "  $0 subgoal_label        # Run all experiments for subgoal_label"
    echo "  $0 helper_aggregate 3   # Run experiment 3 for helper_aggregate"
    exit 1
fi

TASK=$1
EXP_NUM=$2

# Validate task
case $TASK in
    subgoal_label|helper_aggregate|target_pos)
        ;;
    *)
        echo "Error: Unknown task '$TASK'"
        exit 1
        ;;
esac

run_experiment() {
    local exp_name=$1
    shift
    echo "========================================"
    echo "Task: $TASK | Experiment: $exp_name"
    echo "========================================"
    $CMD task=$TASK wandb.name="${TASK}_${exp_name}" "$@"
}

BASE="model.d_model=256 model.nhead=8 model.num_layers=16 model.dim_feedforward=1024 model.dropout=0.1"

# =============================================================================
# JOINT ENCODING - dim 128
# =============================================================================

exp1() {
    run_experiment "joint128_lr001_e200" $BASE \
        data.positional_encoding=learned \
        "++data.positional_encoding_kwargs={encoding_dim: 128, combine_method: joint}" \
        training.max_lr=0.001 trainer.epochs=200
}

exp2() {
    run_experiment "joint128_lr002_e200" $BASE \
        data.positional_encoding=learned \
        "++data.positional_encoding_kwargs={encoding_dim: 128, combine_method: joint}" \
        training.max_lr=0.002 trainer.epochs=200
}

exp3() {
    run_experiment "joint128_lr005_e200" $BASE \
        data.positional_encoding=learned \
        "++data.positional_encoding_kwargs={encoding_dim: 128, combine_method: joint}" \
        training.max_lr=0.005 trainer.epochs=200
}

exp4() {
    run_experiment "joint128_lr001_e100" $BASE \
        data.positional_encoding=learned \
        "++data.positional_encoding_kwargs={encoding_dim: 128, combine_method: joint}" \
        training.max_lr=0.001 trainer.epochs=100
}

exp5() {
    run_experiment "joint128_lr001_e300" $BASE \
        data.positional_encoding=learned \
        "++data.positional_encoding_kwargs={encoding_dim: 128, combine_method: joint}" \
        training.max_lr=0.001 trainer.epochs=300
}

# =============================================================================
# JOINT ENCODING - dim 256
# =============================================================================

exp6() {
    run_experiment "joint256_lr001_e200" $BASE \
        data.positional_encoding=learned \
        "++data.positional_encoding_kwargs={encoding_dim: 256, combine_method: joint}" \
        training.max_lr=0.001 trainer.epochs=200
}

exp7() {
    run_experiment "joint256_lr002_e200" $BASE \
        data.positional_encoding=learned \
        "++data.positional_encoding_kwargs={encoding_dim: 256, combine_method: joint}" \
        training.max_lr=0.002 trainer.epochs=200
}

exp8() {
    run_experiment "joint256_lr005_e200" $BASE \
        data.positional_encoding=learned \
        "++data.positional_encoding_kwargs={encoding_dim: 256, combine_method: joint}" \
        training.max_lr=0.005 trainer.epochs=200
}

# =============================================================================
# ADDITIVE ENCODING - dim 128
# =============================================================================

exp9() {
    run_experiment "add128_lr001_e200" $BASE \
        data.positional_encoding=learned \
        "++data.positional_encoding_kwargs={encoding_dim: 128, combine_method: add}" \
        training.max_lr=0.001 trainer.epochs=200
}

exp10() {
    run_experiment "add128_lr002_e200" $BASE \
        data.positional_encoding=learned \
        "++data.positional_encoding_kwargs={encoding_dim: 128, combine_method: add}" \
        training.max_lr=0.002 trainer.epochs=200
}

exp11() {
    run_experiment "add128_lr005_e200" $BASE \
        data.positional_encoding=learned \
        "++data.positional_encoding_kwargs={encoding_dim: 128, combine_method: add}" \
        training.max_lr=0.005 trainer.epochs=200
}

exp12() {
    run_experiment "add128_lr001_e300" $BASE \
        data.positional_encoding=learned \
        "++data.positional_encoding_kwargs={encoding_dim: 128, combine_method: add}" \
        training.max_lr=0.001 trainer.epochs=300
}

# =============================================================================
# ADDITIVE ENCODING - dim 256
# =============================================================================

exp13() {
    run_experiment "add256_lr001_e200" $BASE \
        data.positional_encoding=learned \
        "++data.positional_encoding_kwargs={encoding_dim: 256, combine_method: add}" \
        training.max_lr=0.001 trainer.epochs=200
}

exp14() {
    run_experiment "add256_lr002_e200" $BASE \
        data.positional_encoding=learned \
        "++data.positional_encoding_kwargs={encoding_dim: 256, combine_method: add}" \
        training.max_lr=0.002 trainer.epochs=200
}

exp15() {
    run_experiment "add256_lr005_e200" $BASE \
        data.positional_encoding=learned \
        "++data.positional_encoding_kwargs={encoding_dim: 256, combine_method: add}" \
        training.max_lr=0.005 trainer.epochs=200
}

# Main execution
if [ -z "$EXP_NUM" ]; then
    echo "Running all 15 experiments for task: $TASK"
    for i in $(seq 1 15); do
        exp$i
    done
else
    echo "Running experiment $EXP_NUM for task: $TASK"
    exp$EXP_NUM
fi
