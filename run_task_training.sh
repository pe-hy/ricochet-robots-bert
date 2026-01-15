#!/bin/bash
# Training script for different prediction tasks
# Usage: ./run_task_training.sh <task_name> [additional_args...]
#
# Tasks:
#   subgoal_label     - Predict subgoal labels (binary classification)
#   helper_aggregate  - Predict helper aggregate goal positions (binary classification)
#   target_pos        - Predict target goal positions (binary classification)
#
# Examples:
#   ./run_task_training.sh subgoal_label
#   ./run_task_training.sh helper_aggregate model.num_layers=12
#   ./run_task_training.sh target_pos training.max_lr=0.002 trainer.epochs=300

set -e

# Check if task name is provided
if [ -z "$1" ]; then
    echo "Error: Task name required"
    echo ""
    echo "Usage: $0 <task_name> [additional_args...]"
    echo ""
    echo "Available tasks:"
    echo "  subgoal_label     - Predict subgoal labels"
    echo "  helper_aggregate  - Predict helper aggregate goal positions"
    echo "  target_pos        - Predict target goal positions"
    echo ""
    echo "Example:"
    echo "  $0 subgoal_label"
    echo "  $0 helper_aggregate model.num_layers=12"
    exit 1
fi

TASK=$1
shift  # Remove first argument, keep the rest

# Validate task name
case $TASK in
    subgoal_label|helper_aggregate|target_pos)
        echo "========================================"
        echo "Training task: $TASK"
        echo "========================================"
        ;;
    *)
        echo "Error: Unknown task '$TASK'"
        echo "Available tasks: subgoal_label, helper_aggregate, target_pos"
        exit 1
        ;;
esac

# Run training with specified task config
python train_node_classifier.py task=$TASK "$@"
