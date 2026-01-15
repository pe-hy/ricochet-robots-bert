#!/bin/bash
# Training experiments - variants of best model (l16_h8_d256)
# Base: d_model=256, nhead=8, num_layers=16, dim_feedforward=1024, dropout=0.1
# Usage: ./run_experiments.sh [experiment_number]

set -e

CMD="python train_node_classifier.py"

run_experiment() {
    local name=$1
    shift
    echo "========================================"
    echo "Running experiment: $name"
    echo "========================================"
    $CMD wandb.name="$name" "$@"
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
if [ -z "$1" ]; then
    echo "Running all 15 experiments..."
    for i in $(seq 1 15); do
        exp$i
    done
else
    echo "Running experiment $1..."
    exp$1
fi
