# Multi-Task Training Guide

The training system now supports training separate models for different prediction tasks using the same architecture.

## Available Tasks

### 1. **subgoal_label** (default)
- **Predicts**: Whether a position is a subgoal (binary classification)
- **Target index**: 19
- **Uses features**: All goal position features [14, 15, 16, 17, 18]

### 2. **helper_aggregate**
- **Predicts**: Helper aggregate goal position (binary classification)
- **Target index**: 17
- **Uses features**: [14, 15, 16, 18, 19] (excludes itself)

### 3. **target_pos**
- **Predicts**: Target goal position (binary classification)
- **Target index**: 18
- **Uses features**: [14, 15, 16, 17, 19] (excludes itself)

## Feature Format (20 features per node)

```
[0] x coordinate
[1] y coordinate
[2-6] robot_type (one-hot: none, target, helper1, helper2, helper3)
[7-8] has_goal (one-hot: no, yes)
[9-13] walls (one-hot: none, top, left, right, bottom)
[14] helper1_goal_pos
[15] helper2_goal_pos
[16] helper3_goal_pos
[17] helper_aggregate_goal_pos
[18] target_goal_pos
[19] subgoal_label
```

## Usage

### Simple Training

Run training for a specific task:

```bash
# Train subgoal_label task (default)
./run_task_training.sh subgoal_label

# Train helper_aggregate task
./run_task_training.sh helper_aggregate

# Train target_pos task
./run_task_training.sh target_pos
```

### Training with Custom Parameters

Override any config parameter:

```bash
./run_task_training.sh helper_aggregate model.num_layers=12 training.max_lr=0.002

./run_task_training.sh target_pos trainer.epochs=300 model.d_model=512
```

### Running Experiments

Run predefined experiments for a task:

```bash
# Run all 15 experiments for subgoal_label
./run_task_experiments.sh subgoal_label

# Run experiment 5 for helper_aggregate
./run_task_experiments.sh helper_aggregate 5

# Run all experiments for target_pos
./run_task_experiments.sh target_pos
```

### Direct Hydra Usage

```bash
# Use task override
python train_node_classifier.py task=helper_aggregate

# Combine with other overrides
python train_node_classifier.py task=target_pos model.num_layers=24 training.max_lr=0.005
```

## Configuration Files

Task configurations are stored in `config/task/`:
- `subgoal_label.yaml`
- `helper_aggregate.yaml`
- `target_pos.yaml`

## Adding a New Task (4th task)

1. Create a new config file: `config/task/my_new_task.yaml`

```yaml
# @package task

name: my_new_task
prediction_type: binary_classification

target_index: 20  # or whatever index in the feature vector

# Include features you want as input (exclude the target)
include_goal_features: [14, 15, 16, 17, 18, 19]
```

2. Use it:

```bash
./run_task_training.sh my_new_task
```

That's it! The system is designed to be extensible for any new prediction task.

## Model Outputs

Each task produces separate:
- **Checkpoints**: `./tmp/checkpoints/{task_name}_...`
- **WandB logs**: Tagged with task name
- **Test predictions**: `./tmp/predictions/test_{iid|ood}_{task_name}_predictions.pkl`

## Notes

- All tasks use the same model architecture (Transformer-based node classifier)
- The difference is only in **what we predict** and **which features we include**
- Feature dimension automatically adjusts based on included goal features
- Both IID and OOD test sets are evaluated for each task
