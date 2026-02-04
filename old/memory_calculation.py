"""
Calculate memory requirements for different architectures.

Based on config/multitask.yaml:
- batch_size: 256
- d_model: 512
- nhead: 16
- num_layers: 20
- dim_feedforward: 1024
- num_comp_vectors: 3
"""

# Configuration
batch_size = 256
d_model = 512
nhead = 16
num_layers = 20
dim_feedforward = 1024
num_comp_vectors = 3
num_nodes = 256

# Bytes per float32
bytes_per_float = 4

def format_bytes(bytes_val):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"

def calculate_attention_memory(batch_size, nhead, seq_len):
    """Calculate memory for attention matrices."""
    # Attention scores: [batch, nhead, seq_len, seq_len]
    attention_scores = batch_size * nhead * seq_len * seq_len * bytes_per_float

    # Need to store for all layers (for gradient computation)
    total = attention_scores * num_layers
    return total

def calculate_activation_memory(batch_size, seq_len, d_model, num_layers):
    """Calculate memory for activations."""
    # Per layer: input, attention output, feedforward intermediate, output
    # Simplified: ~4 × (batch × seq_len × d_model)
    per_layer = 4 * batch_size * seq_len * d_model * bytes_per_float
    total = per_layer * num_layers
    return total

def calculate_feedforward_memory(batch_size, seq_len, dim_feedforward):
    """Calculate memory for feedforward intermediate activations."""
    # FFN intermediate: [batch, seq_len, dim_feedforward]
    per_layer = batch_size * seq_len * dim_feedforward * bytes_per_float
    total = per_layer * num_layers
    return total

def calculate_model_parameters():
    """Calculate model parameter count."""
    # Input projection
    input_proj = num_nodes * d_model

    # Transformer layers
    # Per layer: QKV projections, output projection, 2 FFN layers, 2 layer norms
    per_layer = (
        3 * d_model * d_model +  # Q, K, V
        d_model * d_model +      # Output projection
        d_model * dim_feedforward +  # FFN up
        dim_feedforward * d_model +  # FFN down
        2 * d_model              # Layer norms
    )
    transformer_params = per_layer * num_layers

    # Label embeddings
    label_embeds = 4 * d_model

    # Computation vectors
    comp_embeds = 4 * num_comp_vectors * d_model

    # Classifiers (4 tasks)
    # Each: Linear(2*d_model, d_model) + Linear(d_model, 1)
    classifier_params = 4 * (2 * d_model * d_model + d_model * d_model)

    # Teacher forcing: label encoder
    label_encoder = 2 * d_model  # Embedding(2, d_model)

    total_params = (input_proj + transformer_params + label_embeds +
                   comp_embeds + classifier_params + label_encoder)

    return total_params

def calculate_total_memory(seq_len, architecture_name):
    """Calculate total memory for given sequence length."""
    print(f"\n{'='*60}")
    print(f"Architecture: {architecture_name}")
    print(f"Sequence Length: {seq_len} tokens")
    print(f"{'='*60}")

    # 1. Model parameters
    params = calculate_model_parameters()
    params_memory = params * bytes_per_float
    print(f"Model Parameters: {params:,} ({format_bytes(params_memory)})")

    # 2. Gradients (same size as parameters)
    gradients_memory = params_memory
    print(f"Gradients: {format_bytes(gradients_memory)}")

    # 3. Optimizer states (AdamW: 2 states per parameter)
    optimizer_memory = 2 * params_memory
    print(f"Optimizer States (AdamW): {format_bytes(optimizer_memory)}")

    # 4. Activations
    attention_mem = calculate_attention_memory(batch_size, nhead, seq_len)
    activation_mem = calculate_activation_memory(batch_size, seq_len, d_model, num_layers)
    ffn_mem = calculate_feedforward_memory(batch_size, seq_len, dim_feedforward)

    total_activation = attention_mem + activation_mem + ffn_mem
    print(f"\nActivations:")
    print(f"  Attention matrices: {format_bytes(attention_mem)}")
    print(f"  Layer activations: {format_bytes(activation_mem)}")
    print(f"  FFN intermediates: {format_bytes(ffn_mem)}")
    print(f"  Total activations: {format_bytes(total_activation)}")

    # 5. Total
    total = params_memory + gradients_memory + optimizer_memory + total_activation
    print(f"\n{'='*60}")
    print(f"TOTAL MEMORY: {format_bytes(total)}")
    print(f"{'='*60}")

    return total

if __name__ == "__main__":
    print("="*60)
    print("MEMORY REQUIREMENT CALCULATION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  batch_size: {batch_size}")
    print(f"  d_model: {d_model}")
    print(f"  nhead: {nhead}")
    print(f"  num_layers: {num_layers}")
    print(f"  dim_feedforward: {dim_feedforward}")
    print(f"  num_comp_vectors: {num_comp_vectors}")

    # ComputationVectorMultiTask
    comp_seq_len = num_nodes + 4 * (num_comp_vectors + 1)  # 256 + 4*4 = 272
    comp_memory = calculate_total_memory(comp_seq_len, "ComputationVectorMultiTask")

    # TeacherForcingComputationVectorMultiTask
    # Interleaved structure: [nodes, comp1, label1, GT1, comp2, label2, GT2, ...]
    # Task 1: nodes(256) + comp(3) + label(1) = 260
    # Task 2: nodes(256) + comp(3) + label(1) + GT1(1) + comp(3) + label(1) = 265
    # Task 3: nodes(256) + (comp(3) + label(1) + GT(1)) * 2 + comp(3) + label(1) = 270
    # Task 4: nodes(256) + (comp(3) + label(1) + GT(1)) * 3 + comp(3) + label(1) = 275
    # CRITICAL: Must store activations from ALL 4 passes for backpropagation!

    print("\n" + "="*60)
    print("TEACHER FORCING: ALL 4 PASSES")
    print("="*60)

    tf_seq_lens = [
        num_nodes + num_comp_vectors + 1,  # Task 1: 260
        num_nodes + 2 * (num_comp_vectors + 1) + 1,  # Task 2: 265
        num_nodes + 3 * (num_comp_vectors + 1) + 2,  # Task 3: 270
        num_nodes + 4 * (num_comp_vectors + 1) + 3,  # Task 4: 275
    ]

    print(f"Sequence lengths: {tf_seq_lens}")
    print(f"Total tokens across all passes: {sum(tf_seq_lens)}")

    # Calculate memory for each pass
    tf_memories = []
    for i, seq_len in enumerate(tf_seq_lens, 1):
        memory = calculate_total_memory(seq_len, f"TeacherForcing Task {i}")
        tf_memories.append(memory)

    # Total memory is the sum because we need activations from all passes
    tf_total_memory = sum(tf_memories)

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"ComputationVectorMultiTask (1 pass):     {format_bytes(comp_memory)}")
    print(f"TeacherForcing (4 passes TOTAL):         {format_bytes(tf_total_memory)}")
    print(f"Ratio: {tf_total_memory / comp_memory:.2f}x")
    print(f"\nMemory increase: {format_bytes(tf_total_memory - comp_memory)}")

    print(f"\n{'='*60}")
    print("WHY SO MUCH MORE MEMORY?")
    print(f"{'='*60}")
    print("Teacher forcing requires 4 sequential forward passes.")
    print("All activations must be stored simultaneously for backpropagation!")
    print(f"\nPer-pass breakdown:")
    for i, (seq_len, memory) in enumerate(zip(tf_seq_lens, tf_memories), 1):
        print(f"  Task {i}: {seq_len:3d} tokens → {format_bytes(memory)}")
    print(f"\nTotal: {sum(tf_seq_lens)} tokens across all passes")
