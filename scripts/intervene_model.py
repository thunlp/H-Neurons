import torch
import numpy as np


def get_h_neuron_indices(classifier, config):
    """Identify all H-Neurons (neurons with positive weights in the classifier)."""
    weights = classifier.coef_[0]
    inter_size = config.intermediate_size if hasattr(config, "intermediate_size") else config.text_config.intermediate_size

    # In L1-regularized models, neurons with weights > 0 are associated with hallucinations
    selected_flat_indices = np.where(weights > 0)[0]

    # Map flat indices to {layer_idx: [neuron_indices]}
    neuron_map = {}
    for idx in selected_flat_indices:
        layer_idx = int(idx // inter_size)
        neuron_idx = int(idx % inter_size)
        if layer_idx not in neuron_map:
            neuron_map[layer_idx] = []
        neuron_map[layer_idx].append(neuron_idx)
    
    return neuron_map

def apply_scaling(model, neuron_map, scale_factor):
    """Scale the specific columns of down_proj weights for identified neurons."""
    intervened_count = 0
    for name, module in model.named_modules():
        if 'down_proj' in name and isinstance(module, torch.nn.Linear):
            # Extract layer index from module string
            parts = name.split('.')
            try:
                layer_idx = next(int(p) for p in parts if p.isdigit())
            except StopIteration:
                continue

            if layer_idx in neuron_map:
                target_neurons = neuron_map[layer_idx]
                # Modifying weights in-place
                module.weight.data[:, target_neurons] *= scale_factor
                intervened_count += len(target_neurons)
    
    print(f"Intervention complete. Modified {intervened_count} H-Neurons using scale factor {scale_factor}.")