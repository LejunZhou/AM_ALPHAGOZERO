"""Test loading pretrained reference weights and verifying outputs match."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from am_baseline.config import Config
from am_baseline.model.attention_model import AttentionModel


def build_key_mapping():
    """Map reference model parameter names to our model parameter names."""
    mapping = {}

    # init_embed: same name, direct
    mapping['init_embed.weight'] = 'init_embed.weight'
    mapping['init_embed.bias'] = 'init_embed.bias'

    # Encoder: ref uses 'embedder.layers.N.*', we use the same
    # GraphAttentionEncoder layers: embedder.layers.{0,1,2}.{0,1,2,3}.*
    # 0 = SkipConnection(MHA), 1 = Norm, 2 = SkipConnection(FFN), 3 = Norm
    # These match directly since we kept the same structure

    # Decoder projections: ref has them on AttentionModel, we moved to Decoder
    decoder_params = [
        'project_node_embeddings.weight',
        'project_fixed_context.weight',
        'project_step_context.weight',
        'project_out.weight',
    ]
    for p in decoder_params:
        mapping[p] = 'decoder.' + p

    # W_placeholder: ref has it on AttentionModel, we moved to Decoder
    mapping['W_placeholder'] = 'decoder.W_placeholder'

    return mapping


def load_ref_pretrained(ref_path, config):
    """Load reference pretrained weights into our model."""
    model = AttentionModel(config)

    ref_data = torch.load(ref_path, map_location='cpu', weights_only=False)
    ref_state = ref_data.get('model', ref_data)

    mapping = build_key_mapping()
    our_state = model.state_dict()

    # Map ref keys to our keys
    mapped = {}
    unmapped_ref = []
    for ref_key, ref_val in ref_state.items():
        if ref_key in mapping:
            our_key = mapping[ref_key]
        elif ref_key.startswith('embedder.'):
            # Encoder params pass through directly
            our_key = ref_key
        else:
            unmapped_ref.append(ref_key)
            continue

        if our_key in our_state:
            if our_state[our_key].shape == ref_val.shape:
                mapped[our_key] = ref_val
            else:
                print(f'  Shape mismatch: {ref_key} -> {our_key}: '
                      f'ref {ref_val.shape} vs ours {our_state[our_key].shape}')
        else:
            print(f'  Key not found in our model: {our_key} (from ref {ref_key})')

    if unmapped_ref:
        print(f'  Unmapped ref keys: {unmapped_ref}')

    # Check for missing keys in our model
    missing = set(our_state.keys()) - set(mapped.keys())
    if missing:
        print(f'  Keys in our model not loaded: {missing}')

    our_state.update(mapped)
    model.load_state_dict(our_state)
    print(f'  Loaded {len(mapped)}/{len(our_state)} parameters')
    return model


def test_pretrained():
    ref_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'ref',
                           'attention-learn-to-route-master', 'pretrained')

    for size in [20, 50, 100]:
        print(f'\n=== TSP-{size} ===')
        ref_path = os.path.join(ref_dir, f'tsp_{size}', 'epoch-99.pt')
        if not os.path.exists(ref_path):
            print(f'  Checkpoint not found: {ref_path}')
            continue

        config = Config(graph_size=size, no_cuda=True)
        model = load_ref_pretrained(ref_path, config)
        model.eval()
        model.set_decode_type('greedy')

        # Generate test instances with fixed seed
        torch.manual_seed(1234)
        test_input = torch.rand(1000, size, 2)

        # Greedy evaluation
        with torch.no_grad():
            costs = []
            bs = 200
            for i in range(0, 1000, bs):
                batch = test_input[i:i+bs]
                cost, _ = model(batch)
                costs.append(cost)
            costs = torch.cat(costs)

        avg_cost = costs.mean().item()
        print(f'  Greedy avg cost: {avg_cost:.4f}')
        print(f'  Greedy std: {costs.std().item():.4f}')


if __name__ == '__main__':
    test_pretrained()
