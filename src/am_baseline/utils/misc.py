import os
import json
import torch


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage, weights_only=False)


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    if 'data_distribution' not in args:
        args['data_distribution'] = None
    return args


def _remap_ref_keys(ref_state):
    """Remap parameter names from reference model to our restructured model.
    In our model, decoder projections and W_placeholder live under 'decoder.*'
    instead of at the top level."""
    decoder_keys = {
        'W_placeholder', 'project_node_embeddings.weight',
        'project_fixed_context.weight', 'project_step_context.weight',
        'project_out.weight',
    }
    remapped = {}
    for k, v in ref_state.items():
        if k in decoder_keys:
            remapped['decoder.' + k] = v
        else:
            remapped[k] = v
    return remapped


def load_model(path, epoch=None):
    """Load a trained AttentionModel from a checkpoint directory or file.
    Handles both our own checkpoints and reference pretrained checkpoints."""
    from am_baseline.model.attention_model import AttentionModel
    from am_baseline.config import Config

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        raise ValueError("{} is not a valid directory or file".format(path))

    args = load_args(os.path.join(path, 'args.json'))

    config = Config(
        graph_size=args['graph_size'],
        embedding_dim=args['embedding_dim'],
        hidden_dim=args['hidden_dim'],
        n_encode_layers=args['n_encode_layers'],
        tanh_clipping=args['tanh_clipping'],
        normalization=args['normalization'],
    )

    model = AttentionModel(config)

    load_data = torch_load_cpu(model_filename)
    ref_state = load_data.get('model', {})

    # Try direct load first; if keys don't match, remap from reference format
    our_keys = set(model.state_dict().keys())
    ref_keys = set(ref_state.keys())
    if not ref_keys.issubset(our_keys):
        ref_state = _remap_ref_keys(ref_state)

    model.load_state_dict({**model.state_dict(), **ref_state})

    model.eval()
    return model, args
