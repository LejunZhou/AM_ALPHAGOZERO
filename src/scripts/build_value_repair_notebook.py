"""Build the standalone Colab notebook from the checked-in experiment sources.

Run after editing the runner: python src/scripts/build_value_repair_notebook.py
No checkpoint, data, credentials, or compiled binary is embedded.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def build():
    package = ROOT / 'src/am_baseline'
    paths = [package / '__init__.py']
    for folder in ('model', 'problem', 'utils', 'search', 'experiments'):
        paths.extend(sorted((package / folder).rglob('*.py')))
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)), date_time=(2026, 9, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    data = blob.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    encoded = base64.b64encode(data).decode()
    cells = []
    def add(kind, source):
        source = textwrap.dedent(source).strip() + '\n'
        cell = dict(cell_type=kind, id=f'cell-{len(cells):02d}', metadata={}, source=source.splitlines(True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    def md(source): add('markdown', source)
    def code(source): add('code', source)

    md('''
    # Repair and isolate the TSP value evaluator

    **One fixed policy. Four matched head architectures. No self-play training.**

    This notebook tests whether missing state information explains weak value-based
    decisions. The encoder, policy decoder, and normalization buffers remain frozen.
    Every refit sees the same cached examples and raw greedy-completion targets,
    including the closing edge. Exact optimal completions are held-out evaluation
    oracles, never training labels.

    | Head | Inputs | Purpose |
    |---|---|---|
    | `original` | Decoder glimpse | Retrained architecture control |
    | `original_wide` | Decoder glimpse | Approximately matches repaired parameter count |
    | `repaired` | Glimpse, start/current embeddings and coordinates, unvisited mean embedding/coordinates, remaining fraction, start flag | Tests missing endpoint / set information |
    | `repaired_geo` | `repaired` inputs + 14 tour-geometry features (MST bound of the remaining set, nearest distances from current and start, spread), predicting a **residual over the MST bound** | Tests whether explicit geometry closes the gap to rollout |
    | `checkpoint_head` | Existing checkpoint head | Historical reference, not a matched refit |

    Reference evaluators in the search comparison: greedy rollout, `mst` (the bound
    alone), and `prior_only` (no leaf signal).

    **Two experiments, one switch (`EXPERIMENT` in section 3):**
    - `tsp20_f616` — the Stage 4 checkpoint on TSP-20. Fast pipeline check; every
      sibling decision has an exact oracle. TSP-20 is near-saturated, so treat it as
      a smoke test of the repair, not as the decision.
    - `tsp50_stage1` — the Stage 1 TSP-50 checkpoint (head trained in `bl` units).
      This is the decision run: the value head only matters where rollouts are
      expensive, and TSP-50 is where the current head was measured to be near random
      at early steps. Exact sibling oracles cover the last 20 cities; earlier states
      are compared on search cost and on rollout-target error.

    **Before running:**
    1. Upload this `.ipynb` to [Colab](https://colab.research.google.com/).
    2. Select a GPU runtime for feature generation and head fitting; CPU also works.
    3. Put the checkpoints on Drive:
       - `MyDrive/AM_AlphaGoZero/checkpoints/f616_400iter_step_decay/iter-361_accepted.pt`
         (from `outputs/tsp_20/f616_400iter_step_decay_20260507T101222_20260507T101229/`;
         raw-cost head, `original_value_norm="none"`)
       - `MyDrive/AM_AlphaGoZero/checkpoints/stage1_tsp50_with_value/epoch-99.pt` **and its
         `args.json`** (from `outputs/tsp_50/stage1_tsp50_with_value_20260424T032357/`;
         `bl`-normalized head, the runner checks `args.json` agrees).
    4. Run the cells in order. Results and resumable checkpoints are saved to Drive.

    The source snapshot is embedded below: **no GitHub push, repository clone, W&B
    account, solver license, or C++ compilation is needed.** This notebook contains
    no model weights. Only load checkpoints you trust.
    ''')
    md('''
    ## 1. Runtime and persistent workspace

    Installation keeps Colab's existing PyTorch/CUDA build. This notebook uses
    Python reference search, with CPU search as the default because individual
    tree steps are small. GPU training and CPU search are timed separately.
    ''')
    code('''
    import os, sys, subprocess, platform, importlib.util
    from pathlib import Path
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Skip installed packages; do not upgrade or replace torch.
    required = [p for p in ("numba", "pandas", "matplotlib") if importlib.util.find_spec(p) is None]
    if required:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *required], check=True)
    try:
        from google.colab import drive
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    if IN_COLAB:
        drive.mount('/content/drive')
        WORKSPACE = Path('/content/drive/MyDrive/AM_AlphaGoZero')
        RUNTIME_DIR = Path('/content')
    else:
        WORKSPACE = Path(os.environ.get('AM_VALUE_WORKSPACE', Path.cwd() / 'value_repair_workspace'))
        RUNTIME_DIR = WORKSPACE / 'runtime'
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    import torch
    print('Python:', platform.python_version(), '| torch:', torch.__version__)
    print('Training device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    print('Persistent workspace:', WORKSPACE)
    ''')
    md('''
    ## 2. Restore the bundled source

    The archive hash is checked before extraction. The runner also records hashes
    of the checkpoint and relevant source files in each run manifest. Restart the
    runtime before loading a different notebook source version.
    ''')
    payload_lines = '\n'.join('    ' + repr(encoded[i:i+100]) for i in range(0, len(encoded), 100))
    source = f'''import base64, hashlib, io, zipfile
SNAPSHOT_SHA256 = {digest!r}
SNAPSHOT_B64 = (\n{payload_lines}\n)
snapshot = base64.b64decode(SNAPSHOT_B64)
assert hashlib.sha256(snapshot).hexdigest() == SNAPSHOT_SHA256
BUNDLE_DIR = RUNTIME_DIR / ('value_repair_source_' + SNAPSHOT_SHA256[:12])
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
    for entry in archive.infolist():
        target = (BUNDLE_DIR / entry.filename).resolve()
        if not target.is_relative_to(BUNDLE_DIR.resolve()):
            raise RuntimeError('Unsafe archive path')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(entry))
if 'am_baseline' in sys.modules:
    loaded = Path(sys.modules['am_baseline'].__file__).resolve()
    if not loaded.is_relative_to(BUNDLE_DIR.resolve()):
        raise RuntimeError('Restart the runtime to use this notebook source snapshot.')
sys.path.insert(0, str(BUNDLE_DIR / 'src'))
from am_baseline.experiments.value_repair import ExperimentConfig, run_phase, save_json
print('Bundled source:', SNAPSHOT_SHA256)
'''
    code(source)
    md('''
    ## 3. Choose the bounded experiment

    Set `EXPERIMENT` (`tsp20_f616` or `tsp50_stage1`) and `PROFILE`. `main` runs
    three head seeds; `pilot` checks fitting and runtime with smaller data;
    `cpu_smoke` validates the entire pipeline on six-city instances and **cannot
    answer the research question**. The main preset is a bounded diagnostic, not a
    guarantee of sufficient optimization or statistical power: inspect validation
    curves before interpreting a null result. Rough `main` runtimes: TSP-20 about
    1 h (mostly CPU search); TSP-50 about 3 h (search dominates; the feature cache
    is roughly 1 GB on Drive).

    Separate seeded graphs are used for training, validation, sibling probing,
    timing calibration, and final search. Training/validation use both greedy and
    temperature-3 sampled prefixes (first half of the tour) and enumerate every
    legal child of the selected parent steps. Test probing uses every nontrivial
    step. Loss is weighted equally across parent decisions, rather than giving
    early states more weight just because they have more children.

    Changing protocol settings requires a **new run name**. Re-running unchanged
    cells resumes completed data shards and epoch checkpoints. An interrupted
    epoch is replayed from its beginning with the same minibatch order.
    ''')
    code('''
    from dataclasses import asdict
    import json

    EXPERIMENT = "tsp20_f616"  # "tsp20_f616" (pipeline check) or "tsp50_stage1" (the decision run)
    PROFILE = "main"           # "main", "pilot", or "cpu_smoke"

    experiments = {
        'tsp20_f616': dict(
            checkpoint=WORKSPACE / 'checkpoints/f616_400iter_step_decay/iter-361_accepted.pt',
            checkpoint_key='best_model', original_value_norm='none', graph_size=20,
            main=dict(train_instances=1024, val_instances=128, probe_instances=100,
                      calibration_instances=8, search_instances=100, epochs=30,
                      head_seeds=(0, 1, 2)),
            pilot=dict(train_instances=128, val_instances=32, probe_instances=16,
                       calibration_instances=2, search_instances=16, epochs=10,
                       head_seeds=(0, 1, 2), search_K=20)),
        'tsp50_stage1': dict(
            checkpoint=WORKSPACE / 'checkpoints/stage1_tsp50_with_value/epoch-99.pt',
            checkpoint_key='model', original_value_norm='bl', graph_size=50,
            main=dict(train_instances=512, val_instances=64, probe_instances=64,
                      calibration_instances=4, search_instances=64, epochs=30,
                      head_seeds=(0, 1, 2), feature_batch_size=16),
            pilot=dict(train_instances=64, val_instances=16, probe_instances=8,
                       calibration_instances=2, search_instances=8, epochs=10,
                       head_seeds=(0,), search_K=20, feature_batch_size=16)),
    }
    cpu_smoke = dict(graph_size=6, train_instances=4, val_instances=2, probe_instances=2,
                     calibration_instances=1, search_instances=2, epochs=2, head_seeds=(0,),
                     search_K=2, batch_size=32, feature_batch_size=2, timing_repeats=1, device='cpu')
    assert EXPERIMENT in experiments and PROFILE in ('main', 'pilot', 'cpu_smoke')
    spec = experiments[EXPERIMENT]
    preset = dict(cpu_smoke if PROFILE == 'cpu_smoke' else spec[PROFILE])
    graph_size = preset.pop('graph_size', spec['graph_size'])
    RUN_NAME = f"value_repair_{EXPERIMENT}_v2_{PROFILE}"
    OUTPUT_DIR = WORKSPACE / 'outputs/value_repair' / RUN_NAME
    CHECKPOINT = spec['checkpoint']
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f'Upload the {EXPERIMENT} checkpoint to: {CHECKPOINT}')
    if spec['original_value_norm'] == 'bl' and not (CHECKPOINT.parent / 'args.json').is_file():
        raise FileNotFoundError(f'Upload args.json next to the checkpoint: {CHECKPOINT.parent}')
    cfg = ExperimentConfig(
        checkpoint=str(CHECKPOINT), output_dir=str(OUTPUT_DIR),
        original_value_norm=spec['original_value_norm'], checkpoint_key=spec['checkpoint_key'],
        graph_size=graph_size, search_device='cpu', **preset)
    cfg.validate()
    print(json.dumps(asdict(cfg), indent=2))
    ''')
    md('''
    ## 4. Correctness checks before fitting

    Reproduce the original input alias and verify that repaired inputs distinguish
    those states. Check closing-edge target accounting and exact DP against brute
    force. Compare the evaluator adapter with the unmodified Python MCTS for both
    the existing value head and rollout leaves. Every stage checks that all frozen
    policy parameters and buffers remain unchanged.
    ''')
    code('''
    run_phase(cfg, 'check')
    save_json(Path(cfg.output_dir) / 'config.json', asdict(cfg))
    save_json(Path(cfg.output_dir) / 'notebook_snapshot.json', {'sha256': SNAPSHOT_SHA256})
    ''')
    md('''
    ## 5. Generate targets and train the matched heads

    Frozen greedy completion is computed once per cached child, not inside each
    SGD step. All heads use the same MSE, Adam settings, clipping, epoch budget,
    train-only feature standardization, and minibatch permutations. Head weights
    are selected by validation rollout-target MSE, never by test ranking/search.
    First-time data generation can dominate runtime; progress is printed per shard.
    Cache files can occupy hundreds of MB for the main preset.
    ''')
    code("run_phase(cfg, 'train')")
    code('''
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    history = pd.read_csv(Path(cfg.output_dir) / 'training_history.csv')
    display(history.sort_values('epoch').groupby(['variant', 'seed']).tail(1))
    fig, ax = plt.subplots(figsize=(8, 4))
    for (variant, seed), part in history.groupby(['variant', 'seed']):
        ax.plot(part.epoch, part.val_weighted_mse, label=f'{variant}, seed {seed}', alpha=.8)
    ax.set(xlabel='Epoch', ylabel='Validation MSE (raw remaining cost)', yscale='log')
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(Path(cfg.output_dir) / 'validation_curves.png', dpi=180)
    plt.show()
    ''')
    md('''
    ## 6. Held-out sibling decisions

    TSP-20 is evaluated with exact Held–Karp completion costs. All tied optimal
    children count as correct (tolerance 1e-6). For each decision we report regret,
    optimal-child picks, harmful overrides, fixes conditional on the prior being
    wrong, and Spearman ranking. These use **optimal** completions; the separate
    rollout-target error file evaluates prediction of the frozen policy.

    First use compiles the DP kernel. At 19 to 20 remaining cities its table uses
    about 80 to 170 MB; tables are processed serially. On TSP-50, states with more
    than `oracle_max_remaining` (20) cities left have **no exact ranking result**;
    `exact_states` records the covered denominator and the `21+` bucket stays empty.
    Early TSP-50 decisions are judged through the search comparison in section 7
    and the rollout-target errors, not through exact regret.
    ''')
    code("run_phase(cfg, 'probe')")
    code('''
    sibling = pd.read_csv(Path(cfg.output_dir) / 'sibling_summary.csv')
    display(sibling[sibling.bucket.eq('all')][[
        'method', 'source', 'states', 'exact_states', 'optimal', 'regret',
        'harmful_override', 'prior_wrong_states', 'fix_rate_when_prior_wrong']])
    refits = sibling[sibling.method.str.contains(r'_s\\d+$', regex=True)].copy()
    refits['variant'] = refits.method.str.replace(r'_s\\d+$', '', regex=True)
    display(refits.groupby(['variant', 'source', 'bucket'])[['regret', 'optimal', 'harmful_override']]
            .agg(['mean', 'std', 'count']))  # sample SD across every configured seed
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, source in zip(axes, sorted(refits.source.unique())):
        for variant, part in refits[(refits.source == source) & (refits.bucket != 'all')].groupby('variant'):
            means = part.groupby('bucket').regret.mean()
            ax.plot(means.index, means.values, marker='o', label=variant)
        for reference, style in [('prior', '--'), ('rollout', ':')]:
            part = sibling[(sibling.method == reference) & (sibling.source == source) & (sibling.bucket != 'all')]
            ax.plot(part.bucket, part.regret, style, label=reference, color='black', alpha=.65)
        ax.set(title=source, xlabel='Unvisited cities at parent', ylabel='Exact regret / decision')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(cfg.output_dir) / 'sibling_regret.png', dpi=180)
    plt.show()
    ''')
    md('''
    ## 7. Does the evaluator improve search economically?

    The same frozen prior, PUCT constant, noise-free action selection, terminal
    costs, and subtree reuse are used for all evaluators. `prior_only` assigns
    a constant total cost to nonterminal leaves (no value signal), while still
    backing up exact terminal costs. `mst` uses a geometric lower bound.

    First compare equal K. Then use **separate calibration instances** to choose
    K that approximates rollout search's runtime at the reference K. This is an
    approximate time comparison, not a strict deadline: inspect actual held-out
    seconds and `time_ratio_to_target`. Search timing includes feature computation,
    encoding, and a greedy normalizer pass, but excludes loading and input transfer.
    These CPU/Python timings do not establish C++ or batched GPU efficiency.

    Stop here if training is plainly unfinished. Runtime is measured rather than
    promised; reduce the preset for an initial functionality check. Completed
    method/K outputs are reused on resume. For final timing claims, use one fresh
    uninterrupted session; cross-session timings may differ.
    ''')
    code("run_phase(cfg, 'search')")
    code('''
    search = pd.read_csv(Path(cfg.output_dir) / 'search_summary.csv')
    display(search)
    learned = search[search.method.str.contains(r'_s\\d+$', regex=True)].copy()
    learned['variant'] = learned.method.str.replace(r'_s\\d+$', '', regex=True)
    display(learned.groupby(['mode', 'variant'])[['mean_cost', 'seconds_per_instance']]
            .agg(['mean', 'std', 'count']))
    plotted = search.copy()
    plotted['family'] = plotted.method.str.replace(r'_s\\d+$', '', regex=True)
    families = sorted(plotted.family.unique())
    colors = dict(zip(families, plt.get_cmap('tab10').colors))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mode in zip(axes, ['equal_K', 'calibrated_time']):
        part = plotted[(plotted['mode'] == mode) | (plotted['mode'] == 'greedy')]
        for family, group in part.groupby('family'):
            ax.errorbar(group.seconds_per_instance.mean(), group.mean_cost.mean(),
                        xerr=group.seconds_per_instance.std() if len(group) > 1 else 0,
                        yerr=group.mean_cost.std() if len(group) > 1 else 0,
                        fmt='o', capsize=3, color=colors[family], label=family)
        ax.set(title=mode, xlabel='Measured seconds / instance', ylabel='Mean tour cost (lower is better)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9)
    fig.suptitle('Head seeds: mean ± sample SD; fixed reference methods: single point', fontsize=10)
    fig.tight_layout(rect=(0, .12, 1, .94))
    fig.savefig(Path(cfg.output_dir) / 'search_cost_time.png', dpi=180)
    plt.show()
    ''')
    md('''
    ## 8. Paired evidence and the next decision

    Negative `repaired minus control` and `repaired_geo minus control` differences
    favor the refit named first. Intervals cluster sibling decisions by graph and
    are conditional on each training seed; inspect all seeds, not just seed 0. The
    original-wide contrast separates information from capacity; the
    `repaired_geo minus repaired` contrast shows what explicit geometry adds; the
    `minus mst` search contrast shows what the learned residual adds over the bound
    alone. A one-city input repair alone does not prove that early search improved.

    Continue to a short policy-distillation test only if repaired evaluations
    improve held-out decisions **and** offer a useful measured search quality/time
    tradeoff. If curves are still improving at the budget limit, the outcome is
    inconclusive. A negative result after adequate fitting supports a bounded TSP
    stop, not a theorem that value learning is impossible on deterministic TSP.
    ''')
    code('''
    run_phase(cfg, 'report')
    contrasts = pd.read_csv(Path(cfg.output_dir) / 'paired_contrasts.csv')
    display(contrasts[contrasts.comparison.str.contains('original_wide')])
    display(contrasts[contrasts.comparison.str.contains(r'repaired_geo_s\d+ minus (repaired_s|mst|rollout)', regex=True)])
    print((Path(cfg.output_dir) / 'INTERPRETATION.md').read_text())
    print('All artifacts:', cfg.output_dir)
    ''')
    md('''
    ## 9. Export a compact results bundle

    The full cache and resumable heads stay on Drive. This ZIP contains configuration,
    provenance, all raw child scores/decisions, per-instance search results/tours,
    contrasts, training curves, and plots. It excludes the large feature cache and
    checkpoint weights. Download this bundle when you want to review results locally.
    ''')
    code('''
    from zipfile import ZipFile, ZIP_DEFLATED
    out = Path(cfg.output_dir)
    result_zip = out.parent / (out.name + '_results.zip')
    with ZipFile(result_zip, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            rel = path.relative_to(out)
            if path.is_file() and rel.parts[0] not in {'data', 'heads'} and path.suffix != '.pt':
                archive.write(path, str(rel))
    print('Results ZIP:', result_zip)
    # Optional Colab download:
    # from google.colab import files
    # files.download(str(result_zip))
    ''')
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.12'),
        colab=dict(name='colab_value_evaluator_repair.ipynb', provenance=[]),
        accelerator='GPU', value_repair_source_sha256=digest))
    target = ROOT / 'notebooks/colab_value_evaluator_repair.ipynb'
    target.write_text(json.dumps(notebook, indent=1) + '\n')
    print(f'Wrote {target} ({target.stat().st_size:,} bytes; {len(paths)} source files)')
    return target


if __name__ == '__main__':
    build()
