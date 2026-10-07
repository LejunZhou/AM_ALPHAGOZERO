"""Build the standalone Colab notebook for Stage 5 §I Step 2 (student isolation).

Run after editing the runner or the C++ search:
    python src/scripts/build_student_isolation_notebook.py
No checkpoint, data, credentials or compiled binary is embedded; the C++ search
is compiled inside the Colab runtime from the embedded sources.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'src/am_baseline'


def snapshot_paths():
    paths = [PACKAGE / '__init__.py', PACKAGE / 'training/__init__.py',
             PACKAGE / 'training/trainer.py', PACKAGE / 'experiments/__init__.py',
             PACKAGE / 'experiments/student_isolation.py']
    for folder in ('model', 'problem', 'utils', 'baseline'):
        paths.extend(sorted((PACKAGE / folder).glob('*.py')))
    paths.extend(sorted((PACKAGE / 'search').glob('*.py')))
    for name in ('__init__.py', 'solver.py', 'mcts.cpp', 'mcts.hpp', 'bindings.cpp'):
        paths.append(PACKAGE / 'search/mcts_cpp' / name)
    return sorted(set(paths))


def build():
    paths = snapshot_paths()
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)), date_time=(2026, 10, 3, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    data = blob.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    encoded = base64.b64encode(data).decode()
    cells = []

    def add(kind, source):
        source = textwrap.dedent(source).strip() + '\n'
        cell = dict(cell_type=kind, id=f'cell-{len(cells):02d}', metadata={},
                    source=source.splitlines(True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    def md(source):
        add('markdown', source)

    def code(source):
        add('code', source)

    md('''
    # Step 2 — Can the student absorb its teacher's improvements? (TSP-50)

    Stage 5 §I Step 2. The from-scratch AlphaZero loop stalls once the policy is good
    and the search finds only small improvements. This notebook recreates that endgame
    in one controlled round, starting from the REINFORCE-trained Stage 1 TSP-50 model.

    1. **Teacher (frozen).** Stage 1 policy + batched C++ MCTS (greedy-rollout leaves,
       K=40, c_puct 0.05, tree reuse, no noise). It labels 32,768 training graphs once and
       beats Stage 1 greedy by about 0.06 tour length.
    2. **Three students on identical teacher data**, each starting from Stage 1:

       | Target | What the student imitates |
       |---|---|
       | `visits` | AlphaZero visit counts N(s,a)/ΣN (the current loop's target) |
       | `gumbel_q` | Completed-Q improved policy (Gumbel AlphaZero); unvisited moves valued at the visited children's mean Q |
       | `best_tour` | The better of the MCTS tour and the Stage 1 greedy tour |

       Each target trains with three learning rates; validation picks the checkpoint
       and the learning rate. The test set never selects anything.
    3. **Control: more REINFORCE at the same wall time.** Stage 1 training continues
       from the same checkpoint, with its optimizer and rollout baseline, for exactly the
       teacher's time plus one target's training time.

    **Decision (screen):** a target passes if its greedy test cost beats the control
    with a paired 95% interval below zero **and** it keeps at least a quarter of the
    teacher's gain over Stage 1. `CONTINUE` names the target for the `confirm` profile
    (seeds 1 and 2: new training graphs). `STOP` means one round of search distillation
    does not beat more REINFORCE at matched time.

    **Before running**
    1. Upload this `.ipynb` to [Colab](https://colab.research.google.com/) and pick a
       GPU runtime (L4 recommended; T4 works, slower).
    2. The Step 1 checkpoint must be on Drive (already there if you ran Step 1):
       `MyDrive/AM_AlphaGoZero/checkpoints/stage1_tsp50_with_value/epoch-99.pt` **and**
       `args.json` next to it.
    3. Run all cells. Everything resumes after a disconnect: rerun from the top.

    Expected `main` time on an L4: roughly 2 to 3 hours (teacher ~30 min, students
    ~40 min, control the same wall time as teacher + one target, evaluation a few
    minutes). The first teacher shard prints the measured speed and an ETA. No GitHub
    push, W&B account or solver license is needed; the notebook contains no weights.
    ''')
    md('''
    ## 1. Runtime and persistent workspace
    ''')
    code('''
    import os, sys, subprocess, platform, importlib.util, json
    from pathlib import Path
    missing = [p for p in ('pybind11', 'pandas', 'matplotlib', 'tqdm', 'scipy')
               if importlib.util.find_spec(p) is None]
    if missing:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *missing], check=True)
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
        WORKSPACE = Path(os.environ.get('AM_STEP2_WORKSPACE', Path.cwd() / 'step2_workspace'))
        RUNTIME_DIR = WORKSPACE / 'runtime'
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    import torch
    print('Python', platform.python_version(), '| torch', torch.__version__, '| CPUs', os.cpu_count())
    print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU only)')
    print('Workspace:', WORKSPACE)
    ''')
    md('''
    ## 2. Restore the bundled source and compile the C++ search

    The archive hash is checked before extraction. The C++ MCTS (with the new opt-in
    root-Q export) is compiled in this runtime, about one minute. Restart the runtime
    before loading a different notebook version.
    ''')
    payload = '\n'.join('    ' + repr(encoded[i:i + 100]) for i in range(0, len(encoded), 100))
    code(f'''import base64, hashlib, io, zipfile, sysconfig
SNAPSHOT_SHA256 = {digest!r}
SNAPSHOT_B64 = (
{payload}
)
snapshot = base64.b64decode(SNAPSHOT_B64)
assert hashlib.sha256(snapshot).hexdigest() == SNAPSHOT_SHA256, 'corrupted notebook snapshot'
BUNDLE_DIR = RUNTIME_DIR / ('student_isolation_source_' + SNAPSHOT_SHA256[:12])
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
import pybind11
cpp_dir = BUNDLE_DIR / 'src/am_baseline/search/mcts_cpp'
extension = cpp_dir / ('_mcts_cpp' + sysconfig.get_config_var('EXT_SUFFIX'))
if not extension.exists():
    command = ['c++', '-O3', '-std=c++17', '-shared', '-fPIC',
               '-I' + pybind11.get_include(), '-I' + sysconfig.get_paths()['include'],
               str(cpp_dir / 'bindings.cpp'), str(cpp_dir / 'mcts.cpp'), '-o', str(extension)]
    if sys.platform == 'darwin':
        command[1:1] = ['-undefined', 'dynamic_lookup']
    print('Compiling the C++ search ...', flush=True)
    subprocess.run(command, check=True)
sys.path.insert(0, str(BUNDLE_DIR / 'src'))
from am_baseline.search.mcts_cpp import solver as cpp_solver
assert cpp_solver.HAVE_CPP_MCTS, cpp_solver._IMPORT_ERROR
from am_baseline.experiments.student_isolation import (
    StudentConfig, profile_overrides, run_phase, save_json)
print('Bundled source', SNAPSHOT_SHA256[:12], '| C++ search ready:', extension.name)
''')
    md('''
    ## 3. Choose the profile

    - `main` — the screen: seed 0, all three targets, the control. **Run this.**
    - `confirm` — only after the screen says `CONTINUE`: seeds 1 and 2 (new training
      graphs, same validation/test graphs) for the chosen target and `visits`. Reuses the
      same output folder; the report then covers all seeds.
    - `pilot` — a 15-minute real run on 512 graphs to check the build and measure speed.
    - `cpu_smoke` — a 10-city end-to-end check. **Cannot answer anything.**

    Changing protocol settings needs a new `RUN_NAME`; the run manifest refuses to mix
    protocols in one folder.
    ''')
    code('''
    from dataclasses import asdict
    PROFILE = os.environ.get('AM_STEP2_PROFILE', 'main')   # main | confirm | pilot | cpu_smoke
    CHECKPOINT = WORKSPACE / 'checkpoints/stage1_tsp50_with_value/epoch-99.pt'
    if not CHECKPOINT.is_file() or not (CHECKPOINT.parent / 'args.json').is_file():
        raise FileNotFoundError(f'Upload epoch-99.pt and args.json to {CHECKPOINT.parent}')
    RUN_NAME = 'student_isolation_tsp50_v1'
    if PROFILE not in ('main', 'confirm'):
        RUN_NAME += '_' + PROFILE
    OUTPUT_DIR = WORKSPACE / 'outputs/student_isolation' / RUN_NAME
    overrides = profile_overrides(PROFILE)
    if PROFILE == 'confirm':
        decision_file = OUTPUT_DIR / 'results/decision.json'
        if not decision_file.exists():
            raise FileNotFoundError('Run the main profile through section 8 first.')
        screen = json.loads(decision_file.read_text())
        if screen['screen_outcome'] != 'CONTINUE':
            raise RuntimeError(f"The screen says {screen['screen_outcome']}; confirm is not needed.")
        overrides['targets'] = tuple(sorted({screen['continue_with'], 'visits'}))
    cfg = StudentConfig(checkpoint=str(CHECKPOINT), output_dir=str(OUTPUT_DIR), **overrides)
    cfg.validate()
    print('PROFILE =', PROFILE, '| output:', OUTPUT_DIR)
    print(json.dumps(asdict(cfg), indent=1))
    ''')
    md('''
    ## 4. Correctness checks

    On a CPU copy: the C++ batched teacher matches the Python reference search (tours,
    visit counts, root Q values). Teacher forcing reproduces the policy's own greedy
    decisions. All three targets are proper distributions on legal moves, and the
    best-tour choice picks the cheaper tour. The 10K canonical set reproduces the draw
    behind Stage 1's 5.7999.
    ''')
    code('''
    run_phase(cfg, 'check')
    save_json(OUTPUT_DIR / 'config.json', asdict(cfg))
    save_json(OUTPUT_DIR / 'notebook_snapshot.json', {'sha256': SNAPSHOT_SHA256, 'profile': PROFILE})
    ''')
    md('''
    ## 5. Teacher labels

    Labels the 2,048 test graphs (for the retention denominator) and this profile's
    training graphs, in shards saved to Drive. Completed shards are skipped on rerun.
    ''')
    code('''
    run_phase(cfg, 'teacher')
    for summary in sorted((OUTPUT_DIR / 'teacher').glob('*/summary.json')):
        s = json.loads(summary.read_text())
        print(f"{summary.parent.name}: {s['graphs']} graphs, greedy {s['greedy_mean']:.4f}, "
              f"teacher {s['mcts_mean']:.4f} (gain {s['greedy_mean'] - s['mcts_mean']:+.4f}), "
              f"teacher better on {s['mcts_better_fraction']:.0%} of graphs, "
              f"{s['wall_seconds'] / 60:.1f} min")
    ''')
    md('''
    ## 6. Students

    For every target and learning rate: start from Stage 1, train on the teacher data,
    check validation greedy cost every 32 steps, keep the best checkpoint. Interrupted
    runs resume from their last finished epoch.
    ''')
    code('''
    run_phase(cfg, 'students')
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    rows = []
    for done in sorted((OUTPUT_DIR / 'students').glob('s*/*/lr*/done.json')):
        rows.append(json.loads(done.read_text()))
    display(pd.DataFrame(rows)[['seed', 'target', 'lr', 'initial_val', 'best_val', 'best_step',
                                'steps', 'wall_seconds']])
    fig, ax = plt.subplots(figsize=(9, 4))
    for history in sorted((OUTPUT_DIR / 'students').glob('s*/*/lr*/history.csv')):
        h = pd.read_csv(history)
        ax.plot(h.step, h.val_cost, label=f'{history.parts[-3]} {history.parts[-2]} {history.parts[-4]}')
    ax.set(xlabel='Optimizer step', ylabel='Validation greedy cost')
    ax.legend(fontsize=7, ncol=3)
    plt.show()
    ''')
    md('''
    ## 7. Control: REINFORCE at matched wall time

    Budget = the teacher's time on the training graphs + one target's full training
    time (all learning rates). The control resumes from its last finished epoch; time
    spent in an interrupted epoch is not counted.
    ''')
    code('''
    run_phase(cfg, 'control')
    for done in sorted((OUTPUT_DIR / 'control').glob('s*/done.json')):
        d = json.loads(done.read_text())
        print(f"control {done.parent.name}: val {d['initial_val']:.5f} -> {d['best_val']:.5f} "
              f"after {d['instances']:,} graphs in {d['elapsed_seconds'] / 60:.1f} min "
              f"(budget {d['budget_seconds'] / 60:.1f} min)")
    ''')
    md('''
    ## 8. Evaluate and decide

    Greedy cost of Stage 1, every selected student and the control on the 2,048 test
    graphs (and the canonical 10K set), the absorption diagnostic along the teacher's
    test trajectories, paired contrasts, retention with a bootstrap interval, and the
    screen decision.
    ''')
    code('''
    run_phase(cfg, 'evaluate')
    decision = run_phase(cfg, 'report')
    policies = pd.read_csv(OUTPUT_DIR / 'results/policies.csv')
    display(policies)
    display(pd.read_csv(OUTPUT_DIR / 'results/students.csv'))
    print(f"Teacher gain over Stage 1: {decision['teacher_gain']:+.4f} "
          f"(SE {decision['teacher_gain_se']:.4f})")
    print('SCREEN OUTCOME:', decision['screen_outcome'], '| continue with:', decision['continue_with'])
    if decision['confirm']:
        print('Confirm seeds:', json.dumps(decision['confirm'], indent=1))
    from IPython.display import Image
    display(Image(str(OUTPUT_DIR / 'results/test_deltas.png')))
    display(Image(str(OUTPUT_DIR / 'results/training_curves.png')))
    print((OUTPUT_DIR / 'INTERPRETATION.md').read_text())
    ''')
    md('''
    ## 9. Export a compact results bundle

    Configuration, provenance, the test-set teacher labels, every training curve,
    per-graph test costs, tables, decision and plots. Excludes model weights and the
    training-graph teacher shards. Share the ZIP (or the Drive folder) for review.
    ''')
    code('''
    from zipfile import ZipFile, ZIP_DEFLATED
    result_zip = OUTPUT_DIR.parent / (OUTPUT_DIR.name + '_results.zip')
    with ZipFile(result_zip, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT_DIR.rglob('*')):
            rel = path.relative_to(OUTPUT_DIR)
            if not path.is_file() or path.suffix == '.pt':
                continue
            if rel.parts[0] == 'teacher' and rel.parts[1].startswith('train') and path.suffix == '.npz':
                continue
            archive.write(path, str(rel))
    print('Results ZIP:', result_zip, f'({result_zip.stat().st_size / 1e6:.1f} MB)')
    ''')
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.12'),
        colab=dict(name='colab_student_isolation.ipynb', provenance=[]),
        accelerator='GPU', student_isolation_source_sha256=digest))
    target = ROOT / 'notebooks/colab_student_isolation.ipynb'
    target.write_text(json.dumps(notebook, indent=1) + '\n')
    print(f'Wrote {target} ({target.stat().st_size:,} bytes; {len(paths)} source files; '
          f'snapshot {digest[:12]})')
    return target


if __name__ == '__main__':
    build()
