"""Watch a Step 2 (student isolation) Colab run through a shared Google Drive folder.

The folder must be shared as "Anyone with the link: Viewer". Pass either the
`outputs/student_isolation` folder or the run folder itself. Uses only the public
folder listing (no login), polls until `results/decision.json` appears, then
downloads the small result files and prints the screen decision.

  python src/scripts/watch_student_isolation.py --folder <drive folder URL or id> \
      --interval 600 --timeout_hours 6 --out <local dir>
  python src/scripts/watch_student_isolation.py --folder <...> --once   # one status line

Exit codes: 0 finished, 2 timed out, 3 folder not readable.
"""
import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LIST_URL = 'https://drive.google.com/embeddedfolderview?id={}'
FILE_URL = 'https://drive.google.com/uc?export=download&id={}'
ENTRY = re.compile(
    r'<div class="flip-entry" id="entry-(?P<id>[^"]+)".*?'
    r'href="https://drive\.google\.com/(?P<kind>drive/folders|file/d)/[^"]*".*?'
    r'<div class="flip-entry-title">(?P<title>[^<]*)</div>.*?'
    r'<div class="flip-entry-last-modified"><div>(?P<modified>[^<]*)</div>', re.S)
RUN_PREFIX = 'student_isolation_tsp50_v1'
TARGETS = ('visits', 'gumbel_q', 'best_tour')


def folder_id(text):
    match = re.search(r'folders/([A-Za-z0-9_-]{10,})', text) or re.search(r'id=([A-Za-z0-9_-]{10,})', text)
    return match.group(1) if match else text.strip()


def fetch(url, timeout=60):
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def list_folder(fid):
    page = fetch(LIST_URL.format(fid)).decode('utf-8', 'replace')
    if 'flip-entries' not in page and 'flip-entry' not in page:
        raise PermissionError('folder listing not readable; share it as "Anyone with the link"')
    return [dict(id=m['id'], folder=m['kind'] == 'drive/folders', title=html.unescape(m['title']),
                 modified=html.unescape(m['modified'])) for m in ENTRY.finditer(page)]


def crawl(fid, prefix='', depth=6):
    """{relative path: entry} for the whole tree; folders end with '/'."""
    out = {}
    for entry in list_folder(fid):
        path = f'{prefix}{entry["title"]}'
        out[path + ('/' if entry['folder'] else '')] = entry
        if entry['folder'] and depth > 0:
            out.update(crawl(entry['id'], path + '/', depth - 1))
    return out


def run_root(paths):
    """Locate the run folder prefix inside the crawl (run folder itself or its parent)."""
    for path in sorted(paths):
        if path.rstrip('/').endswith(RUN_PREFIX) and path.endswith('/'):
            return path, any(p.endswith(RUN_PREFIX + '_results.zip') for p in paths)
    return '', any(p.endswith('_results.zip') for p in paths)


def status(paths, seed=0):
    root, zipped = run_root(paths)
    rel = sorted(p[len(root):] for p in paths if p.startswith(root))
    has = lambda name: name in rel
    shards = lambda split: sum(1 for p in rel if p.startswith(f'teacher/{split}/shard_') and p.endswith('.npz'))
    done = [p for p in rel if p.startswith(f'students/s{seed}/') and p.endswith('/done.json')]
    running = [p for p in rel if p.startswith(f'students/s{seed}/') and p.endswith('/resume.pt')]
    state = dict(
        started=has('manifest.json'), checked=has('invariants.json'),
        test_shards=shards('test'), test_done=has('teacher/test/summary.json'),
        train_shards=shards(f'train_s{seed}'), train_done=has(f'teacher/train_s{seed}/summary.json'),
        students_done=len(done), students_running=[p.split('/')[2] + '/' + p.split('/')[3] for p in running],
        control_running=has(f'control/s{seed}/resume.pt'), control_done=has(f'control/s{seed}/done.json'),
        evaluated=has('results/policies.csv'), decided=has('results/decision.json'), exported=zipped)
    if state['decided']:
        phase = 'DONE: decision written' + (' and results ZIP exported' if zipped else '')
    elif state['evaluated']:
        phase = 'report'
    elif state['control_done']:
        phase = 'evaluation'
    elif state['students_done'] >= 9 or state['control_running']:
        phase = 'REINFORCE control'
    elif state['train_done']:
        phase = f'students ({state["students_done"]}/9 done)'
    elif state['test_done'] or state['train_shards']:
        phase = f'teacher (test {state["test_shards"]}/2, train {state["train_shards"]}/32 shards)'
    elif state['checked']:
        phase = 'teacher (starting)'
    elif state['started']:
        phase = 'correctness checks'
    else:
        phase = 'not started (no manifest yet)'
    return root, phase, state


def download_results(paths, root, out):
    out.mkdir(parents=True, exist_ok=True)
    wanted = [p for p in paths if p.startswith(root) and not paths[p]['folder'] and (
        p.endswith('.json') or p.endswith('.csv') or p.endswith('.png') or p.endswith('.md'))
        and '/shard_' not in p]
    zip_path = next((p for p in paths if p.endswith(RUN_PREFIX + '_results.zip')), None)
    if zip_path:
        wanted.append(zip_path)
    for p in wanted:
        target = out / p[len(root):] if p.startswith(root) else out / Path(p).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fetch(FILE_URL.format(paths[p]['id']), timeout=300))
    return wanted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folder', required=True)
    ap.add_argument('--interval', type=float, default=600)
    ap.add_argument('--timeout_hours', type=float, default=6)
    ap.add_argument('--out', default='student_isolation_results')
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    fid = folder_id(args.folder)
    deadline = time.time() + args.timeout_hours * 3600
    last = None
    while True:
        try:
            paths = crawl(fid)
        except PermissionError as exc:
            print(time.strftime('%H:%M:%S'), exc, flush=True)
            return 3
        except (urllib.error.URLError, TimeoutError) as exc:
            print(time.strftime('%H:%M:%S'), 'network error, retrying:', exc, flush=True)
            paths = None
        if paths is not None:
            root, phase, state = status(paths)
            line = f'{phase} | students done {state["students_done"]}/9' \
                   f'{" running " + ",".join(state["students_running"]) if state["students_running"] else ""}' \
                   f' | control {"done" if state["control_done"] else "running" if state["control_running"] else "-"}'
            if line != last or args.once:
                print(time.strftime('%Y-%m-%d %H:%M:%S'), line, flush=True)
                last = line
            if state['decided']:
                files = download_results(paths, root, Path(args.out))
                print(f'downloaded {len(files)} files to {args.out}', flush=True)
                decision = json.loads((Path(args.out) / 'results/decision.json').read_text())
                print('SCREEN OUTCOME:', decision['screen_outcome'], '| continue with:',
                      decision['continue_with'], flush=True)
                return 0
            if args.once:
                return 0
        if time.time() > deadline:
            print('timed out; last status:', last, flush=True)
            return 2
        time.sleep(args.interval)


if __name__ == '__main__':
    sys.exit(main())
