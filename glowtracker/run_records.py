"""Where GlowTracker keeps experiments, recordings and scripts, and what each recording records
about itself. No GUI code, so it can be tested on its own.

Layout (the data root is a setting, default Documents/GlowTracker):

    <data root>/experiments/<YYYY-MM-DD>_<name>/            one experiment ("New experiment")
        <YYYYMMDD_HHMMSS>_<label>/                          one recording (each Record)
            *-coords.txt, images, plugin_log_*.jsonl          as before
            run.json                                          what, when, how; see start_run()
            settings.ini                                      every setting when Record was pressed
            protocol/<script or plugin>                       frozen copy of what ran
        runs.csv                                            one row per recording, for analysis
    <data root>/scripts/<readable-name>.py | .txt           scripts the assistant wrote

Nothing is ever overwritten: names get -2, -3, ... when taken.
"""
from __future__ import annotations

import configparser
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess

RUN_FILE = 'run.json'
RUNS_TABLE = 'runs.csv'
SETTINGS_FILE = 'settings.ini'
PROTOCOL_DIR = 'protocol'
SCHEMA = 1


# --- folders and names ---------------------------------------------------------------------------

def default_data_root() -> str:
    documents = os.path.join(os.path.expanduser('~'), 'Documents')
    base = documents if os.path.isdir(documents) else os.path.expanduser('~')
    return os.path.join(base, 'GlowTracker')


def data_root(configured: str = '') -> str:
    return os.path.abspath(os.path.expanduser(configured.strip())) if configured and configured.strip() \
        else default_data_root()


def experiments_dir(root: str) -> str:
    return os.path.join(root, 'experiments')


def scripts_dir(root: str) -> str:
    return os.path.join(root, 'scripts')


def slug(text: str, fallback: str = 'untitled', limit: int = 60) -> str:
    """A readable, file-system-safe name: lowercase words joined by hyphens."""
    text = os.path.splitext(os.path.basename(str(text or '')))[0] if text else ''
    words = re.sub(r'[^0-9A-Za-z]+', ' ', text).strip().lower().split()
    name = '-'.join(words)[:limit].strip('-')
    return name or fallback


def unique_path(folder: str, stem: str, ext: str = '') -> str:
    """folder/stem+ext, or stem-2, stem-3, ... when taken. Does not create anything."""
    candidate = os.path.join(folder, f'{stem}{ext}')
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f'{stem}-{n}{ext}')
        n += 1
    return candidate


def make_unique_dir(folder: str, name: str) -> str:
    """Create folder/name (or name-2, ...) and return it; safe if two are made in the same second."""
    os.makedirs(folder, exist_ok=True)
    n = 1
    while True:
        path = os.path.join(folder, name if n == 1 else f'{name}-{n}')
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            n += 1


def new_experiment(root: str, name: str, now: _dt.datetime | None = None) -> str:
    """Create <root>/experiments/<YYYY-MM-DD>_<name> and return it."""
    now = now or _dt.datetime.now()
    return make_unique_dir(experiments_dir(root), f'{now:%Y-%m-%d}_{slug(name, "experiment")}')


def new_recording_dir(experiment: str, label: str = '', now: _dt.datetime | None = None) -> str:
    """Create <experiment>/<YYYYMMDD_HHMMSS>_<label> for one recording and return it."""
    now = now or _dt.datetime.now()
    return make_unique_dir(experiment, f'{now:%Y%m%d_%H%M%S}_{slug(label, "rec")}')


def script_path(root: str, kind_ext: str, name: str, fallback: str) -> str:
    """Where a new script goes: <root>/scripts/<readable name><ext>, never an existing file."""
    folder = scripts_dir(root)
    os.makedirs(folder, exist_ok=True)
    return unique_path(folder, slug(name, fallback), kind_ext)


# --- what a recording records about itself -------------------------------------------------------

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def freeze(src: str, run_dir: str, role: str) -> dict | None:
    """Copy a script or plugin that this recording uses into run_dir/protocol/, with its checksum."""
    if not src or not os.path.isfile(src):
        return None
    dest_dir = os.path.join(run_dir, PROTOCOL_DIR)
    os.makedirs(dest_dir, exist_ok=True)
    dest = unique_path(dest_dir, os.path.splitext(os.path.basename(src))[0], os.path.splitext(src)[1])
    shutil.copy2(src, dest)
    return {'role': role, 'source': os.path.abspath(src), 'copy': os.path.relpath(dest, run_dir),
            'sha256': sha256(dest)}


def config_dict(config) -> dict:
    """All settings as {section: {key: value}} (a ConfigParser or Kivy config)."""
    return {s: {k: v for k, v in config.items(s)} for s in config.sections()}


def write_settings(settings: dict, run_dir: str) -> None:
    parser = configparser.ConfigParser(interpolation=None)
    for section, values in settings.items():
        parser[section] = {k: str(v) for k, v in values.items()}
    with open(os.path.join(run_dir, SETTINGS_FILE), 'w', encoding='utf-8') as f:
        parser.write(f)


_VERSION_CACHE: dict = {}


def software_version(package_dir: str) -> dict:
    """GlowTracker version and, when run from a git checkout, the commit (and whether it had
    uncommitted changes)."""
    if package_dir in _VERSION_CACHE:
        return _VERSION_CACHE[package_dir]
    info: dict = {}
    try:
        from importlib.metadata import version
        info['glowtracker'] = version('glowtracker')
    except Exception:
        pass
    try:
        run = lambda *args: subprocess.run(['git', *args], cwd=package_dir, capture_output=True,
                                           text=True, timeout=5)
        commit = run('rev-parse', 'HEAD')
        if commit.returncode == 0:
            info['git_commit'] = commit.stdout.strip()
            info['git_uncommitted_changes'] = bool(run('status', '--porcelain', '--untracked-files=no').stdout.strip())
    except Exception:
        pass
    _VERSION_CACHE[package_dir] = info
    return info


def _write_json(path: str, data: dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, default=str)
    os.replace(tmp, path)


def start_run(run_dir: str, *, experiment: str, settings: dict, software: dict, acquisition: dict,
              stimulation: dict, frozen: list, context: dict | None = None,
              conditions: dict | None = None, now: _dt.datetime | None = None) -> dict:
    """Write run.json and settings.ini when a recording starts. Returns the record."""
    now = now or _dt.datetime.now()
    record = {
        'schema': SCHEMA,
        'recording': os.path.basename(run_dir),
        'experiment': os.path.basename(os.path.normpath(experiment)),
        'started': now.isoformat(timespec='seconds'),
        'ended': None,
        'computer': platform.node(),
        'software': software,
        'acquisition': acquisition,
        'stimulation': stimulation,
        'protocol_files': [f for f in frozen if f],
        'conditions': conditions or {},
        'context': context or {},
        'result': None,
    }
    write_settings(settings, run_dir)
    _write_json(os.path.join(run_dir, RUN_FILE), record)
    return record


def finish_run(run_dir: str, result: dict, experiment: str | None = None,
               now: _dt.datetime | None = None) -> dict | None:
    """Complete run.json when the recording stops and add its row to the experiment's runs.csv."""
    path = os.path.join(run_dir, RUN_FILE)
    try:
        with open(path, encoding='utf-8') as f:
            record = json.load(f)
    except (OSError, ValueError):
        return None
    record['ended'] = (now or _dt.datetime.now()).isoformat(timespec='seconds')
    record['result'] = result
    _write_json(path, record)
    if experiment:
        append_runs_table(experiment, table_row(record))
    return record


def table_row(record: dict) -> dict:
    """One flat row of a run.json for runs.csv."""
    row = {'recording': record.get('recording'), 'started': record.get('started'),
           'ended': record.get('ended'), 'computer': record.get('computer')}
    for group in ('conditions', 'acquisition', 'stimulation', 'result'):
        for key, value in (record.get(group) or {}).items():
            if not isinstance(value, (dict, list)):
                row[f'{group}.{key}' if group != 'conditions' else key] = value
    for f in record.get('protocol_files') or []:
        row[f"{f.get('role', 'file')}_file"] = os.path.basename(f.get('source', ''))
        row[f"{f.get('role', 'file')}_sha256"] = f.get('sha256', '')[:12]
    row['git_commit'] = (record.get('software') or {}).get('git_commit', '')[:12]
    return row


def append_runs_table(experiment: str, row: dict) -> None:
    """Add a row to <experiment>/runs.csv; columns are the union over all rows so far, so runs
    with different fields still line up."""
    path = os.path.join(experiment, RUNS_TABLE)
    rows, columns = [], []
    if os.path.isfile(path):
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
    for key in row:
        if key not in columns:
            columns.append(key)
    rows.append({k: '' if v is None else v for k, v in row.items()})
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
