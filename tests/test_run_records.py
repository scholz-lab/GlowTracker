"""Experiment and recording folders, run.json, runs.csv and script names (run_records)."""
import configparser
import csv
import datetime as dt
import json
import os

import run_records as rr

NOW = dt.datetime(2026, 10, 28, 14, 30, 12)


def test_readable_names():
    assert rr.slug('AWA 1 s pulses, every 30 s!') == 'awa-1-s-pulses-every-30-s'
    assert rr.slug('../evil/path.py') == 'path'
    assert rr.slug('', 'plugin') == 'plugin'


def test_experiment_and_recording_folders_never_collide(tmp_path):
    exp = rr.new_experiment(str(tmp_path), 'AWA guidance', now=NOW)
    assert exp == os.path.join(str(tmp_path), 'experiments', '2026-10-28_awa-guidance')
    assert rr.new_experiment(str(tmp_path), 'AWA guidance', now=NOW).endswith('2026-10-28_awa-guidance-2')
    a = rr.new_recording_dir(exp, 'bandit_zapper_v4', now=NOW)
    b = rr.new_recording_dir(exp, 'bandit_zapper_v4', now=NOW)       # same second
    assert os.path.basename(a) == '20261028_143012_bandit-zapper-v4' and b == a + '-2'
    assert os.path.basename(rr.new_recording_dir(exp, '', now=NOW)) == '20261028_143012_rec'


def test_script_paths_are_readable_and_never_overwrite(tmp_path):
    first = rr.script_path(str(tmp_path), '.py', 'Buzz at 10, 30 and 50 s', 'plugin')
    assert first == os.path.join(str(tmp_path), 'scripts', 'buzz-at-10-30-and-50-s.py')
    open(first, 'w').close()
    assert rr.script_path(str(tmp_path), '.py', 'buzz at 10 30 and 50 s', 'plugin').endswith('buzz-at-10-30-and-50-s-2.py')


def test_run_record_freeze_settings_and_table(tmp_path):
    exp = rr.new_experiment(str(tmp_path), 'test', now=NOW)
    run = rr.new_recording_dir(exp, 'buzzer', now=NOW)
    plugin = tmp_path / 'buzzer.py'
    plugin.write_text('def update(state, scope):\n    pass\n')
    config = configparser.ConfigParser()
    config['Tracking'] = {'binning': '8', 'continuoustracking': 'false'}
    frozen = [rr.freeze(str(plugin), run, 'plugin'), rr.freeze('', run, 'sequencer')]
    rr.start_run(run, experiment=exp, settings=rr.config_dict(config), software={'git_commit': 'abc123def4567'},
                 acquisition={'framerate': 30.0}, stimulation={'daq_mode': 'Plugin'}, frozen=frozen,
                 conditions={'strain': 'AWA::Chrimson'}, now=NOW)
    plugin.write_text('# edited later\n')                                  # editing the original...
    record = json.load(open(os.path.join(run, 'run.json')))
    copy = os.path.join(run, record['protocol_files'][0]['copy'])
    assert open(copy).read().startswith('def update')                     # ...does not change the copy
    assert record['protocol_files'][0]['sha256'] == rr.sha256(copy)
    assert len(record['protocol_files']) == 1 and record['ended'] is None
    saved = configparser.ConfigParser(); saved.read(os.path.join(run, 'settings.ini'))
    assert saved['Tracking']['binning'] == '8'

    rr.finish_run(run, {'frames_saved': 900, 'frames_dropped': 0}, experiment=exp, now=NOW)
    run2 = rr.new_recording_dir(exp, 'buzzer', now=NOW)
    rr.start_run(run2, experiment=exp, settings={}, software={}, acquisition={}, stimulation={},
                 frozen=[], conditions={'strain': 'N2', 'food': 'off'}, now=NOW)
    rr.finish_run(run2, {'frames_saved': 10}, experiment=exp, now=NOW)
    rows = list(csv.DictReader(open(os.path.join(exp, 'runs.csv'), newline='')))
    assert [r['recording'] for r in rows] == [os.path.basename(run), os.path.basename(run2)]
    assert rows[0]['strain'] == 'AWA::Chrimson' and rows[0]['food'] == ''   # columns are the union
    assert rows[1]['food'] == 'off' and rows[0]['plugin_file'] == 'buzzer.py'
    assert rows[0]['result.frames_saved'] == '900'


def test_a_continuous_recording_claims_no_planned_length():
    text = rr.coords_plan(True, 840.0, 4200)
    assert 'continuous True' in text and '840' not in text and '4200' not in text


def test_a_fixed_recording_states_its_plan():
    assert rr.coords_plan(False, 840.0, 4200).split('\n')[:3] == \
        ['continuous False', 'duration 840.0', 'nframes 4200']


def test_the_real_length_closes_the_file_as_comments():
    lines = rr.coords_result(6123, 1224.56).splitlines()
    assert lines == ['# frames_recorded 6123', '# duration_recorded_s 1224.560']
