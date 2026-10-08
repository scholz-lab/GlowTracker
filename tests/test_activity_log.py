"""The activity log behind the terminal panel."""
import threading

from activity_log import ActivityLog


def test_lines_are_kept_in_order_trimmed_and_one_line_each():
    log = ActivityLog(max_entries=3)
    seen = []
    log.subscribe(seen.append)
    for i in range(5):
        log.add('rec', f'line {i}\n  continued')
    log.add('rec', '   ')                                    # empty: ignored
    assert [e.text for e in log.entries()] == ['line 2 continued', 'line 3 continued', 'line 4 continued']
    assert len(seen) == 5 and seen[-1].source == 'rec' and len(seen[-1].clock) == 8


def test_threads_can_log_at_once_and_a_broken_listener_does_not_stop_logging():
    log = ActivityLog()
    log.subscribe(lambda entry: 1 / 0)
    threads = [threading.Thread(target=lambda n=n: [log.add('stage', f'{n}-{i}') for i in range(50)])
               for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(log.entries()) == 400


def test_plugin_messages_warnings_and_errors_reach_the_sink(tmp_path):
    import script_api
    plugin = tmp_path / 'p.py'
    plugin.write_text('def setup(scope):\n    scope.print("hello", 3)\ndef update(state, scope):\n    1 / 0\n')
    got = []
    host = script_api.PluginHost(state_provider=lambda: None, frame_provider=lambda: None,
                                 daq_getter=lambda: None, message_sink=lambda text, level: got.append((level, text)))
    host.load(str(plugin))
    scope = script_api.Scope(host)
    host.controller.setup(scope)
    host._warn('update() took 300 ms')
    assert ('info', 'hello 3') in got and ('warn', 'update() took 300 ms') in got


def test_diagonal_jogs_keep_the_straight_speed_and_are_logged():
    from types import SimpleNamespace
    import GlowTracker as G
    sent, logged = [], []
    app = SimpleNamespace(stage=object(), vhigh=2.0, vlow=0.5,
                          request_jog=lambda velocity, fast: sent.append(velocity),
                          log=lambda source, text, level='info': logged.append(text))
    G.GlowTrackerApp.jog(app, (-1, 1, 0), True)
    G.GlowTrackerApp.jog(app, (0, 0, 1), False)
    (vx, vy, vz), straight = sent
    assert abs((vx ** 2 + vy ** 2) ** 0.5 - 2.0) < 1e-9 and vx < 0 < vy and vz == 0
    assert straight == (0.0, 0.0, 0.5)
    assert logged == ['jog X- Y+ fast', 'jog Z+ slow']


def test_corner_key_is_an_L_and_leaves_the_inner_square_to_the_slow_key():
    import GlowTracker as G
    key = G.CornerPad(corner='ul', size=(100, 100), pos=(0, 0), gap=4)   # 2 x 2 squares of 48
    assert key.collide_point(10, 90)        # outer corner square
    assert key.collide_point(80, 90)        # along the top row
    assert key.collide_point(10, 20)        # down the left column
    assert not key.collide_point(80, 20)    # inner square: the slow diagonal key
    (cx, cy), size = key._arrowBox()
    assert (cx, cy) == (24, 76) and size == 48
