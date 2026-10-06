buzz_at_s = [10, 30, 50]    
buzz_s = 1.0                

QUIET = 4.5
BUZZ = 0.0
idle_voltage = QUIET       

start = None


def setup(scope):
    scope.set_voltage(QUIET)


def update(state, scope):
    global start
    if not state.is_recording:
        start = None
        scope.set_voltage(QUIET)
        return
    if start is None:
        start = state.wall_time
    t = state.wall_time - start
    buzzing = any(b <= t < b + buzz_s for b in buzz_at_s)
    scope.set_voltage(BUZZ if buzzing else QUIET)


def teardown(scope):
    scope.set_voltage(QUIET)
