from threading import Event
import time

from MacroScript import MacroScriptExecutor


def test_macro_wait_is_interruptible():
    finished = Event()
    executor = MacroScriptExecutor()
    executor.executeScript('wait(5)', finished.set)
    time.sleep(0.05)
    assert executor.stop(timeout=1.0)
    assert finished.wait(0.5)
