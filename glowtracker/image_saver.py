import os
import time
from queue import Empty
from threading import Lock, Thread

import tifffile


def close_file_with_timeout(file_object, timeout):
    result = {'error': None}

    def close_file():
        try:
            file_object.close()
        except Exception as error:
            result['error'] = error

    thread = Thread(target=close_file, daemon=True, name='CoordinateFileClose')
    thread.start()
    thread.join(timeout)
    return not thread.is_alive() and result['error'] is None, result['error'], thread


class SaveAcknowledgements:
    def __init__(self, coordinate_file):
        self.coordinate_file = coordinate_file
        self.pending = {}
        self.next_index = 0
        self.saved_frames = 0
        self.failed_frames = 0
        self.lock = Lock()

    def add(self, index, coordinate_row, channels):
        channels = set(channels)
        if not channels:
            raise ValueError('a frame must have at least one save channel')
        with self.lock:
            if index in self.pending or index < self.next_index:
                raise ValueError(f'frame {index} is already registered')
            self.pending[index] = {
                'coordinate_row': coordinate_row,
                'channels': channels,
                'failed': False,
            }

    def saved(self, index, channel):
        with self.lock:
            frame = self.pending.get(index)
            if frame is None or frame['failed']:
                return False
            if channel not in frame['channels']:
                raise ValueError(
                    f'frame {index} did not expect channel {channel}'
                )
            frame['channels'].discard(channel)
            self._flush()
            return True

    def failed(self, index):
        with self.lock:
            frame = self.pending.get(index)
            if frame is None or frame['failed']:
                return False
            frame['failed'] = True
            frame['channels'].clear()
            self.failed_frames += 1
            self._flush()
            return True

    def discard_pending(self):
        with self.lock:
            discarded = 0
            for frame in self.pending.values():
                if not frame['failed']:
                    frame['failed'] = True
                    frame['channels'].clear()
                    self.failed_frames += 1
                    discarded += 1
            self._flush()
            return discarded

    @property
    def pending_count(self):
        with self.lock:
            return len(self.pending)

    def _flush(self):
        while self.next_index in self.pending:
            frame = self.pending[self.next_index]
            if frame['channels']:
                break
            if not frame['failed']:
                self.coordinate_file.write(frame['coordinate_row'])
                self.saved_frames += 1
            del self.pending[self.next_index]
            self.next_index += 1


def _report(status_queue, status):
    if status_queue is None:
        return True
    try:
        status_queue.put(status)
        return True
    except Exception:
        return False


def _set_failure(failure_event):
    if failure_event is not None:
        try:
            failure_event.set()
        except Exception:
            pass


def save_worker(
        image_queue, save_dir, filename_format, stop_event,
        status_queue=None, failure_event=None):
    get = getattr(image_queue, 'get_nowait', image_queue.get)
    while True:
        if failure_event is not None:
            try:
                if failure_event.is_set():
                    break
            except Exception:
                pass
        try:
            data = get()
        except Empty:
            try:
                if stop_event.is_set():
                    break
            except Exception as e:
                error = f'{type(e).__name__}: {e}'
                _set_failure(failure_event)
                _report(status_queue, ('failed', -1, -1, error))
                break
            time.sleep(0.001)
            continue
        except Exception as e:
            error = f'{type(e).__name__}: {e}'
            _set_failure(failure_event)
            _report(status_queue, ('failed', -1, -1, error))
            break

        index = -1
        channel = -1
        temporary = None
        try:
            index = int(data['idx'])
            channel = int(data.get('channel', 0))
            fname = filename_format.format(index)
            if channel:
                root, extension = os.path.splitext(fname)
                suffix = '-main' if channel == 1 else '-minor'
                fname = root + suffix + extension

            destination = os.path.join(save_dir, fname)
            root, extension = os.path.splitext(destination)
            temporary = root + '.part' + extension
            tifffile.imwrite(temporary, data['img'])
            os.replace(temporary, destination)
        except Exception as e:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
            error = f'{type(e).__name__}: {e}'
            _set_failure(failure_event)
            _report(status_queue, ('failed', index, channel, error))
            break

        if not _report(status_queue, ('saved', index, channel, '')):
            _set_failure(failure_event)
            break
