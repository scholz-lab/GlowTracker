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


class DropBudget:
    """Tolerate short saver back-pressure instead of ending the recording.

    A frame that cannot be queued is dropped (it gets no coordinate row, like any failed frame).
    Only when more than `max_consecutive` frames in a row are dropped does the saver count as
    failed, which is when the recording stops.
    """

    def __init__(self, max_consecutive):
        self.max_consecutive = max(0, int(max_consecutive))
        self.consecutive = 0
        self.total = 0
        self.lock = Lock()

    def dropped(self):
        """Record one dropped frame. Returns True once the run of drops exceeds the budget."""
        with self.lock:
            self.consecutive += 1
            self.total += 1
            return self.consecutive > self.max_consecutive

    def passed(self):
        """A frame went through: the run of consecutive drops is over."""
        with self.lock:
            self.consecutive = 0


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


def _save_loop(image_queue, stop_event, status_queue, failure_event, write_frame):
    """Take frames off the queue until stopped, write each with write_frame(index, channel, image),
    and report ('saved' | 'failed', index, channel, error) on status_queue."""
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
        try:
            index = int(data['idx'])
            channel = int(data.get('channel', 0))
            write_frame(index, channel, data['img'])
        except Exception as e:
            error = f'{type(e).__name__}: {e}'
            _set_failure(failure_event)
            _report(status_queue, ('failed', index, channel, error))
            break

        if not _report(status_queue, ('saved', index, channel, '')):
            _set_failure(failure_event)
            break


def _channel_name(fname, channel):
    """Split dual-colour recordings get -main / -minor before the extension."""
    if not channel:
        return fname
    root, extension = os.path.splitext(fname)
    return root + ('-main' if channel == 1 else '-minor') + extension


def save_worker(
        image_queue, save_dir, filename_format, stop_event,
        status_queue=None, failure_event=None):
    """One TIFF file per frame (written to a temporary name, then renamed into place)."""

    def write_frame(index, channel, image):
        destination = os.path.join(save_dir, _channel_name(filename_format.format(index), channel))
        root, extension = os.path.splitext(destination)
        temporary = root + '.part' + extension
        try:
            tifffile.imwrite(temporary, image)
            os.replace(temporary, destination)
        except Exception:
            try:
                os.unlink(temporary)
            except Exception:
                pass
            raise

    _save_loop(image_queue, stop_event, status_queue, failure_event, write_frame)


# A new stack file is started once the current one reaches this size, so a single file stays
# manageable to copy and open, and one damaged file never holds a whole long recording.
STACK_MAX_BYTES = 2 * 1024 ** 3


class StackWriter:
    """Append frames as pages of BigTIFF stacks, one stack series per channel.

    Files are <prefix>stack_000.tiff, _001, ... (with -main / -minor for split dual colour).
    Each page is a complete TIFF page (not 'contiguous' mode), so every page written before a
    crash stays readable. Next to each stack, <stack>_frames.txt lists the recording frame index
    of each page, one per line, so pages map to coordinate rows even when frames were dropped.
    """

    def __init__(self, save_dir, filename_format, max_bytes=STACK_MAX_BYTES):
        self.save_dir = save_dir
        base = filename_format.format('stack')
        root, _ = os.path.splitext(base)
        self.base = root
        self.max_bytes = max_bytes
        self.files = {}          # channel -> dict(writer, index_file, bytes, part, path)
        self.paths = []

    def _open(self, channel, part):
        name = _channel_name(f'{self.base}_{part:03d}.tiff', channel)
        path = os.path.join(self.save_dir, name)
        writer = tifffile.TiffWriter(path, bigtiff=True)
        index_file = open(os.path.splitext(path)[0] + '_frames.txt', 'w', encoding='utf-8')
        self.files[channel] = {'writer': writer, 'index_file': index_file, 'bytes': 0, 'part': part, 'pages': 0}
        self.paths.append(path)
        return self.files[channel]

    def _close_one(self, entry):
        error = None
        for closer in (entry['writer'].close, entry['index_file'].close):
            try:
                closer()
            except Exception as e:
                error = error or e
        if error is not None:
            raise error

    def write(self, index, channel, image):
        entry = self.files.get(channel)
        if entry is not None and entry['bytes'] + image.nbytes > self.max_bytes and entry['pages']:
            self._close_one(entry)
            entry = self._open(channel, entry['part'] + 1)
        elif entry is None:
            entry = self._open(channel, 0)
        entry['writer'].write(image, contiguous=False, metadata=None)
        entry['index_file'].write(f'{index}\n')
        entry['bytes'] += image.nbytes
        entry['pages'] += 1
        if entry['pages'] % 30 == 0:
            entry['index_file'].flush()

    def close(self):
        error = None
        for entry in self.files.values():
            try:
                self._close_one(entry)
            except Exception as e:
                error = error or e
        self.files = {}
        if error is not None:
            raise error


def stack_save_worker(
        image_queue, save_dir, filename_format, stop_event,
        status_queue=None, failure_event=None, max_bytes=STACK_MAX_BYTES):
    """Write frames into multi-page BigTIFF stacks (see StackWriter). Needs a single worker,
    so frames are written in the order they were queued."""
    stack = StackWriter(save_dir, filename_format, max_bytes)
    try:
        _save_loop(image_queue, stop_event, status_queue, failure_event, stack.write)
    finally:
        try:
            stack.close()
        except Exception as e:
            # Pages already written stay readable; only the file trailer is affected.
            _report(status_queue, ('failed', -1, -1, f'Closing the image stack failed: {type(e).__name__}: {e}'))
