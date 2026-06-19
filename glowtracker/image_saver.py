import os
import time
from queue import Empty

import tifffile


def save_worker(image_queue, name_queue, save_dir, stop_event):
    """Runs in its own process. Pulls frames out of the shared-memory queue and
    their filenames from name_queue, and writes them to disk. Exits once
    stop_event is set and the queue is drained."""
    while True:
        try:
            data = image_queue.get()
        except Empty:
            if stop_event.is_set():
                break
            time.sleep(0.001)
            continue
        fname = name_queue.get()
        tifffile.imwrite(os.path.join(save_dir, fname), data['img'])
