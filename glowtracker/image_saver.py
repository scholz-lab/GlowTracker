import os
import time
from queue import Empty

import tifffile


def save_worker(image_queue, save_dir, filename_format, stop_event):
    get = getattr(image_queue, 'get_nowait', image_queue.get)
    while True:
        try:
            data = get()
        except Empty:
            if stop_event.is_set():
                break
            time.sleep(0.001)
            continue
        fname = filename_format.format(int(data['idx']))
        tifffile.imwrite(os.path.join(save_dir, fname), data['img'])
