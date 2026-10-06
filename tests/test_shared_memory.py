import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from SharedMemory import SharedMemoryQueue


def consume_frame(queue, result_queue):
    result_queue.put(queue.get())


def test_queue_transfers_frames_to_spawned_process():
    context = mp.get_context('spawn')
    manager = SharedMemoryManager(ctx=context)
    manager.start()
    result_queue = context.Queue()
    try:
        image = np.arange(12, dtype=np.uint16).reshape(3, 4)
        queue = SharedMemoryQueue.create_from_examples(
            manager,
            {'img': image, 'idx': 0, 'channel': 0},
            buffer_size=2,
            context=context,
        )
        queue.put({'img': image, 'idx': 7, 'channel': 2})
        process = context.Process(
            target=consume_frame,
            args=(queue, result_queue),
        )
        process.start()
        process.join(5)
        assert process.exitcode == 0
        result = result_queue.get(timeout=1)
        np.testing.assert_array_equal(result['img'], image)
        assert result['idx'] == 7
        assert result['channel'] == 2
        assert queue.empty()
    finally:
        result_queue.close()
        manager.shutdown()


def test_stack_saver_runs_in_spawned_process(tmp_path):
    """The Linux recording path: a spawned save process fed through the shared-memory queue."""
    import tifffile
    import image_saver
    context = mp.get_context('spawn')
    manager = SharedMemoryManager(ctx=context)
    manager.start()
    try:
        image = np.zeros((16, 16), dtype=np.uint8)
        queue = SharedMemoryQueue.create_from_examples(
            manager, {'img': image, 'idx': 0, 'channel': 0}, buffer_size=8, context=context)
        for i in range(5):
            queue.put({'img': np.full((16, 16), i, np.uint8), 'idx': i, 'channel': 0})
        stop, failure, status = context.Event(), context.Event(), context.Queue()
        stop.set()
        process = context.Process(target=image_saver.stack_save_worker,
                                  args=(queue, str(tmp_path), 'rec-basler_{}.tiff', stop, status, failure))
        process.start()
        process.join(20)
        assert process.exitcode == 0 and not failure.is_set()
        pages = tifffile.imread(tmp_path / 'rec-basler_stack_000.tiff')
        assert [int(p[0, 0]) for p in pages] == [0, 1, 2, 3, 4]
    finally:
        manager.shutdown()
