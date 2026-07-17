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
