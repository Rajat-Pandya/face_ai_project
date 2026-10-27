import concurrent.futures
from src.stats import StatsManager, get_stats_manager


def test_stats_manager_singleton_and_keys():
    manager1 = get_stats_manager()
    manager2 = get_stats_manager()
    assert manager1 is manager2

    stats = manager1.get_stats()
    assert set(stats.keys()) == {"faces", "emotion", "mask"}


def test_stats_manager_concurrent_updates():
    manager = StatsManager()

    def worker(idx):
        for i in range(50):
            manager.update_faces(i)
            manager.update_emotion("Happy" if i % 2 == 0 else "Neutral")
            manager.update_mask("MASK" if i % 2 == 0 else "NO MASK")
            s = manager.get_stats()
            assert set(s.keys()) == {"faces", "emotion", "mask"}
        if idx == 0:
            manager.reset()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    final_stats = manager.get_stats()
    assert set(final_stats.keys()) == {"faces", "emotion", "mask"}
