"""Tests for file locking functionality in responseLocal.

This module tests the file locking mechanisms used to synchronize
download and delete operations when running with multiple workers.
"""

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest
from filelock import FileLock


class TestFileLocking:
    """Tests for file locking behavior."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def test_filelock_basic_functionality(self, temp_dir):
        """Test that FileLock prevents concurrent access."""
        lock_file = os.path.join(temp_dir, "test.lock")
        shared_counter = [0]
        access_times = []

        def worker(worker_id, delay=0.1):
            """Worker that increments counter under lock."""
            lock = FileLock(lock_file, timeout=10)
            with lock:
                start_time = time.time()
                # Critical section
                current = shared_counter[0]
                time.sleep(delay)  # Simulate work
                shared_counter[0] = current + 1
                end_time = time.time()
                access_times.append((worker_id, start_time, end_time))

        # Run 3 workers concurrently
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Counter should be incremented correctly
        assert shared_counter[0] == 3

        # Verify no time overlap (serialized execution)
        access_times.sort(key=lambda x: x[1])
        for i in range(len(access_times) - 1):
            current_end = access_times[i][2]
            next_start = access_times[i + 1][1]
            # Next access should start after current ends
            assert next_start >= current_end - 0.01

    def test_download_lock_pattern(self, temp_dir):
        """Test the download locking pattern."""
        source_file = os.path.join(temp_dir, "source.txt")
        Path(source_file).write_text("test content")

        download_count = [0]

        def simulated_download(worker_id):
            """Simulate download with locking."""
            local_file_path = os.path.join(temp_dir, f"worker_{worker_id}.txt")
            lock_file = local_file_path + ".download.lock"
            lock = FileLock(lock_file, timeout=10)

            with lock:
                # Simulate download
                shutil.copy(source_file, local_file_path)
                time.sleep(0.05)
                download_count[0] += 1

            # Lock should be released
            assert not os.path.exists(lock_file + ".lock")

        threads = []
        for i in range(3):
            t = threading.Thread(target=simulated_download, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert download_count[0] == 3

    def test_delete_lock_pattern(self, temp_dir):
        """Test the delete locking pattern."""
        # Create directories to delete
        dirs_to_delete = []
        for i in range(3):
            dir_path = os.path.join(temp_dir, f"old_dir_{i}")
            os.makedirs(dir_path, exist_ok=True)
            dirs_to_delete.append(f"old_dir_{i}")

        delete_count = [0]

        def simulated_delete(dir_name):
            """Simulate directory deletion with locking."""
            lock_file = os.path.join(temp_dir, ".delete.lock")
            lock = FileLock(lock_file, timeout=10)

            with lock:
                dir_path = os.path.join(temp_dir, dir_name)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    time.sleep(0.05)
                    delete_count[0] += 1

        threads = []
        for dir_name in dirs_to_delete:
            t = threading.Thread(target=simulated_delete, args=(dir_name,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert delete_count[0] == 3

        # All directories should be deleted
        for dir_name in dirs_to_delete:
            assert not os.path.exists(os.path.join(temp_dir, dir_name))

    def test_delete_same_resource_concurrently(self, temp_dir):
        """Test multiple workers trying to delete the same resource."""
        dir_name = "shared_dir"
        dir_path = os.path.join(temp_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)

        attempt_count = [0]

        def try_delete():
            """Try to delete the shared directory."""
            lock_file = os.path.join(temp_dir, ".delete.lock")
            lock = FileLock(lock_file, timeout=10)

            with lock:
                # Check if directory still exists before deleting
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                attempt_count[0] += 1

        # Start 3 threads trying to delete the same directory
        threads = []
        for _ in range(3):
            t = threading.Thread(target=try_delete)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All attempts should complete (3 workers all tried)
        assert attempt_count[0] == 3

        # Directory should be deleted
        assert not os.path.exists(dir_path)

    def test_lock_timeout_behavior(self, temp_dir):
        """Test that lock timeout works correctly."""
        lock_file = os.path.join(temp_dir, "timeout.lock")

        def hold_lock_long():
            """Hold lock for a long time."""
            lock = FileLock(lock_file, timeout=10)
            with lock:
                time.sleep(2)

        def try_acquire_with_timeout():
            """Try to acquire lock with short timeout."""
            lock = FileLock(lock_file, timeout=0.5)
            try:
                with lock:
                    return "acquired"
            except Exception:
                return "timeout"

        # Start thread that holds lock
        t1 = threading.Thread(target=hold_lock_long)
        t1.start()

        # Give it time to acquire the lock
        time.sleep(0.1)

        # Try to acquire with short timeout (should fail)
        result_container = []

        def run_timeout_test():
            result_container.append(try_acquire_with_timeout())

        t2 = threading.Thread(target=run_timeout_test)
        t2.start()
        t2.join()

        assert result_container[0] == "timeout"

        t1.join()
