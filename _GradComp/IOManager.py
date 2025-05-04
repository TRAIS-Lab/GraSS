from __future__ import annotations

# from typing import TYPE_CHECKING, Any, Dict, Literal, List, Optional, Union, Tuple, TypedDict, cast
import os
import time
import threading
import queue
import psutil

import torch

class TwoStageBuffer:
    """Two-stage buffer for efficient disk I/O."""

    def __init__(self, cache_dir, buffer_size_mb=1024):
        self.cache_dir = cache_dir
        self.memory_buffer = {}  # In-memory buffer
        self.buffer_size = buffer_size_mb * 1024 * 1024  # Convert to bytes
        self.current_size = 0
        self.buffer_lock = threading.Lock()
        self.flush_thread = None
        self.running = False
        self.flush_event = threading.Event()

    def start(self):
        """Start background flushing thread."""
        self.running = True
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()

    def stop(self):
        """Stop background flushing thread and flush remaining items."""
        self.running = False
        self.flush_event.set()  # Signal thread to exit
        if self.flush_thread:
            self.flush_thread.join()
        self._flush_all()

    def add(self, tensor, file_path):
        """Add tensor to buffer."""
        tensor_size = tensor.element_size() * tensor.nelement()

        with self.buffer_lock:
            # Add to memory buffer
            self.memory_buffer[file_path] = tensor
            self.current_size += tensor_size

            # Check if buffer is full
            if self.current_size > self.buffer_size * 0.7:  # 70% threshold
                # Signal flush thread
                self.flush_event.set()

    def get(self, file_path):
        """Get tensor from buffer or disk."""
        with self.buffer_lock:
            # Check if in memory buffer
            if file_path in self.memory_buffer:
                return self.memory_buffer[file_path]

        # Not in memory, try disk
        if os.path.exists(file_path):
            return torch.load(file_path)

        raise FileNotFoundError(f"Could not find tensor file: {file_path}")

    def _flush_worker(self):
        """Background worker that flushes items to disk."""
        while self.running:
            # Wait for flush signal or periodic check
            self.flush_event.wait(timeout=1.0)
            self.flush_event.clear()

            # Check if buffer needs flushing
            should_flush = False
            with self.buffer_lock:
                if self.current_size > self.buffer_size * 0.5:  # 50% threshold
                    should_flush = True

            if should_flush:
                self._flush_some()

    def _flush_some(self):
        """Flush some items from memory buffer to disk."""
        items_to_flush = {}

        with self.buffer_lock:
            # Take up to 30% of buffer items
            num_items = len(self.memory_buffer)
            flush_size = max(1, int(num_items * 0.3))

            # Get oldest items
            keys = list(self.memory_buffer.keys())[:flush_size]

            # Move to flush list
            for key in keys:
                items_to_flush[key] = self.memory_buffer.pop(key)

            # Recalculate current size
            self.current_size = sum(t.element_size() * t.nelement() for t in self.memory_buffer.values())

        # Now flush items outside of lock
        for file_path, tensor in items_to_flush.items():
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Save to disk
            torch.save(tensor, file_path)

    def _flush_all(self):
        """Flush all items from memory buffer to disk."""
        items_to_flush = {}

        with self.buffer_lock:
            # Get all items
            items_to_flush = self.memory_buffer
            self.memory_buffer = {}
            self.current_size = 0

        # Flush all items
        for file_path, tensor in items_to_flush.items():
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Save to disk
            torch.save(tensor, file_path)


class IOManager:
    """Enhanced asynchronous I/O manager with dynamic throttling."""

    def __init__(self, num_threads=32, max_queue_size=100, high_watermark=0.8, low_watermark=0.4,
                use_buffer=False, cache_dir=None, buffer_size_mb=1024, is_tmpfs=False):
        self.read_queue = queue.Queue()
        self.write_queue = queue.Queue()
        self.read_results = {}  # Maps request_id to result
        self.read_condition = threading.Condition()

        # Adjust thread count based on storage type
        self.num_read_threads = max(8, num_threads // 4)
        self.num_write_threads = num_threads - self.num_read_threads

        # Increase thread count for non-tmpfs storage
        if not is_tmpfs:
            self.num_write_threads = min(128, self.num_write_threads * 2)

        self.max_queue_size = max_queue_size
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self.throttling = False
        self.throttle_lock = threading.Lock()

        self.workers = []
        self.running = False

        # Two-stage buffer for non-tmpfs storage
        self.use_buffer = use_buffer and not is_tmpfs
        if self.use_buffer and cache_dir:
            self.buffer = TwoStageBuffer(cache_dir, buffer_size_mb)
        else:
            self.buffer = None

    def start(self):
        """Start worker threads."""
        self.running = True

        # Start two-stage buffer if used
        if self.buffer:
            self.buffer.start()

        # Start read workers
        for i in range(self.num_read_threads):
            thread = threading.Thread(target=self._read_worker, daemon=True)
            thread.start()
            self.workers.append(thread)

        # Start write workers
        for i in range(self.num_write_threads):
            thread = threading.Thread(target=self._write_worker, daemon=True)
            thread.start()
            self.workers.append(thread)

    def stop(self):
        """Stop all worker threads."""
        self.running = False

        # Send stop signals
        for _ in range(self.num_read_threads):
            self.read_queue.put(None)

        for _ in range(self.num_write_threads):
            self.write_queue.put(None)

        # Wait for threads to complete
        for thread in self.workers:
            thread.join()

        self.workers = []

        # Stop buffer if used
        if self.buffer:
            self.buffer.stop()

    def _read_worker(self):
        """Worker thread for file reading."""
        while self.running:
            job = self.read_queue.get()
            if job is None:
                self.read_queue.task_done()
                break

            request_id, file_path = job

            try:
                # Check if file is in buffer
                if self.buffer:
                    try:
                        tensor = self.buffer.get(file_path)
                    except FileNotFoundError:
                        # Not in buffer, load from disk
                        tensor = torch.load(file_path)
                else:
                    # Read directly from disk
                    tensor = torch.load(file_path)

                # Store result
                with self.read_condition:
                    self.read_results[request_id] = tensor
                    self.read_condition.notify_all()
            except Exception as e:
                # Store error as result
                with self.read_condition:
                    self.read_results[request_id] = e
                    self.read_condition.notify_all()
            finally:
                self.read_queue.task_done()

    def _write_worker(self):
        """Worker thread for file writing."""
        while self.running:
            job = self.write_queue.get()
            if job is None:
                self.write_queue.task_done()
                break

            tensor, file_path = job

            try:
                # If using buffer, add to buffer
                if self.buffer:
                    self.buffer.add(tensor, file_path)
                else:
                    # Write directly to disk
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    # Write the file
                    torch.save(tensor, file_path)

            except Exception as e:
                print(f"Error in write worker: {e}")
            finally:
                self.write_queue.task_done()

    def async_read(self, file_path, request_id=None):
        """Queue a file to be read asynchronously."""
        if request_id is None:
            request_id = file_path

        # Check if queue is full
        if self.read_queue.qsize() >= self.max_queue_size:
            self.read_queue.join()

        # Queue read operation
        self.read_queue.put((request_id, file_path))
        return request_id

    def async_write(self, tensor, file_path):
        """Queue a tensor to be written asynchronously with throttling."""
        # Check if queue is too full
        queue_fullness = self.write_queue.qsize() / self.max_queue_size

        with self.throttle_lock:
            # If we're not throttling but queue is getting full, start throttling
            if not self.throttling and queue_fullness > self.high_watermark:
                self.throttling = True

            # If we're throttling but queue is getting empty, stop throttling
            elif self.throttling and queue_fullness < self.low_watermark:
                self.throttling = False

        # If throttling, wait for some items to be processed
        if self.throttling:
            # Wait for queue to decrease to low watermark
            while self.write_queue.qsize() > self.low_watermark * self.max_queue_size:
                time.sleep(0.01)  # Small sleep to avoid busy waiting

        # Queue write operation
        self.write_queue.put((tensor, file_path))

    def wait_for_read(self, request_id, timeout=None):
        """Wait for a read operation to complete and return the result."""
        with self.read_condition:
            # Wait for result to be available
            while request_id not in self.read_results:
                if not self.read_condition.wait(timeout):
                    raise TimeoutError(f"Read operation timed out: {request_id}")

            # Get and remove result
            result = self.read_results.pop(request_id)

            # If result is an exception, raise it
            if isinstance(result, Exception):
                raise result

            return result

    def wait_all(self):
        """Wait for all I/O operations to complete."""
        self.read_queue.join()
        self.write_queue.join()