"""
Utilities for managing file handles and preventing "too many open files" errors.
Simplified version for single-threaded operation (num_workers=0).
"""

import resource
import os
import gc
import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

def increase_file_descriptor_limit(target_limit: int = 8192) -> tuple:
    """
    Increase the file descriptor limit for the current process.

    Args:
        target_limit: Desired soft limit for file descriptors

    Returns:
        Tuple of (old_soft_limit, old_hard_limit, new_soft_limit, new_hard_limit)
    """
    try:
        # Get current limits
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        old_limits = (soft, hard)

        # Calculate new soft limit (can't exceed hard limit)
        new_soft = min(target_limit, hard)

        # Set new limits
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))

        # Verify the change
        new_soft_actual, new_hard_actual = resource.getrlimit(resource.RLIMIT_NOFILE)

        logger.info(f"File descriptor limits: {soft} -> {new_soft_actual} (soft), {hard} -> {new_hard_actual} (hard)")

        return old_limits + (new_soft_actual, new_hard_actual)

    except (ValueError, OSError) as e:
        logger.warning(f"Could not increase file descriptor limit: {e}")
        return resource.getrlimit(resource.RLIMIT_NOFILE) * 2

def get_open_file_count() -> int:
    """
    Get the current number of open file descriptors for this process.

    Returns:
        Number of open file descriptors, or -1 if unable to determine
    """
    try:
        # Linux/Unix method
        proc_fd_path = f"/proc/{os.getpid()}/fd"
        if os.path.exists(proc_fd_path):
            return len(os.listdir(proc_fd_path))
    except (OSError, PermissionError):
        pass

    try:
        # Alternative method using lsof (if available)
        import subprocess
        result = subprocess.run(['lsof', '-p', str(os.getpid())],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return len(result.stdout.strip().split('\n')) - 1  # -1 for header
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return -1

def monitor_file_descriptors(threshold: float = 0.8) -> dict:
    """
    Monitor file descriptor usage and return statistics.

    Args:
        threshold: Warning threshold as fraction of soft limit

    Returns:
        Dictionary with file descriptor statistics
    """
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    open_count = get_open_file_count()

    stats = {
        'soft_limit': soft_limit,
        'hard_limit': hard_limit,
        'open_count': open_count,
        'usage_ratio': open_count / soft_limit if open_count >= 0 else -1,
        'warning': False
    }

    if open_count >= 0 and open_count > (soft_limit * threshold):
        stats['warning'] = True
        logger.warning(f"High file descriptor usage: {open_count}/{soft_limit} ({stats['usage_ratio']:.1%})")

    return stats

@contextmanager
def managed_file_descriptors(target_limit: int = 4096, cleanup_threshold: float = 0.8):
    """
    Context manager for managing file descriptors during operations.

    Args:
        target_limit: Target soft limit for file descriptors
        cleanup_threshold: Threshold for triggering cleanup
    """
    # Increase limits at start
    old_limits = increase_file_descriptor_limit(target_limit)

    try:
        yield
    finally:
        # Monitor usage and cleanup if needed
        stats = monitor_file_descriptors(cleanup_threshold)
        if stats['warning']:
            logger.info("Forcing garbage collection due to high file descriptor usage")
            gc.collect()

            # Check again after cleanup
            new_stats = monitor_file_descriptors(cleanup_threshold)
            if new_stats['open_count'] >= 0:
                logger.info(f"File descriptors after cleanup: {new_stats['open_count']}")

def setup_file_handle_limits():
    """
    Setup optimal file handle limits for the chunked I/O system.
    Should be called at the start of your program.
    """
    try:
        # Get current limits
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

        logger.info(f"Initial file descriptor limits: soft={soft}, hard={hard}")

        # Try to increase to a reasonable level
        target_soft = min(8192, hard)  # Target 8K file descriptors or hard limit

        if soft < target_soft:
            increase_file_descriptor_limit(target_soft)

        # Monitor initial state
        stats = monitor_file_descriptors()
        logger.info(f"File descriptor monitoring: {stats['open_count']} open, "
                   f"{stats['soft_limit']} soft limit, {stats['hard_limit']} hard limit")

        return True

    except Exception as e:
        logger.error(f"Failed to setup file handle limits: {e}")
        return False

class FileHandleManager:
    """
    Simplified manager for file handle limits (since we use num_workers=0).
    """

    def __init__(self, max_workers: int = 0, fd_per_worker: int = 0):
        self.max_workers = 0  # Always 0 for single-threaded operation
        self.fd_per_worker = 0  # Always 0 since no workers
        self.setup_done = False

    def setup(self):
        """Setup file handle limits (simplified since no workers)."""
        if self.setup_done:
            return

        # Just setup basic limits for single process
        setup_file_handle_limits()
        self.setup_done = True
        logger.info("FileHandleManager setup complete for single-threaded operation")

    def get_safe_worker_count(self) -> int:
        """
        Always return 0 since we use single-threaded operation.

        Returns:
            Always 0 to prevent multiprocessing issues
        """
        return 0

    def worker_init_fn(self, worker_id: int):
        """Worker initialization function (not used since num_workers=0)."""
        pass  # Not needed since we don't use workers