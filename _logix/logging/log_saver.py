# Copyright 2023-present the LogIX team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any

import torch

from _logix.logging.mmap import MemoryMapHandler
from _logix.tensor_log import TensorLog
from _logix.utils import get_rank, to_numpy


class LogSaver:
    """
    Efficient tensor-based log saver for storing logs to disk.
    """
    def __init__(self, config, state):
        """
        Initialize the LogSaver.

        Args:
            config: Configuration object
            state: LogIXState object
        """
        self.log_dir = config.log_dir
        self.state = state
        self.model_module = self.state.get_state("model_module")
        self.file_prefix = f"log_rank_{get_rank()}_chunk_"

        self.max_worker = config.num_workers
        self.allow_async = True if self.max_worker > 1 else False

        self.flush_threshold = config.flush_threshold
        self.flush_count = 0

        # Initialize buffer as an empty dictionary mapping data_id to TensorLog
        self.buffer: Dict[str, Dict[Tuple[str, str], torch.Tensor]] = {}
        self.buffer_size = 0

    def buffer_write(self, binfo):
        """
        Add log state to buffer.

        Args:
            binfo: BatchInfo object containing logs
        """
        data_ids = binfo.data_id
        tensor_log = binfo.log

        # For each data_id in the batch, add its tensors to the buffer
        for idx, data_id in enumerate(data_ids):
            # Initialize a new tensor dictionary for this data_id if needed
            if data_id not in self.buffer:
                self.buffer[data_id] = {}

            # For each module and log type, extract the corresponding tensor
            for (module_name, log_type), tensor in tensor_log.items():
                # Get the tensor for this specific data point (by index)
                single_tensor = tensor[idx:idx+1].detach().cpu()

                # Add to buffer
                self.buffer[data_id][(module_name, log_type)] = single_tensor
                self.buffer_size += single_tensor.numel() * single_tensor.element_size()

    def _flush_unsafe(self, log_dir, buffer, flush_count) -> str:
        """
        Thread unsafe flush of current buffer. No shared variable must be allowed.

        Args:
            log_dir: Directory to save logs
            buffer: Buffer to flush
            flush_count: Current flush count

        Returns:
            str: Filename of the saved log
        """
        filename = self.file_prefix + f"{flush_count}.mmap"
        buffer_list = [(k, v) for k, v in buffer.items()]
        MemoryMapHandler.write_tensor_log(log_dir, filename, buffer_list)
        return filename

    def _flush_safe(self, log_dir) -> str:
        """
        Thread safe flush of current buffer.

        Args:
            log_dir: Directory to save logs

        Returns:
            str: Filename of the saved log
        """
        buffer_copy = self.buffer.copy()
        flush_count_copy = self.flush_count
        self.flush_count += 1
        self.buffer_clear()
        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            save_path = executor.submit(
                self._flush_unsafe, log_dir, buffer_copy, flush_count_copy
            )
        return save_path

    def _flush_serialized(self, log_dir) -> str:
        """
        Execute the flushing of buffers in serialized manner.

        Args:
            log_dir: Directory to save logs

        Returns:
            str: Log directory
        """
        if len(self.buffer) == 0:
            return log_dir

        buffer_list = [(k, v) for k, v in self.buffer.items()]
        MemoryMapHandler.write_tensor_log(
            log_dir,
            self.file_prefix + f"{self.flush_count}.mmap",
            buffer_list,
            dtype="uint8",
        )

        self.flush_count += 1
        self.buffer_clear()
        del buffer_list
        return log_dir

    def flush(self) -> None:
        """
        Flush the buffer if it exceeds the threshold.
        """
        if 0 < self.flush_threshold < self.buffer_size:
            if self.allow_async:
                self._flush_safe(self.log_dir)
                return
            self._flush_serialized(self.log_dir)

    def finalize(self):
        """
        Dump everything in the buffer to disk when `logix.finalize()` is called.
        """
        self._flush_serialized(self.log_dir)

    def buffer_clear(self):
        """
        Clear the buffer.
        """
        self.buffer.clear()
        self.buffer_size = 0