"""Memory management utilities for efficient resource usage."""

import gc
import os
import mmap
import psutil
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from contextlib import contextmanager

from .logger import get_module_logger
from .config import config_manager


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    process_memory: int
    process_memory_percent: float


class MemoryManager:
    """Memory management system for efficient resource usage."""
    
    def __init__(self, max_memory_usage: Optional[int] = None):
        """Initialize memory manager.
        
        Args:
            max_memory_usage: Maximum memory usage in bytes. If None, uses config value.
        """
        self.logger = get_module_logger("memory_manager")
        self.config = config_manager.config
        self.max_memory_usage = max_memory_usage or self.config.max_memory_usage
        self.process = psutil.Process()
        self._monitoring_active = False
        self._monitor_thread = None
        self._memory_callbacks = []
        
        # Memory thresholds
        self.warning_threshold = 0.8  # 80% of max memory
        self.critical_threshold = 0.9  # 90% of max memory
        
        self.logger.info(f"MemoryManager initialized with max memory: {self.max_memory_usage / (1024**3):.2f} GB")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Process memory
        process_memory_info = self.process.memory_info()
        process_memory = process_memory_info.rss  # Resident Set Size
        
        return MemoryStats(
            total_memory=system_memory.total,
            available_memory=system_memory.available,
            used_memory=system_memory.used,
            memory_percent=system_memory.percent,
            process_memory=process_memory,
            process_memory_percent=(process_memory / system_memory.total) * 100
        )
    
    def get_optimal_chunk_size(self, file_size: int, base_chunk_size: int = None) -> int:
        """Calculate optimal chunk size based on available memory and file size.
        
        Args:
            file_size: Size of the file to be processed in bytes
            base_chunk_size: Base chunk size to adjust from
            
        Returns:
            Optimal chunk size in number of rows
        """
        if base_chunk_size is None:
            base_chunk_size = self.config.chunk_size
        
        stats = self.get_memory_stats()
        available_memory = stats.available_memory
        
        # Reserve 20% of available memory for other operations
        usable_memory = int(available_memory * 0.8)
        
        # Estimate memory per row (rough estimate: 1KB per row for typical e-commerce data)
        estimated_memory_per_row = 1024
        
        # Calculate maximum rows that can fit in memory
        max_rows_in_memory = usable_memory // estimated_memory_per_row
        
        # Adjust chunk size based on available memory
        if max_rows_in_memory < base_chunk_size:
            # Reduce chunk size if memory is limited
            optimal_chunk_size = max(1000, max_rows_in_memory // 2)  # Minimum 1000 rows
            self.logger.warning(f"Reducing chunk size from {base_chunk_size} to {optimal_chunk_size} due to memory constraints")
        elif available_memory > self.max_memory_usage * 0.5:
            # Increase chunk size if we have plenty of memory
            optimal_chunk_size = min(base_chunk_size * 2, max_rows_in_memory // 4)
            self.logger.info(f"Increasing chunk size from {base_chunk_size} to {optimal_chunk_size} due to abundant memory")
        else:
            optimal_chunk_size = base_chunk_size
        
        self.logger.debug(f"Optimal chunk size calculated: {optimal_chunk_size} rows")
        return optimal_chunk_size
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage and return status information."""
        stats = self.get_memory_stats()
        
        # Calculate usage ratios
        system_usage_ratio = stats.memory_percent / 100
        process_usage_ratio = stats.process_memory / self.max_memory_usage
        
        status = {
            'system_memory_percent': stats.memory_percent,
            'process_memory_mb': stats.process_memory / (1024**2),
            'process_memory_percent': stats.process_memory_percent,
            'available_memory_gb': stats.available_memory / (1024**3),
            'usage_ratio': process_usage_ratio,
            'status': 'normal'
        }
        
        # Determine status based on thresholds
        if process_usage_ratio >= self.critical_threshold:
            status['status'] = 'critical'
            self.logger.critical(f"Critical memory usage: {process_usage_ratio:.1%} of max allowed")
        elif process_usage_ratio >= self.warning_threshold:
            status['status'] = 'warning'
            self.logger.warning(f"High memory usage: {process_usage_ratio:.1%} of max allowed")
        
        return status
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return collection statistics."""
        self.logger.debug("Forcing garbage collection")
        
        # Get memory before GC
        before_stats = self.get_memory_stats()
        
        # Force garbage collection
        collected = {
            'generation_0': gc.collect(0),
            'generation_1': gc.collect(1),
            'generation_2': gc.collect(2)
        }
        
        # Get memory after GC
        after_stats = self.get_memory_stats()
        
        memory_freed = before_stats.process_memory - after_stats.process_memory
        
        self.logger.info(f"Garbage collection completed. Memory freed: {memory_freed / (1024**2):.2f} MB")
        
        return {
            **collected,
            'memory_freed_mb': memory_freed / (1024**2),
            'memory_before_mb': before_stats.process_memory / (1024**2),
            'memory_after_mb': after_stats.process_memory / (1024**2)
        }
    
    def create_memory_mapped_file(self, file_path: Union[str, Path], mode: str = 'r') -> mmap.mmap:
        """Create a memory-mapped file for efficient large file handling.
        
        Args:
            file_path: Path to the file
            mode: File access mode ('r' for read, 'r+' for read/write)
            
        Returns:
            Memory-mapped file object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Open file in appropriate mode
        if mode == 'r':
            file_obj = open(file_path, 'rb')
            mmap_obj = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        elif mode == 'r+':
            file_obj = open(file_path, 'r+b')
            mmap_obj = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_WRITE)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        self.logger.info(f"Created memory-mapped file: {file_path} ({file_path.stat().st_size / (1024**2):.2f} MB)")
        
        return mmap_obj
    
    @contextmanager
    def memory_checkpoint(self, operation_name: str):
        """Context manager for memory checkpoints during processing.
        
        Args:
            operation_name: Name of the operation for logging
        """
        # Record memory before operation
        before_stats = self.get_memory_stats()
        start_time = time.time()
        
        self.logger.info(f"Memory checkpoint START - {operation_name}: {before_stats.process_memory / (1024**2):.2f} MB")
        
        try:
            yield
        finally:
            # Record memory after operation
            after_stats = self.get_memory_stats()
            end_time = time.time()
            
            memory_delta = after_stats.process_memory - before_stats.process_memory
            duration = end_time - start_time
            
            self.logger.info(
                f"Memory checkpoint END - {operation_name}: "
                f"{after_stats.process_memory / (1024**2):.2f} MB "
                f"(Delta{memory_delta / (1024**2):+.2f} MB) "
                f"Duration: {duration:.2f}s"
            )
            
            # Force GC if memory usage is high
            if after_stats.process_memory / self.max_memory_usage > self.warning_threshold:
                self.force_garbage_collection()
    
    def register_memory_callback(self, callback: Callable[[Dict[str, Any]], None], threshold: float = 0.8):
        """Register a callback to be called when memory usage exceeds threshold.
        
        Args:
            callback: Function to call when threshold is exceeded
            threshold: Memory usage threshold (0.0 to 1.0)
        """
        self._memory_callbacks.append((callback, threshold))
        self.logger.debug(f"Registered memory callback with threshold: {threshold:.1%}")
    
    def start_memory_monitoring(self, interval: float = 5.0):
        """Start continuous memory monitoring in a separate thread.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring_active:
            self.logger.warning("Memory monitoring is already active")
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._memory_monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info(f"Started memory monitoring with {interval}s interval")
    
    def stop_memory_monitoring(self):
        """Stop continuous memory monitoring."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        self.logger.info("Stopped memory monitoring")
    
    def _memory_monitor_loop(self, interval: float):
        """Memory monitoring loop (runs in separate thread)."""
        while self._monitoring_active:
            try:
                status = self.check_memory_usage()
                usage_ratio = status['usage_ratio']
                
                # Check callbacks
                for callback, threshold in self._memory_callbacks:
                    if usage_ratio >= threshold:
                        try:
                            callback(status)
                        except Exception as e:
                            self.logger.error(f"Error in memory callback: {e}")
                
                # Auto garbage collection at high usage
                if usage_ratio >= self.critical_threshold:
                    self.force_garbage_collection()
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(interval)
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report."""
        stats = self.get_memory_stats()
        gc_stats = gc.get_stats()
        
        report = {
            'timestamp': time.time(),
            'system_memory': {
                'total_gb': stats.total_memory / (1024**3),
                'available_gb': stats.available_memory / (1024**3),
                'used_gb': stats.used_memory / (1024**3),
                'percent_used': stats.memory_percent
            },
            'process_memory': {
                'rss_mb': stats.process_memory / (1024**2),
                'percent_of_system': stats.process_memory_percent,
                'percent_of_limit': (stats.process_memory / self.max_memory_usage) * 100
            },
            'garbage_collection': {
                'generation_0': gc_stats[0] if gc_stats else {},
                'generation_1': gc_stats[1] if len(gc_stats) > 1 else {},
                'generation_2': gc_stats[2] if len(gc_stats) > 2 else {}
            },
            'configuration': {
                'max_memory_gb': self.max_memory_usage / (1024**3),
                'warning_threshold': self.warning_threshold,
                'critical_threshold': self.critical_threshold
            }
        }
        
        return report
    
    def optimize_for_large_dataset(self, estimated_dataset_size: int) -> Dict[str, Any]:
        """Optimize memory settings for processing large datasets.
        
        Args:
            estimated_dataset_size: Estimated dataset size in bytes
            
        Returns:
            Dictionary with optimization recommendations
        """
        stats = self.get_memory_stats()
        available_memory = stats.available_memory
        
        # Calculate optimal settings
        recommendations = {
            'original_chunk_size': self.config.chunk_size,
            'estimated_dataset_gb': estimated_dataset_size / (1024**3),
            'available_memory_gb': available_memory / (1024**3)
        }
        
        # Adjust chunk size based on dataset size and available memory
        if estimated_dataset_size > available_memory * 0.5:
            # Large dataset - use smaller chunks
            optimal_chunk_size = max(1000, self.config.chunk_size // 4)
            recommendations['recommended_chunk_size'] = optimal_chunk_size
            recommendations['strategy'] = 'small_chunks'
            recommendations['reason'] = 'Dataset is large relative to available memory'
        elif estimated_dataset_size < available_memory * 0.1:
            # Small dataset - can use larger chunks
            optimal_chunk_size = min(self.config.chunk_size * 4, 500000)
            recommendations['recommended_chunk_size'] = optimal_chunk_size
            recommendations['strategy'] = 'large_chunks'
            recommendations['reason'] = 'Dataset is small relative to available memory'
        else:
            # Medium dataset - use default settings
            recommendations['recommended_chunk_size'] = self.config.chunk_size
            recommendations['strategy'] = 'default'
            recommendations['reason'] = 'Dataset size is appropriate for default settings'
        
        # Additional recommendations
        recommendations['enable_memory_mapping'] = estimated_dataset_size > (1024**3)  # > 1GB
        recommendations['gc_frequency'] = 'high' if estimated_dataset_size > available_memory * 0.3 else 'normal'
        
        self.logger.info(f"Memory optimization recommendations: {recommendations['strategy']} strategy")
        
        return recommendations


# Global memory manager instance
memory_manager = MemoryManager()