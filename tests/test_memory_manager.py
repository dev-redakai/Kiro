"""Tests for memory management utilities."""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from src.utils.memory_manager import MemoryManager, MemoryStats


class TestMemoryManager:
    """Test cases for MemoryManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.memory_manager = MemoryManager(max_memory_usage=1024*1024*1024)  # 1GB for testing
    
    def test_initialization(self):
        """Test MemoryManager initialization."""
        assert self.memory_manager.max_memory_usage == 1024*1024*1024
        assert self.memory_manager.warning_threshold == 0.8
        assert self.memory_manager.critical_threshold == 0.9
        assert not self.memory_manager._monitoring_active
    
    def test_get_memory_stats(self):
        """Test memory statistics retrieval."""
        stats = self.memory_manager.get_memory_stats()
        
        assert isinstance(stats, MemoryStats)
        assert stats.total_memory > 0
        assert stats.available_memory > 0
        assert stats.used_memory > 0
        assert 0 <= stats.memory_percent <= 100
        assert stats.process_memory > 0
        assert stats.process_memory_percent >= 0
    
    def test_get_optimal_chunk_size(self):
        """Test optimal chunk size calculation."""
        # Test with small file
        small_file_size = 1024 * 1024  # 1MB
        chunk_size = self.memory_manager.get_optimal_chunk_size(small_file_size, 10000)
        assert chunk_size > 0
        
        # Test with large file
        large_file_size = 10 * 1024 * 1024 * 1024  # 10GB
        chunk_size = self.memory_manager.get_optimal_chunk_size(large_file_size, 10000)
        assert chunk_size > 0
    
    def test_check_memory_usage(self):
        """Test memory usage checking."""
        status = self.memory_manager.check_memory_usage()
        
        assert 'system_memory_percent' in status
        assert 'process_memory_mb' in status
        assert 'process_memory_percent' in status
        assert 'available_memory_gb' in status
        assert 'usage_ratio' in status
        assert 'status' in status
        assert status['status'] in ['normal', 'warning', 'critical']
    
    def test_force_garbage_collection(self):
        """Test garbage collection functionality."""
        # Create some objects to collect
        test_data = [list(range(1000)) for _ in range(100)]
        del test_data
        
        result = self.memory_manager.force_garbage_collection()
        
        assert 'generation_0' in result
        assert 'generation_1' in result
        assert 'generation_2' in result
        assert 'memory_freed_mb' in result
        assert 'memory_before_mb' in result
        assert 'memory_after_mb' in result
    
    def test_memory_checkpoint(self):
        """Test memory checkpoint context manager."""
        with self.memory_manager.memory_checkpoint("test_operation"):
            # Simulate some memory usage
            test_data = [i for i in range(10000)]
            assert len(test_data) == 10000
    
    def test_create_memory_mapped_file(self):
        """Test memory-mapped file creation."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"Hello, World!" * 1000)
            temp_file_path = temp_file.name
        
        try:
            # Test read-only memory mapping
            mmap_obj = self.memory_manager.create_memory_mapped_file(temp_file_path, 'r')
            assert mmap_obj is not None
            assert len(mmap_obj) > 0
            mmap_obj.close()
            
        finally:
            # Clean up
            Path(temp_file_path).unlink()
    
    def test_create_memory_mapped_file_nonexistent(self):
        """Test memory-mapped file creation with non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.memory_manager.create_memory_mapped_file("nonexistent_file.txt")
    
    def test_register_memory_callback(self):
        """Test memory callback registration."""
        callback_called = False
        
        def test_callback(status):
            nonlocal callback_called
            callback_called = True
        
        self.memory_manager.register_memory_callback(test_callback, 0.5)
        assert len(self.memory_manager._memory_callbacks) == 1
        assert self.memory_manager._memory_callbacks[0][1] == 0.5
    
    def test_memory_monitoring_start_stop(self):
        """Test memory monitoring start and stop."""
        # Start monitoring
        self.memory_manager.start_memory_monitoring(0.1)  # Very short interval for testing
        assert self.memory_manager._monitoring_active
        assert self.memory_manager._monitor_thread is not None
        
        # Wait a bit to let monitoring run
        time.sleep(0.2)
        
        # Stop monitoring
        self.memory_manager.stop_memory_monitoring()
        assert not self.memory_manager._monitoring_active
    
    def test_get_memory_report(self):
        """Test memory report generation."""
        report = self.memory_manager.get_memory_report()
        
        assert 'timestamp' in report
        assert 'system_memory' in report
        assert 'process_memory' in report
        assert 'garbage_collection' in report
        assert 'configuration' in report
        
        # Check system memory section
        sys_mem = report['system_memory']
        assert 'total_gb' in sys_mem
        assert 'available_gb' in sys_mem
        assert 'used_gb' in sys_mem
        assert 'percent_used' in sys_mem
        
        # Check process memory section
        proc_mem = report['process_memory']
        assert 'rss_mb' in proc_mem
        assert 'percent_of_system' in proc_mem
        assert 'percent_of_limit' in proc_mem
    
    def test_optimize_for_large_dataset(self):
        """Test dataset optimization recommendations."""
        # Test with small dataset
        small_dataset_size = 100 * 1024 * 1024  # 100MB
        recommendations = self.memory_manager.optimize_for_large_dataset(small_dataset_size)
        
        assert 'original_chunk_size' in recommendations
        assert 'estimated_dataset_gb' in recommendations
        assert 'available_memory_gb' in recommendations
        assert 'recommended_chunk_size' in recommendations
        assert 'strategy' in recommendations
        assert 'reason' in recommendations
        assert 'enable_memory_mapping' in recommendations
        assert 'gc_frequency' in recommendations
        
        # Test with large dataset
        large_dataset_size = 50 * 1024 * 1024 * 1024  # 50GB
        recommendations = self.memory_manager.optimize_for_large_dataset(large_dataset_size)
        
        assert recommendations['strategy'] in ['small_chunks', 'large_chunks', 'default']
        assert recommendations['enable_memory_mapping'] is True  # Should be True for large datasets
    
    def test_memory_stats_dataclass(self):
        """Test MemoryStats dataclass."""
        stats = MemoryStats(
            total_memory=8*1024*1024*1024,
            available_memory=4*1024*1024*1024,
            used_memory=4*1024*1024*1024,
            memory_percent=50.0,
            process_memory=100*1024*1024,
            process_memory_percent=1.25
        )
        
        assert stats.total_memory == 8*1024*1024*1024
        assert stats.available_memory == 4*1024*1024*1024
        assert stats.used_memory == 4*1024*1024*1024
        assert stats.memory_percent == 50.0
        assert stats.process_memory == 100*1024*1024
        assert stats.process_memory_percent == 1.25


class TestMemoryManagerIntegration:
    """Integration tests for MemoryManager."""
    
    def test_memory_manager_with_callback(self):
        """Test memory manager with callback integration."""
        memory_manager = MemoryManager(max_memory_usage=100*1024*1024)  # 100MB limit for testing
        
        callback_status = None
        
        def memory_callback(status):
            nonlocal callback_status
            callback_status = status
        
        # Register callback with very low threshold to trigger it
        memory_manager.register_memory_callback(memory_callback, 0.01)
        
        # Start monitoring briefly
        memory_manager.start_memory_monitoring(0.1)
        time.sleep(0.2)  # Let it run briefly
        memory_manager.stop_memory_monitoring()
        
        # Callback should have been triggered due to low threshold
        assert callback_status is not None
        assert 'usage_ratio' in callback_status
    
    @patch('psutil.virtual_memory')
    @patch('psutil.Process')
    def test_memory_manager_with_mocked_psutil(self, mock_process_class, mock_virtual_memory):
        """Test MemoryManager with mocked psutil for controlled testing."""
        # Mock virtual memory
        mock_vm = Mock()
        mock_vm.total = 8 * 1024 * 1024 * 1024  # 8GB
        mock_vm.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_vm.used = 4 * 1024 * 1024 * 1024  # 4GB
        mock_vm.percent = 50.0
        mock_virtual_memory.return_value = mock_vm
        
        # Mock process
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process
        
        # Create memory manager with mocked psutil
        memory_manager = MemoryManager(max_memory_usage=1024*1024*1024)
        
        # Test get_memory_stats with mocked data
        stats = memory_manager.get_memory_stats()
        
        assert stats.total_memory == 8 * 1024 * 1024 * 1024
        assert stats.available_memory == 4 * 1024 * 1024 * 1024
        assert stats.used_memory == 4 * 1024 * 1024 * 1024
        assert stats.memory_percent == 50.0
        assert stats.process_memory == 100 * 1024 * 1024