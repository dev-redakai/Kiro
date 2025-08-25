"""Tests for the error handling system."""

import pytest
import time
from unittest.mock import patch, MagicMock

from src.utils.error_handler import (
    ErrorHandler, DataQualityError, PipelineMemoryError, ProcessingError, DashboardError,
    ErrorCategory, ErrorSeverity, error_handler
)


class TestErrorHandler:
    """Test cases for the ErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = ErrorHandler()
        self.handler.clear_error_history()
    
    def test_data_quality_error_handling(self):
        """Test data quality error handling with correction."""
        error = DataQualityError(
            "Invalid quantity",
            field="quantity",
            value="-5",
            correction=0
        )
        
        result = self.handler.handle_data_quality_error(error)
        
        # Should return the correction value
        assert result == 0
        
        # Should record the error
        assert len(self.handler.error_history) == 1
        error_record = self.handler.error_history[0]
        assert error_record.category == ErrorCategory.DATA_QUALITY
        assert error_record.resolved == True
    
    def test_data_quality_error_skip_record(self):
        """Test data quality error handling with record skipping."""
        error = DataQualityError(
            "Invalid email format",
            field="email",
            value="invalid-email"
        )
        
        result = self.handler.handle_data_quality_error(error)
        
        # Should skip the record
        assert result == "SKIP_RECORD"
        
        # Should record the error
        assert len(self.handler.error_history) == 1
    
    def test_data_quality_error_default_value(self):
        """Test data quality error handling with default values."""
        error = DataQualityError(
            "Missing product name",
            field="product_name",
            value=None
        )
        
        result = self.handler.handle_data_quality_error(error)
        
        # Should return default value
        assert result == "Unknown Product"
    
    def test_memory_error_handling(self):
        """Test memory error handling with chunk size reduction."""
        error = PipelineMemoryError(
            "Out of memory",
            current_usage=7.5,
            max_usage=8.0
        )
        
        context_chunk_size = 100000
        result = self.handler.handle_memory_error(error, {'chunk_size': context_chunk_size})
        
        # Should reduce chunk size
        assert result is not None
        assert 'new_chunk_size' in result
        assert result['new_chunk_size'] < context_chunk_size
        assert result['new_chunk_size'] == 50000  # Should be half of 100000
        
        # Should record the error
        assert len(self.handler.error_history) == 1
        error_record = self.handler.error_history[0]
        assert error_record.category == ErrorCategory.MEMORY
    
    @patch('gc.collect')
    def test_memory_error_garbage_collection(self, mock_gc):
        """Test memory error handling with garbage collection."""
        mock_gc.return_value = 150  # Mock collected objects
        
        error = PipelineMemoryError("Memory full")
        
        # Force garbage collection strategy by making chunk size very small
        with patch.object(self.handler, '_reduce_chunk_size', return_value=None):
            result = self.handler.handle_memory_error(error)
        
        # Should trigger garbage collection
        mock_gc.assert_called_once()
        assert result is not None
        assert 'objects_collected' in result
        assert result['objects_collected'] == 150
    
    def test_processing_error_retry(self):
        """Test processing error handling with retry."""
        error = ProcessingError(
            "Transformation failed",
            operation="data_transformation"
        )
        
        result = self.handler.handle_processing_error(error, {'retry_count': 0})
        
        # Should suggest retry
        assert result is not None
        assert result.get('retry') == True
        assert result.get('retry_count') == 1
    
    def test_processing_error_skip_chunk(self):
        """Test processing error handling with chunk skipping."""
        error = ProcessingError(
            "Corrupted data chunk",
            operation="data_cleaning"
        )
        
        # Simulate max retries reached
        with patch.object(self.handler, '_retry_operation', return_value=None):
            result = self.handler.handle_processing_error(error)
        
        # Should skip chunk
        assert result is not None
        assert result.get('skip_chunk') == True
    
    def test_dashboard_error_cached_data(self):
        """Test dashboard error handling with cached data."""
        error = DashboardError(
            "Failed to load fresh data",
            component="revenue_chart",
            fallback_available=True
        )
        
        result = self.handler.handle_dashboard_error(error)
        
        # Should use cached data
        assert result is not None
        assert result.get('use_cache') == True
    
    def test_dashboard_error_fallback_visualization(self):
        """Test dashboard error handling with fallback visualization."""
        error = DashboardError(
            "Complex chart rendering failed",
            component="interactive_map"
        )
        
        # Force fallback visualization strategy
        with patch.object(self.handler, '_use_cached_data', return_value=None):
            with patch.object(self.handler, '_show_error_message', return_value=None):
                result = self.handler.handle_dashboard_error(error)
        
        # Should use fallback visualization
        assert result is not None
        assert result.get('use_fallback_viz') == True
    
    def test_error_summary(self):
        """Test error summary generation."""
        # Generate some test errors
        errors = [
            DataQualityError("Invalid data", field="test"),
            PipelineMemoryError("Out of memory"),
            ProcessingError("Processing failed", operation="test")
        ]
        
        for error in errors:
            self.handler.handle_error(
                error, 
                ErrorCategory.DATA_QUALITY if isinstance(error, DataQualityError)
                else ErrorCategory.MEMORY if isinstance(error, PipelineMemoryError)
                else ErrorCategory.PROCESSING,
                ErrorSeverity.MEDIUM
            )
        
        summary = self.handler.get_error_summary()
        
        assert summary['total_errors'] == 3
        assert 'by_category' in summary
        assert 'by_severity' in summary
        assert 'recent_errors' in summary
        assert len(summary['recent_errors']) == 3
    
    def test_error_history_management(self):
        """Test error history management."""
        # Add some errors
        for i in range(5):
            error = DataQualityError(f"Test error {i}")
            self.handler.handle_data_quality_error(error)
        
        assert len(self.handler.error_history) == 5
        
        # Clear history
        self.handler.clear_error_history()
        assert len(self.handler.error_history) == 0
    
    def test_error_export(self, tmp_path):
        """Test error log export."""
        # Add some test errors
        error = DataQualityError("Test export error", field="test")
        self.handler.handle_data_quality_error(error)
        
        # Export to temporary file
        export_path = tmp_path / "test_errors.json"
        self.handler.export_error_log(str(export_path))
        
        # Verify file was created
        assert export_path.exists()
        
        # Verify content
        import json
        with open(export_path) as f:
            exported_data = json.load(f)
        
        assert len(exported_data) == 1
        assert exported_data[0]['error_type'] == 'DataQualityError'
        assert exported_data[0]['message'] == 'Test export error'
    
    def test_critical_error_reraise(self):
        """Test that critical errors are re-raised."""
        error = ProcessingError("Critical system failure")
        
        with pytest.raises(ProcessingError):
            self.handler.handle_error(error, ErrorCategory.PROCESSING, ErrorSeverity.CRITICAL)
    
    def test_error_context_preservation(self):
        """Test that error context is preserved."""
        error = DataQualityError("Test context", field="test_field")
        context = {
            'record_id': 123,
            'batch_id': 'batch_001',
            'processing_stage': 'validation'
        }
        
        self.handler.handle_data_quality_error(error, context)
        
        error_record = self.handler.error_history[0]
        assert error_record.details['record_id'] == 123
        assert error_record.details['batch_id'] == 'batch_001'
        assert error_record.details['processing_stage'] == 'validation'
        assert error_record.details['field'] == 'test_field'


class TestGlobalErrorHandler:
    """Test the global error handler instance."""
    
    def test_global_error_handler_functions(self):
        """Test global error handler convenience functions."""
        from src.utils.error_handler import (
            handle_data_quality_error, handle_memory_error,
            handle_processing_error, handle_dashboard_error
        )
        
        # Test data quality error
        error = DataQualityError("Global test", field="quantity", correction=5)
        result = handle_data_quality_error(error)
        assert result == 5
        
        # Test memory error
        error = PipelineMemoryError("Global memory test")
        result = handle_memory_error(error)
        assert result is not None
        
        # Test processing error
        error = ProcessingError("Global processing test")
        result = handle_processing_error(error)
        assert result is not None
        
        # Test dashboard error
        error = DashboardError("Global dashboard test")
        result = handle_dashboard_error(error)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])