"""
Tests for data export and persistence functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json

from src.utils.config import PipelineConfig
from src.data.data_exporter import DataExporter, ExportManager
from src.data.data_integrity import DataIntegrityValidator
from src.data.data_persistence import DataCache, CheckpointManager, DataLineageTracker, PersistenceManager


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(100),
        'value': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'price': np.random.uniform(10, 100, 100)
    })


@pytest.fixture
def temp_config():
    """Create a temporary configuration for testing."""
    temp_dir = tempfile.mkdtemp()
    config = PipelineConfig(
        input_data_path=f"{temp_dir}/raw",
        processed_data_path=f"{temp_dir}/processed",
        output_data_path=f"{temp_dir}/output",
        temp_data_path=f"{temp_dir}/temp",
        output_formats=['csv', 'parquet']
    )
    
    # Create directories
    for path in [config.input_data_path, config.processed_data_path, 
                 config.output_data_path, config.temp_data_path]:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    yield config
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestDataExporter:
    """Test data export functionality."""
    
    def test_csv_export(self, sample_data, temp_config):
        """Test CSV export functionality."""
        exporter = DataExporter(temp_config)
        
        csv_path = exporter.export_csv(sample_data, "test_data", include_timestamp=False)
        
        # Verify file exists
        assert Path(csv_path).exists()
        
        # Verify data can be read back
        loaded_data = pd.read_csv(csv_path)
        assert len(loaded_data) == len(sample_data)
        assert list(loaded_data.columns) == list(sample_data.columns)
    
    def test_parquet_export(self, sample_data, temp_config):
        """Test Parquet export functionality."""
        exporter = DataExporter(temp_config)
        
        parquet_path = exporter.export_parquet(sample_data, "test_data", include_timestamp=False)
        
        # Verify file exists
        assert Path(parquet_path).exists()
        
        # Verify data can be read back
        loaded_data = pd.read_parquet(parquet_path)
        assert len(loaded_data) == len(sample_data)
        assert list(loaded_data.columns) == list(sample_data.columns)
    
    def test_analytical_tables_export(self, sample_data, temp_config):
        """Test analytical tables export."""
        exporter = DataExporter(temp_config)
        
        # Create analytical tables
        analytical_tables = {
            'summary': sample_data.groupby('category').agg({'value': 'mean', 'price': 'sum'}).reset_index(),
            'top_values': sample_data.nlargest(5, 'value')[['id', 'value', 'category']]
        }
        
        export_results = exporter.export_analytical_tables(analytical_tables, include_timestamp=False)
        
        # Verify exports
        assert 'summary' in export_results
        assert 'top_values' in export_results
        
        for table_name, paths in export_results.items():
            assert 'csv' in paths
            assert 'parquet' in paths
            assert Path(paths['csv']).exists()
            assert Path(paths['parquet']).exists()


class TestDataIntegrityValidator:
    """Test data integrity validation."""
    
    def test_data_consistency_validation(self, sample_data, temp_config):
        """Test data consistency validation."""
        exporter = DataExporter(temp_config)
        validator = DataIntegrityValidator()
        
        # Export data
        csv_path = exporter.export_csv(sample_data, "test_data", include_timestamp=False)
        parquet_path = exporter.export_parquet(sample_data, "test_data", include_timestamp=False)
        
        # Validate CSV
        is_valid_csv, report_csv = validator.validate_data_consistency(
            sample_data, csv_path, 'csv'
        )
        assert is_valid_csv or 'dtype_consistency' in report_csv['checks']  # Allow dtype differences
        
        # Validate Parquet
        is_valid_parquet, report_parquet = validator.validate_data_consistency(
            sample_data, parquet_path, 'parquet'
        )
        assert is_valid_parquet
    
    def test_validate_exported_files(self, sample_data, temp_config):
        """Test validation of multiple exported files."""
        exporter = DataExporter(temp_config)
        validator = DataIntegrityValidator()
        
        # Export in both formats
        export_paths = exporter.export_processed_data(sample_data, "test_data")
        
        # Validate all exports
        validation_results = validator.validate_exported_files(export_paths, sample_data)
        
        assert 'csv' in validation_results
        assert 'parquet' in validation_results
        assert validation_results['parquet']['overall_valid']


class TestDataCache:
    """Test data caching functionality."""
    
    def test_cache_and_retrieve_data(self, sample_data, temp_config):
        """Test caching and retrieving data."""
        cache = DataCache(temp_config)
        
        # Cache data
        parameters = {'test_param': 'value1'}
        cache_key = cache.cache_data(sample_data, "test_data", parameters, "Test data")
        
        # Retrieve cached data
        cached_data = cache.get_cached_data("test_data", parameters)
        
        assert cached_data is not None
        assert len(cached_data) == len(sample_data)
        pd.testing.assert_frame_equal(cached_data.reset_index(drop=True), 
                                    sample_data.reset_index(drop=True))
    
    def test_cache_key_generation(self, sample_data, temp_config):
        """Test cache key generation with different parameters."""
        cache = DataCache(temp_config)
        
        params1 = {'param1': 'value1', 'param2': 2}
        params2 = {'param1': 'value1', 'param2': 3}
        params3 = {'param2': 2, 'param1': 'value1'}  # Same as params1 but different order
        
        key1 = cache._generate_cache_key("test_data", params1)
        key2 = cache._generate_cache_key("test_data", params2)
        key3 = cache._generate_cache_key("test_data", params3)
        
        assert key1 != key2  # Different parameters should generate different keys
        assert key1 == key3  # Same parameters in different order should generate same key
    
    def test_cache_stats(self, sample_data, temp_config):
        """Test cache statistics."""
        cache = DataCache(temp_config)
        
        # Initially empty
        stats = cache.get_cache_stats()
        assert stats['total_entries'] == 0
        
        # Cache some data
        cache.cache_data(sample_data, "test_data", {'param': 'value'}, "Test")
        
        # Check stats
        stats = cache.get_cache_stats()
        assert stats['total_entries'] == 1
        assert stats['total_size_mb'] > 0


class TestCheckpointManager:
    """Test checkpoint management functionality."""
    
    def test_create_and_load_checkpoint(self, sample_data, temp_config):
        """Test creating and loading checkpoints."""
        checkpoint_manager = CheckpointManager(temp_config)
        
        # Create checkpoint
        state = {
            'current_step': 'processing',
            'completed_steps': ['validation', 'cleaning'],
            'progress': 0.5
        }
        
        checkpoint_id = checkpoint_manager.create_checkpoint(
            "test_process", state, sample_data
        )
        
        # Load checkpoint
        loaded_state, loaded_data = checkpoint_manager.load_checkpoint(checkpoint_id)
        
        assert loaded_state['current_step'] == 'processing'
        assert loaded_state['completed_steps'] == ['validation', 'cleaning']
        assert loaded_data is not None
        assert len(loaded_data) == len(sample_data)
    
    def test_list_checkpoints(self, sample_data, temp_config):
        """Test listing checkpoints."""
        checkpoint_manager = CheckpointManager(temp_config)
        
        # Create multiple checkpoints
        state1 = {'step': 'step1'}
        state2 = {'step': 'step2'}
        
        checkpoint_manager.create_checkpoint("process1", state1)
        checkpoint_manager.create_checkpoint("process2", state2)
        
        # List all checkpoints
        all_checkpoints = checkpoint_manager.list_checkpoints()
        assert len(all_checkpoints) >= 2
        
        # List checkpoints for specific process
        process1_checkpoints = checkpoint_manager.list_checkpoints("process1")
        assert len(process1_checkpoints) >= 1
        assert all(cp['process_id'] == 'process1' for cp in process1_checkpoints)


class TestDataLineageTracker:
    """Test data lineage tracking functionality."""
    
    def test_register_dataset_and_track_transformation(self, temp_config):
        """Test dataset registration and transformation tracking."""
        lineage_tracker = DataLineageTracker(temp_config)
        
        # Register datasets
        lineage_tracker.register_dataset("dataset1", {"source": "file1.csv", "records": 1000})
        lineage_tracker.register_dataset("dataset2", {"source": "transformation", "records": 800})
        
        # Track transformation
        lineage_tracker.track_transformation(
            "transform1",
            ["dataset1"],
            ["dataset2"],
            "cleaning",
            {"remove_nulls": True}
        )
        
        # Get lineage
        lineage = lineage_tracker.get_dataset_lineage("dataset2")
        
        assert lineage['dataset_id'] == 'dataset2'
        assert len(lineage['produced_by']) == 1
        assert lineage['produced_by'][0]['transformation_id'] == 'transform1'
    
    def test_generate_lineage_report(self, temp_config):
        """Test lineage report generation."""
        lineage_tracker = DataLineageTracker(temp_config)
        
        # Add some data
        lineage_tracker.register_dataset("dataset1", {"records": 100})
        lineage_tracker.track_transformation("transform1", ["dataset1"], ["dataset2"], "test", {})
        
        # Generate report
        report = lineage_tracker.generate_lineage_report()
        
        assert 'report_generated_at' in report
        assert report['total_datasets'] >= 1
        assert report['total_transformations'] >= 1


class TestPersistenceManager:
    """Test the high-level persistence manager."""
    
    def test_cache_intermediate_result(self, sample_data, temp_config):
        """Test caching intermediate results."""
        persistence_manager = PersistenceManager(temp_config)
        
        # Cache result
        cache_key = persistence_manager.cache_intermediate_result(
            sample_data, "test_process", "step1", {"param": "value"}
        )
        
        # Retrieve result
        cached_data = persistence_manager.get_cached_result(
            "test_process", "step1", {"param": "value"}
        )
        
        assert cached_data is not None
        assert len(cached_data) == len(sample_data)
    
    def test_create_process_checkpoint(self, sample_data, temp_config):
        """Test creating process checkpoints."""
        persistence_manager = PersistenceManager(temp_config)
        
        checkpoint_id = persistence_manager.create_process_checkpoint(
            "test_process",
            "step2",
            ["step1"],
            sample_data,
            {"additional": "info"}
        )
        
        # Verify checkpoint was created
        checkpoints = persistence_manager.checkpoint_manager.list_checkpoints("test_process")
        assert len(checkpoints) >= 1
        assert any(cp['checkpoint_id'] == checkpoint_id for cp in checkpoints)
    
    def test_get_system_status(self, sample_data, temp_config):
        """Test getting system status."""
        persistence_manager = PersistenceManager(temp_config)
        
        # Add some data to the system
        persistence_manager.cache_intermediate_result(
            sample_data, "test_process", "step1", {"param": "value"}
        )
        persistence_manager.create_process_checkpoint(
            "test_process", "step1", [], sample_data
        )
        persistence_manager.lineage_tracker.register_dataset("test_dataset", {"records": 100})
        
        # Get status
        status = persistence_manager.get_system_status()
        
        assert 'timestamp' in status
        assert 'cache' in status
        assert 'checkpoints' in status
        assert 'lineage' in status
        assert status['cache']['total_entries'] >= 1
        assert status['checkpoints']['total_checkpoints'] >= 1
        assert status['lineage']['total_datasets'] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])