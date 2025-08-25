"""
Data Persistence and Caching Module

This module provides functionality for caching intermediate results,
checkpoint management, and data lineage tracking.
"""

import os
import pickle
import json
import hashlib
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import shutil

from ..utils.config import PipelineConfig
from ..utils.logger import get_logger


class DataCache:
    """
    Manages caching of intermediate processing results for large datasets.
    """
    
    def __init__(self, config: PipelineConfig, cache_dir: Optional[str] = None):
        """
        Initialize the DataCache.
        
        Args:
            config: Pipeline configuration
            cache_dir: Custom cache directory path
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.cache_dir = Path(cache_dir) if cache_dir else Path(config.temp_data_path) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata file
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_metadata = self._load_cache_metadata()
        
        # Default cache settings
        self.max_cache_size_gb = 2.0  # Maximum cache size in GB
        self.cache_ttl_hours = 24     # Time to live in hours
        
    def _load_cache_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from file."""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load cache metadata: {e}")
        
        return {"entries": {}, "total_size": 0}
    
    def _save_cache_metadata(self):
        """Save cache metadata to file."""
        try:
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Could not save cache metadata: {e}")
    
    def _generate_cache_key(self, data_identifier: str, parameters: Dict[str, Any]) -> str:
        """
        Generate a unique cache key based on data identifier and parameters.
        
        Args:
            data_identifier: Unique identifier for the data
            parameters: Processing parameters that affect the result
            
        Returns:
            Cache key string
        """
        # Create a string representation of parameters
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        combined_str = f"{data_identifier}_{param_str}"
        
        # Generate hash
        return hashlib.md5(combined_str.encode()).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str, file_format: str = 'parquet') -> Path:
        """Get the file path for a cache entry."""
        return self.cache_dir / f"{cache_key}.{file_format}"
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        if file_path.exists():
            return file_path.stat().st_size / (1024 * 1024)
        return 0.0
    
    def _cleanup_expired_cache(self):
        """Remove expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for cache_key, metadata in self.cache_metadata["entries"].items():
            created_time = datetime.fromisoformat(metadata["created_at"])
            if current_time - created_time > timedelta(hours=self.cache_ttl_hours):
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            self._remove_cache_entry(cache_key)
        
        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _cleanup_oversized_cache(self):
        """Remove oldest cache entries if cache size exceeds limit."""
        total_size_gb = self.cache_metadata["total_size"] / (1024 * 1024 * 1024)
        
        if total_size_gb <= self.max_cache_size_gb:
            return
        
        # Sort entries by creation time (oldest first)
        entries = list(self.cache_metadata["entries"].items())
        entries.sort(key=lambda x: x[1]["created_at"])
        
        # Remove oldest entries until under size limit
        removed_count = 0
        for cache_key, metadata in entries:
            self._remove_cache_entry(cache_key)
            removed_count += 1
            
            total_size_gb = self.cache_metadata["total_size"] / (1024 * 1024 * 1024)
            if total_size_gb <= self.max_cache_size_gb * 0.8:  # Leave some buffer
                break
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} cache entries due to size limit")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove a cache entry and its files."""
        if cache_key not in self.cache_metadata["entries"]:
            return
        
        metadata = self.cache_metadata["entries"][cache_key]
        
        # Remove files
        for file_format in metadata.get("formats", []):
            cache_file = self._get_cache_file_path(cache_key, file_format)
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Could not remove cache file {cache_file}: {e}")
        
        # Update metadata
        self.cache_metadata["total_size"] -= metadata.get("size_bytes", 0)
        del self.cache_metadata["entries"][cache_key]
    
    def cache_data(self, data: pd.DataFrame, data_identifier: str,
                   parameters: Dict[str, Any], description: str = "") -> str:
        """
        Cache a DataFrame with associated metadata.
        
        Args:
            data: DataFrame to cache
            data_identifier: Unique identifier for the data
            parameters: Processing parameters
            description: Optional description of the cached data
            
        Returns:
            Cache key for the stored data
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(data_identifier, parameters)
            
            # Check if already cached
            if cache_key in self.cache_metadata["entries"]:
                self.logger.info(f"Data already cached with key: {cache_key}")
                return cache_key
            
            # Clean up cache if needed
            self._cleanup_expired_cache()
            self._cleanup_oversized_cache()
            
            # Save data in Parquet format for efficiency
            cache_file = self._get_cache_file_path(cache_key, 'parquet')
            data.to_parquet(cache_file, compression='snappy')
            
            # Calculate file size
            file_size = cache_file.stat().st_size
            
            # Update metadata
            cache_metadata = {
                "cache_key": cache_key,
                "data_identifier": data_identifier,
                "parameters": parameters,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "data_shape": list(data.shape),
                "columns": list(data.columns),
                "size_bytes": file_size,
                "formats": ["parquet"]
            }
            
            self.cache_metadata["entries"][cache_key] = cache_metadata
            self.cache_metadata["total_size"] += file_size
            
            self._save_cache_metadata()
            
            self.logger.info(f"Cached data with key: {cache_key} (size: {file_size / 1024 / 1024:.2f} MB)")
            return cache_key
            
        except Exception as e:
            self.logger.error(f"Failed to cache data: {e}")
            raise
    
    def get_cached_data(self, data_identifier: str, parameters: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data if available and not expired.
        
        Args:
            data_identifier: Unique identifier for the data
            parameters: Processing parameters
            
        Returns:
            Cached DataFrame or None if not found/expired
        """
        try:
            cache_key = self._generate_cache_key(data_identifier, parameters)
            
            if cache_key not in self.cache_metadata["entries"]:
                return None
            
            metadata = self.cache_metadata["entries"][cache_key]
            
            # Check if expired
            created_time = datetime.fromisoformat(metadata["created_at"])
            if datetime.now() - created_time > timedelta(hours=self.cache_ttl_hours):
                self._remove_cache_entry(cache_key)
                return None
            
            # Load data
            cache_file = self._get_cache_file_path(cache_key, 'parquet')
            if not cache_file.exists():
                self.logger.warning(f"Cache file missing: {cache_file}")
                self._remove_cache_entry(cache_key)
                return None
            
            data = pd.read_parquet(cache_file)
            self.logger.info(f"Retrieved cached data with key: {cache_key}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve cached data: {e}")
            return None
    
    def clear_cache(self, data_identifier: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            data_identifier: If provided, only clear entries for this identifier
        """
        if data_identifier is None:
            # Clear all cache
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_metadata = {"entries": {}, "total_size": 0}
            self._save_cache_metadata()
            self.logger.info("Cleared all cache entries")
        else:
            # Clear specific identifier
            keys_to_remove = [
                key for key, metadata in self.cache_metadata["entries"].items()
                if metadata["data_identifier"] == data_identifier
            ]
            
            for key in keys_to_remove:
                self._remove_cache_entry(key)
            
            self._save_cache_metadata()
            self.logger.info(f"Cleared {len(keys_to_remove)} cache entries for {data_identifier}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size_mb = self.cache_metadata["total_size"] / (1024 * 1024)
        
        return {
            "total_entries": len(self.cache_metadata["entries"]),
            "total_size_mb": round(total_size_mb, 2),
            "cache_directory": str(self.cache_dir),
            "max_size_gb": self.max_cache_size_gb,
            "ttl_hours": self.cache_ttl_hours
        }


class CheckpointManager:
    """
    Manages checkpoints for long-running processes to enable recovery.
    """
    
    def __init__(self, config: PipelineConfig, checkpoint_dir: Optional[str] = None):
        """
        Initialize the CheckpointManager.
        
        Args:
            config: Pipeline configuration
            checkpoint_dir: Custom checkpoint directory path
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(config.temp_data_path) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def create_checkpoint(self, process_id: str, state: Dict[str, Any],
                         data: Optional[pd.DataFrame] = None) -> str:
        """
        Create a checkpoint for a process.
        
        Args:
            process_id: Unique identifier for the process
            state: Process state information
            data: Optional DataFrame to checkpoint
            
        Returns:
            Checkpoint ID
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_id = f"{process_id}_{timestamp}"
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            checkpoint_path.mkdir(exist_ok=True)
            
            # Save state information
            state_file = checkpoint_path / "state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            # Save data if provided
            if data is not None:
                data_file = checkpoint_path / "data.parquet"
                data.to_parquet(data_file, compression='snappy')
            
            # Create checkpoint metadata
            metadata = {
                "checkpoint_id": checkpoint_id,
                "process_id": process_id,
                "created_at": datetime.now().isoformat(),
                "state": state,
                "has_data": data is not None,
                "data_shape": list(data.shape) if data is not None else None
            }
            
            metadata_file = checkpoint_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"Created checkpoint: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_id: str) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint identifier
            
        Returns:
            Tuple of (state, data)
        """
        try:
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
            
            # Load state
            state_file = checkpoint_path / "state.json"
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Load data if available
            data = None
            data_file = checkpoint_path / "data.parquet"
            if data_file.exists():
                data = pd.read_parquet(data_file)
            
            self.logger.info(f"Loaded checkpoint: {checkpoint_id}")
            return state, data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def list_checkpoints(self, process_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available checkpoints.
        
        Args:
            process_id: Filter by process ID
            
        Returns:
            List of checkpoint metadata
        """
        checkpoints = []
        
        for checkpoint_path in self.checkpoint_dir.iterdir():
            if not checkpoint_path.is_dir():
                continue
            
            metadata_file = checkpoint_path / "metadata.json"
            if not metadata_file.exists():
                continue
            
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                if process_id is None or metadata.get("process_id") == process_id:
                    checkpoints.append(metadata)
                    
            except Exception as e:
                self.logger.warning(f"Could not read checkpoint metadata: {e}")
        
        return sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5, older_than_hours: int = 48):
        """
        Clean up old checkpoints.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep per process
            older_than_hours: Remove checkpoints older than this many hours
        """
        try:
            checkpoints = self.list_checkpoints()
            
            # Group by process_id
            process_checkpoints = {}
            for checkpoint in checkpoints:
                process_id = checkpoint["process_id"]
                if process_id not in process_checkpoints:
                    process_checkpoints[process_id] = []
                process_checkpoints[process_id].append(checkpoint)
            
            removed_count = 0
            current_time = datetime.now()
            
            for process_id, proc_checkpoints in process_checkpoints.items():
                # Sort by creation time (newest first)
                proc_checkpoints.sort(key=lambda x: x["created_at"], reverse=True)
                
                # Keep the most recent ones
                to_remove = proc_checkpoints[keep_last_n:]
                
                # Also remove old ones regardless of count
                for checkpoint in proc_checkpoints:
                    created_time = datetime.fromisoformat(checkpoint["created_at"])
                    if current_time - created_time > timedelta(hours=older_than_hours):
                        if checkpoint not in to_remove:
                            to_remove.append(checkpoint)
                
                # Remove checkpoints
                for checkpoint in to_remove:
                    checkpoint_path = self.checkpoint_dir / checkpoint["checkpoint_id"]
                    if checkpoint_path.exists():
                        shutil.rmtree(checkpoint_path)
                        removed_count += 1
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old checkpoints")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup checkpoints: {e}")


class DataLineageTracker:
    """
    Tracks data lineage and transformation history.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the DataLineageTracker.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.lineage_file = Path(config.output_data_path) / "data_lineage.json"
        self.lineage_data = self._load_lineage_data()
    
    def _load_lineage_data(self) -> Dict[str, Any]:
        """Load lineage data from file."""
        if self.lineage_file.exists():
            try:
                with open(self.lineage_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load lineage data: {e}")
        
        return {"datasets": {}, "transformations": []}
    
    def _save_lineage_data(self):
        """Save lineage data to file."""
        try:
            self.lineage_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.lineage_file, 'w') as f:
                json.dump(self.lineage_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Could not save lineage data: {e}")
    
    def register_dataset(self, dataset_id: str, metadata: Dict[str, Any]):
        """
        Register a dataset in the lineage tracking.
        
        Args:
            dataset_id: Unique identifier for the dataset
            metadata: Dataset metadata
        """
        self.lineage_data["datasets"][dataset_id] = {
            "dataset_id": dataset_id,
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata
        }
        self._save_lineage_data()
        self.logger.info(f"Registered dataset: {dataset_id}")
    
    def track_transformation(self, transformation_id: str, input_datasets: List[str],
                           output_datasets: List[str], transformation_type: str,
                           parameters: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """
        Track a data transformation.
        
        Args:
            transformation_id: Unique identifier for the transformation
            input_datasets: List of input dataset IDs
            output_datasets: List of output dataset IDs
            transformation_type: Type of transformation
            parameters: Transformation parameters
            metadata: Additional metadata
        """
        transformation_record = {
            "transformation_id": transformation_id,
            "timestamp": datetime.now().isoformat(),
            "input_datasets": input_datasets,
            "output_datasets": output_datasets,
            "transformation_type": transformation_type,
            "parameters": parameters,
            "metadata": metadata or {}
        }
        
        self.lineage_data["transformations"].append(transformation_record)
        self._save_lineage_data()
        self.logger.info(f"Tracked transformation: {transformation_id}")
    
    def get_dataset_lineage(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get the lineage information for a specific dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Lineage information
        """
        # Find transformations that produced this dataset
        producing_transformations = [
            t for t in self.lineage_data["transformations"]
            if dataset_id in t["output_datasets"]
        ]
        
        # Find transformations that consumed this dataset
        consuming_transformations = [
            t for t in self.lineage_data["transformations"]
            if dataset_id in t["input_datasets"]
        ]
        
        return {
            "dataset_id": dataset_id,
            "dataset_info": self.lineage_data["datasets"].get(dataset_id, {}),
            "produced_by": producing_transformations,
            "consumed_by": consuming_transformations
        }
    
    def generate_lineage_report(self) -> Dict[str, Any]:
        """Generate a comprehensive lineage report."""
        return {
            "report_generated_at": datetime.now().isoformat(),
            "total_datasets": len(self.lineage_data["datasets"]),
            "total_transformations": len(self.lineage_data["transformations"]),
            "datasets": self.lineage_data["datasets"],
            "transformations": self.lineage_data["transformations"]
        }


class PersistenceManager:
    """
    High-level manager for data persistence, caching, and lineage tracking.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the PersistenceManager.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.cache = DataCache(config)
        self.checkpoint_manager = CheckpointManager(config)
        self.lineage_tracker = DataLineageTracker(config)
    
    def cache_intermediate_result(self, data: pd.DataFrame, process_name: str,
                                 step_name: str, parameters: Dict[str, Any]) -> str:
        """
        Cache an intermediate processing result.
        
        Args:
            data: DataFrame to cache
            process_name: Name of the process
            step_name: Name of the processing step
            parameters: Processing parameters
            
        Returns:
            Cache key
        """
        data_identifier = f"{process_name}_{step_name}"
        description = f"Intermediate result for {process_name} - {step_name}"
        
        return self.cache.cache_data(data, data_identifier, parameters, description)
    
    def get_cached_result(self, process_name: str, step_name: str,
                         parameters: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Retrieve a cached intermediate result.
        
        Args:
            process_name: Name of the process
            step_name: Name of the processing step
            parameters: Processing parameters
            
        Returns:
            Cached DataFrame or None
        """
        data_identifier = f"{process_name}_{step_name}"
        return self.cache.get_cached_data(data_identifier, parameters)
    
    def create_process_checkpoint(self, process_name: str, current_step: str,
                                 completed_steps: List[str], data: Optional[pd.DataFrame] = None,
                                 additional_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a checkpoint for a long-running process.
        
        Args:
            process_name: Name of the process
            current_step: Current processing step
            completed_steps: List of completed steps
            data: Optional current data state
            additional_state: Additional state information
            
        Returns:
            Checkpoint ID
        """
        state = {
            "process_name": process_name,
            "current_step": current_step,
            "completed_steps": completed_steps,
            "additional_state": additional_state or {}
        }
        
        return self.checkpoint_manager.create_checkpoint(process_name, state, data)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status including cache and checkpoint information."""
        cache_stats = self.cache.get_cache_stats()
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cache": cache_stats,
            "checkpoints": {
                "total_checkpoints": len(checkpoints),
                "recent_checkpoints": checkpoints[:5]  # Show 5 most recent
            },
            "lineage": {
                "total_datasets": len(self.lineage_tracker.lineage_data["datasets"]),
                "total_transformations": len(self.lineage_tracker.lineage_data["transformations"])
            }
        }