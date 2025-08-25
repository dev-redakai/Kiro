"""
Data ingestion module for memory-efficient CSV reading and data loading.

This module provides classes for chunked CSV reading to handle large datasets
without exceeding memory constraints, along with progress tracking and error handling.
"""

import os
import sys
import csv
import logging
from typing import Iterator, Dict, Any, Optional, List, Tuple
from pathlib import Path
import pandas as pd
import psutil
from datetime import datetime

from ..utils import memory_manager, performance_manager, get_module_logger

# Configure logging
logger = get_module_logger("ingestion")


class ChunkedCSVReader:
    """
    Memory-efficient CSV reader that processes large files in configurable chunks.
    
    This class handles CSV files that may be too large to fit in memory by reading
    them in chunks, with progress tracking and memory usage estimation.
    """
    
    def __init__(self, file_path: str, chunk_size: int = 50000, encoding: str = 'utf-8'):
        """
        Initialize the ChunkedCSVReader.
        
        Args:
            file_path: Path to the CSV file to read
            chunk_size: Number of rows to read per chunk (default: 50000)
            encoding: File encoding (default: 'utf-8')
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.encoding = encoding
        self._total_rows = None
        self._file_size = None
        self._columns = None
        self._current_chunk = 0
        self._bytes_processed = 0
        
        # Validate file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        # Get file size for progress tracking
        self._file_size = self.file_path.stat().st_size
        
        logger.info(f"Initialized ChunkedCSVReader for {file_path} with chunk_size={chunk_size}")
    
    def _detect_delimiter(self) -> str:
        """
        Detect the CSV delimiter by examining the first few lines.
        
        Returns:
            The detected delimiter character
        """
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as file:
                # Read first few lines to detect delimiter
                sample = file.read(8192)  # Read first 8KB
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                logger.debug(f"Detected delimiter: '{delimiter}'")
                return delimiter
        except Exception as e:
            logger.warning(f"Could not detect delimiter, using comma: {e}")
            return ','
    
    def _get_column_names(self) -> List[str]:
        """
        Get column names from the CSV header.
        
        Returns:
            List of column names
        """
        if self._columns is not None:
            return self._columns
            
        try:
            delimiter = self._detect_delimiter()
            with open(self.file_path, 'r', encoding=self.encoding) as file:
                reader = csv.reader(file, delimiter=delimiter)
                self._columns = next(reader)
                logger.debug(f"Detected columns: {self._columns}")
                return self._columns
        except Exception as e:
            logger.error(f"Error reading column names: {e}")
            raise
    
    def get_total_rows(self) -> int:
        """
        Estimate total number of rows in the CSV file.
        
        Returns:
            Estimated total number of rows
        """
        if self._total_rows is not None:
            return self._total_rows
        
        try:
            logger.info("Estimating total rows in CSV file...")
            
            # For very large files, estimate based on sample
            if self._file_size > 100 * 1024 * 1024:  # 100MB
                return self._estimate_rows_from_sample()
            else:
                return self._count_rows_exactly()
                
        except Exception as e:
            logger.error(f"Error estimating total rows: {e}")
            # Return a conservative estimate
            return max(1, self._file_size // 100)  # Assume ~100 bytes per row
    
    def _estimate_rows_from_sample(self) -> int:
        """
        Estimate total rows by sampling the file.
        
        Returns:
            Estimated number of rows
        """
        sample_size = min(1024 * 1024, self._file_size // 10)  # 1MB or 10% of file
        
        with open(self.file_path, 'r', encoding=self.encoding) as file:
            # Skip header
            next(file)
            
            # Read sample
            sample_data = file.read(sample_size)
            sample_rows = sample_data.count('\n')
            
            # Estimate total rows
            if sample_rows > 0:
                bytes_per_row = sample_size / sample_rows
                estimated_rows = int((self._file_size - len(next(open(self.file_path, 'r', encoding=self.encoding)))) / bytes_per_row)
                self._total_rows = max(1, estimated_rows)
            else:
                self._total_rows = 1
                
        logger.info(f"Estimated total rows: {self._total_rows}")
        return self._total_rows
    
    def _count_rows_exactly(self) -> int:
        """
        Count rows exactly for smaller files.
        
        Returns:
            Exact number of rows
        """
        with open(self.file_path, 'r', encoding=self.encoding) as file:
            # Skip header
            next(file)
            row_count = sum(1 for _ in file)
            
        self._total_rows = row_count
        logger.info(f"Counted exact rows: {self._total_rows}")
        return self._total_rows
    
    def estimate_memory_usage(self) -> Dict[str, int]:
        """
        Estimate memory usage for processing chunks.
        
        Returns:
            Dictionary with memory usage estimates in bytes
        """
        try:
            # Get column info
            columns = self._get_column_names()
            num_columns = len(columns)
            
            # Estimate bytes per row (rough approximation)
            # Assume average of 50 bytes per column for mixed data types
            estimated_bytes_per_row = num_columns * 50
            
            # Calculate memory for one chunk
            chunk_memory = self.chunk_size * estimated_bytes_per_row
            
            # Add pandas overhead (approximately 2x for DataFrame)
            pandas_overhead = chunk_memory * 2
            
            # Get current system memory info
            memory_info = psutil.virtual_memory()
            
            return {
                'estimated_bytes_per_row': estimated_bytes_per_row,
                'chunk_memory_bytes': chunk_memory,
                'pandas_overhead_bytes': pandas_overhead,
                'total_chunk_memory': chunk_memory + pandas_overhead,
                'system_available_memory': memory_info.available,
                'system_total_memory': memory_info.total,
                'memory_usage_percentage': ((chunk_memory + pandas_overhead) / memory_info.available) * 100
            }
            
        except Exception as e:
            logger.error(f"Error estimating memory usage: {e}")
            return {
                'estimated_bytes_per_row': 0,
                'chunk_memory_bytes': 0,
                'pandas_overhead_bytes': 0,
                'total_chunk_memory': 0,
                'system_available_memory': 0,
                'system_total_memory': 0,
                'memory_usage_percentage': 0
            }
    
    def read_chunks(self) -> Iterator[pd.DataFrame]:
        """
        Read the CSV file in chunks with performance and memory optimization.
        
        Yields:
            DataFrame chunks of the specified size
        """
        try:
            delimiter = self._detect_delimiter()
            
            # Use memory manager for monitoring
            with memory_manager.memory_checkpoint("csv_chunk_reading"):
                # Log memory usage before starting
                memory_info = self.estimate_memory_usage()
                logger.info(f"Starting chunked reading. Estimated memory per chunk: "
                           f"{memory_info['total_chunk_memory'] / (1024*1024):.1f} MB "
                           f"({memory_info['memory_usage_percentage']:.1f}% of available memory)")
                
                # Create progress tracker
                total_rows = self.get_total_rows()
                progress_tracker = performance_manager.create_progress_tracker(total_rows)
                
                # Read CSV in chunks using pandas
                chunk_reader = pd.read_csv(
                    self.file_path,
                    chunksize=self.chunk_size,
                    encoding=self.encoding,
                    delimiter=delimiter,
                    on_bad_lines='warn',  # Handle malformed lines gracefully
                    low_memory=False  # Prevent mixed type warnings
                )
                
                self._current_chunk = 0
                
                for chunk in chunk_reader:
                    self._current_chunk += 1
                    
                    # Optimize DataFrame for better performance
                    optimized_chunk = performance_manager.optimize_dataframe_operations(chunk)
                    
                    # Update progress tracking
                    progress_tracker.update(len(optimized_chunk))
                    self._bytes_processed += len(optimized_chunk) * memory_info['estimated_bytes_per_row']
                    
                    # Log progress with ETA
                    if self._current_chunk % 10 == 0 or progress_tracker.should_update_display():
                        progress_msg = progress_tracker.get_progress_message()
                        memory_status = memory_manager.check_memory_usage()
                        logger.info(f"Chunk {self._current_chunk}: {progress_msg}, "
                                   f"Memory: {memory_status['process_memory_mb']:.1f} MB")
                    
                    # Force garbage collection if memory usage is high
                    memory_status = memory_manager.check_memory_usage()
                    if memory_status['status'] == 'warning':
                        memory_manager.force_garbage_collection()
                    
                    yield optimized_chunk
                
        except Exception as e:
            logger.error(f"Error reading CSV chunks: {e}")
            raise
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress information.
        
        Returns:
            Dictionary with progress metrics
        """
        try:
            total_rows = self.get_total_rows()
            processed_rows = self._current_chunk * self.chunk_size
            
            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'current_chunk': self._current_chunk,
                'processed_rows': min(processed_rows, total_rows),
                'total_rows': total_rows,
                'percentage_complete': min((processed_rows / total_rows) * 100, 100.0) if total_rows > 0 else 0,
                'bytes_processed': self._bytes_processed,
                'file_size_bytes': self._file_size,
                'current_memory_mb': memory_info.rss / (1024 * 1024),
                'chunk_size': self.chunk_size
            }
        except Exception as e:
            logger.error(f"Error getting progress: {e}")
            return {
                'current_chunk': self._current_chunk,
                'processed_rows': 0,
                'total_rows': 0,
                'percentage_complete': 0,
                'bytes_processed': 0,
                'file_size_bytes': self._file_size or 0,
                'current_memory_mb': 0,
                'chunk_size': self.chunk_size
            }
    
    def adjust_chunk_size_for_memory(self, target_memory_mb: int = 500) -> int:
        """
        Automatically adjust chunk size based on available memory using MemoryManager.
        
        Args:
            target_memory_mb: Target memory usage per chunk in MB
            
        Returns:
            Adjusted chunk size
        """
        try:
            # Use the memory manager for optimal chunk size calculation
            file_size = self._file_size or 0
            optimal_chunk_size = memory_manager.get_optimal_chunk_size(file_size, self.chunk_size)
            
            if optimal_chunk_size != self.chunk_size:
                logger.info(f"MemoryManager adjusted chunk size from {self.chunk_size} to {optimal_chunk_size}")
                self.chunk_size = optimal_chunk_size
            
            return self.chunk_size
            
        except Exception as e:
            logger.error(f"Error adjusting chunk size: {e}")
            return self.chunk_size

class DataIngestionManager:
    """
    Orchestrates the data ingestion process with file validation, statistics, and error handling.
    
    This class manages the overall data loading workflow, including file validation,
    format checking, progress monitoring, and error recovery.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DataIngestionManager.
        
        Args:
            config: Configuration dictionary with ingestion settings
        """
        self.config = config or {}
        self.default_chunk_size = self.config.get('chunk_size', 50000)
        self.default_encoding = self.config.get('encoding', 'utf-8')
        self.max_retries = self.config.get('max_retries', 3)
        self.supported_formats = self.config.get('supported_formats', ['.csv', '.tsv', '.txt'])
        
        # Statistics tracking
        self.ingestion_stats = {
            'files_processed': 0,
            'total_rows_processed': 0,
            'total_chunks_processed': 0,
            'errors_encountered': 0,
            'start_time': None,
            'end_time': None,
            'processing_time_seconds': 0,
            'average_rows_per_second': 0,
            'files_with_errors': [],
            'memory_peak_mb': 0
        }
        
        logger.info("Initialized DataIngestionManager")
    
    def validate_file_format(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate if the file format is supported for ingestion.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            file_path_obj = Path(file_path)
            
            # Check if file exists
            if not file_path_obj.exists():
                return False, f"File does not exist: {file_path}"
            
            # Check if it's a file (not directory)
            if not file_path_obj.is_file():
                return False, f"Path is not a file: {file_path}"
            
            # Check file extension
            file_extension = file_path_obj.suffix.lower()
            if file_extension not in self.supported_formats:
                return False, f"Unsupported file format: {file_extension}. Supported formats: {self.supported_formats}"
            
            # Check file size (warn if very large)
            file_size = file_path_obj.stat().st_size
            if file_size > 10 * 1024 * 1024 * 1024:  # 10GB
                logger.warning(f"Very large file detected: {file_size / (1024**3):.1f} GB")
            
            # Check if file is readable
            try:
                with open(file_path, 'r', encoding=self.default_encoding) as f:
                    f.read(1024)  # Try to read first 1KB
            except UnicodeDecodeError:
                # Try with different encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            f.read(1024)
                        logger.info(f"File requires encoding: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    return False, f"Unable to read file with any supported encoding"
            
            # Basic CSV structure validation
            if file_extension in ['.csv', '.tsv']:
                return self._validate_csv_structure(file_path)
            
            return True, "File validation passed"
            
        except Exception as e:
            logger.error(f"Error validating file format: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _validate_csv_structure(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate basic CSV file structure.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Try to read first few lines to validate structure
            with open(file_path, 'r', encoding=self.default_encoding) as file:
                # Check if file is empty
                first_line = file.readline().strip()
                if not first_line:
                    return False, "CSV file appears to be empty"
                
                # Check for basic CSV structure
                sniffer = csv.Sniffer()
                sample = first_line + '\n' + file.read(8192)
                
                try:
                    dialect = sniffer.sniff(sample)
                    has_header = sniffer.has_header(sample)
                    
                    # Count columns in header
                    reader = csv.reader([first_line], delimiter=dialect.delimiter)
                    header_columns = len(next(reader))
                    
                    if header_columns == 0:
                        return False, "No columns detected in CSV header"
                    
                    logger.debug(f"CSV validation: {header_columns} columns, has_header={has_header}")
                    
                except csv.Error as e:
                    return False, f"CSV structure validation failed: {e}"
            
            return True, "CSV structure validation passed"
            
        except Exception as e:
            return False, f"CSV validation error: {str(e)}"
    
    def ingest_data(self, file_path: str, chunk_size: Optional[int] = None, 
                   encoding: Optional[str] = None) -> Iterator[pd.DataFrame]:
        """
        Ingest data from a file with error handling, performance tracking, and memory optimization.
        
        Args:
            file_path: Path to the file to ingest
            chunk_size: Override default chunk size
            encoding: Override default encoding
            
        Yields:
            DataFrame chunks from the file
        """
        def _ingest_operation():
            """Internal ingestion operation for performance benchmarking."""
            # Start timing
            if self.ingestion_stats['start_time'] is None:
                self.ingestion_stats['start_time'] = datetime.now()
            
            chunk_size_to_use = chunk_size or self.default_chunk_size
            encoding_to_use = encoding or self.default_encoding
            
            # Validate file format
            is_valid, error_msg = self.validate_file_format(file_path)
            if not is_valid:
                self.ingestion_stats['errors_encountered'] += 1
                self.ingestion_stats['files_with_errors'].append({
                    'file': file_path,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                })
                raise ValueError(f"File validation failed: {error_msg}")
            
            logger.info(f"Starting data ingestion for: {file_path}")
            
            # Get file size for memory optimization
            file_size = Path(file_path).stat().st_size
            optimization_recommendations = memory_manager.optimize_for_large_dataset(file_size)
            
            if optimization_recommendations['strategy'] != 'default':
                chunk_size_to_use = optimization_recommendations['recommended_chunk_size']
                logger.info(f"Applied memory optimization: {optimization_recommendations['strategy']} "
                           f"(chunk_size: {chunk_size_to_use})")
            
            # Attempt ingestion with retries
            for attempt in range(self.max_retries):
                try:
                    # Create chunked reader
                    reader = ChunkedCSVReader(file_path, chunk_size_to_use, encoding_to_use)
                    
                    # Adjust chunk size based on memory if needed
                    reader.adjust_chunk_size_for_memory()
                    
                    chunk_count = 0
                    total_rows = 0
                    chunks_yielded = []
                    
                    # Process chunks with memory monitoring
                    with memory_manager.memory_checkpoint(f"ingestion_{Path(file_path).name}"):
                        for chunk in reader.read_chunks():
                            chunk_count += 1
                            total_rows += len(chunk)
                            
                            # Update statistics
                            self.ingestion_stats['total_chunks_processed'] += 1
                            
                            # Monitor memory usage
                            memory_status = memory_manager.check_memory_usage()
                            self.ingestion_stats['memory_peak_mb'] = max(
                                self.ingestion_stats['memory_peak_mb'], 
                                memory_status['process_memory_mb']
                            )
                            
                            chunks_yielded.append(chunk)
                    
                    # Update final statistics for this file
                    self.ingestion_stats['files_processed'] += 1
                    self.ingestion_stats['total_rows_processed'] += total_rows
                    
                    logger.info(f"Successfully ingested {total_rows:,} rows in {chunk_count} chunks from {file_path}")
                    return chunks_yielded  # Return for benchmarking
                    
                except Exception as e:
                    logger.error(f"Ingestion attempt {attempt + 1} failed for {file_path}: {e}")
                    
                    if attempt == self.max_retries - 1:
                        # Final attempt failed
                        self.ingestion_stats['errors_encountered'] += 1
                        self.ingestion_stats['files_with_errors'].append({
                            'file': file_path,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat(),
                            'attempts': self.max_retries
                        })
                        raise
                    else:
                        # Wait before retry
                        import time
                        time.sleep(2 ** attempt)  # Exponential backoff
        
        # Benchmark the ingestion operation
        file_name = Path(file_path).name
        metrics = performance_manager.benchmark_operation(f"ingest_{file_name}", _ingest_operation)
        
        logger.info(f"Ingestion performance: {metrics.throughput_description}, "
                   f"Duration: {metrics.duration:.2f}s, "
                   f"Memory: {metrics.memory_usage_mb:+.2f} MB")
        
        # Yield the chunks (re-run the operation since benchmark consumed the generator)
        chunk_size_to_use = chunk_size or self.default_chunk_size
        encoding_to_use = encoding or self.default_encoding
        
        # Get optimized chunk size
        file_size = Path(file_path).stat().st_size
        optimization_recommendations = memory_manager.optimize_for_large_dataset(file_size)
        if optimization_recommendations['strategy'] != 'default':
            chunk_size_to_use = optimization_recommendations['recommended_chunk_size']
        
        reader = ChunkedCSVReader(file_path, chunk_size_to_use, encoding_to_use)
        reader.adjust_chunk_size_for_memory()
        
        for chunk in reader.read_chunks():
            yield chunk
    
    def ingest_multiple_files(self, file_paths: List[str], 
                            chunk_size: Optional[int] = None,
                            encoding: Optional[str] = None) -> Iterator[Tuple[str, pd.DataFrame]]:
        """
        Ingest data from multiple files sequentially.
        
        Args:
            file_paths: List of file paths to ingest
            chunk_size: Override default chunk size
            encoding: Override default encoding
            
        Yields:
            Tuples of (file_path, DataFrame chunk)
        """
        logger.info(f"Starting batch ingestion of {len(file_paths)} files")
        
        for file_path in file_paths:
            try:
                for chunk in self.ingest_data(file_path, chunk_size, encoding):
                    yield file_path, chunk
            except Exception as e:
                logger.error(f"Failed to ingest file {file_path}: {e}")
                # Continue with next file
                continue
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive ingestion statistics.
        
        Returns:
            Dictionary with ingestion statistics and metrics
        """
        # Update end time and calculate processing time
        if self.ingestion_stats['start_time'] is not None:
            end_time = datetime.now()
            self.ingestion_stats['end_time'] = end_time
            
            processing_time = (end_time - self.ingestion_stats['start_time']).total_seconds()
            self.ingestion_stats['processing_time_seconds'] = processing_time
            
            # Calculate average rows per second
            if processing_time > 0:
                self.ingestion_stats['average_rows_per_second'] = (
                    self.ingestion_stats['total_rows_processed'] / processing_time
                )
        
        # Add system information
        memory_info = psutil.virtual_memory()
        
        stats = self.ingestion_stats.copy()
        stats.update({
            'system_memory_total_gb': memory_info.total / (1024**3),
            'system_memory_available_gb': memory_info.available / (1024**3),
            'system_memory_usage_percent': memory_info.percent,
            'success_rate': (
                (self.ingestion_stats['files_processed'] / 
                 max(1, self.ingestion_stats['files_processed'] + len(self.ingestion_stats['files_with_errors']))) * 100
            ) if self.ingestion_stats['files_processed'] > 0 or self.ingestion_stats['files_with_errors'] else 0
        })
        
        return stats
    
    def reset_stats(self):
        """Reset ingestion statistics for a new ingestion session."""
        self.ingestion_stats = {
            'files_processed': 0,
            'total_rows_processed': 0,
            'total_chunks_processed': 0,
            'errors_encountered': 0,
            'start_time': None,
            'end_time': None,
            'processing_time_seconds': 0,
            'average_rows_per_second': 0,
            'files_with_errors': [],
            'memory_peak_mb': 0
        }
        logger.info("Ingestion statistics reset")
    
    def generate_ingestion_report(self) -> str:
        """
        Generate a human-readable ingestion report.
        
        Returns:
            Formatted string report of ingestion statistics
        """
        stats = self.get_ingestion_stats()
        
        report = []
        report.append("=" * 60)
        report.append("DATA INGESTION REPORT")
        report.append("=" * 60)
        
        # Summary statistics
        report.append(f"Files processed successfully: {stats['files_processed']}")
        report.append(f"Total rows processed: {stats['total_rows_processed']:,}")
        report.append(f"Total chunks processed: {stats['total_chunks_processed']:,}")
        report.append(f"Processing time: {stats['processing_time_seconds']:.1f} seconds")
        report.append(f"Average processing rate: {stats['average_rows_per_second']:,.0f} rows/second")
        report.append(f"Success rate: {stats['success_rate']:.1f}%")
        
        # Memory usage
        report.append(f"\nMemory Usage:")
        report.append(f"Peak memory usage: {stats['memory_peak_mb']:.1f} MB")
        report.append(f"System memory total: {stats['system_memory_total_gb']:.1f} GB")
        report.append(f"System memory available: {stats['system_memory_available_gb']:.1f} GB")
        
        # Errors
        if stats['errors_encountered'] > 0:
            report.append(f"\nErrors encountered: {stats['errors_encountered']}")
            report.append("Files with errors:")
            for error_info in stats['files_with_errors']:
                report.append(f"  - {error_info['file']}: {error_info['error']}")
        
        report.append("=" * 60)
        
        return "\n".join(report)


# Utility functions for ingestion
def create_sample_csv(file_path: str, num_rows: int = 1000, include_errors: bool = False):
    """
    Create a sample CSV file for testing ingestion functionality.
    
    Args:
        file_path: Path where to create the sample CSV
        num_rows: Number of rows to generate
        include_errors: Whether to include data quality issues
    """
    import random
    from datetime import datetime, timedelta
    
    logger.info(f"Creating sample CSV with {num_rows} rows at {file_path}")
    
    # Sample data
    products = ["Laptop", "Smartphone", "Tablet", "Headphones", "Camera", "Watch", "Speaker"]
    categories = ["Electronics", "Fashion", "Home", "Sports", "Books"]
    regions = ["North", "South", "East", "West", "Central"]
    
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            'order_id', 'product_name', 'category', 'quantity', 'unit_price',
            'discount_percent', 'region', 'sale_date', 'customer_email'
        ])
        
        # Generate sample data
        for i in range(num_rows):
            order_id = f"ORD{i+1:06d}"
            product = random.choice(products)
            category = random.choice(categories)
            quantity = random.randint(1, 10)
            unit_price = round(random.uniform(10.0, 1000.0), 2)
            discount = round(random.uniform(0.0, 0.3), 3)
            region = random.choice(regions)
            
            # Generate date within last year
            base_date = datetime.now() - timedelta(days=365)
            sale_date = base_date + timedelta(days=random.randint(0, 365))
            sale_date_str = sale_date.strftime('%Y-%m-%d')
            
            email = f"customer{i+1}@example.com"
            
            # Introduce errors if requested
            if include_errors and random.random() < 0.1:  # 10% error rate
                if random.random() < 0.3:
                    quantity = "invalid"  # Invalid quantity
                elif random.random() < 0.3:
                    discount = 1.5  # Invalid discount > 1
                elif random.random() < 0.3:
                    unit_price = -10.0  # Negative price
                else:
                    email = "invalid-email"  # Invalid email
            
            writer.writerow([
                order_id, product, category, quantity, unit_price,
                discount, region, sale_date_str, email
            ])
    
    logger.info(f"Sample CSV created successfully at {file_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
        # Create ingestion manager
        manager = DataIngestionManager()
        
        try:
            # Ingest data
            total_rows = 0
            for chunk in manager.ingest_data(file_path):
                total_rows += len(chunk)
                print(f"Processed chunk with {len(chunk)} rows")
            
            # Print statistics
            print(manager.generate_ingestion_report())
            
        except Exception as e:
            print(f"Ingestion failed: {e}")
            sys.exit(1)
    else:
        print("Usage: python ingestion.py <csv_file_path>")
        print("Or create sample data: python -c \"from src.pipeline.ingestion import create_sample_csv; create_sample_csv('sample.csv', 10000)\"")