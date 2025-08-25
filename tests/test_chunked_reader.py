#!/usr/bin/env python3
"""
Test script to verify ChunkedCSVReader functionality
"""

import sys
import os
sys.path.append('src')

from pipeline.ingestion import ChunkedCSVReader, create_sample_csv
import tempfile
import pandas as pd

def test_chunked_csv_reader():
    """Test the ChunkedCSVReader implementation"""
    print("Testing ChunkedCSVReader implementation...")
    
    # Create a temporary CSV file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Create sample data
        print("Creating sample CSV file...")
        create_sample_csv(temp_path, num_rows=1000, include_errors=False)
        
        # Test ChunkedCSVReader
        print("Initializing ChunkedCSVReader...")
        reader = ChunkedCSVReader(temp_path, chunk_size=100)
        
        # Test memory estimation
        print("Testing memory estimation...")
        memory_info = reader.estimate_memory_usage()
        print(f"Estimated memory per chunk: {memory_info['total_chunk_memory'] / (1024*1024):.2f} MB")
        
        # Test row counting
        print("Testing row counting...")
        total_rows = reader.get_total_rows()
        print(f"Total rows detected: {total_rows}")
        
        # Test chunked reading
        print("Testing chunked reading...")
        chunk_count = 0
        total_rows_read = 0
        
        for chunk in reader.read_chunks():
            chunk_count += 1
            total_rows_read += len(chunk)
            
            # Test progress tracking
            progress = reader.get_progress()
            print(f"Chunk {chunk_count}: {len(chunk)} rows, "
                  f"Progress: {progress['percentage_complete']:.1f}%")
            
            # Only process first few chunks for testing
            if chunk_count >= 5:
                break
        
        print(f"\nTest completed successfully!")
        print(f"Processed {chunk_count} chunks with {total_rows_read} total rows")
        
        # Test automatic chunk size adjustment
        print("\nTesting automatic chunk size adjustment...")
        original_chunk_size = reader.chunk_size
        adjusted_size = reader.adjust_chunk_size_for_memory(target_memory_mb=100)
        print(f"Original chunk size: {original_chunk_size}")
        print(f"Adjusted chunk size: {adjusted_size}")
        
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    success = test_chunked_csv_reader()
    sys.exit(0 if success else 1)