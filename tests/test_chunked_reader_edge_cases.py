#!/usr/bin/env python3
"""
Test edge cases for ChunkedCSVReader functionality
"""

import sys
import os
sys.path.append('src')

from pipeline.ingestion import ChunkedCSVReader, create_sample_csv
import tempfile
import pandas as pd

def test_edge_cases():
    """Test ChunkedCSVReader with edge cases"""
    print("Testing ChunkedCSVReader edge cases...")
    
    # Test 1: Non-existent file
    print("\n1. Testing non-existent file...")
    try:
        reader = ChunkedCSVReader("non_existent_file.csv")
        print("ERROR: Should have raised FileNotFoundError")
        return False
    except FileNotFoundError:
        print("✓ Correctly handled non-existent file")
    
    # Test 2: CSV with different delimiters
    print("\n2. Testing CSV with different delimiters...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        temp_path = temp_file.name
        # Create CSV with semicolon delimiter
        temp_file.write("order_id;product_name;quantity\n")
        temp_file.write("1;Product A;10\n")
        temp_file.write("2;Product B;20\n")
    
    try:
        reader = ChunkedCSVReader(temp_path, chunk_size=10)
        chunks = list(reader.read_chunks())
        print(f"✓ Successfully read {len(chunks)} chunks with semicolon delimiter")
        print(f"  First chunk shape: {chunks[0].shape}")
    except Exception as e:
        print(f"ERROR: Failed to handle semicolon delimiter: {e}")
        return False
    finally:
        os.unlink(temp_path)
    
    # Test 3: CSV with encoding issues
    print("\n3. Testing CSV with different encoding...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='latin-1') as temp_file:
        temp_path = temp_file.name
        temp_file.write("order_id,product_name,quantity\n")
        temp_file.write("1,Café Product,10\n")
        temp_file.write("2,Naïve Product,20\n")
    
    try:
        reader = ChunkedCSVReader(temp_path, chunk_size=10, encoding='latin-1')
        chunks = list(reader.read_chunks())
        print(f"✓ Successfully read {len(chunks)} chunks with latin-1 encoding")
    except Exception as e:
        print(f"ERROR: Failed to handle latin-1 encoding: {e}")
        return False
    finally:
        os.unlink(temp_path)
    
    # Test 4: Very small chunk size
    print("\n4. Testing very small chunk size...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        create_sample_csv(temp_path, num_rows=50, include_errors=False)
        reader = ChunkedCSVReader(temp_path, chunk_size=1)  # Very small chunks
        
        chunk_count = 0
        for chunk in reader.read_chunks():
            chunk_count += 1
            if chunk_count > 10:  # Only test first 10 chunks
                break
        
        print(f"✓ Successfully processed {chunk_count} chunks with chunk_size=1")
        
    except Exception as e:
        print(f"ERROR: Failed with small chunk size: {e}")
        return False
    finally:
        os.unlink(temp_path)
    
    # Test 5: Memory estimation accuracy
    print("\n5. Testing memory estimation...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        create_sample_csv(temp_path, num_rows=1000, include_errors=False)
        reader = ChunkedCSVReader(temp_path, chunk_size=100)
        
        memory_info = reader.estimate_memory_usage()
        required_keys = [
            'estimated_bytes_per_row', 'chunk_memory_bytes', 
            'pandas_overhead_bytes', 'total_chunk_memory',
            'system_available_memory', 'memory_usage_percentage'
        ]
        
        for key in required_keys:
            if key not in memory_info:
                print(f"ERROR: Missing key in memory info: {key}")
                return False
        
        print("✓ Memory estimation provides all required metrics")
        print(f"  Estimated bytes per row: {memory_info['estimated_bytes_per_row']}")
        print(f"  Memory usage percentage: {memory_info['memory_usage_percentage']:.2f}%")
        
    except Exception as e:
        print(f"ERROR: Memory estimation failed: {e}")
        return False
    finally:
        os.unlink(temp_path)
    
    print("\n✓ All edge case tests passed!")
    return True

if __name__ == "__main__":
    success = test_edge_cases()
    sys.exit(0 if success else 1)