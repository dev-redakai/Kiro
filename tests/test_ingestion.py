#!/usr/bin/env python3
"""
Test script to verify DataIngestionManager integration with ChunkedCSVReader
"""

import sys
import os
sys.path.append('src')

from pipeline.ingestion import DataIngestionManager, create_sample_csv
import tempfile

def test_ingestion_manager():
    """Test the DataIngestionManager with ChunkedCSVReader"""
    print("Testing DataIngestionManager integration...")
    
    # Create a temporary CSV file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Create sample data
        print("Creating sample CSV file...")
        create_sample_csv(temp_path, num_rows=500, include_errors=False)
        
        # Test DataIngestionManager
        print("Initializing DataIngestionManager...")
        manager = DataIngestionManager(config={'chunk_size': 50})
        
        # Test file validation
        print("Testing file validation...")
        is_valid, message = manager.validate_file_format(temp_path)
        print(f"File validation: {is_valid} - {message}")
        
        if not is_valid:
            print("ERROR: File validation failed")
            return False
        
        # Test data ingestion
        print("Testing data ingestion...")
        chunk_count = 0
        total_rows = 0
        
        for chunk in manager.ingest_data(temp_path):
            chunk_count += 1
            total_rows += len(chunk)
            print(f"Processed chunk {chunk_count}: {len(chunk)} rows")
            
            # Only process first few chunks for testing
            if chunk_count >= 5:
                break
        
        # Get ingestion statistics
        print("\nGetting ingestion statistics...")
        stats = manager.get_ingestion_stats()
        print(f"Files processed: {stats['files_processed']}")
        print(f"Total rows processed: {stats['total_rows_processed']}")
        print(f"Total chunks processed: {stats['total_chunks_processed']}")
        print(f"Processing time: {stats['processing_time_seconds']:.2f} seconds")
        print(f"Average rows per second: {stats['average_rows_per_second']:.0f}")
        
        # Generate report
        print("\nGenerating ingestion report...")
        report = manager.generate_ingestion_report()
        print("Report generated successfully")
        
        print(f"\nTest completed successfully!")
        print(f"Processed {chunk_count} chunks with {total_rows} total rows")
        
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
    success = test_ingestion_manager()
    sys.exit(0 if success else 1)