#!/usr/bin/env python3
"""
Test runner for validation fixes test suite.

This script runs all the validation fix tests and provides a comprehensive report.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_test_suite(test_file, description):
    """Run a test suite and return results."""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            test_file, 
            '-v', 
            '--tb=short',
            '--disable-warnings'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Duration: {duration:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        return result.returncode == 0, duration, result.stdout
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False, 0, str(e)

def main():
    """Run all validation fix tests."""
    print("ETL Validation Fixes - Comprehensive Test Suite")
    print("=" * 60)
    
    test_suites = [
        ("tests/test_validation_fixes.py", "Core Validation Fixes Tests"),
        ("tests/test_validation_edge_cases.py", "Edge Case Tests"),
        ("tests/test_validation_performance.py", "Performance Tests")
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_file, description in test_suites:
        success, duration, output = run_test_suite(test_file, description)
        results.append((test_file, description, success, duration, output))
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUITE SUMMARY")
    print(f"{'='*60}")
    
    passed_count = 0
    failed_count = 0
    
    for test_file, description, success, duration, output in results:
        status = "PASSED" if success else "FAILED"
        print(f"{description}: {status} ({duration:.2f}s)")
        
        if success:
            passed_count += 1
        else:
            failed_count += 1
    
    print(f"\nTotal Duration: {total_duration:.2f} seconds")
    print(f"Test Suites Passed: {passed_count}")
    print(f"Test Suites Failed: {failed_count}")
    
    # Extract test counts from output
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    for test_file, description, success, duration, output in results:
        if "passed" in output:
            # Try to extract test counts
            lines = output.split('\n')
            for line in lines:
                if "passed" in line and ("failed" in line or "error" in line):
                    # Parse pytest summary line
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "passed" in part:
                                passed = int(parts[i-1]) if i > 0 else 0
                                total_passed += passed
                            elif "failed" in part:
                                failed = int(parts[i-1]) if i > 0 else 0
                                total_failed += failed
                    except:
                        pass
    
    if total_passed > 0 or total_failed > 0:
        print(f"\nIndividual Test Results:")
        print(f"Tests Passed: {total_passed}")
        print(f"Tests Failed: {total_failed}")
        print(f"Total Tests: {total_passed + total_failed}")
    
    # Overall result
    if failed_count == 0:
        print(f"\n🎉 ALL TEST SUITES PASSED! 🎉")
        return 0
    else:
        print(f"\n❌ {failed_count} TEST SUITE(S) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())