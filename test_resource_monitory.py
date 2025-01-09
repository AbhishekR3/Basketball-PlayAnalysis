"""
Test script for resource_monitor.py
"""
import time
import numpy as np
from memory_profiler import profile

@profile
def memory_intensive_function():
    """
    Objective:
    Create a memory-intensive operation to test monitoring
    
    Returns:
    [ndarray] large_array - Large numpy array for testing
    """
    try:
        # Create large array
        large_array = np.zeros((1000, 1000, 100))
        time.sleep(2)  # Simulate processing
        return large_array
    except Exception as e:
        print(f"Error in memory intensive function: {e}")
        raise

@profile
def cpu_intensive_function():
    """
    Objective:
    Create a CPU-intensive operation to test monitoring
    
    Returns:
    [float] result - Result of CPU intensive calculation
    """
    try:
        result = 0
        for i in range(10):
            result += i * i
        time.sleep(1)  # Simulate processing
        return result
    except Exception as e:
        print(f"Error in CPU intensive function: {e}")
        raise

def main():
    """
    Objective:
    Main function to run test cases
    """
    try:
        print("Starting memory test...")
        arr = memory_intensive_function()
        print(f"Memory test complete. Array shape: {arr.shape}")
        
        print("\nStarting CPU test...")
        result = cpu_intensive_function()
        print(f"CPU test complete. Result: {result}")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()