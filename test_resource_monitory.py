import numpy as np
import torch
import os
import time
from resource_monitor import monitor_resources

@monitor_resources("./test_outputs")
def test_memory_intensive():
    """
    Objective:
    Test memory monitoring with large array operations
    
    Returns:
    [dict] metrics - Memory usage metrics
    """
    try:
        # Create and manipulate large arrays
        arrays = []
        for _ in range(5):
            arr = np.random.rand(1000, 1000)
            arrays.append(arr)
            time.sleep(1)  # Allow monitor to collect metrics
            
        return {"arrays_created": len(arrays)}
        
    except Exception as e:
        print(f"Error in memory test: {e}")
        raise

@monitor_resources("./test_outputs")
def test_gpu_intensive():
    """
    Objective:
    Test GPU monitoring with tensor operations
    
    Returns:
    [dict] metrics - GPU usage metrics
    """
    try:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            tensors = []
            for _ in range(5):
                tensor = torch.randn(1000, 1000, device=device)
                tensors.append(tensor)
                time.sleep(1)
                
            return {"tensors_created": len(tensors)}
            
    except Exception as e:
        print(f"Error in GPU test: {e}")
        raise

@monitor_resources("./test_outputs")
def test_disk_intensive():
    """
    Objective:
    Test disk I/O monitoring with file operations
    
    Returns:
    [dict] metrics - Disk I/O metrics
    """
    try:
        files_created = 0
        for i in range(5):
            with open(f"test_file_{i}.txt", "w") as f:
                f.write("x" * 1000000)  # Write 1MB
                files_created += 1
                time.sleep(1)
                
        # Cleanup
        for i in range(files_created):
            os.remove(f"test_file_{i}.txt")
            
        return {"files_processed": files_created}
        
    except Exception as e:
        print(f"Error in disk test: {e}")
        raise

def validate_metrics(test_name):
    """
    Objective:
    Validate metrics collected during tests
    
    Parameters:
    [str] test_name - Name of the test to validate
    
    Returns:
    [bool] is_valid - Whether metrics are valid
    """
    try:
        import pandas as pd
        metrics_file = f"./test_outputs/{test_name}_resources.csv"
        
        if not os.path.exists(metrics_file):
            print(f"No metrics file found for {test_name}")
            return False
            
        df = pd.read_csv(metrics_file)
        
        # Basic validation
        required_columns = [
            'timestamp', 'cpu_percent', 'memory_percent',
            'gpu_memory_allocated', 'disk_io_read', 'disk_io_write'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                print(f"Missing column: {col}")
                return False
                
        return True
        
    except Exception as e:
        print(f"Error validating metrics: {e}")
        return False

def main():
    """
    Objective:
    Run all tests and validate results
    """
    try:
        os.makedirs("./test_outputs", exist_ok=True)
        
        print("Running memory test...")
        test_memory_intensive()
        assert validate_metrics("test_memory_intensive"), "Memory metrics validation failed"
        
        print("Running GPU test...")
        test_gpu_intensive()
        assert validate_metrics("test_gpu_intensive"), "GPU metrics validation failed"
        
        print("Running disk I/O test...")
        test_disk_intensive()
        assert validate_metrics("test_disk_intensive"), "Disk I/O metrics validation failed"
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test suite failed: {e}")
        raise

if __name__ == "__main__":
    main()