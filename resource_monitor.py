import psutil
import torch
from memory_profiler import profile
import time
import pandas as pd
import os
from functools import wraps
from datetime import datetime

class ResourceMonitor:
    """
    Objective:
    Monitor and track system resources during script execution
    
    Attributes:
    [str] output_dir - Directory to save monitoring results
    [dict] metrics - Dictionary to store collected metrics
    """
    
    def __init__(self, output_dir):
        """
        Objective:
        Initialize the resource monitor
        
        Parameters:
        [str] output_dir - Directory path to save monitoring results
        
        Returns:
        None
        """
        try:
            self.output_dir = output_dir
            self.metrics = {
                'timestamp': [],
                'cpu_percent': [],
                'memory_percent': [],
                'gpu_memory_allocated': [],
                'disk_io_read': [],
                'disk_io_write': [],
                'power_consumption': [],
                'cpu_energy': [],
                'gpu_energy': [],
                'total_energy': []
            }
            
            # Initialize energy monitoring
            self.last_energy_check = time.time()
            self.energy_command = None
            
            # Check if we're on macOS with Apple Silicon
            if self.is_apple_silicon():
                # Get username
                try:
                    import pwd
                    username = pwd.getpwuid(os.getuid())[0]
                    # Set up powermetrics permissions
                    setup_command = f"sudo chmod +a 'user:{username} allow read,write' /private/var/db/powermetricsdb"
                    os.system(setup_command)
                except Exception as e:
                    print(f"Error setting up powermetrics permissions: {e}")
                
                self.energy_command = "sudo powermetrics -n 1 -i 1000 --samplers cpu_power,gpu_power"
            
            # Initialize disk I/O counters
            self.disk_io_start = psutil.disk_io_counters()
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
        except Exception as e:
            print(f"Error initializing ResourceMonitor: {e}")
            raise

    def is_apple_silicon(self):
        """
        Objective:
        Check if running on Apple Silicon Mac
        
        Returns:
        [bool] is_apple_silicon - True if running on Apple Silicon
        """
        try:
            import platform
            return (platform.system() == 'Darwin' and 
                   platform.machine().startswith('arm64'))
        except Exception as e:
            print(f"Error checking for Apple Silicon: {e}")
            return False

    def get_energy_metrics(self):
        """
        Objective:
        Get energy consumption metrics
        
        Returns:
        [tuple] energy metrics - (cpu_energy, gpu_energy, total_energy)
        """
        try:
            if not self.energy_command:
                return 0, 0, 0
                
            import subprocess
            result = subprocess.run(
                self.energy_command.split(),
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return 0, 0, 0
                
            output = result.stdout
            
            # Parse powermetrics output
            cpu_energy = 0
            gpu_energy = 0
            
            for line in output.split('\n'):
                if 'CPU Power' in line:
                    cpu_energy = float(line.split(':')[1].strip().split()[0])
                elif 'GPU Power' in line:
                    gpu_energy = float(line.split(':')[1].strip().split()[0])
            
            total_energy = cpu_energy + gpu_energy
            return cpu_energy, gpu_energy, total_energy
            
        except Exception as e:
            print(f"Error collecting energy metrics: {e}")
            return 0, 0, 0

    def collect_metrics(self):
        """
        Objective:
        Collect current system metrics
        
        Returns:
        None
        """
        try:
            current_time = datetime.now()
            
            # Collect CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Collect memory metrics
            memory_percent = psutil.virtual_memory().percent
            
            # Collect GPU metrics if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
            else:
                gpu_memory = 0
                
            # Collect disk I/O metrics
            disk_io_current = psutil.disk_io_counters()
            disk_read = disk_io_current.read_bytes - self.disk_io_start.read_bytes
            disk_write = disk_io_current.write_bytes - self.disk_io_start.write_bytes
            
            # Collect power metrics (if available)
            try:
                battery = psutil.sensors_battery()
                power_consumption = battery.power_plugged if battery else 0
            except:
                power_consumption = 0
                
            # Get energy metrics
            cpu_energy, gpu_energy, total_energy = self.get_energy_metrics()
            
            # Store metrics
            self.metrics['timestamp'].append(current_time)
            self.metrics['cpu_percent'].append(cpu_percent)
            self.metrics['memory_percent'].append(memory_percent)
            self.metrics['gpu_memory_allocated'].append(gpu_memory)
            self.metrics['disk_io_read'].append(disk_read)
            self.metrics['disk_io_write'].append(disk_write)
            self.metrics['power_consumption'].append(power_consumption)
            self.metrics['cpu_energy'].append(cpu_energy)
            self.metrics['gpu_energy'].append(gpu_energy)
            self.metrics['total_energy'].append(total_energy)
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            raise

    def save_metrics(self, script_name):
        """
        Objective:
        Save collected metrics to CSV file
        
        Parameters:
        [str] script_name - Name of the script being monitored
        
        Returns:
        None
        """
        try:
            df = pd.DataFrame(self.metrics)
            output_file = os.path.join(self.output_dir, f"{script_name}_resources.csv")
            df.to_csv(output_file, index=False)
            
        except Exception as e:
            print(f"Error saving metrics: {e}")
            raise

def monitor_resources(output_dir):
    """
    Objective:
    Decorator to monitor resources during function execution
    
    Parameters:
    [str] output_dir - Directory to save monitoring results
    
    Returns:
    [function] wrapper - Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = ResourceMonitor(output_dir)
            
            # Start monitoring thread
            import threading
            stop_monitoring = threading.Event()
            
            def monitoring_task():
                while not stop_monitoring.is_set():
                    monitor.collect_metrics()
                    time.sleep(1)  # Collect metrics every second
                    
            monitor_thread = threading.Thread(target=monitoring_task)
            monitor_thread.start()
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Stop monitoring
                stop_monitoring.set()
                monitor_thread.join()
                
                # Save metrics
                script_name = func.__name__
                monitor.save_metrics(script_name)
                
                return result
                
            except Exception as e:
                stop_monitoring.set()
                monitor_thread.join()
                raise e
                
        return wrapper
    return decorator