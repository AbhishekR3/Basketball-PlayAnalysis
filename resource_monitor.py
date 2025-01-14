import psutil
import torch
import pandas as pd
import os
import time
from datetime import datetime
from threading import Event, Thread
import subprocess
from functools import wraps

class ResourceMonitor:
    """
    Objective:
    Monitor system resources including CPU, memory, GPU, and disk I/O
    Specifically optimized for M1 Mac environments
    """
    
    def __init__(self, output_dir):
        """
        Objective:
        Initialize the resource monitor with appropriate metric collection capabilities

        Parameters:
        [str] output_dir - Directory where monitoring results will be saved

        Returns:
        None
        """
        try:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize base metrics that should always be available
            self.metrics = {
                'timestamp': [],
                'cpu_percent': [],
                'memory_percent': [],
                'disk_io_read': [],
                'disk_io_write': []
            }
            
            # Initialize GPU metrics if available
            if torch.backends.mps.is_available():
                self.metrics['gpu_memory_allocated'] = []
            
            # Initialize power metrics if on M1
            self.has_power_metrics = self._check_power_metrics()
            if self.has_power_metrics:
                self.metrics.update({
                    'power_consumption': [],
                    'cpu_energy': [],
                    'gpu_energy': [],
                    'total_energy': []
                })
            
            # Initialize disk I/O baseline
            self.disk_io_start = psutil.disk_io_counters()
            
        except Exception as e:
            print(f"Error initializing ResourceMonitor: {e}")
            raise

    def _check_power_metrics(self):
        """
        Objective:
        Check if power metrics are available and accessible

        Returns:
        [bool] has_power_metrics - Whether power metrics are available
        """
        try:
            # Check if we're on macOS ARM
            if not (os.uname().sysname == 'Darwin' and os.uname().machine == 'arm64'):
                return False
                
            # Check powermetrics access
            try:
                result = subprocess.run(['powermetrics', '-n', '0'], 
                                     capture_output=True, 
                                     timeout=1)
                return result.returncode == 0
            except:
                return False
                
        except Exception as e:
            print(f"Error checking power metrics: {e}")
            return False

    def collect_metrics(self):
        """
        Objective:
        Collect current system resource metrics

        Returns:
        None
        """
        try:
            current_time = datetime.now()
            
            # Always collect base metrics
            self.metrics['timestamp'].append(current_time)
            self.metrics['cpu_percent'].append(psutil.cpu_percent(interval=0.1))
            self.metrics['memory_percent'].append(psutil.virtual_memory().percent)
            
            # Collect disk I/O
            current_io = psutil.disk_io_counters()
            self.metrics['disk_io_read'].append(
                current_io.read_bytes - self.disk_io_start.read_bytes)
            self.metrics['disk_io_write'].append(
                current_io.write_bytes - self.disk_io_start.write_bytes)
            
            # Collect GPU metrics if available
            if 'gpu_memory_allocated' in self.metrics and torch.backends.mps.is_available():
                self.metrics['gpu_memory_allocated'].append(
                    torch.mps.current_allocated_memory() / 1024**2)  # Convert to MB
            
            # Collect power metrics if available
            if self.has_power_metrics:
                power_data = self._get_power_metrics()
                if power_data:
                    self.metrics['power_consumption'].append(power_data.get('power', 0))
                    self.metrics['cpu_energy'].append(power_data.get('cpu_energy', 0))
                    self.metrics['gpu_energy'].append(power_data.get('gpu_energy', 0))
                    self.metrics['total_energy'].append(power_data.get('total_energy', 0))
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")

    def _get_power_metrics(self):
        """
        Objective:
        Get power consumption metrics for M1 Macs

        Returns:
        [dict] metrics - Power consumption metrics
        """
        try:
            if not self.has_power_metrics:
                return None
                
            cmd = ['powermetrics', '-n', '1', '-i', '100', '--show-process-energy']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1)
            
            if result.returncode != 0:
                return None
                
            metrics = {}
            for line in result.stdout.split('\n'):
                if 'CPU Power' in line:
                    metrics['cpu_energy'] = float(line.split(':')[1].strip().split()[0])
                elif 'GPU Power' in line:
                    metrics['gpu_energy'] = float(line.split(':')[1].strip().split()[0])
                    
            metrics['total_energy'] = metrics.get('cpu_energy', 0) + metrics.get('gpu_energy', 0)
            metrics['power'] = metrics['total_energy']  # Current power consumption
            
            return metrics
            
        except Exception as e:
            print(f"Error getting power metrics: {e}")
            return None

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
            stop_monitoring = Event()
            
            def monitoring_task():
                while not stop_monitoring.is_set():
                    try:
                        monitor.collect_metrics()
                        time.sleep(1)
                    except Exception as e:
                        print(f"Error in monitoring task: {e}")
                        break
            
            monitor_thread = Thread(target=monitoring_task)
            monitor_thread.daemon = True  # Ensure thread terminates with main program
            monitor_thread.start()
            
            try:
                result = func(*args, **kwargs)
                stop_monitoring.set()
                monitor_thread.join(timeout=2)
                monitor.save_metrics(func.__name__)
                return result
            except Exception as e:
                stop_monitoring.set()
                monitor_thread.join(timeout=2)
                raise e
                
        return wrapper
    return decorator