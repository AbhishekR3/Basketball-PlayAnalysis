import subprocess
import sys
import os
from resource_monitor import monitor_resources
import pandas as pd

def run_script(script_name):
    """
    Objective:
    Run a Python script and capture its output and return code
    
    Parameters:
    [str] script_name - Name of the Python script to run
    
    Returns:
    [bool] success - Whether the script executed successfully
    """
    try:
        result = subprocess.run(['python3', script_name], 
                              capture_output=True, 
                              text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Errors from {script_name}:", result.stderr, file=sys.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False

@monitor_resources("/app/output/logs")
def run_passing_simulation():
    """
    Objective:
    Run the passing simulation with resource monitoring
    
    Returns:
    [bool] success - Whether the simulation completed successfully
    """
    try:
        return run_script('Passing_Simulation.py')
    except Exception as e:
        print(f"Error in passing simulation: {e}")
        return False

@monitor_resources("/app/output/logs")
def run_object_tracking():
    """
    Objective:
    Run object tracking with resource monitoring
    
    Returns:
    [bool] success - Whether the tracking completed successfully
    """
    try:
        return run_script('Object_Tracking.py')
    except Exception as e:
        print(f"Error in object tracking: {e}")
        return False

@monitor_resources("/app/output/logs")
def run_feature_engineering():
    """
    Objective:
    Run feature engineering with resource monitoring
    
    Returns:
    [bool] success - Whether the feature engineering completed successfully
    """
    try:
        return run_script('Feature_Engineering.py')
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return False

def aggregate_metrics():
    """
    Objective:
    Aggregate and summarize resource usage metrics from all stages
    
    Returns:
    [dict] summary - Dictionary containing resource usage summaries
    """
    try:
        log_dir = "/app/output/logs"
        stages = ['run_passing_simulation', 'run_object_tracking', 'run_feature_engineering']
        summary = {}
        
        for stage in stages:
            metrics_file = os.path.join(log_dir, f"{stage}_resources.csv")
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                summary[stage] = {
                    'avg_cpu': df['cpu_percent'].mean(),
                    'max_cpu': df['cpu_percent'].max(),
                    'avg_memory': df['memory_percent'].mean(),
                    'max_memory': df['memory_percent'].max(),
                    'total_disk_write': df['disk_io_write'].max(),
                    'total_disk_read': df['disk_io_read'].max()
                }
                
                if 'gpu_memory_allocated' in df.columns:
                    summary[stage].update({
                        'avg_gpu': df['gpu_memory_allocated'].mean(),
                        'max_gpu': df['gpu_memory_allocated'].max()
                    })
        
        # Save summary to file
        summary_file = os.path.join(log_dir, 'resource_summary.txt')
        with open(summary_file, 'w') as f:
            for stage, metrics in summary.items():
                f.write(f"\n{stage} Summary:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.2f}\n")
                    
        return summary
    except Exception as e:
        print(f"Error aggregating metrics: {e}")
        return {}

def main():
    """
    Objective:
    Main execution function that runs all stages with monitoring
    """
    try:
        stages = [
            ('Passing Simulation', run_passing_simulation),
            ('Object Tracking', run_object_tracking),
            ('Feature Engineering', run_feature_engineering)
        ]
        
        for name, func in stages:
            print(f"\nStarting {name}...")
            if not func():
                print(f"{name} failed!")
                sys.exit(1)
            print(f"{name} completed successfully!")
        
        print("\nGenerating resource usage summary...")
        aggregate_metrics()
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()