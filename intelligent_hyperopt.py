#!/usr/bin/env python3
"""
Intelligent hyperparameter optimization script that:
- Analyzes existing results and logs
- Skips completed experiments 
- Identifies failed experiments and reasons
- Resumes from where it stopped
- Runs only missing configurations
"""

import os
import subprocess
import time
import glob
from datetime import datetime
import re

def analyze_experiment_status():
    """Analyze the status of all experiments."""
    
    # Configuration
    datasets = ["Citeseer", "Texas", "Wisconsin", "Cora"]
    gnns = ["GPS"]
    embedding_dims = [32, 64, 128]
    num_layers = [2, 3]
    seeds = [3, 42, 9]
    k_max_values = [3, 5, 7, 9, 11]
    
    results_dir = "results_diffLifting_point_cloud_hyperopt"
    
    # Generate all expected experiments
    all_experiments = []
    for dataset in datasets:
        for gnn in gnns:
            for embed_dim in embedding_dims:
                for layers in num_layers:
                    for seed in seeds:
                        for k_max in k_max_values:
                            exp_name = f"{dataset}_{gnn}_{embed_dim}_{layers}_{seed}_kmax{k_max}"
                            all_experiments.append({
                                'name': exp_name,
                                'dataset': dataset,
                                'gnn': gnn,
                                'embed_dim': embed_dim,
                                'layers': layers,
                                'seed': seed,
                                'k_max': k_max
                            })
    
    print(f"📊 ANALYZING {len(all_experiments)} TOTAL EXPERIMENTS")
    print("=" * 80)
    
    completed = []
    failed = []
    missing = []
    
    for exp in all_experiments:
        exp_path = os.path.join(results_dir, exp['name'])
        
        if not os.path.exists(exp_path):
            missing.append(exp)
            continue
            
        # Check for .results file (success indicator)
        results_files = glob.glob(os.path.join(exp_path, "*.results"))
        
        if len(results_files) == 1:
            # Experiment completed successfully
            results_file = results_files[0]
            mtime = os.path.getmtime(results_file)
            exp['completion_time'] = datetime.fromtimestamp(mtime)
            exp['results_file'] = results_file
            completed.append(exp)
            
        elif len(results_files) == 0:
            # Check if there are logs (failed experiment)
            log_files = glob.glob(os.path.join(exp_path, "*.log"))
            if log_files:
                # Analyze the log to understand failure
                log_file = log_files[0]
                failure_reason = analyze_log_failure(log_file)
                exp['log_file'] = log_file
                exp['failure_reason'] = failure_reason
                exp['log_mtime'] = datetime.fromtimestamp(os.path.getmtime(log_file))
                failed.append(exp)
            else:
                # No results, no logs - treat as missing
                missing.append(exp)
        else:
            # Multiple results files - something weird happened
            print(f"⚠️  Multiple results files found for {exp['name']}")
            failed.append(exp)
    
    return completed, failed, missing, all_experiments

def analyze_log_failure(log_file):
    """Analyze log file to determine failure reason."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Common failure patterns
        if "CUDA out of memory" in content:
            return "CUDA OOM"
        elif "Killed" in content:
            return "Process killed (OOM?)"
        elif "KeyboardInterrupt" in content:
            return "Manual interruption"
        elif "ConnectionError" in content:
            return "Network/connection error"
        elif "FileNotFoundError" in content:
            return "Missing files"
        elif "RuntimeError" in content:
            return "Runtime error"
        elif "Traceback" in content:
            return "Python exception"
        elif len(content.strip()) == 0:
            return "Empty log (early termination)"
        elif "Epoch" not in content:
            return "Training never started"
        else:
            # Check if training was progressing
            epoch_matches = re.findall(r'Epoch\s+(\d+):', content)
            if epoch_matches:
                last_epoch = max([int(e) for e in epoch_matches])
                return f"Training stopped at epoch {last_epoch}"
            else:
                return "Unknown failure"
                
    except Exception as e:
        return f"Log analysis error: {str(e)}"

def print_analysis_report(completed, failed, missing):
    """Print detailed analysis report."""
    
    print(f"✅ COMPLETED: {len(completed)}")
    print(f"❌ FAILED: {len(failed)}")
    print(f"⏳ MISSING: {len(missing)}")
    print(f"📈 Success rate: {len(completed)/(len(completed)+len(failed)+len(missing))*100:.1f}%")
    print("")
    
    # Completed experiments summary
    if completed:
        print("✅ COMPLETED EXPERIMENTS:")
        print("-" * 60)
        completed_by_dataset = {}
        for exp in completed:
            dataset = exp['dataset']
            if dataset not in completed_by_dataset:
                completed_by_dataset[dataset] = []
            completed_by_dataset[dataset].append(exp)
        
        for dataset, exps in completed_by_dataset.items():
            print(f"  {dataset}: {len(exps)} experiments")
            for exp in sorted(exps, key=lambda x: x['completion_time'], reverse=True)[:3]:
                time_str = exp['completion_time'].strftime("%m-%d %H:%M")
                print(f"    • {exp['name']} ({time_str})")
        print("")
    
    # Failed experiments analysis
    if failed:
        print("❌ FAILED EXPERIMENTS ANALYSIS:")
        print("-" * 60)
        failure_reasons = {}
        for exp in failed:
            reason = exp.get('failure_reason', 'Unknown')
            if reason not in failure_reasons:
                failure_reasons[reason] = []
            failure_reasons[reason].append(exp)
        
        for reason, exps in failure_reasons.items():
            print(f"  {reason}: {len(exps)} experiments")
            for exp in exps[:2]:  # Show first 2 examples
                time_str = exp.get('log_mtime', datetime.now()).strftime("%m-%d %H:%M")
                print(f"    • {exp['name']} ({time_str})")
        print("")
    
    # Missing experiments by configuration
    if missing:
        print("⏳ MISSING EXPERIMENTS:")
        print("-" * 60)
        missing_by_dataset = {}
        for exp in missing:
            dataset = exp['dataset']
            if dataset not in missing_by_dataset:
                missing_by_dataset[dataset] = 0
            missing_by_dataset[dataset] += 1
        
        for dataset, count in missing_by_dataset.items():
            print(f"  {dataset}: {count} experiments")
        print("")

def run_missing_experiments(missing, prevent_sleep=True):
    """Run only the missing experiments."""
    
    if not missing:
        print("🎉 No missing experiments! All configurations completed.")
        return
    
    print(f"🚀 RUNNING {len(missing)} MISSING EXPERIMENTS")
    print("=" * 80)
    
    # Prevent sleep before starting
    if prevent_sleep:
        print("🛡️ Setting up sleep prevention...")
        try:
            subprocess.run(["xset", "s", "off"], check=False, capture_output=True)
            subprocess.run(["xset", "-dpms"], check=False, capture_output=True)
            print("  ✓ Display power management disabled")
            
            # Try to mask sleep targets
            result = subprocess.run(
                ["sudo", "systemctl", "mask", "sleep.target", "suspend.target"], 
                check=False, capture_output=True, input=b'\n'
            )
            if result.returncode == 0:
                print("  ✓ System suspend targets masked")
            else:
                print("  ⚠️ Could not mask suspend targets (sudo required)")
                
        except Exception as e:
            print(f"  ⚠️ Sleep prevention setup failed: {e}")
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Start keep-alive process
    keep_alive_process = None
    if prevent_sleep:
        keep_alive_script = """
while true; do
    sleep 300
    echo "$(date): Hyperopt keep-alive - $(pgrep -f 'python.*main.py' | wc -l) processes running" > /tmp/hyperopt_keepalive.txt
    if [ ! -z "$DISPLAY" ]; then
        xdotool key shift 2>/dev/null || true
    fi
done
"""
        keep_alive_process = subprocess.Popen(["bash", "-c", keep_alive_script])
        print(f"  ✓ Keep-alive process started (PID: {keep_alive_process.pid})")
    
    try:
        start_time = time.time()
        success_count = 0
        fail_count = 0
        
        for i, exp in enumerate(missing, 1):
            print(f"\n🔄 [{i}/{len(missing)}] Running: {exp['name']}")
            
            # Build command
            cmd = [
                "python", "main.py",
                "--dataset", exp['dataset'],
                "--seed", str(exp['seed']),
                "--tnn", "UniGCNII",
                "--gnn", exp['gnn'],
                "--hidden_dim", "64",
                "--gnn_embedding_dim", str(exp['embed_dim']),
                "--num_layers", str(exp['layers']),
                "--k_max", str(exp['k_max']),
                "--lr", "0.005",
                "--weight_decay", "0.0",
                "--batch_size", "128",
                "--max_epochs", "200",
                "--early_stop_patience", "25",
                "--point_cloud",
                "--logdir", f"results_diffLifting_point_cloud_hyperopt/{exp['name']}"
            ]
            
            log_file = f"logs/{exp['name']}.log"
            
            print(f"  ⚡ Starting experiment (timeout: 30min)")
            start_exp = time.time()
            
            try:
                # Run with timeout
                with open(log_file, 'w') as f:
                    result = subprocess.run(
                        cmd, 
                        stdout=f, 
                        stderr=subprocess.STDOUT,
                        timeout=1800,  # 30 minutes
                        cwd=os.getcwd()
                    )
                
                duration = time.time() - start_exp
                
                if result.returncode == 0:
                    print(f"  ✅ Completed successfully ({duration:.0f}s)")
                    success_count += 1
                else:
                    print(f"  ❌ Failed with exit code {result.returncode} ({duration:.0f}s)")
                    fail_count += 1
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏰ Timeout after 30 minutes")
                fail_count += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
                fail_count += 1
            
            # Progress update
            progress = i * 100 / len(missing)
            elapsed = time.time() - start_time
            if i > 1:
                eta_seconds = elapsed / i * (len(missing) - i)
                eta_str = f"{eta_seconds/3600:.1f}h"
            else:
                eta_str = "calculating..."
                
            print(f"  📊 Progress: {progress:.1f}% | ✅ {success_count} | ❌ {fail_count} | ETA: {eta_str}")
            
            # Small delay between experiments
            time.sleep(3)
            
    finally:
        # Cleanup
        if keep_alive_process:
            keep_alive_process.terminate()
            print(f"\n🧹 Keep-alive process stopped")
        
        if prevent_sleep:
            # Restore power management
            try:
                subprocess.run(["xset", "s", "on"], check=False, capture_output=True)
                subprocess.run(["xset", "+dpms"], check=False, capture_output=True)
                subprocess.run(
                    ["sudo", "systemctl", "unmask", "sleep.target", "suspend.target"], 
                    check=False, capture_output=True
                )
                print("🔄 Power management restored")
            except:
                pass
        
        total_time = time.time() - start_time
        print(f"\n🎯 HYPERPARAMETER OPTIMIZATION COMPLETED!")
        print(f"  ✅ Successful: {success_count}")
        print(f"  ❌ Failed: {fail_count}")
        print(f"  ⏱️ Total time: {total_time/3600:.1f}h")

def main():
    """Main execution function."""
    print("🔍 INTELLIGENT HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Step 1: Analyze existing experiments
    completed, failed, missing, all_experiments = analyze_experiment_status()
    
    # Step 2: Print analysis report
    print_analysis_report(completed, failed, missing)
    
    # Step 3: Ask user what to do
    if missing:
        response = input(f"🤔 Found {len(missing)} missing experiments. Run them now? [y/N]: ").lower()
        if response in ['y', 'yes']:
            run_missing_experiments(missing)
        else:
            print("👋 Skipping missing experiments.")
    
    # Step 4: Handle failed experiments
    if failed:
        print("\n" + "=" * 80)
        print("🔄 FAILED EXPERIMENTS ANALYSIS")
        
        # Categorize failures
        retryable_failures = []
        permanent_failures = []
        
        for exp in failed:
            reason = exp.get('failure_reason', '')
            if any(keyword in reason.lower() for keyword in ['killed', 'oom', 'timeout', 'stopped at epoch']):
                retryable_failures.append(exp)
            else:
                permanent_failures.append(exp)
        
        if retryable_failures:
            print(f"🔄 Found {len(retryable_failures)} potentially retryable failures")
            response = input("🤔 Retry failed experiments? [y/N]: ").lower()
            if response in ['y', 'yes']:
                # Clean up failed experiments and retry
                for exp in retryable_failures:
                    exp_path = os.path.join("results_diffLifting_point_cloud_hyperopt", exp['name'])
                    if os.path.exists(exp_path):
                        subprocess.run(["rm", "-rf", exp_path])
                
                run_missing_experiments(retryable_failures)
    
    print(f"\n🎉 ANALYSIS COMPLETE! Run 'python analyze_hyperopt_results.py' for detailed results.")

if __name__ == "__main__":
    main()