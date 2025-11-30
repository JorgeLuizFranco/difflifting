#!/usr/bin/env python3
"""
Analyze hyperparameter optimization results for DiffLifting Point Cloud experiments.
Based on the existing analysis code but adapted for our specific experiment structure.
"""

import os
import torch
import numpy as np
from collections import defaultdict

# Configuration
datasets = ["Citeseer", "Texas", "Wisconsin", "Cora"]
gnns = ["GPS"]  # We used GPS in our hyperopt
embedding_dims = [32, 64, 128]
num_layers = [2, 3]
seeds = [3, 42, 9]
k_max_values = [3, 5, 7, 9, 11]

# Results directory
results_dir = "results_diffLifting_point_cloud_hyperopt_by_tnn/UniGCNII/results_diffLifting_point_cloud_hyperopt"

def find_first_best_val_and_test(logdir):
    """Find the first occurrence of best validation accuracy and corresponding test accuracy."""
    try:
        data = torch.load(logdir, map_location=torch.device('cpu'))
        val_accuracies = data["val_accuracies"]
        test_accuracies = data["test_accuracies"]

        max_val = torch.max(val_accuracies)
        first_best_idx = (val_accuracies == max_val).nonzero()[0].item()
        return val_accuracies[first_best_idx].item(), test_accuracies[first_best_idx].item()
    except Exception as e:
        print(f"❌ Error processing {logdir}: {e}")
        return None, None

def analyze_results():
    """Analyze hyperparameter optimization results."""
    print("🔍 ANALYZING DIFFLIFTING POINT CLOUD HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Track results by dataset
    final_results = []
    all_experiments = []

    for dataset in datasets:
        print(f"\n📊 Processing dataset: {dataset}")
        seed_results = []

        for seed in seeds:
            best_val_acc = -1
            best_test_acc = -1
            best_hyperparams = None
            best_config = None

            # Iterate over hyperparameters
            for gnn in gnns:
                for embed_dim in embedding_dims:
                    for layers in num_layers:
                        for k_max in k_max_values:
                            folder_name = f"{dataset}_{gnn}_{embed_dim}_{layers}_{seed}_kmax{k_max}"
                            folder_path = os.path.join(results_dir, folder_name)

                            if not os.path.exists(folder_path):
                                print(f"   ⚠️  Missing: {folder_name}")
                                continue

                            # Find results files
                            results_files = [f for f in os.listdir(folder_path) if f.endswith(".results")]
                            if len(results_files) != 1:
                                print(f"   ⚠️  Invalid results in {folder_name} (found {len(results_files)} .results files)")
                                continue

                            logdir = os.path.join(folder_path, results_files[0])
                            val_acc, test_acc = find_first_best_val_and_test(logdir)
                            
                            if val_acc is not None:
                                print(f"   📈 {dataset} | seed={seed} | gnn={gnn} | dim={embed_dim} | layers={layers} | k_max={k_max} | val={val_acc:.4f} | test={test_acc:.4f}")
                                
                                # Store all experiments for detailed analysis
                                all_experiments.append({
                                    'dataset': dataset,
                                    'seed': seed,
                                    'gnn': gnn,
                                    'embed_dim': embed_dim,
                                    'layers': layers,
                                    'k_max': k_max,
                                    'val_acc': val_acc,
                                    'test_acc': test_acc
                                })
                                
                                # Update best for this seed if better validation accuracy
                                # In case of tie, choose better test accuracy
                                if (val_acc > best_val_acc) or (val_acc == best_val_acc and test_acc > best_test_acc):
                                    best_val_acc = val_acc
                                    best_test_acc = test_acc
                                    best_hyperparams = (gnn, embed_dim, layers, k_max)
                                    best_config = {
                                        'gnn': gnn,
                                        'embed_dim': embed_dim,
                                        'layers': layers,
                                        'k_max': k_max
                                    }

            # Store best result for this seed
            if best_hyperparams:
                seed_results.append({
                    "seed": seed,
                    "test_acc": best_test_acc * 100,
                    "val_acc": best_val_acc * 100,
                    "config": best_config
                })
                print(f"   🏆 Best for seed {seed}: val={best_val_acc*100:.2f}%, test={best_test_acc*100:.2f}%, config={best_config}")

        # Calculate mean and std for dataset
        if seed_results:
            test_accs = [res["test_acc"] for res in seed_results]
            val_accs = [res["val_acc"] for res in seed_results]
            mean_test_acc = np.mean(test_accs)
            std_test_acc = np.std(test_accs)
            mean_val_acc = np.mean(val_accs)
            std_val_acc = np.std(val_accs)
            
            final_results.append({
                "dataset": dataset,
                "mean_test_acc": mean_test_acc,
                "std_test_acc": std_test_acc,
                "mean_val_acc": mean_val_acc,
                "std_val_acc": std_val_acc,
                "details": seed_results,
            })

    # Print summary
    print("\n" + "=" * 80)
    print("📋 FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"{'Dataset':<12} {'Test Acc (Mean±Std)':<20} {'Val Acc (Mean±Std)':<20} {'Details'}")
    print("-" * 80)
    
    for res in final_results:
        print(f"{res['dataset']:<12} {res['mean_test_acc']:>6.2f}±{res['std_test_acc']:<5.2f}      {res['mean_val_acc']:>6.2f}±{res['std_val_acc']:<5.2f}")
        
        # Show best config for each seed
        for detail in res["details"]:
            config = detail["config"]
            print(f"  Seed {detail['seed']:>2}: {detail['test_acc']:>5.2f}% | dim={config['embed_dim']}, layers={config['layers']}, k_max={config['k_max']}")

    # Print LaTeX format
    print("\n" + "=" * 80)
    print("📄 RESULTS IN LATEX FORMAT")
    print("=" * 80)
    
    for res in final_results:
        print(f"{res['dataset']} & ${res['mean_test_acc']:.2f} \\textcolor{{gray}}{{\\scriptstyle{{\\pm {res['std_test_acc']:.2f}}}}}$ \\\\")

    # Hyperparameter analysis
    print("\n" + "=" * 80)
    print("🔧 HYPERPARAMETER ANALYSIS")
    print("=" * 80)
    
    # Group by hyperparameters to see trends
    hp_analysis = defaultdict(list)
    
    for exp in all_experiments:
        hp_analysis['embed_dim'].append((exp['embed_dim'], exp['test_acc']))
        hp_analysis['layers'].append((exp['layers'], exp['test_acc']))
        hp_analysis['k_max'].append((exp['k_max'], exp['test_acc']))
    
    for hp_name, values in hp_analysis.items():
        print(f"\n📊 {hp_name.upper()} Analysis:")
        hp_stats = defaultdict(list)
        
        for val, acc in values:
            hp_stats[val].append(acc)
        
        for val in sorted(hp_stats.keys()):
            accs = hp_stats[val]
            mean_acc = np.mean(accs) * 100
            std_acc = np.std(accs) * 100
            print(f"  {hp_name}={val}: {mean_acc:.2f}±{std_acc:.2f}% (n={len(accs)})")

    # Overall statistics
    all_test_accs = [exp['test_acc'] * 100 for exp in all_experiments]
    print(f"\n📈 Overall Statistics:")
    print(f"  Total experiments: {len(all_experiments)}")
    print(f"  Mean test accuracy: {np.mean(all_test_accs):.2f}±{np.std(all_test_accs):.2f}%")
    print(f"  Best single result: {np.max(all_test_accs):.2f}%")
    print(f"  Worst single result: {np.min(all_test_accs):.2f}%")

if __name__ == "__main__":
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        print(f"   Run the hyperparameter optimization first with: bash run_hyperopt.sh")
        exit(1)
    
    analyze_results()
    print(f"\n🎉 Analysis completed!")