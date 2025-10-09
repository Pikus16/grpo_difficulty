#!/usr/bin/env python3
"""
Run all validation experiments for the scaling law model
This is the main entry point for final validation
"""

import os
import sys
import subprocess
from datetime import datetime


def run_script(script_name):
    """Run a Python script and capture output"""
    print(f"\n{'='*70}")
    print(f"Running {script_name}...")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False


def check_data_files():
    """Check if required data files exist"""
    required_files = [
        '../scaling_analysis_results.csv',
        '../held_out_scaling_numbers.csv'
    ]
    
    print("Checking data files...")
    all_exist = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} NOT FOUND")
            all_exist = False
    
    return all_exist


def main():
    """Run all validation experiments"""
    print("="*70)
    print("GRPO SCALING LAW - FINAL VALIDATION SUITE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check data files
    if not check_data_files():
        print("\n❌ Missing required data files. Please ensure:")
        print("   - scaling_analysis_results.csv")
        print("   - held_out_scaling_numbers.csv")
        print("   are in the parent directory.")
        return False
    
    # List of scripts to run in order
    scripts = [
        ('smoke_tests.py', 'Comprehensive smoke tests'),
        ('calibration_ablations.py', 'Calibration and ablation analyses'),
        ('final_readiness_checklist.py', 'Final readiness verification')
    ]
    
    print(f"\nWill run {len(scripts)} validation scripts:")
    for script, desc in scripts:
        print(f"  - {script}: {desc}")
    
    # Run each script
    all_success = True
    for script, desc in scripts:
        if not run_script(script):
            print(f"\n❌ Failed to run {script}")
            all_success = False
            # Continue anyway to see what we can salvage
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUITE COMPLETE")
    print("="*70)
    
    if all_success:
        print("\n✅ All validations completed successfully!")
        
        # List generated files
        print("\nGenerated files:")
        output_files = [
            # From smoke tests
            'time_forward_results.csv',
            'decision_utility_results.csv',
            'time_forward_evaluation.png',
            'decision_utility_curves.png',
            
            # From calibration/ablations
            'calibration_ablation_summary.csv',
            'calibration_analysis.png',
            'ablation_study.png',
            
            # From readiness checklist
            'readiness_report.csv',
            'latex_summary.tex'
        ]
        
        for file in output_files:
            if os.path.exists(file):
                print(f"  ✓ {file}")
            else:
                print(f"  - {file} (not generated)")
        
        print("\nNext steps:")
        print("1. Review final_experiments_summary.md for overall results")
        print("2. Check readiness_report.csv for publication readiness")
        print("3. Use latex_summary.tex for paper tables/equations")
        
    else:
        print("\n⚠️  Some validations failed. Check individual outputs.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
