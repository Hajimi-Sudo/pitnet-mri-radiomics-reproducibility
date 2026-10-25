#!/usr/bin/env python3
"""
Main Workflow Runner for PitNET MRI Radiomics Reproducibility & Stability Pipeline.
Provides a unified command-line interface to execute end-to-end data processing,
perturbation generation, IBSI extraction, reproducibility auditing, stress testing,
and table/figure generation.
"""

import os
import sys
import argparse
import yaml

def load_config(config_path="code/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_step(step_name, script_path, extra_args=""):
    print(f"\n{'='*70}")
    print(f"  [Pipeline Step] Executing: {step_name}")
    print(f"  Script: {script_path} {extra_args}")
    print(f"{'='*70}\n")
    ret = os.system(f"python3 {script_path} {extra_args}")
    if ret != 0:
        print(f"[Warning] Step {step_name} exited with status {ret}.")
    return ret

def main():
    parser = argparse.ArgumentParser(
        description="PitNET MRI Radiomics Reproducibility & Robustness Pipeline"
    )
    parser.add_argument(
        "--step",
        choices=["audit_data", "perturb", "extract", "icc_audit", "stress_test", "export_tables", "all"],
        default="all",
        help="Pipeline stage to execute (default: all)"
    )
    parser.add_argument(
        "--config",
        default="code/config.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    step_map = {
        "audit_data": os.path.join(base_dir, "01_dataset_loader.py"),
        "perturb": os.path.join(base_dir, "02_perturbation_engine.py"),
        "extract": os.path.join(base_dir, "03_radiomic_extraction.py"),
        "icc_audit": os.path.join(base_dir, "04_icc_reproducibility_audit.py"),
        "stress_test": os.path.join(base_dir, "05_clinical_stress_testing.py"),
        "export_tables": os.path.join(base_dir, "06_generate_figures_and_tables.py"),
    }

    if args.step == "all":
        for name, path in step_map.items():
            run_step(name, path)
    else:
        run_step(args.step, step_map[args.step])

    print("\n[Done] Pipeline execution sequence completed.\n")

if __name__ == "__main__":
    main()
