#!/usr/bin/env python
import subprocess
import sys
import os
import time
import argparse
import math
import yaml
import re
import tempfile

# Predefined bundle to num_clusters mapping
BUNDLE_CONFIG = {
    'CG_left': '12', 'CG_right': '12', 'FPT_right': '12', 'ILF_left': '15',
    'MLF_right': '15', 'OR_right': '12', 'SCP_left': '12', 'SCP_right': '12',
    'ST_FO_left': '12', 'ST_FO_right': '12', 'ST_OCC_left': '12', 'ST_OCC_right': '12',
    'ST_PAR_left': '12', 'ST_POSTC_right': '12', 'ST_PREF_left': '12', 'ST_PREF_right': '12',
    'ST_PREC_left': '12', 'ST_PREC_right': '12', 'ST_PREM_right': '12',
    'T_OCC_right': '12', 'T_OCC_left': '12', 'T_PAR_left': '12', 'T_PAR_right': '12',
    'T_PREM_right': '12', 'OR_left': '12', 'OR_right': '12', 'UF_left': '7',
    'SLF_II_right': '12', 'SLF_II_left': '12', 'SLF_I_left': '12', 'SLF_I_right': '12',
    'AF_left': '12', 'AF_right': '12', 'ATR_left': '10', 'ATR_right': '10',
    'CST_left': '12', 'CST_right': '12',
    "SLF_III_left": "12", "SLF_III_right": "12",
    "CC_1": "12", "CC_2": "12", "CC_3": "12", "CC_4": "12", "CC_5": "12", "CC_6": "12", "CC_7": "12",
}

DEFAULT_CONFIG_PATH = "/example/path/to/config/train.yaml"
TEMP_CONFIG_DIR = "/example/path/to/temp_fmri_configs"

def get_bundles_from_directory(directory, pattern=None):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return []
    bundles = set()
    for item in os.listdir(directory):
        if pattern:
            match = re.search(pattern, item)
            if match:
                bundle_name = match.group(1) if match.groups() else match.group(0)
                bundles.add(bundle_name)
        else:
            name_part = os.path.splitext(item)[0]
            if os.path.isdir(os.path.join(directory, item)) or \
               any(item.endswith(ext) for ext in ['.tck', '.trk', '.fib', '.mat']):
                bundles.add(name_part)
            elif not os.path.splitext(item)[1]:
                bundles.add(item)
    return sorted(list(bundles))

def parse_args():
    parser = argparse.ArgumentParser(description='Run fMRI training command')
    bundle_group = parser.add_mutually_exclusive_group()
    bundle_group.add_argument('--bundle', type=str, nargs='+', default=None, help='List of bundle names to process')
    bundle_group.add_argument('--bundle-dir', type=str, default='/example/path/to/fmri_correlation', help='Directory containing all bundles, auto extract bundle names')
    bundle_group.add_argument('--bundle-pattern', type=str, help='Regex pattern to extract bundle names from --bundle-dir')
    parser.add_argument('--run-direct', action='store_true', help='Run command directly instead of in background')
    parser.add_argument('--log', type=str, help='Log file path (for main log in background mode)')
    parser.add_argument('--gpu', type=str, nargs='+', default=['0'], help='List of GPU IDs')
    parser.add_argument('--num_clusters', type=str, nargs='+', default=[], help='List of cluster numbers, overrides config if provided')
    parser.add_argument('--parallel', type=int, default=1, help='Number of tasks to run in parallel (suggested to match GPU count)')
    parser.add_argument('--use-config', action='store_true', help='Use predefined BUNDLE_CONFIG for cluster numbers, ignore num_clusters argument')
    parser.add_argument('--base-config', type=str, default=DEFAULT_CONFIG_PATH, help='Base config file path')
    args = parser.parse_args()
    if args.bundle_dir and not args.bundle:
        print(f"Getting bundle list from directory '{args.bundle_dir}'...")
        args.bundle = get_bundles_from_directory(args.bundle_dir, args.bundle_pattern)
        if not args.bundle:
            print(f"Warning: No bundles found in directory '{args.bundle_dir}'. Using default bundle 'SLF_II_right'.")
            args.bundle = ['SLF_II_right']
        else:
            print(f"Found bundles: {args.bundle}")
    elif not args.bundle:
        args.bundle = ['SLF_II_right']
    return args

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
        return {}
    except Exception as e:
        print(f"Failed to load config file '{config_path}': {e}")
        return {}

def run_command(base_config_path=DEFAULT_CONFIG_PATH, bundle='SLF_II_right', gpu='0', num_clusters='12'):
    print(f"Run_command: bundle={bundle}, gpu={gpu}, num_clusters={num_clusters}, base_config={base_config_path}")
    current_task_config = load_config(base_config_path)
    if not current_task_config:
        print(f"Error: Could not load base config file {base_config_path} for bundle {bundle}")
        sys.exit(1)
    print(f"Process started, PID: {os.getpid()} for bundle {bundle}")
    current_task_config['bundle'] = bundle
    current_task_config['num_clusters'] = int(num_clusters)
    current_task_config['GPU'] = int(gpu)
    current_task_config['output_name'] = bundle
    current_task_config['fmri_path'] = f"/example/path/to/TractSeg_mat_test/{bundle}.tck/train"
    current_task_config['fmri_similarity_path'] = f"/example/path/to/fmri_correlation/{bundle}"
    if 'outputDirectory' not in current_task_config or not current_task_config['outputDirectory']:
        current_task_config['outputDirectory'] = "/example/path/to/output"
    os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)
    temp_config_filename = f"temp_config_{bundle}_gpu{gpu}_{time.strftime('%Y%m%d%H%M%S')}.yaml"
    temp_config_filepath = os.path.join(TEMP_CONFIG_DIR, temp_config_filename)
    try:
        with open(temp_config_filepath, 'w') as f:
            yaml.dump(current_task_config, f, sort_keys=False)
        print(f"Generated temp config file for bundle {bundle}: {temp_config_filepath}")
    except Exception as e:
        print(f"Error: Could not write temp config file {temp_config_filepath} for bundle {bundle}: {e}")
        sys.exit(1)
    cmd = [
        "/example/path/to/python",
        "/example/path/to/fmri_train/train_bilateral_Fan_bundle_fmri_origin.py",
        "--config", temp_config_filepath,
    ]
    try:
        print(f"Starting command for bundle {bundle}...")
        print(f"Command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        if process.stdout:
            for line in process.stdout:
                print(line, end='', flush=True)
        process.wait()
        print(f"Command finished for bundle {bundle}, exit code: {process.returncode}")
        if process.returncode != 0:
            print(f"Warning: bundle {bundle} command failed, exit code {process.returncode}")
    except FileNotFoundError:
        print(f"Error: Could not find Python interpreter or target script for bundle {bundle}.")
        print(f"Python interpreter: {cmd[0]}")
        print(f"Target script: {cmd[1]}")
        sys.exit(1)
    except Exception as e:
        print(f"Command failed for bundle {bundle}: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(temp_config_filepath):
            print(f"Temp config file kept for debugging: {temp_config_filepath}")


def run_in_background(base_config_path=DEFAULT_CONFIG_PATH, log_file=None, bundles=['SLF_II_right'],
                     gpus=['0'], num_clusters=[], parallel=1, use_config=False, max_tasks_per_gpu=2):
    current_script = os.path.abspath(__file__)
    tasks_to_run = []
    if use_config or not num_clusters:
        for bundle_item in bundles:
            num_cluster = BUNDLE_CONFIG.get(bundle_item, '12')
            tasks_to_run.append((bundle_item, num_cluster))
    else:
        min_len = min(len(bundles), len(num_clusters))
        if len(bundles) > len(num_clusters) and num_clusters:
            num_clusters_filled = num_clusters + [num_clusters[-1]] * (len(bundles) - len(num_clusters))
            tasks_to_run = list(zip(bundles, num_clusters_filled))
        elif len(bundles) < len(num_clusters):
            tasks_to_run = list(zip(bundles, num_clusters[:len(bundles)]))
        elif not num_clusters and bundles:
             for bundle_item in bundles:
                 tasks_to_run.append((bundle_item, BUNDLE_CONFIG.get(bundle_item, '12')))
        else:
            tasks_to_run = list(zip(bundles, num_clusters))
    if not tasks_to_run and bundles:
        for bundle_item in bundles:
            tasks_to_run.append((bundle_item, BUNDLE_CONFIG.get(bundle_item, '12')))

    total_initial_tasks = len(tasks_to_run)
    if total_initial_tasks == 0:
        print("No tasks to execute. Please check the bundle list.")
        return
    print(f"Total {total_initial_tasks} tasks to execute: {tasks_to_run}")

    main_log_dir = "/example/path/to/logs_fmri"
    if log_file:
        main_log_dir = os.path.dirname(os.path.abspath(log_file))
    os.makedirs(main_log_dir, exist_ok=True)
    os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)

    gpu_task_counts = {gpu_id: 0 for gpu_id in gpus}
    active_processes_info = []
    task_queue = list(tasks_to_run)
    launched_task_count = 0

    print(f"Max {max_tasks_per_gpu} tasks per GPU allowed.")
    print(f"Available GPU list: {gpus}")

    while launched_task_count < total_initial_tasks or active_processes_info:
        if active_processes_info:
            completed_pids_info = []
            remaining_active_pids_info = []
            for pid_file, assigned_gpu in active_processes_info:
                if not os.path.exists(pid_file):
                    print(f"Task (PID file {pid_file}, GPU {assigned_gpu}) finished, GPU released.")
                    gpu_task_counts[assigned_gpu] = max(0, gpu_task_counts[assigned_gpu] - 1)
                    completed_pids_info.append((pid_file, assigned_gpu))
                else:
                    try:
                        with open(pid_file, 'r') as f_pid:
                            pid = int(f_pid.read().strip())
                        subprocess.check_call(f"kill -0 {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        remaining_active_pids_info.append((pid_file, assigned_gpu))
                    except (FileNotFoundError, ValueError, subprocess.CalledProcessError):
                        print(f"Task (PID file {pid_file}, GPU {assigned_gpu}) seems finished, GPU released.")
                        gpu_task_counts[assigned_gpu] = max(0, gpu_task_counts[assigned_gpu] - 1)
                        completed_pids_info.append((pid_file, assigned_gpu))
                        if os.path.exists(pid_file):
                            try: os.remove(pid_file)
                            except OSError as e: print(f"Failed to remove PID file {pid_file}: {e}")
            active_processes_info = remaining_active_pids_info

        can_launch_more_in_parallel = len(active_processes_info) < parallel
        while task_queue and can_launch_more_in_parallel:
            assigned_gpu_for_task = None
            for gpu_id in gpus:
                if gpu_task_counts[gpu_id] < max_tasks_per_gpu:
                    assigned_gpu_for_task = gpu_id
                    break
            if assigned_gpu_for_task:
                current_task_bundle, current_task_num_clusters = task_queue.pop(0)
                gpu_task_counts[assigned_gpu_for_task] += 1
                launched_task_count += 1
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                task_log_filename = f"train_{current_task_bundle}_gpu{assigned_gpu_for_task}_{timestamp}.log"
                current_task_log_path = os.path.join(main_log_dir, task_log_filename)
                pid_file_name = f"pid_{current_task_bundle}_gpu{assigned_gpu_for_task}_{timestamp}.txt"
                pid_file_path = os.path.join(main_log_dir, pid_file_name)
                nohup_cmd_parts = [
                    "nohup", "python", current_script, "--run-direct",
                    "--base-config", base_config_path,
                    "--bundle", current_task_bundle,
                    "--gpu", assigned_gpu_for_task,
                    "--num_clusters", str(current_task_num_clusters),
                    ">", current_task_log_path, "2>&1", "&",
                    "echo $! >", pid_file_path
                ]
                nohup_cmd_str = " ".join(nohup_cmd_parts)
                print(f"Start background task [{launched_task_count}/{total_initial_tasks}]: {current_task_bundle} on GPU {assigned_gpu_for_task} (current GPU task count: {gpu_task_counts[assigned_gpu_for_task]})")
                print(f"Command: {nohup_cmd_str}")
                print(f"Task log: {current_task_log_path}")
                try:
                    subprocess.run(nohup_cmd_str, shell=True, check=True)
                    active_processes_info.append((pid_file_path, assigned_gpu_for_task))
                except subprocess.CalledProcessError as e:
                    print(f"Error: Failed to start background task for {current_task_bundle}. Error: {e}")
                    gpu_task_counts[assigned_gpu_for_task] = max(0, gpu_task_counts[assigned_gpu_for_task] - 1)
                    launched_task_count -=1
                print(f"Check progress: tail -f {current_task_log_path}")
                print(f"PID file: {pid_file_path}")
                can_launch_more_in_parallel = len(active_processes_info) < parallel
            else:
                break
        if not task_queue and not active_processes_info:
            break
        print(f"Current GPU task distribution: {gpu_task_counts}. Active processes: {len(active_processes_info)}. Remaining task queue: {len(task_queue)}")
        time.sleep(10)
    print("All tasks have been processed.")


def wait_for_batch_completion(pid_files_with_gpu):
    print("wait_for_batch_completion has been replaced by the new task management logic and should not be called directly.")
    pass

if __name__ == "__main__":
    args = parse_args()
    max_tasks_per_gpu_val = 2
    if hasattr(args, 'max_tasks_per_gpu'):
        max_tasks_per_gpu_val = args.max_tasks_per_gpu
    if args.run_direct:
        print("Direct execution mode...")
        current_bundle = args.bundle[0] if args.bundle else 'SLF_II_right'
        if args.num_clusters:
            current_num_clusters = args.num_clusters[0]
        elif args.use_config:
            current_num_clusters = BUNDLE_CONFIG.get(current_bundle, '12')
        else:
            current_num_clusters = BUNDLE_CONFIG.get(current_bundle, '12')
        current_gpu = args.gpu[0] if args.gpu else '0'
        run_command(base_config_path=args.base_config, bundle=current_bundle, gpu=current_gpu, num_clusters=current_num_clusters)
    else:
        print("Background execution mode...")
        main_log_dir_for_nohup = "/example/path/to/logs_fmri"
        if args.log:
            main_log_dir_for_nohup = os.path.dirname(os.path.abspath(args.log))
        os.makedirs(main_log_dir_for_nohup, exist_ok=True)
        run_in_background(
            base_config_path=args.base_config,
            log_file=args.log,
            bundles=args.bundle,
            gpus=args.gpu,
            num_clusters=args.num_clusters,
            parallel=args.parallel,
            use_config=args.use_config,
            max_tasks_per_gpu=max_tasks_per_gpu_val
        )
