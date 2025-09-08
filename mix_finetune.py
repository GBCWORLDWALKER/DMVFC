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
import glob # 新增导入

BUNDLE_CONFIG = {
    'UF_left': '7'
}

DEFAULT_CONFIG_PATH = "/example/path/to/config/trained.yaml"
TEMP_CONFIG_DIR = "/example/path/to/temp_fmri_configs"

def get_bundles_from_directory(directory, pattern=None):
    if not os.path.exists(directory):
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
    parser = argparse.ArgumentParser()
    bundle_group = parser.add_mutually_exclusive_group()
    bundle_group.add_argument('--bundle', type=str, nargs='+', default=None)
    bundle_group.add_argument('--bundle-dir', type=str, default='/example/path/to/fmri_correlation')
    bundle_group.add_argument('--bundle-pattern', type=str)
    parser.add_argument('--run-direct', action='store_true')
    parser.add_argument('--log', type=str)
    parser.add_argument('--gpu', type=str, nargs='+', default=['0'])
    parser.add_argument('--num_clusters', type=str, nargs='+', default=[])
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--use-config', action='store_true')
    parser.add_argument('--base-config', type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--max-tasks-per-gpu', type=int, default=2)
    args = parser.parse_args()
    if args.bundle_dir and not args.bundle:
        args.bundle = get_bundles_from_directory(args.bundle_dir, args.bundle_pattern)
        if not args.bundle:
            args.bundle = ['SLF_II_right']
    elif not args.bundle:
        args.bundle = ['SLF_II_right']
    return args

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        return {}

def run_command(base_config_path=DEFAULT_CONFIG_PATH, bundle='SLF_II_right', gpu='0', num_clusters='12'):
    current_task_config = load_config(base_config_path)
    if not current_task_config:
        sys.exit(1)
    current_task_config['bundle'] = bundle
    current_task_config['num_clusters'] = int(num_clusters)
    current_task_config['GPU'] = [int(gpu)]
    current_task_config['fmri_path'] = f"/example/path/to/TractSeg_mat_test/{bundle}.tck/train"
    current_task_config['fmri_similarity_path'] = f"/example/path/to/fmri_correlation/{bundle}"
    model_b_type_for_output = current_task_config.get('model_b', 'unknown_model_b')
    current_task_config['output_name'] = f"{bundle}_{model_b_type_for_output}"
    if 'outputDirectory' not in current_task_config or not current_task_config['outputDirectory']:
        current_task_config['outputDirectory'] = "/example/path/to/output"
    current_task_config['pretrained_net_a'] = f"/example/path/to/out_full_train/models/{bundle}_baseline_final_k{num_clusters}.pt"
    model_b_type = current_task_config.get('model_b', 'dgcnn')
    base_output_dir_for_fmri_models = current_task_config.get('outputDirectory', "/example/path/to/output")
    fmri_model_filename = f"{bundle}_sim_scale_100_{model_b_type}_pretrain_b_k{num_clusters}.pt"
    experiment_dir_core_pattern = f"_{bundle}_sim_scale_100_{model_b_type}"
    search_path_pattern = os.path.join(base_output_dir_for_fmri_models, f"*{experiment_dir_core_pattern}")
    found_experiment_dirs = []
    if os.path.exists(base_output_dir_for_fmri_models):
        potential_dirs = glob.glob(os.path.join(base_output_dir_for_fmri_models, "*"))
        for d_path in potential_dirs:
            if os.path.isdir(d_path) and os.path.basename(d_path).endswith(experiment_dir_core_pattern):
                if re.match(r"^\d{2,}_\d{2,}_\d{2,}_.*", os.path.basename(d_path)):
                    found_experiment_dirs.append(d_path)
    fmri_pretrain_exp_name = None
    if len(found_experiment_dirs) == 1:
        fmri_pretrain_exp_name = os.path.basename(found_experiment_dirs[0])
        current_task_config['pretrained_net_b'] = os.path.join(base_output_dir_for_fmri_models, fmri_pretrain_exp_name, "output", "models", fmri_model_filename)
    else:
        fmri_pretrain_exp_name_placeholder = f"EXPECTED_FMRIPRETAIN_EXP_DIR_ENDING_WITH{experiment_dir_core_pattern}"
        current_task_config['pretrained_net_b'] = os.path.join(base_output_dir_for_fmri_models, fmri_pretrain_exp_name_placeholder, "output", "models", fmri_model_filename)
    os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)
    temp_config_filename = f"temp_config_{bundle}_gpu{gpu}_{time.strftime('%Y%m%d%H%M%S')}.yaml"
    temp_config_filepath = os.path.join(TEMP_CONFIG_DIR, temp_config_filename)
    try:
        with open(temp_config_filepath, 'w') as f:
            yaml.dump(current_task_config, f, sort_keys=False)
    except Exception as e:
        sys.exit(1)
    cmd = [
        "/example/path/to/python", 
        "/example/path/to/finetune/train_bilateral_Fan_bundle_fmri_origin.py",
        "--config", temp_config_filepath,
    ]
    try:
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
    except FileNotFoundError:
        sys.exit(1)
    except Exception as e:
        sys.exit(1)
    finally:
        if os.path.exists(temp_config_filepath):
            pass


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
        return
    main_log_dir = "/example/path/to/logs_finetune"
    if log_file:
        main_log_dir = os.path.dirname(os.path.abspath(log_file))
    os.makedirs(main_log_dir, exist_ok=True)
    os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)
    gpu_task_counts = {gpu_id: 0 for gpu_id in gpus}
    active_processes_info = []
    task_queue = list(tasks_to_run)
    launched_task_count = 0
    while launched_task_count < total_initial_tasks or active_processes_info:
        if active_processes_info:
            remaining_active_pids_info = []
            for pid_file, assigned_gpu in active_processes_info:
                try:
                    with open(pid_file, 'r') as f_pid:
                        pid = int(f_pid.read().strip())
                    subprocess.check_call(f"kill -0 {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    remaining_active_pids_info.append((pid_file, assigned_gpu))
                except (FileNotFoundError, ValueError, subprocess.CalledProcessError):
                    gpu_task_counts[assigned_gpu] = max(0, gpu_task_counts[assigned_gpu] - 1)
                    if os.path.exists(pid_file):
                        try: os.remove(pid_file)
                        except OSError as e: pass
            active_processes_info = remaining_active_pids_info
        can_launch_more_in_parallel = len(active_processes_info) < parallel
        while task_queue and can_launch_more_in_parallel:
            assigned_gpu_for_task = None
            gpus_sorted_by_load = sorted(gpus, key=lambda g: gpu_task_counts.get(g, 0))
            for gpu_id in gpus_sorted_by_load:
                if gpu_task_counts.get(gpu_id, 0) < max_tasks_per_gpu:
                    assigned_gpu_for_task = gpu_id
                    break
            if assigned_gpu_for_task:
                current_task_bundle, current_task_num_clusters = task_queue.pop(0)
                gpu_task_counts[assigned_gpu_for_task] = gpu_task_counts.get(assigned_gpu_for_task, 0) + 1
                launched_task_count += 1
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                task_log_filename = f"train_{current_task_bundle}_gpu{assigned_gpu_for_task}_{timestamp}.log"
                current_task_log_path = os.path.join(main_log_dir, task_log_filename)
                pid_file_name = f"pid_{current_task_bundle}_gpu{assigned_gpu_for_task}_{timestamp}.txt"
                pid_file_path = os.path.join(main_log_dir, pid_file_name)
                direct_run_args = [
                    "python", current_script, "--run-direct",
                    "--base-config", base_config_path,
                    "--bundle", current_task_bundle,
                    "--gpu", assigned_gpu_for_task,
                    "--num_clusters", str(current_task_num_clusters)
                ]
                nohup_cmd_str = f"nohup {' '.join(direct_run_args)} > {current_task_log_path} 2>&1 & echo $! > {pid_file_path}"
                try:
                    subprocess.run(nohup_cmd_str, shell=True, check=True, executable='/bin/bash')
                    active_processes_info.append((pid_file_path, assigned_gpu_for_task))
                except subprocess.CalledProcessError as e:
                    gpu_task_counts[assigned_gpu_for_task] = max(0, gpu_task_counts.get(assigned_gpu_for_task, 0) - 1)
                    launched_task_count -=1
                can_launch_more_in_parallel = len(active_processes_info) < parallel
            else:
                break
        if not task_queue and not active_processes_info:
            break
        time.sleep(10)
    

if __name__ == "__main__":
    args = parse_args()
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
        main_log_dir_for_nohup = "/example/path/to/logs"
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
