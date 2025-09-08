import subprocess
import sys
import os
import time
import argparse
import math

# Predefined bundle to num_clusters mapping
BUNDLE_CONFIG = {
    'UF_left': '7'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Run training command')
    parser.add_argument('--run-direct', action='store_true', help='Run command directly instead of in background')
    parser.add_argument('--log', type=str, help='Log file path')
    parser.add_argument('--bundle', type=str, nargs='+', default=['CC_1'], help='List of bundle names to process')
    parser.add_argument('--gpu', type=str, nargs='+', default=['0'], help='List of GPU IDs')
    parser.add_argument('--num_clusters', type=str, nargs='+', default=[], help='List of cluster numbers, overrides config if provided')
    parser.add_argument('--parallel', type=int, default=8, help='Number of tasks to run in parallel')
    parser.add_argument('--use-config', action='store_true', help='Use predefined bundle config, ignore num_clusters argument')
    return parser.parse_args()

def run_in_background(log_file=None, bundles=['CC_1'], gpus=['0'], num_clusters=[], parallel=2, use_config=False):
    current_script = os.path.abspath(__file__)
    tasks = []
    if use_config or not num_clusters:
        for bundle in bundles:
            num_cluster = BUNDLE_CONFIG.get(bundle, '12')
            tasks.append((bundle, num_cluster))
    else:
        if len(bundles) > len(num_clusters):
            num_clusters = num_clusters + [num_clusters[-1]] * (len(bundles) - len(num_clusters))
        elif len(bundles) < len(num_clusters):
            print(f"Warning: bundle list ({len(bundles)}) is shorter than num_clusters list ({len(num_clusters)}), extra cluster numbers will be ignored")
            num_clusters = num_clusters[:len(bundles)]
        tasks = list(zip(bundles, num_clusters))
    total_tasks = len(tasks)
    print(f"Total {total_tasks} tasks to run")
    batches = math.ceil(total_tasks / parallel)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    else:
        os.makedirs("/example/path/to/logs_new", exist_ok=True)
    for batch in range(batches):
        start_idx = batch * parallel
        end_idx = min(start_idx + parallel, total_tasks)
        batch_tasks = tasks[start_idx:end_idx]
        print(f"Running batch {batch+1}/{batches}, number of tasks: {len(batch_tasks)}")
        batch_pid_files = []
        for i, (bundle, num_cluster) in enumerate(batch_tasks):
            gpu = gpus[i % len(gpus)]
            if log_file is None:
                current_log = f"/example/path/to/logs_new/train_{bundle}_gpu{gpu}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            else:
                log_dir = os.path.dirname(log_file)
                log_name = os.path.basename(log_file)
                base_name, ext = os.path.splitext(log_name)
                current_log = os.path.join(log_dir, f"{base_name}_{bundle}_gpu{gpu}{ext}")
            pid_file = f"/example/path/to/logs_new/pid_{bundle}_gpu{gpu}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            batch_pid_files.append(pid_file)
            nohup_cmd = f"nohup python {current_script} --run-direct --bundle {bundle} --gpu {gpu} --num_clusters {num_cluster} > {current_log} 2>&1 & echo $! > {pid_file}"
            print(f"Start background task [{start_idx+i+1}/{total_tasks}]: {bundle}, GPU: {gpu}, num_clusters: {num_cluster}")
            print(f"Command: {nohup_cmd}")
            subprocess.run(nohup_cmd, shell=True)
            print(f"Check progress: tail -f {current_log}")
            print(f"PID file: {pid_file}")
        if batch < batches - 1:
            print(f"Batch {batch+1}/{batches} started, waiting for all tasks to finish before next batch...")
            wait_for_batch_completion(batch_pid_files)
            print(f"Batch {batch+1}/{batches} all tasks finished, preparing next batch...")

def wait_for_batch_completion(pid_files):
    """Wait for all processes recorded in the given PID files to finish"""
    all_done = False
    pids = []
    for pid_file in pid_files:
        max_retries = 10
        retries = 0
        while not os.path.exists(pid_file) and retries < max_retries:
            time.sleep(1)
            retries += 1
        if os.path.exists(pid_file):
            with open(pid_file, 'r') as f:
                try:
                    pid = int(f.read().strip())
                    pids.append(pid)
                except:
                    print(f"Warning: Could not read valid PID from {pid_file}")
    print(f"Monitoring {len(pids)} processes: {pids}")
    while not all_done:
        time.sleep(10)
        all_done = True
        for pid in pids:
            try:
                subprocess.check_call(f"kill -0 {pid}", shell=True, stderr=subprocess.DEVNULL)
                all_done = False
                break
            except subprocess.CalledProcessError:
                continue
    print("All processes finished")
    for pid_file in pid_files:
        if os.path.exists(pid_file):
            os.remove(pid_file)

def run_command(bundle='CC_1', gpu='0', num_clusters='12'):
    if not num_clusters:
        num_clusters = BUNDLE_CONFIG.get(bundle, '12')
    print(f"Process started, PID: {os.getpid()}")
    cmd = [
        "/example/path/to/python",
        "/example/path/to/dMRI_train/train_bilateral_Fan_bundle.py",
        "-indir", "/example/path/to/tractseg_MNI_vox_fMRI_vtk/train",
        "-outdir", "/example/path/to/out_full_train/",
        "-fmri_path", "/example/path/to/fmri_train",
        "-p", "25",
        "--epochs_pretrain", "450",
        "--epochs", "20",
        "--loss_surf", "False",
        "--embedding_surf", "False",
        "--pretrain", "True",
        "--dataset_prepared", "False",
        "--alpha", "0",
        "--similarity_path", "/example/path/to/similarity",
        "--fmri_similarity_path", "/example/path/to/pkl",
        "--dmri_similarity_path", "/example/path/to/dmri_distance_output",
        "--num_clusters", num_clusters,
        "--GPU", gpu,
        "--pretrained_net", f"/example/path/to/out_full_train/models/{bundle}_baseline_pretrain_k{num_clusters}.pt",
        "--bundle", bundle,
        "--output_name", f"{bundle}_baseline"
    ]
    try:
        print("Starting command...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        for line in process.stdout:
            print(line, end='', flush=True)
        process.wait()
        print(f"Command finished, exit code: {process.returncode}")
    except Exception as e:
        print(f"Command failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    
    if args.run_direct:
        if isinstance(args.bundle, list):
            bundle = args.bundle[0]
        else:
            bundle = args.bundle
            
        if isinstance(args.num_clusters, list) and args.num_clusters:
            num_clusters = args.num_clusters[0]
        elif args.use_config:
            num_clusters = BUNDLE_CONFIG.get(bundle, '12')
        else:
            num_clusters = args.num_clusters[0] if args.num_clusters else BUNDLE_CONFIG.get(bundle, '12')
            
        if isinstance(args.gpu, list):
            gpu = args.gpu[0]
        else:
            gpu = args.gpu
            
        run_command(bundle=bundle, gpu=gpu, num_clusters=num_clusters)
    else:
        run_in_background(
            log_file=args.log, 
            bundles=args.bundle, 
            gpus=args.gpu, 
            num_clusters=args.num_clusters,
            parallel=args.parallel,
            use_config=args.use_config
        )
