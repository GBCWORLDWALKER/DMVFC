import h5py
import torch
import os
import glob
import numpy as np

# Directory containing the h5 files
def get_corr(h5_directory,device):

    # Find all FMRI_clusters.h5 files

    h5_files = glob.glob(os.path.join(h5_directory, "*_fmri_clusters_endpoint.h5"))

    all_correlations = []

    for h5_file_path in h5_files:
        print(f"Processing file: {os.path.basename(h5_file_path)}")
        with h5py.File(h5_file_path, 'r') as f:
            cluster_keys = list(f.keys())
            num_clusters = len(cluster_keys)
            correlations=[]
            for i in range(num_clusters):
                corr=f[cluster_keys[i]]
                # Calculate correlation matrix
                corr_tensor = torch.tensor(corr[:], device=device)
                
                if corr_tensor.shape[0]==1 or corr_tensor.shape[0]==0:
                    continue
                else:
                    process=torch.cat([corr_tensor,corr_tensor.flip(dims=[1])],dim=0)
                    cor_matrix = torch.corrcoef(process.view(process.shape[0],-1))
                l=len(corr_tensor)
                # Get the upper triangular part of the matrix (excluding diagonal)
                matrix = torch.max(cor_matrix[:l,:l],cor_matrix[:l,l:2*l])
                if torch.isnan(matrix).any():
                    continue
                correlations.append(matrix)
            if correlations:
                all_values = torch.cat([torch.tensor(corr).flatten() for corr in correlations])
                cor_average = torch.mean(all_values)
            all_correlations.append(cor_average)
            
            print(f"Average correlation for {os.path.basename(h5_file_path)}: {cor_average}")
            print("="*50)

    # Calculate overall average correlation
    overall_avg_correlation = torch.mean(torch.tensor(all_correlations))
    print(f"\nOverall average correlation across all subjects: {overall_avg_correlation}")
    # Save results to original folder
    results_file = os.path.join(h5_directory, "correlation_results_endpoint.txt")

    with open(results_file, 'w') as f:
        for h5_file, correlation in zip(h5_files, all_correlations):
            f.write(f"{os.path.basename(h5_file)}: {correlation.item()}\n")
        f.write(f"\nOverall average correlation: {overall_avg_correlation.item()}")

    print(f"Results saved to {results_file}")

    print("Processing complete.")

if __name__ == "__main__":
    h5_directory = "/data06/jinwang/isbi/data/bundle_full/out/AF_left_full_train_baseline"
    get_corr(h5_directory)
