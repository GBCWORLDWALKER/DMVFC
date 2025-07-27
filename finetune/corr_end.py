import h5py
import torch
import os
import glob
import numpy as np

# Directory containing the h5 files
def get_corr(h5_directory):

    # Find all FMRI_clusters.h5 files
    h5_files = glob.glob(os.path.join(h5_directory, "*_fmri_clusters.h5"))

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                corr_tensor = torch.tensor(corr[:], device=device)[:, [0, 1], :]
                
                if corr_tensor.shape[0]==1 or corr_tensor.shape[0]==0:
                    continue
                else:
                    cor_matrix = torch.corrcoef(corr_tensor.view(corr_tensor.shape[0],-1))
                    cor_matrix_1=torch.corrcoef(corr_tensor.flip(dims=[1]).view(corr_tensor.shape[0],-1))
                
                # Set diagonal to -inf and get max correlation for each row
                cor_matrix.fill_diagonal_(-float('inf'))
                cor_matrix_1.fill_diagonal_(-float('inf'))
                upper_triangle = torch.max(cor_matrix, dim=1)[0]
                upper_triangle_1 = torch.max(cor_matrix_1, dim=1)[0]
                # Calculate the average of the upper triangular part

                cor_average=torch.mean(torch.max(upper_triangle_1,upper_triangle))
                if torch.isnan(cor_average):
                    continue
                correlations.append(cor_average)
            cor_average=torch.mean(torch.tensor(correlations))
            all_correlations.append(cor_average)
            
            print(f"Average correlation for {os.path.basename(h5_file_path)}: {cor_average}")
            print("="*50)

    # Calculate overall average correlation
    overall_avg_correlation = torch.mean(torch.tensor(all_correlations))
    print(f"\nOverall average correlation across all subjects: {overall_avg_correlation}")
    # Save results to original folder
    results_file = os.path.join(h5_directory, "correlation_results_gm_pca.txt")

    with open(results_file, 'w') as f:
        for h5_file, correlation in zip(h5_files, all_correlations):
            f.write(f"{os.path.basename(h5_file)}: {correlation.item()}\n")
        f.write(f"\nOverall average correlation: {overall_avg_correlation.item()}")

    print(f"Results saved to {results_file}")

    print("Processing complete.")

if __name__ == "__main__":
    h5_directory = "/data06/jinwang/isbi/data/bundle_full/out/CST_left_baseline_501"
    get_corr(h5_directory)
