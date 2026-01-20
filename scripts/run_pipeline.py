
import sys
import os
import time
import pandas as pd
import numpy as np

# Add parent directory to path to import llmcer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmcer.config import DATASET_PATH, GROUND_TRUTH_PATH, OPENAI_MODEL
from llmcer.data_utils import get_ground_truth
from llmcer.vectorization import cal_total_simi_vector
from llmcer.clustering import lsh_block
from llmcer.pipeline import seperate_parallel
from llmcer.llm_interaction import merge_2
from llmcer.metrics import calculate_purity, calculate_inverse_purity, calculate_fp_measure, calculate_ari
from llmcer.id_utils import get_id_column

def convert_xlsx_to_csv(xlsx_path):
    csv_path = xlsx_path.replace('.xlsx', '.csv')
    if not os.path.exists(csv_path):
        print(f"Converting {xlsx_path} to {csv_path}...")
        df = pd.read_excel(xlsx_path)
        
        # Check for ID column using case-insensitive check
        id_col = get_id_column(df)
        
        if id_col:
            # Check if IDs are integers (optional verification)
            try:
                first_id = int(df[id_col].iloc[0])
            except:
                pass
        else:
            print("Warning: No ID column found (case-insensitive search for 'id').")
            
        df.to_csv(csv_path, index=False)
    return csv_path

def main():
    print("Starting LLMCER Pipeline...")
    
    # 0. Prepare Data
    if DATASET_PATH.endswith('.xlsx'):
        dataset_csv_path = convert_xlsx_to_csv(DATASET_PATH)
    else:
        dataset_csv_path = DATASET_PATH
        
    print(f"Using dataset: {dataset_csv_path}")
    
    # Load Ground Truth
    print(f"Loading ground truth from {GROUND_TRUTH_PATH}...")
    try:
        ground_truth = get_ground_truth(GROUND_TRUTH_PATH)
        print(f"Ground truth loaded, {len(ground_truth)} clusters.")
    except Exception as e:
        print(f"Warning: Could not load ground truth: {e}")
        ground_truth = []

    # 1. Vectorization & Similarity Matrix
    print("Calculating vectors and similarity matrix...")
    vectors, simi_matrix, data = cal_total_simi_vector(dataset_csv_path)
    
    # Dynamic Threshold Calculation
    sim_mean = np.mean(simi_matrix)
    sim_std = np.std(simi_matrix)
    # User Request: 
    # Low threshold (Merge) = Mean + 1.5 * Std
    # High threshold (Block) = Mean + 2.5 * Std
    block_threshold = sim_mean + 2.5 * sim_std
    merge_threshold_lower = sim_mean + 3.0 * sim_std
    
    # Cap thresholds at 0.99 to avoid floating point issues or impossible matches
    block_threshold = min(block_threshold, 0.99)
    merge_threshold_lower = min(merge_threshold_lower, 0.90)
    
    # print(f"Dynamic Thresholds: Merge (Lower)={merge_threshold_lower:.4f}, Block (Upper)={block_threshold:.4f}")
    
    # 2. Blocking (LSH)
    print("Running LSH Blocking...")
    # Use block_threshold for LSH
    lsh_threshold = block_threshold
    # print(f"Using dynamic LSH threshold: {lsh_threshold:.4f}")
    
    merge_clusters_pre = lsh_block(vectors, data, lsh_threshold)
    print(f"LSH Blocking done. Found {len(merge_clusters_pre)} blocks.")
    
    # 3. Separation
    print("Running Separation (Cluster Splitting)...")
    # Use block_threshold for separation/merge internal logic as well
    separation_threshold = block_threshold 
    # print(f"Using dynamic Separation threshold: {separation_threshold:.4f}")
    
    result_sep, api_calls, sep_time, sep_tokens, in_tokens, out_tokens, mdg_fails = seperate_parallel(
        vectors, simi_matrix, merge_clusters_pre, data, separation_threshold
    )
    print(f"Separation done. Resulting clusters: {len(result_sep)}")
    print(f"Stats: API Calls={api_calls}, Time={sep_time:.2f}s, Tokens={sep_tokens}")
    print(f"MDG Interventions: {mdg_fails}")
    
    # 4. Merging
    print("Running Merging...")
    
    # Thresholds are already calculated in Step 1
    # print(f"Using Dynamic Thresholds: Merge (Lower)={merge_threshold_lower:.4f}, Block (Upper)={block_threshold:.4f}")
    
    final_result, merge_api_calls, merge_time, merge_tokens, m_in_tok, m_out_tok = merge_2(
        result_sep, simi_matrix, data, block_threshold, merge_threshold_lower
    )
    print(f"Merging done. Final clusters: {len(final_result)}")
    print(f"Stats: API Calls={merge_api_calls}, Time={merge_time:.2f}s, Tokens={merge_tokens}")

    # 5. Metrics
    print("="*40)
    print("FINAL METRICS REPORT")
    print("="*40)
    
    if ground_truth:
        # Augment Ground Truth with Singletons for missing items
        if hasattr(data, 'iloc'):
             # Assuming 1st column is ID as used in lsh_block
             all_ids = data.iloc[:, 0].tolist()
        else:
             all_ids = []
             
        # Extract existing IDs in GT (normalize to str for comparison)
        gt_ids_str = set()
        for cluster in ground_truth:
            for item in cluster:
                gt_ids_str.add(str(item).strip())
                
        missing_count = 0
        for item in all_ids:
            if str(item).strip() not in gt_ids_str:
                ground_truth.append([item])
                missing_count += 1
                
        print(f"Augmented Ground Truth with {missing_count} singletons (total items: {len(all_ids)}).")

        from llmcer.metrics import calculate_pairwise_metrics, calculate_tolerant_purity, calculate_bcubed_metrics, calculate_macro_purity, calculate_pure_cluster_ratio
        
        # Standard Metrics
        purity = calculate_purity(ground_truth, final_result)
        inv_purity = calculate_inverse_purity(ground_truth, final_result)
        f_measure = calculate_fp_measure(ground_truth, final_result) # Default beta=1.0
        ari = calculate_ari(ground_truth, final_result)
        
        # New Metrics (Pairwise & Tolerant)
        pairwise = calculate_pairwise_metrics(ground_truth, final_result)
        tolerant_purity = calculate_tolerant_purity(ground_truth, final_result, tolerance=1)
        
        # BCubed Metrics (Better for entity-centric evaluation)
        bcubed = calculate_bcubed_metrics(ground_truth, final_result)

        # User-Requested "Loose" Metrics
        f_beta_05 = calculate_fp_measure(ground_truth, final_result, beta=0.5)
        macro_purity = calculate_macro_purity(ground_truth, final_result)
        pure_cluster_ratio = calculate_pure_cluster_ratio(ground_truth, final_result)
        
        print(f"Purity:             {purity:.4f}")
        print(f"Tolerant Purity (1):{tolerant_purity:.4f}")
        print(f"Inverse Purity:     {inv_purity:.4f}")
        print(f"F-Measure:          {f_measure:.4f}")
        print(f"ARI:                {ari:.4f}")
        
        print("-" * 20)
        print("Loose Metrics (User Preference):")
        print(f"  F-0.5 Measure:      {f_beta_05:.4f} (Bias towards Precision)")
        print(f"  Macro Purity:       {macro_purity:.4f} (Avg Purity per Cluster)")
        print(f"  Pure Cluster Ratio: {pure_cluster_ratio:.4f} (% of Perfect Clusters)")
        
        print("-" * 20)
        print("BCubed Metrics (Entity-Centric):")
        print(f"  Precision: {bcubed['precision']:.4f}")
        print(f"  Recall:    {bcubed['recall']:.4f}")
        print(f"  F1 Score:  {bcubed['f1']:.4f}")
        
        if pairwise:
            print("-" * 20)
            print("Pairwise Metrics:")
            print(f"  Accuracy:  {pairwise['accuracy']:.4f}")
            print(f"  Precision: {pairwise['precision']:.4f}")
            print(f"  Recall:    {pairwise['recall']:.4f}")
            print(f"  F1 Score:  {pairwise['f1']:.4f}")
            print(f"  TP: {pairwise['tp']}, TN: {pairwise['tn']}")
            print(f"  FP: {pairwise['fp']}, FN: {pairwise['fn']}")
    else:
        print("No ground truth provided. Skipping accuracy metrics.")

    # Total Stats
    total_api_calls = api_calls + merge_api_calls
    total_time = sep_time + merge_time
    total_tokens = sep_tokens + merge_tokens
    total_in_tokens = in_tokens + m_in_tok
    total_out_tokens = out_tokens + m_out_tok
    
    print("-" * 40)
    print(f"Total API Calls:     {total_api_calls}")
    print(f"Total Execution Time: {total_time:.2f} s")
    print(f"Total Tokens:        {total_tokens}")
    print(f"  - Input Tokens:    {total_in_tokens}")
    print(f"  - Output Tokens:   {total_out_tokens}")
    print(f"Total MDG Interventions: {mdg_fails}")
    print("="*40)
    
    # Save results
    output_path = "final_results.txt"
    with open(output_path, "w") as f:
        for cluster in final_result:
            f.write(" ".join(map(str, cluster)) + "\n")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
