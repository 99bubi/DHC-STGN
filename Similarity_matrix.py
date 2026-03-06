import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter1d
from fastdtw import fastdtw
import time
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')


class SimilarityMatrixConfig:
    DATA_PATH = "./data/solar_radiation.csv"
    OUTPUT_DIR = "./similarity_matrices"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/full_matrices", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)
    
    WINDOW_SIZE = 30
    DTW_WINDOW = 365
    SMOOTHING_SCALES = [1, 7, 30]
    BATCH_SIZE = 20
    
    PLOT_FULL_MATRICES = True
    PLOT_STATISTICS = True
    SAVE_RAW_DATA = True


config = SimilarityMatrixConfig()

print("=" * 80)
print("加载数据...")
print("=" * 80)

start_time = time.time()

data = pd.read_csv(config.DATA_PATH, header=None)
site_info = data.iloc[:, :4].values
temporal_data = data.iloc[:, 4:].values

n_sites = site_info.shape[0]
n_timesteps = temporal_data.shape[1]

print(f"站点数量: {n_sites}")
print(f"时间步长: {n_timesteps}")

print("\n处理缺失值...")
def impute_missing_values(data_matrix):
    n_sites, n_timesteps = data_matrix.shape
    data_filled = data_matrix.copy()
    
    for t in range(n_timesteps):
        missing_mask = np.isnan(data_matrix[:, t])
        if np.any(missing_mask):
            valid_mask = ~missing_mask
            if np.sum(valid_mask) > 0:
                for i in np.where(missing_mask)[0]:
                    valid_indices = np.where(valid_mask)[0]
                    if len(valid_indices) > 0:
                        distances = np.sqrt(
                            (site_info[i, 1] - site_info[valid_indices, 1])**2 +
                            (site_info[i, 2] - site_info[valid_indices, 2])**2
                        )
                        nearest_idx = valid_indices[np.argsort(distances)[:5]]
                        if len(nearest_idx) > 0:
                            values = data_matrix[nearest_idx, t]
                            valid_values = values[~np.isnan(values)]
                            if len(valid_values) > 0:
                                data_filled[i, t] = np.mean(valid_values)
    
    for i in range(n_sites):
        site_series = data_filled[i]
        mask = np.isnan(site_series)
        if np.any(mask):
            indices = np.where(~mask)[0]
            if len(indices) > 1:
                data_filled[i, mask] = np.interp(
                    np.where(mask)[0],
                    indices,
                    site_series[indices]
                )
            elif len(indices) == 1:
                data_filled[i, mask] = site_series[indices[0]]
            else:
                data_filled[i, mask] = np.nanmean(data_matrix)
    
    return data_filled

temporal_data_filled = impute_missing_values(temporal_data)
nan_ratio = np.isnan(temporal_data_filled).sum() / temporal_data_filled.size


def compute_mahalanobis_similarity_paper(spatial_coords):
    print("\n" + "="*60)
    print("1. 计算马氏距离相似性矩阵")
    print("="*60)
    
    start_time = time.time()
    n = spatial_coords.shape[0]
    
    coords_centered = spatial_coords - spatial_coords.mean(axis=0)
    cov_matrix = np.cov(coords_centered.T)
    reg = np.eye(cov_matrix.shape[0]) * 1e-8
    
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix + reg)
    except:
        inv_cov_matrix = np.linalg.pinv(cov_matrix + reg)
    
    dist_matrix = np.zeros((n, n))
    
    for i in tqdm(range(n), desc="计算马氏距离"):
        for j in range(i, n):
            if i == j:
                dist_matrix[i, j] = 0
                continue
            
            diff = spatial_coords[i] - spatial_coords[j]
            dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    off_diag_dist = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    sigma = np.median(off_diag_dist)
    print(f"自适应尺度参数 σ = {sigma:.4f}")
    
    similarity_matrix = np.exp(-dist_matrix / sigma)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    elapsed = time.time() - start_time
    print(f"计算完成! 耗时: {elapsed:.1f}秒")
    print(f"相似性范围: [{similarity_matrix.min():.6f}, {similarity_matrix.max():.6f}]")
    print(f"平均相似性: {similarity_matrix.mean():.6f}")
    print(f"标准差: {similarity_matrix.std():.6f}")
    
    return similarity_matrix, dist_matrix


def compute_dtw_similarity_paper(temporal_data, n_workers=8, use_threads=True):
    print("\n" + "="*60)
    print("2. 计算DTW相似性矩阵")
    print("="*60)
    
    start_time = time.time()
    n_sites = temporal_data.shape[0]
    
    use_days = min(365, temporal_data.shape[1])
    recent_data = temporal_data[:, -use_days:].copy()
    
    print(f"站点数量: {n_sites}")
    print(f"序列长度: {use_days}天")
    print(f"需要计算的站点对: {n_sites * (n_sites - 1) // 2:,}对")
    
    similarity_matrix = np.eye(n_sites, dtype=np.float32)
    
    if n_workers is None:
        if use_threads:
            n_workers = min(mp.cpu_count() * 2, 16)
        else:
            n_workers = min(mp.cpu_count(), 8)
    
    print(f"使用{'线程' if use_threads else '进程'}池，工作数: {n_workers}")
    
    tasks = []
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            tasks.append((i, j, recent_data[i], recent_data[j]))
    
    print(f"任务总数: {len(tasks):,}")
    
    def compute_single_dtw(task):
        i, j, ts_i, ts_j = task
        try:
            distance, _ = fastdtw(
                ts_i.reshape(-1, 1),
                ts_j.reshape(-1, 1),
                dist=euclidean,
                radius=7
            )
            similarity = np.exp(-distance / len(ts_i))
            return i, j, similarity
        except Exception as e:
            corr = np.corrcoef(ts_i, ts_j)[0, 1]
            if np.isnan(corr):
                similarity = 0.1
            else:
                similarity = max(0.0, (corr + 1) / 2)
            return i, j, similarity
    
    print(f"开始并行计算...")
    
    results = []
    
    if use_threads:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for result in tqdm(executor.map(compute_single_dtw, tasks),
                             total=len(tasks),
                             desc="DTW计算进度"):
                results.append(result)
    else:
        batch_size = 5000
        n_batches = (len(tasks) + batch_size - 1) // batch_size
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(tasks))
                batch_tasks = tasks[batch_start:batch_end]
                
                batch_results = list(executor.map(compute_single_dtw, batch_tasks))
                results.extend(batch_results)
                
                progress = (batch_idx + 1) / n_batches * 100
                print(f"批处理进度: {progress:.1f}% ({batch_end}/{len(tasks)})")
    
    print("填充相似性矩阵...")
    for i, j, similarity in tqdm(results, desc="填充矩阵"):
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity
    
    np.fill_diagonal(similarity_matrix, 1.0)
    
    elapsed_time = time.time() - start_time
    off_diag_mask = ~np.eye(n_sites, dtype=bool)
    off_diag_values = similarity_matrix[off_diag_mask]
    
    print("\n" + "=" * 70)
    print("DTW计算完成!")
    print("=" * 70)
    print(f"总计算时间: {elapsed_time:.1f}秒 ({elapsed_time/60:.1f}分钟)")
    print(f"相似性矩阵形状: {similarity_matrix.shape}")
    print(f"相似性范围: [{similarity_matrix.min():.6f}, {similarity_matrix.max():.6f}]")
    print(f"非对角线均值: {off_diag_values.mean():.6f}")
    print(f"非对角线标准差: {off_diag_values.std():.6f}")
    print(f"计算速度: {len(tasks)/elapsed_time:.0f} 对/秒")
    
    return similarity_matrix


def compute_source_reliability_paper(temporal_data):
    print("\n" + "="*60)
    print("3. 计算源头可靠性向量")
    print("="*60)
    
    start_time = time.time()
    n_sites = temporal_data.shape[0]
    
    reliability = np.ones(n_sites, dtype=np.float64)
    eval_window = min(2 * 365, temporal_data.shape[1])
    
    for j in tqdm(range(n_sites), desc="计算站点可靠性"):
        eval_data = temporal_data[j, -eval_window:].copy()
        eval_data = eval_data[~np.isnan(eval_data)]
        
        if len(eval_data) < 100:
            reliability[j] = 0.3
            continue
        
        noise_estimates = []
        for sigma in config.SMOOTHING_SCALES:
            if sigma < len(eval_data):
                try:
                    smoothed = gaussian_filter1d(eval_data, sigma=sigma)
                    residual = eval_data - smoothed
                    noise_var = np.var(residual)
                    noise_estimates.append(noise_var)
                except:
                    pass
        
        if noise_estimates:
            sigma_noise = np.sqrt(np.min(noise_estimates))
        else:
            sigma_noise = np.std(eval_data)
        
        mu_j = np.mean(np.abs(eval_data))
        
        diff_series = np.diff(eval_data)
        diff_std = np.std(diff_series)
        diff_mean_abs = np.mean(np.abs(diff_series))
        
        epsilon = 1e-8
        lambda_val = 1.0
        alpha = 1.0
        
        reliability_score = np.exp(-lambda_val * (
            sigma_noise / (mu_j + epsilon) + 
            alpha * diff_std / (diff_mean_abs + epsilon)
        ))
        
        reliability[j] = np.clip(reliability_score, 0.1, 0.95)
    
    elapsed = time.time() - start_time
    print(f"计算完成! 耗时: {elapsed:.1f}秒")
    print(f"可靠性范围: [{reliability.min():.6f}, {reliability.max():.6f}]")
    print(f"平均可靠性: {reliability.mean():.6f}")
    print(f"标准差: {reliability.std():.6f}")
    
    print("\n可靠性分布:")
    quantiles = [0, 0.25, 0.5, 0.75, 1.0]
    for q in quantiles:
        value = np.quantile(reliability, q)
        print(f"  {q*100:.0f}%分位数: {value:.4f}")
    
    return reliability


def compute_cross_interpretability_paper(temporal_data):
    print("\n" + "="*60)
    print("4. 计算交叉解释性矩阵")
    print("="*60)
    
    start_time = time.time()
    n_sites = temporal_data.shape[0]
    
    use_length = min(config.WINDOW_SIZE * 2, temporal_data.shape[1])
    recent_data = temporal_data[:, -use_length:].copy()
    
    interpretability_matrix = np.eye(n_sites, dtype=np.float64)
    seq_means = np.nanmean(recent_data, axis=1)
    seq_stds = np.nanstd(recent_data, axis=1)
    
    total_pairs = n_sites * n_sites - n_sites
    processed = 0
    
    for i in tqdm(range(n_sites), desc="计算交叉解释性"):
        if seq_stds[i] < 1e-8:
            continue
        
        ts_i = recent_data[i]
        valid_i = ~np.isnan(ts_i)
        
        if np.sum(valid_i) < 10:
            continue
        
        ts_i_valid = ts_i[valid_i]
        
        batch_indices = []
        batch_data = []
        
        for j in range(n_sites):
            if i == j:
                continue
            
            if seq_stds[j] < 1e-8:
                interpretability_matrix[i, j] = 0.1
                continue
            
            ts_j = recent_data[j]
            valid_j = ~np.isnan(ts_j)
            
            common_valid = valid_i & valid_j
            if np.sum(common_valid) < 10:
                interpretability_matrix[i, j] = 0.1
                continue
            
            batch_indices.append(j)
            batch_data.append(ts_j[common_valid])
        
        if not batch_data:
            continue
        
        ts_i_standardized = (ts_i_valid - seq_means[i]) / (seq_stds[i] + 1e-8)
        
        for idx, (j, ts_j_common) in enumerate(zip(batch_indices, batch_data)):
            ts_j_standardized = (ts_j_common - seq_means[j]) / (seq_stds[j] + 1e-8)
            
            X = ts_j_standardized.reshape(-1, 1)
            y = ts_i_standardized[:len(X)]
            
            try:
                alpha = 0.1
                X_with_const = np.hstack([X, np.ones((len(X), 1))])
                beta = np.linalg.inv(X_with_const.T @ X_with_const + alpha * np.eye(2)) @ X_with_const.T @ y
                y_pred = X_with_const @ beta
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                
                if ss_tot > 1e-8:
                    r2 = max(0, 1 - ss_res / ss_tot)
                else:
                    r2 = 0.1
            except:
                r2 = 0.1
            
            interpretability_matrix[i, j] = np.clip(r2, 0.05, 0.95)
            processed += 1
    
    sym_matrix = 0.5 * (interpretability_matrix + interpretability_matrix.T)
    np.fill_diagonal(sym_matrix, 1.0)
    
    elapsed = time.time() - start_time
    print(f"计算完成! 耗时: {elapsed:.1f}秒")
    print(f"解释性范围: [{sym_matrix.min():.6f}, {sym_matrix.max():.6f}]")
    print(f"平均解释性: {sym_matrix.mean():.6f}")
    print(f"标准差: {sym_matrix.std():.6f}")
    
    asymmetry = np.mean(np.abs(interpretability_matrix - interpretability_matrix.T))
    print(f"平均不对称性: {asymmetry:.6f}")
    
    return sym_matrix


def main():
    print("=" * 80)
    print("动态层次置信图 - 四个相似性矩阵计算")
    print("=" * 80)
    
    total_start_time = time.time()
    
    try:
        g_ij, mahalanobis_dist = compute_mahalanobis_similarity_paper(site_info[:, 1:4])
        m_ij = compute_dtw_similarity_paper(temporal_data_filled)
        r_j = compute_source_reliability_paper(temporal_data_filled)
        c_ij = compute_cross_interpretability_paper(temporal_data_filled)
        
        pd.DataFrame(g_ij).to_csv(f"{config.OUTPUT_DIR}/full_matrices/mahalanobis_similarity.csv", 
                                index=False, header=False)
        pd.DataFrame(m_ij).to_csv(f"{config.OUTPUT_DIR}/full_matrices/dtw_similarity.csv", 
                                index=False, header=False)
        pd.DataFrame(r_j).to_csv(f"{config.OUTPUT_DIR}/full_matrices/source_reliability.csv", 
                                index=False, header=False)
        pd.DataFrame(c_ij).to_csv(f"{config.OUTPUT_DIR}/full_matrices/cross_interpretability.csv", 
                                index=False, header=False)
        
        np.save(f"{config.OUTPUT_DIR}/full_matrices/mahalanobis_similarity.npy", g_ij)
        np.save(f"{config.OUTPUT_DIR}/full_matrices/dtw_similarity.npy", m_ij)
        np.save(f"{config.OUTPUT_DIR}/full_matrices/source_reliability.npy", r_j)
        np.save(f"{config.OUTPUT_DIR}/full_matrices/cross_interpretability.npy", c_ij)
        
        print("所有矩阵已保存!")
        
        total_elapsed = time.time() - total_start_time
        
        print(f"""
        ===================================================================
                        相似性矩阵计算完成
        ===================================================================
        
        数据信息:
        ----------
        站点数量: {n_sites} 个
        时间长度: {n_timesteps} 天
        处理时间: {total_elapsed:.1f} 秒
        
        矩阵维度验证:
        ----------
        1. 马氏距离相似性矩阵: {g_ij.shape}
        2. DTW相似性矩阵: {m_ij.shape}
        3. 源头可靠性向量: {r_j.shape}
        4. 交叉解释性矩阵: {c_ij.shape}
        
        输出文件:
        ----------
        {config.OUTPUT_DIR}/full_matrices/
        """)
        
    except Exception as e:
        print(f"计算过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
