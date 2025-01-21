'''
LIRA for large-scale datasets
1. Full redundancy with probing model is used on every data point, which means that each data point is assigned to two partitions.
2. Internal index is recommended to be HNSW, which is more efficient than FLAT.
cluster, bucket, partition are used interchangeably.
'''
import numpy as np
import os
num_threads = 12
os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
import faiss
import time
from utils import *
import pandas as pd
from dataclasses import dataclass
from transformers import HfArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import numpy as np
from prettytable import PrettyTable
from model_probing import *
from tqdm import tqdm

@dataclass
class Config:
    method_name: str = 'LIRA_fullRE_subtrain'
    dataset: str = 'deep50M' # sift, glove, bigann, deep50M, text2image50M, deep100M
    dis_metric: str = None
    k: int = 100
    n_bkt: int = 1024 # 64 for small-scale; 1024 for large-scale
    n_epoch: int = 30 # model training epoch, 10 for small-scale; 30 for large-scale
    batch_size: int = 512
    n_mul: int = 2
    inner_index_type: str = 'HNSW' # 'HNSW', 'FLAT' # type of internal index

    repa_step: int = 1 # full redundancy
    duplicate_type = 'model' # 'None' 'model'
    ef_fixed: int = 128
    hnsw_M: int = 32

    pth_log: str = f'./logs/{dataset}/{method_name}_{inner_index_type}/'
    file_name: str = f'{dataset}-k={k}-ML_kmeans={n_bkt}_{inner_index_type}_ReType={duplicate_type}'
    log_name: str = f'{file_name}.txt'
    df_name: str = f'{file_name}.csv'

    def update(self):
        if self.dis_metric == None:
            self.dis_metric = 'inner_product' if self.dataset in ['glove', 'text2image50M'] else 'L2'
    
def mul_partition_by_model(data_partition_score, data_predicts, global_xd_ids, start_idx, data_2_bkt, cluster_cnts, cluster_ids):
    _, n_mul = data_2_bkt.shape
    for idx_global in global_xd_ids:
        idx_local = idx_global - start_idx # idx for data_partition_score and data_predicts
        cur_c_id = data_2_bkt[idx_global, 0] # current partition id
        sorted_candi_partition = torch.argsort(data_partition_score[idx_local], descending=True)
        n_effective_partition = len(torch.where(data_predicts[idx_local] != False)[0]) # effective partition id where the probing probability > 0
        n_actual_partition = np.min([n_mul-1, n_effective_partition])
        cur_c_id_loc = torch.where(sorted_candi_partition == cur_c_id)[0] # the location of current cluster in the candidate partition
        if cur_c_id_loc.numel() == 0 or cur_c_id_loc[0] >= n_actual_partition:
            actual_partition = sorted_candi_partition[0 : n_actual_partition].cpu().numpy()
            data_2_bkt[idx_global, 1 : n_actual_partition + 1] = actual_partition
        elif n_effective_partition == n_actual_partition:
            actual_partition = sorted_candi_partition[0 : n_actual_partition].cpu().numpy()
            data_2_bkt[idx_global, 0 : n_actual_partition] = actual_partition
        else:
            actual_partition = sorted_candi_partition[0 : n_actual_partition + 1].cpu().numpy()
            data_2_bkt[idx_global, 0 : n_actual_partition + 1] = actual_partition
        for c_id in actual_partition:
            if c_id != cur_c_id:
                cluster_cnts[c_id] += 1
                cluster_ids[c_id].append(int(idx_global))

def cal_metrics(all_predicts, all_targets, epoch, knn_distr_id, cluster_id, results_df, loss, knn=100):
    '''
    Calculation metrics.
    '''
    # metrics for partitions probing
    nprobe_output_mean = torch.mean(torch.sum(all_predicts, dim=1).float()).item()
    nprobe_target_mean = torch.mean(torch.sum(all_targets, dim=1).float()).item()
    accuracy = accuracy_score(all_targets.numpy().flatten(), all_predicts.numpy().flatten())
    hit_rates = torch.logical_and(all_targets, all_predicts).sum(dim=1).float() / all_targets.sum(dim=1).float()  # TP / (TP + FN)
    hit_rate = torch.nanmean(hit_rates).item()  # mean hit rates, using nanmean in case 0

    #  metrics for KNN recall and computations
    n_q = len(all_targets)
    recalls = np.zeros(n_q)
    knn_computations = np.zeros(n_q)
    for qid in range(n_q):
        probing_bids = all_predicts[qid].nonzero(as_tuple=True)[0].numpy()
        knn_id_to_concate = [knn_distr_id[qid, q_c] for q_c in probing_bids]
        if knn_id_to_concate:
            aknn = np.unique(np.concatenate(knn_id_to_concate))
        else:
            aknn = np.array([])
        recalls[qid] = len(aknn) / knn

    recall_avg = np.mean(recalls)
    cmp_avg = np.mean(knn_computations)

    table = PrettyTable(['Epoch', 'Loss', 'Accuracy', 'Hit Rate', 'nprobe predict', 'nprobe target', 'KNN Recall', 'KNN Computations'])
    table.float_format = "4.4"
    table.add_row([epoch, loss, accuracy, hit_rate, nprobe_output_mean, nprobe_target_mean, recall_avg, cmp_avg])
    fprint(table, fw)
    new_data = pd.DataFrame({
        'Epoch': [epoch],
        'Loss': [loss],
        'Accuracy': [accuracy],
        'Hit Rate': [hit_rate],
        'nprobe predict': [nprobe_output_mean],
        'nprobe target': [nprobe_target_mean],
        'KNN Recall': [recall_avg],
        'KNN Computations': [cmp_avg]
    }).round(4)
    results_df = pd.concat([results_df, new_data], ignore_index=True)

    return results_df


def get_cmp_recall(inner_indexes, x_q, xd_id_bkt, cfg):
    """
    search in each partition for each query
    return the search time, distance comparisons, and found aknn id (which need to be deduplicated later)
    """
    n_bkt = cfg.n_bkt
    n_q = len(x_q)
    search_time = np.zeros((n_q, n_bkt))  # record the search time
    cmp_distr_all = np.zeros((n_q, n_bkt), dtype=int)  # record the distance comparisons
    found_aknn_id = np.full((n_q, n_bkt, cfg.k), -1)  # record the found aknn id
    xd_id_bkt_int = [np.array(bkt).astype(int) for bkt in xd_id_bkt]

    # loop for each bucket
    for b_id in tqdm(range(n_bkt)): 
        inner_index = inner_indexes[b_id]
        xd_id_bid = xd_id_bkt_int[b_id] # the data id in the current bucket
        if inner_index is None or len(xd_id_bkt_int[b_id]) == 0:
            continue
        
        # loop for each query
        for q_id in range(n_q):
            q = x_q[q_id].reshape(1, -1)
            t_start = time.time()
            if cfg.inner_index_type == 'HNSW':
                inner_index.hnsw.efSearch = cfg.ef_fixed
                hnsw_stats = faiss.cvar.hnsw_stats
                hnsw_stats.reset()
                _, idx_in_bkt = inner_index.search(q, cfg.k) # get the top-k id in the current bucket
                aknn_id = xd_id_bid[idx_in_bkt.reshape(-1)] # Note that idx_in_bkt is relative to the bkt, convert to the global id.
                found_aknn_id[q_id][b_id] = aknn_id
                cmp_distr_all[q_id][b_id] = int(hnsw_stats.ndis)
            elif cfg.inner_index_type == 'FLAT':
                _, idx_in_bkt = inner_index.search(q, cfg.k)
                aknn_id = xd_id_bid[idx_in_bkt.reshape(-1)]
                found_aknn_id[q_id][b_id] = aknn_id
                cmp_distr_all[q_id][b_id] = inner_index.ntotal
            
            search_time[q_id][b_id] = time.time() - t_start

    return search_time, cmp_distr_all, found_aknn_id

def query_tuning(all_outputs, knn_distr_id, found_aknn_id, cfg, part=0):
    n_q = len(all_outputs)
    n_s = n_q
    
    pd_cols = ['threshold', 'nprobe', 'Recall', 'Computations']
    df_threshold = pd.DataFrame(columns=pd_cols)
    for threshold in np.arange(0.1, 1.0, 0.02):
        thre_recall = np.zeros(n_s)
        thre_cmp = np.zeros(n_s)
        thre_nprobe = np.zeros(n_s)
        for i in range(n_s):
            # get the bucket with probing probability > threshold
            top_m_indices = np.where(all_outputs[i] > threshold)[0]
            thre_nprobe[i] = len(top_m_indices)
            thre_cmp[i] = np.sum(cmp_distr_all[i, top_m_indices])
            found_knn = set()
            for q_c in top_m_indices:
                aknn_c = set(knn_distr_id[i][q_c]).intersection(found_aknn_id[i][q_c])
                found_knn.update(aknn_c)
            thre_recall[i] = len(found_knn) / cfg.k

        thre_recall_avg = np.mean(thre_recall)
        thre_cmp_avg = np.mean(thre_cmp)
        thre_nprobe_avg = np.mean(thre_nprobe)
        print(f'threshold: {threshold:.3f}, nprobe: {thre_nprobe_avg}, KNN Recall: {thre_recall_avg:.4f}, KNN Computations: {thre_cmp_avg:.4f}')
        new_data = pd.DataFrame([{'threshold': threshold, 'nprobe': thre_nprobe_avg, 'Recall': thre_recall_avg, 'Computations': thre_cmp_avg}])
        df_threshold = pd.concat([df_threshold, new_data], ignore_index=True)
    os.makedirs(cfg.pth_log + cfg.file_name + f'_tuning_threshold/', exist_ok=True)
    df_threshold.to_csv(cfg.pth_log + cfg.file_name + f'_tuning_threshold/{cfg.duplicate_type}_{part}.csv', index=False)

# ============================================================================================
# ============================================================================================

if __name__ == "__main__":
    parser = HfArgumentParser(Config)
    cfg = parser.parse_args_into_dataclasses()[0]
    cfg.update()
    n_bkt = cfg.n_bkt

    os.makedirs(cfg.pth_log, exist_ok=True)
    fw = open(cfg.pth_log + cfg.log_name, 'a', encoding='utf-8')
    
    # (1) load data and query
    x_d, x_q, _, _ = load_data(cfg.dataset)
    fprint(f">> dataset: {cfg.dataset}, data_sizes: {x_d.shape}, query_size: {x_q.shape}, n_bkt: {n_bkt}, knn: {cfg.k}, num_threads: {num_threads}", fw)
    n_d, dim = x_d.shape
    n_q, dim = x_q.shape
    device = f"cuda:{get_idle_gpu()}"
    fprint(f">> device: {device}", fw)

    # get a subset of the data
    nd_sub = int(n_d / 100)
    seed = 43
    np.random.seed(seed)
    sub_idx = np.random.choice(range(len(x_d)), nd_sub, replace=False)
    xd_sub = x_d[sub_idx]

    fprint(">> begin data preprocessing (please wait a few minutes)", fw)
    knn_data_sub = get_flat_GT_knn(xd_sub, xd_sub, cfg, datatype="data")
    knn_query_sub = get_flat_GT_knn(xd_sub, x_q, cfg)

    # (2) initial partitioning
    data_2_bkt_sub = np.full((nd_sub, cfg.n_mul), -1) # -1 for empty value
    time_start = time.perf_counter()
    kmeans, data_sub_2_single_bkt, cluster_cnts, cluster_ids = build_kmeans_index(xd_sub, n_bkt)
    data_2_bkt_sub[:, :1] = data_sub_2_single_bkt
    time_build_kmeans = time.perf_counter() - time_start
    fprint(f'>> build kmeans index time: {time_build_kmeans}', fw)

    # (3) probing model
    time_start = time.perf_counter()
    print(">> begin get knn distribution and repartition")
    knn_distr_cnt_data_sub, knn_distr_id_data_sub = get_knn_distr_redundancy(knn_data_sub, data_2_bkt_sub, cfg)
    time_knn_distr = time.perf_counter() - time_start
    fprint(f'>> get knn distribution time: {time_knn_distr}', fw)
    knn_distr_cnt_query_sub, knn_distr_id_query_sub = get_knn_distr_redundancy(knn_query_sub, data_2_bkt_sub, cfg)
    labels_data = np.where(knn_distr_cnt_data_sub != 0, 1, knn_distr_cnt_data_sub)
    labels_query = np.where(knn_distr_cnt_query_sub != 0, 1, knn_distr_cnt_query_sub)
    # ------------------------------------------------------------------------------------------
    # observe_knn_tail(knn_distr_cnt, knn_distr_id)
    # ------------------------------------------------------------------------------------------
    # get the distance as input
    time_start = time.perf_counter()
    distances_data_scaled_sub, distances_query_scaled_sub = get_scaled_dist(xd_sub, x_q, kmeans, n_bkt)
    time_dist = time.perf_counter() - time_start
    fprint(f'>> get scaled distance time: {time_dist}', fw)
    fprint(f'>> get scaled distance time of queries: {time_dist * n_q / (n_q + n_d)}', fw)

    train_tensor = TensorDataset(torch.tensor(distances_data_scaled_sub, dtype=torch.float32), torch.tensor(xd_sub, dtype=torch.float32), torch.tensor(labels_data, dtype=torch.float32))
    test_tensor = TensorDataset(torch.tensor(distances_query_scaled_sub, dtype=torch.float32), torch.tensor(x_q, dtype=torch.float32), torch.tensor(labels_query, dtype=torch.float32))
    train_loader = DataLoader(train_tensor, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_tensor, batch_size=cfg.batch_size, shuffle=False)

    # model training and evaluation
    model = MLP_2_Input(input_dim1=n_bkt, input_dim2=dim, output_dim=n_bkt).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch = -1
    time_start = time.perf_counter()
    all_targets, all_predicts, loss_test, all_outputs = model_evaluate(model, test_loader, criterion, device)
    time_test = time.perf_counter() - time_start
    fprint(f'Epoch {epoch}, Test Loss: {loss_test}, time_test: {time_test}', fw)
    pd_cols = ['Epoch', 'Accuracy', 'Hit Rate', 'nprobe predict', 'nprobe target', 'KNN Recall', 'KNN Computations']
    results_df = pd.DataFrame(columns=pd_cols)
    results_df = cal_metrics(all_predicts, all_targets, epoch, knn_distr_id_query_sub, cluster_ids, results_df, loss_test, knn=cfg.k)
    for epoch in range(cfg.n_epoch):
        time_start = time.perf_counter()
        loss_train = model_train(model, train_loader, device, optimizer, criterion)
        time_train = time.perf_counter() - time_start

        time_start = time.perf_counter()
        all_targets, all_predicts, loss_test, all_outputs = model_evaluate(model, test_loader, criterion, device)
        time_test = time.perf_counter() - time_start
        fprint(f'Epoch {epoch}, Train Loss: {loss_train}, Test Loss: {loss_test}, time_train: {time_train}, time_test: {time_test}', fw)
        results_df = cal_metrics(all_predicts, all_targets, epoch, knn_distr_id_query_sub, cluster_ids, results_df, loss_test, knn=cfg.k)

    # (4) partition the full data
    data_2_bkt = np.full((n_d, cfg.n_mul), -1) # -1 for empty value
    _, data_2_single_bkt = kmeans.index.search(x_d, 1)
    data_2_bkt[:, :1] = data_2_single_bkt
    cluster_cnts = np.bincount(data_2_single_bkt.flatten()) 
    cluster_ids = [[] for _ in range(cfg.n_bkt)] # the data id in each bucket
    for idx, cid in enumerate(data_2_single_bkt.flatten()):
        cluster_ids[cid].append(idx)

    fprint(">> begin data preprocessing (please wait a few minutes)", fw)
    knn_query = get_flat_GT_knn(x_d, x_q, cfg)
    print(">> begin get knn distribution")
    knn_distr_cnt_query, knn_distr_id_query = get_knn_distr_redundancy(knn_query, data_2_bkt, cfg)

    # (4) redundancy with probing model
    fprint(f">> begin redundancy with {cfg.duplicate_type}", fw)
    if cfg.duplicate_type == 'model':
        print(">> before redundancy")
        time_start = time.perf_counter()
        inner_indexes = create_inner_indexes(x_d, cluster_ids, cfg)
        time_build_hnsw = time.perf_counter() - time_start
        fprint(f'>> build hnsw index time: {time_build_hnsw}', fw)
        search_time, cmp_distr_all, found_aknn_id = get_cmp_recall(inner_indexes, x_q, cluster_ids, cfg)
        # query process with different threshold, to get the curve of recall VS. computations
        query_tuning(all_outputs, knn_distr_id_query, found_aknn_id, cfg)

        batch_redundancy = 1000000
        for start_idx in tqdm(range(0, n_d, batch_redundancy)):
            end_idx = min(start_idx + batch_redundancy, n_d)
            xd_batch = x_d[start_idx:end_idx]
            distances_data_scaled = get_scaled_dist_data(xd_batch, kmeans, n_bkt)
            train_tensor = TensorDataset(torch.tensor(distances_data_scaled, dtype=torch.float32), torch.tensor(xd_batch, dtype=torch.float32))
            train_loader = DataLoader(train_tensor, batch_size=cfg.batch_size, shuffle=False)
            data_predicts, data_partition_score = model_infer(model, train_loader, device) # model infers the batch data
            global_xd_ids = np.arange(start_idx, end_idx)
            mul_partition_by_model(data_partition_score, data_predicts, global_xd_ids, start_idx, data_2_bkt, cluster_cnts, cluster_ids)

        knn_distr_cnt_query, knn_distr_id_query = get_knn_distr_redundancy(knn_query, data_2_bkt, cfg)
        # build inner indexes
        inner_indexes = create_inner_indexes(x_d, cluster_ids, cfg)
        search_time, cmp_distr_all, found_aknn_id = get_cmp_recall(inner_indexes, x_q, cluster_ids, cfg)
        print(">> after redundancy")
        query_tuning(all_outputs, knn_distr_id_query, found_aknn_id, cfg, part=1)
        # The QPS can be obtained through search_time
    
    elif cfg.duplicate_type == 'None':
        # build inner indexes
        time_start = time.perf_counter()
        inner_indexes = create_inner_indexes(x_d, cluster_ids, cfg)
        time_build_hnsw = time.perf_counter() - time_start
        fprint(f'>> build hnsw index time: {time_build_hnsw}', fw)

        time_start = time.perf_counter()
        search_time, cmp_distr_all, found_aknn_id = get_cmp_recall(inner_indexes, x_q, cluster_ids, cfg)
        time_search = time.perf_counter() - time_start
        fprint(f'>> search time: {time_search}', fw)
        query_tuning(all_outputs, knn_distr_id_query, found_aknn_id, cfg)

    fprint("finish!", fw)
    results_df.to_csv(cfg.pth_log + cfg.df_name, index=False)  # Save DataFrame to CSV
    fw.close()

