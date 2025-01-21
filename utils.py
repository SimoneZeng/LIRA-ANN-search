import sys
import numpy as np
import os
import numpy as np
import h5py
import subprocess
import re
from sklearn.preprocessing import StandardScaler
import faiss
import torch
import pandas as pd
from tqdm import tqdm
import h5py
import random
np.random.seed(43)
seed = 43
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

def mmap_fvecs(fname, dtype='float32'):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view(dtype).reshape(-1, d + 1)[:, 1:]

def read_ivecs(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()

    data = np.frombuffer(raw_data, dtype=np.int32) # tranfer to int32
    vectors = []
    i = 0
    while i < len(data):
        dimension = data[i]  # the first integer is the dimension of the vector
        i += 1
        vectors.append(data[i:i+dimension])  # get the vector
        i += dimension

    return np.array(vectors)

def norm_glove(arr):
    '''
    norm to 1
    '''
    norms = np.linalg.norm(arr, axis=1)
    norm_arr = arr / norms[:, np.newaxis]
    return norm_arr

def load_data(DB_name):
    '''
    load dataset
    '''
    if DB_name == "gist":
        DB_DIR = './dataset/gist/'
        x_d = mmap_fvecs('{}gist_base.fvecs'.format(DB_DIR)) # data (1000000, 960)
        x_q = mmap_fvecs('{}gist_query.fvecs'.format(DB_DIR)) # query (1000, 960)
        xt = mmap_fvecs('{}gist_learn.fvecs'.format(DB_DIR)) # (500000, 960)
        gt_ids = None
    elif DB_name == "sift":
        DB_DIR = './dataset/sift/'
        x_d = mmap_fvecs('{}sift_base.fvecs'.format(DB_DIR)) # data (1000000, 128)
        x_q = mmap_fvecs('{}sift_query.fvecs'.format(DB_DIR)) # query (10000, 128)
        xt = mmap_fvecs('{}sift_learn.fvecs'.format(DB_DIR)) # (100000, 128)
        gt_ids = read_ivecs('{}sift_groundtruth.ivecs'.format(DB_DIR))
    elif DB_name == "glove":
        DB_DIR = './dataset/glove-100-angular.hdf5'
        dataset = h5py.File(DB_DIR, 'r')
        xd_ori = np.array(dataset["train"]) # ndarray (1183514, 100)
        xq_ori = np.array(dataset['test']) # (10000, 100)
        gt_ids = np.array(dataset['neighbors']) # (10000, 100)
        x_d = norm_glove(xd_ori) # np.linalg.norm(x_d[0]) = 1
        x_q = norm_glove(xq_ori)
        xt = None
    elif DB_name == "bigann":
        DB_DIR = './dataset/bigann/'
        x_d = mmap_fvecs('{}bigann_base_50m.bvecs'.format(DB_DIR), dtype=np.uint8)
        x_q = mmap_fvecs('{}bigann_query.bvecs'.format(DB_DIR), dtype=np.uint8) 
        gt_ids = None
        xt = None
    elif DB_name == 'deep50M':
        DB_DIR = './dataset/deep50M/'
        # DB_DIR = '/data/cph/Project/balanced-multi-probe-ANN/dataset/deep50M/'
        x_d = mmap_fvecs('{}deep_base_50m.fvecs'.format(DB_DIR)) # data (50M, 96)
        x_q = mmap_fvecs('{}deep_query.fvecs'.format(DB_DIR)) # query (10000, 96)
        gt_ids = None
        xt = None
    elif DB_name == 'text2image50M':
        DB_DIR = './dataset/text2image/'
        x_d = mmap_fvecs('{}text2image_base_50m.fvecs'.format(DB_DIR)) # data (50M, 200)
        x_q = mmap_fvecs('{}text2image_query.fvecs'.format(DB_DIR)) # query (100000, 200)
        gt_ids = None
        xt = None
    elif DB_name == 'deep100M':
        DB_DIR = './dataset/deep100M/'
        xd_ori = mmap_fvecs('{}deep_base_100m.fvecs'.format(DB_DIR)) # data (100M, 96)
        xq_ori = mmap_fvecs('{}deep_query.fvecs'.format(DB_DIR)) # query (10000, 96)
        x_d = norm_glove(xd_ori)
        x_q = norm_glove(xq_ori)
        gt_ids = None
        xt = None
    else:
        print("unknown dataset")
    # for faiss1.5.2
    x_q = np.ascontiguousarray(x_q)
    x_d = np.ascontiguousarray(x_d)
    return x_d, x_q, xt, gt_ids

def get_idle_gpu():
    # get GPU status
    smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits']).decode()
    gpu_usage = [int(re.search(r'\d+$', line).group()) for line in smi_output.strip().split('\n')]
    idle_gpu_index = gpu_usage.index(min(gpu_usage))

    return idle_gpu_index

def get_dist_cid(data, kmeans, n_bkt):
    # batch distance calculation
    batch_size = 512
    n_samples = data.shape[0]
    n_centroids = kmeans.centroids.shape[0]
    distances = np.zeros((n_samples, n_centroids), dtype=np.float32)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        data_batch = data[start_idx:end_idx]
        distances_batch = np.linalg.norm(data_batch[:, None] - kmeans.centroids, axis=2)
        distances[start_idx:end_idx] = distances_batch

    return distances

def get_scaled_dist(x_d, x_q, kmeans, n_bkt):
    distances_data = get_dist_cid(x_d, kmeans, n_bkt)
    distances_query = get_dist_cid(x_q, kmeans, n_bkt)
    scaler = StandardScaler()
    distances_data_scaled = scaler.fit_transform(distances_data)
    distances_query_scaled = scaler.transform(distances_query)

    return distances_data_scaled, distances_query_scaled

def get_scaled_dist_data(x_d, kmeans, n_bkt):
    distances_data = get_dist_cid(x_d, kmeans, n_bkt)
    scaler = StandardScaler()
    distances_data_scaled = scaler.fit_transform(distances_data)

    return distances_data_scaled

def fprint(message, file=None):
    print(message)  # print to terminal
    if file:
        print(message, file=file)  # print to file


def get_flat_GT_knn(x_data, x_query, cfg, datatype="query"):
    '''
    get the top-k knn of query or data on the dataset with len(x_data)
    x_data can be a subset of the whole dataset
    '''
    pth_query_knn = f'./dataset/{cfg.dataset}/{cfg.dataset}-xd{len(x_data)}_{datatype}_knn{cfg.k}.npy'
    if not os.path.exists(pth_query_knn):
        dim = x_data.shape[1]
        index_flat = faiss.IndexFlatL2(dim)
        index_flat.add(x_data)
        _, knn_query = index_flat.search(x_query, cfg.k)
        np.save(pth_query_knn, knn_query)
    else:
        knn_query = np.load(pth_query_knn)
        knn_query = knn_query.astype(int) # will be loaded as float

    return knn_query

def build_kmeans_index(x_data, n_bkt):
    n_d, dim = x_data.shape
    kmeans = faiss.Kmeans(dim, n_bkt, niter=20, verbose=True)
    kmeans.train(x_data)
    _, data_2_bkt = kmeans.index.search(x_data, 1)  # get the cluster index of each data point
    cluster_cnts = np.bincount(data_2_bkt.flatten())  # get the number of data points in each cluster
    cluster_ids = [[] for _ in range(n_bkt)]
    for i in range(n_d):
        cluster_ids[data_2_bkt[i, 0]].append(i)
    return kmeans, data_2_bkt, cluster_cnts, cluster_ids

def get_knn_distr(knn, data_2_bkt, cfg):
    '''
    knn_distr. the distribution of knn counts of each query
    knn_id. the knn ids of each query in each cluster
    '''
    n_data = knn.shape[0]
    knn_distr_cnt = np.zeros((n_data, cfg.n_bkt), dtype=int)
    knn_distr_id = np.empty((n_data, cfg.n_bkt), dtype=object)
    for i in range(n_data):
        for j in range(cfg.n_bkt):
            knn_distr_id[i, j] = []
    
    for v_idx in tqdm(range(n_data)):
        v_knn_ids = knn[v_idx] # the knn ids of a query
        v_knn_bkts = data_2_bkt[v_knn_ids].flatten() # the cluster id of each knn of a query
        unique_bkts, counts = np.unique(v_knn_bkts, return_counts=True)
        knn_distr_cnt[v_idx, unique_bkts] = counts
        for bkt in unique_bkts:
            knn_distr_id[v_idx, bkt] = v_knn_ids[v_knn_bkts == bkt].tolist()
    
    return knn_distr_cnt, knn_distr_id

def get_knn_distr_redundancy(knn, data_2_bkt, cfg):
    '''
    knn_distr. the distribution of knn counts of each query
    knn_id. the knn ids of each query in each cluster
    '''
    _, n_mul = data_2_bkt.shape
    n_data = knn.shape[0]
    knn_distr_cnt = np.zeros((n_data, cfg.n_bkt), dtype=int)
    knn_distr_id = np.empty((n_data, cfg.n_bkt), dtype=object)
    for i in range(n_data):
        for j in range(cfg.n_bkt):
            knn_distr_id[i, j] = []
    
    for v_idx in tqdm(range(n_data)):
        v_knn_ids = knn[v_idx] # the knn ids of a query
        v_knn_bkts = data_2_bkt[v_knn_ids].flatten() # the cluster id of each knn of a query
        unique_bkts, counts = np.unique(v_knn_bkts, return_counts=True)
        if unique_bkts[0] == -1:
            unique_bkts = unique_bkts[1:]
            counts = counts[1:]
        knn_distr_cnt[v_idx, unique_bkts] = counts
        for bkt in unique_bkts:
            v_mul_knn_ids = np.repeat(v_knn_ids, n_mul) # [1,3,...] -> [1,...,1,3,...,3,...]
            knn_distr_id[v_idx, bkt] = v_mul_knn_ids[v_knn_bkts == bkt].tolist()
    
    return knn_distr_cnt, knn_distr_id

def create_hnsw_indexes(x_d, xd_id_bkts, cfg, dis_metric: str = 'L2'):
    """
    create a hnsw index for each bucket
    """
    hnsw_indexes = []
    for i in range(cfg.n_bkt):
        xd_id_bkt = np.array(xd_id_bkts[i])
        xd_bkt = x_d[xd_id_bkt]
        hnsw_index = faiss.IndexHNSWFlat(xd_bkt.shape[1], cfg.hnsw_M)
        if dis_metric == 'inner_product':
            hnsw_index.metric_type = faiss.METRIC_INNER_PRODUCT
        hnsw_index.add(xd_bkt)
        hnsw_indexes.append(hnsw_index)

    return hnsw_indexes

def create_flat_indexes(x_d, xd_id_bkts, cfg, dis_metric: str = 'L2'):
    """
    create a flat index for each bucket
    """
    flat_indexes = []
    for i in range(cfg.n_bkt):
        xd_id_bkt = np.array(xd_id_bkts[i])
        xd_bkt = x_d[xd_id_bkt]
        if dis_metric == 'inner_product':
            flat_index = faiss.IndexFlatIP(xd_bkt.shape[1])
        else:
            flat_index = faiss.IndexFlatL2(xd_bkt.shape[1])
        flat_index.add(xd_bkt)
        flat_indexes.append(flat_index)

    return flat_indexes

def create_inner_indexes(x_d, cluster_ids, cfg):
    if cfg.inner_index_type == 'FLAT':
        inner_indexes = create_flat_indexes(x_d, cluster_ids, cfg)
    elif cfg.inner_index_type == 'HNSW':
        inner_indexes = create_hnsw_indexes(x_d, cluster_ids, cfg)
    return inner_indexes

def min_exclude_zero(row):
    non_zero_elements = row[row != 0]
    if non_zero_elements.size > 0:
        return non_zero_elements.min()
    else:
        return np.nan

def observe_knn_tail(knn_distr_cnt, knn_distr_id, n_d, cfg, model, distances_data_scaled, x_d, data_2_bkt, device):
    '''
    observe the distribution of long-tail knn
    '''
    min_values = np.apply_along_axis(min_exclude_zero, 1, knn_distr_cnt)
    unique_elements, counts = np.unique(min_values, return_counts=True)
    print(np.asarray((unique_elements, counts)).T)

    # analyze the long-tail data. When a data is a long-tail data in a knn distribution, get the replica buckets where # of knn >= 2
    tail_ids = []
    tail_id_other_bkts = np.zeros((n_d, cfg.n_bkt), dtype=bool)
    for q_id in range(len(knn_distr_cnt)):
        bkt_tail = np.where(knn_distr_cnt[q_id] == 1)[0] # the bucket with only one knn
        if len(bkt_tail) > 0:
            bkt_non_tail = np.where(knn_distr_cnt[q_id] > 1)[0] # the bucket with more than one knn
            vec_tail = np.concatenate(knn_distr_id[q_id][bkt_tail]) # the long-tail data id
            tail_ids.extend(vec_tail)
            for vec in vec_tail:
                tail_id_other_bkts[vec][bkt_non_tail] = 1
    
    tail_id = np.where(np.any(tail_id_other_bkts, axis=1))[0] # the long-tail data id after removing duplicates
    # np.save(f'./dataset/{cfg.dataset}/{cfg.dataset}-bkt{cfg.n_bkt}query_longtail_id.npy', tail_id_test)
    # np.save(f'./dataset/{cfg.dataset}/{cfg.dataset}-bkt{cfg.n_bkt}data_longtail_id.npy', tail_id_train)
    
    n_tail_id = len(tail_id) # 5, len(tail_id)
    output_rank_replica = np.zeros((n_tail_id, cfg.n_bkt), dtype=int) # the ranking of replica buckets by model output (probing rank)
    dist_rank_replica = np.zeros((n_tail_id, cfg.n_bkt), dtype=int) # the ranking of replica buckets by centroids distance in IVF/kmeans (distance rank)
    output_matrix = np.zeros((n_tail_id, cfg.n_bkt)) # model output
    replica_matrix = np.zeros((n_tail_id, cfg.n_bkt), dtype=int) # the replica buckets for each long-tail data

    for idx, vid in enumerate(tail_id[:n_tail_id]):
        model.eval()
        with torch.no_grad():
            output = model(
                torch.from_numpy(distances_data_scaled[vid]).unsqueeze(0).to(device), 
                torch.from_numpy(x_d[vid]).unsqueeze(0).to(device)
            ).cpu()[0]
        output_matrix[idx] = output
        vec_tail_bkt = np.where(tail_id_other_bkts[vid] == True)[0]
        replica_matrix[idx][vec_tail_bkt] = 1
        output_sorted_idx = np.argsort(-output) # model output sorted index (probing rank)
        distance_sorted_idx = np.argsort(distances_data_scaled[vid]) # centroids distance sorted in ivf/kmeans (distance rank)
        bkt_output_pairs = [(bkt, output[bkt].item(), np.where(output_sorted_idx == bkt)[0][0], np.where(distance_sorted_idx == bkt)[0][0]) for bkt in vec_tail_bkt]
        sorted_bkt_output_pairs = sorted(bkt_output_pairs, key=lambda x: x[2])
        
        print('-' * 40)
        print(f'vec [{vid}] prob of tail_to_replica_bkt, self bkt_id {data_2_bkt[vid]}')
        for bkt, probi, p_rank, dis_rank in sorted_bkt_output_pairs:
            output_rank_replica[idx][p_rank] = 1
            dist_rank_replica[idx][dis_rank] = 1
            print(f"bkt_Id: {bkt}, output: {probi:.4f}, output rank: {p_rank}, dist rank: {dis_rank}")

    # analysis of the nprobe reduction when putting the long-tail data into the replica buckets with probing rank and distance rank
    output_rank_replica_cum = np.maximum.accumulate(output_rank_replica, axis=1)
    dist_rank_replica_cum = np.maximum.accumulate(dist_rank_replica, axis=1)

    output_rank_valid = np.sum(output_rank_replica_cum, axis=0) / n_tail_id # the effectiveness of long-tail data
    dist_rank_valid = np.sum(dist_rank_replica_cum, axis=0) / n_tail_id
    print("output_rank_valid", output_rank_valid)
    print("dist_rank_valid", dist_rank_valid)

    output_matrix_masked = output_matrix * replica_matrix
    replica_values = output_matrix_masked[output_matrix_masked != 0]

def per_query(all_outputs, knn_distr_cnt_query, cluster_cnts, n_bkt, cfg):
    nq_test = 100
    cmp_per_q = np.zeros(nq_test, dtype=int)
    nprobe_per_q = np.zeros(nq_test, dtype=int)
    recall_target = 0.98

    df_result_perquery = pd.DataFrame(columns=['q_id', 'nprobe', 'cmp'])

    for q_id in range(nq_test):
        for probeM in range(1, 20):
            Mbkt = all_outputs[q_id].topk(probeM).indices.cpu().numpy()
            match_count = knn_distr_cnt_query[q_id, Mbkt].sum()
            if match_count / cfg.k >= recall_target:
                cmp_per_q[q_id] = cluster_cnts[Mbkt].sum()
                nprobe_per_q[q_id] = probeM
                break
        df_result_perquery = pd.concat([df_result_perquery, pd.DataFrame({'q_id': [q_id], 'nprobe': [nprobe_per_q[q_id]], 'cmp': [cmp_per_q[q_id]]})], ignore_index=True)
    df_result_perquery.to_csv(cfg.pth_log + f'{cfg.dataset}-k={cfg.k}-ML_kmeans={n_bkt}_perquery.csv', index=False)


