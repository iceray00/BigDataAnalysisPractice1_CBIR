import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from tqdm import tqdm


# ----------------------
# 模块3：LSH索引
# ----------------------
class LSHIndexer:
    def __init__(self, num_tables=10, hash_size=8, random_seed=4254):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.hash_tables = []
        self.random_seed = random_seed
        self.data = None

    def build_index(self, data, lsh_method="stable"):
        if lsh_method == "stable":
            self.build_index_stable(data)
        # else:
        #     self.build_index_random(data)

    def build_index_stable(self, data):
        np.random.seed(self.random_seed)
        self.data = data
        dim = data.shape[1]

        for _ in tqdm(range(self.num_tables), desc='Building LSH Tables'):
            # 生成随机投影向量并归一化
            projections = np.random.randn(self.hash_size, dim)
            projections /= np.linalg.norm(projections, axis=1, keepdims=True)

            table = {
                'projections': projections,
                'buckets': defaultdict(list)
            }

            # 计算哈希值并存入哈希表
            for idx, vec in enumerate(data):
                hash_key = tuple((np.dot(vec, projections.T) > 0).astype(int))
                table['buckets'][hash_key].append(idx)

            self.hash_tables.append(table)


    def query(self, query_vec, k=5):
        candidates = set()
        query_vec = query_vec / np.linalg.norm(query_vec)  # 归一化

        # for table in tqdm(self.hash_tables, desc='Querying LSH Tables'):
        for table in self.hash_tables:
            hash_key = tuple((np.dot(query_vec, table['projections'].T) > 0).astype(int))
            candidates.update(table['buckets'].get(hash_key, []))

        # 计算余弦相似度
        similarities = cosine_similarity([query_vec], self.data[list(candidates)])[0]

        # similarities = np.linalg.norm(self.data[list(candidates)] - query_vec, axis=1)

        # 根据相似度对候选集进行排序
        sorted_indices = np.argsort(-similarities)[:k]

        return [list(candidates)[i] for i in sorted_indices]
