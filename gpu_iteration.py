import argparse
import logging
import os
from typing import Literal
import numpy as np
import torch
import torch.cuda
import torchvision.models as models
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from tqdm import tqdm
from annoy import AnnoyIndex
from PIL import Image
from datetime import datetime
from skimage.feature import hog, local_binary_pattern
from LSH import LSHIndexer


# 添加GPU信息打印函数
def print_gpu_info():
    """打印GPU信息"""
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPU可用! 当前设备: {current_device}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 可用GPU数量: {device_count}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPU名称: {device_name}")
        # 打印GPU内存信息
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPU总内存: {torch.cuda.get_device_properties(current_device).total_memory / 1024 ** 3:.2f} GB")
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 当前分配的GPU内存: {torch.cuda.memory_allocated(current_device) / 1024 ** 3:.2f} GB")
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 缓存的GPU内存: {torch.cuda.memory_reserved(current_device) / 1024 ** 3:.2f} GB")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPU不可用，将使用CPU运行!")


# ----------------------
# 模块1：数据预处理
# ----------------------
class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_paths = []
        self.image_names = []
        self.labels = []

    def load_data(self):
        for class_dir in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_dir)
            for img_file in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_file))
                self.image_names.append(img_file)  # 文件名
                self.labels.append(class_dir)  # 标签，也就是前面的文件夹名
        return self.image_paths, self.image_names, self.labels


# ----------------------
# 模块2：特征提取
# ----------------------
class FeatureExtractor:
    def __init__(self, method='hog'):
        self.method = method
        self.cnn_model = None
        self.preprocess = None
        # 检测GPU是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 使用设备: {self.device}")
        if method == 'cnn':
            self._init_cnn()

    def _init_cnn(self):
        self.cnn_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # self.cnn_model = models.vgg16(pretrained=True)
        self.cnn_model = torch.nn.Sequential(*(list(self.cnn_model.children())[:-1]))
        # 将模型移动到相应设备(GPU或CPU)
        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image_path):
        img = Image.open(image_path).convert('RGB')
        # 统一图像方向
        if img.height > img.width:
            img = img.rotate(90, expand=True)
        if self.method == 'hog':
            return self._extract_hog(img)
        if self.method == 'hog_p':
            return self._extract_hog_plus(img)
        elif self.method == 'lbp':
            return self._extract_lbp(img)
        elif self.method == 'lbp_p':
            return self._extract_lbp_plus(img)
        elif self.method == 'cnn':
            return self._extract_cnn(img)
        else:
            raise ValueError("Unsupported feature method")

    def _extract_hog(self, img):
        img = img.resize((64, 64))  # 统一尺寸
        gray = np.array(img.convert('L'))
        fd = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(4, 4), block_norm='L2-Hys')
        return fd

    def _extract_hog_plus(self, img):
        # 增加图像的分辨率
        img = img.resize((128, 128))  # 更高分辨率的图像
        # 将图像转换为灰度图
        gray = np.array(img.convert('L'))
        # 增加方向数，并减小像素单元大小
        fd = hog(gray, orientations=8, pixels_per_cell=(21, 21), cells_per_block=(5, 5), block_norm='L2-Hys',
                 visualize=False)
        return fd

    def _extract_lbp(self, img):
        img = img.resize((128, 128))
        gray = np.array(img.convert('L'))
        lbp = local_binary_pattern(gray, 24, 3, method='default')  # 使用uniform的话，会导致只有前26个非零，而其他全为 0
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        return hist

    def _extract_lbp_plus(self, img):
        img = img.resize((128, 128))
        gray = np.array(img.convert('L'))
        # 修改LBP的参数
        lbp = local_binary_pattern(gray, P=8, R=1, method='default')  # 或者可以试试 'uniform'
        # 创建直方图
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))  # 使用合适的bins数量
        hist = hist.astype('float32')
        hist /= hist.sum()  # 归一化
        return hist

    def _extract_cnn(self, img):
        img_t = self.preprocess(img)
        img_t = img_t.unsqueeze(0)
        # 将输入张量移动到相应设备
        img_t = img_t.to(self.device)
        with torch.no_grad():
            features = self.cnn_model(img_t)
        # 确保特征返回到CPU以便后续处理
        return features.squeeze().cpu().numpy()

    def batch_extract(self, image_paths, batch_size=16):
        """
        批量提取特征以加速处理

        Args:
            image_paths: 图像路径列表
            batch_size: 批处理大小

        Returns:
            features_list: 特征列表
        """
        if self.method != 'cnn':
            # 如果不是CNN方法，则逐个处理
            return [self.extract(path) for path in tqdm(image_paths, desc="Extracting Features")]

        features_list = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Batch Processing"):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            valid_indices = []  # 记录有效的图像索引

            # 预处理每个图像并收集到一个批次中
            for j, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert('RGB')
                    # 统一图像方向
                    if img.height > img.width:
                        img = img.rotate(90, expand=True)
                    img_t = self.preprocess(img)
                    batch_tensors.append(img_t)
                    valid_indices.append(j)
                except Exception as e:
                    print(f"Error processing image {path}: {e}")
                    # 跳过错误的图像

            # 将批次堆叠成一个张量
            if batch_tensors:
                batch_tensor = torch.stack(batch_tensors).to(self.device)

                # 一次处理整个批次
                with torch.no_grad():
                    batch_features = self.cnn_model(batch_tensor)

                # 将特征分配回原始顺序
                batch_features_cpu = batch_features.cpu().numpy()
                for idx, feat in zip(valid_indices, batch_features_cpu):
                    while len(features_list) < i + idx:
                        # 填充任何缺失的特征（如果有）
                        features_list.append(None)
                    features_list.append(feat.squeeze())

                # 确保列表长度与当前批次末尾对齐
                while len(features_list) < i + len(batch_paths):
                    features_list.append(None)

        # 过滤掉可能的None值（由于处理错误）
        features_list = [f for f in features_list if f is not None]
        return features_list


# 优化的特征提取函数
def extract_features_optimized(extractor, image_paths, args):
    """优化的特征提取过程"""
    try:
        if args.feature_method == 'cnn':
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 使用批处理加速CNN特征提取")
            # 可以根据GPU内存和图像大小调整批处理大小
            if torch.cuda.is_available():
                # 如果有大量GPU内存，可以使用更大的批处理大小
                total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                if total_memory_gb > 10:  # 超过10GB
                    batch_size = 32
                elif total_memory_gb > 6:  # 超过6GB
                    batch_size = 16
                else:  # 较小内存
                    batch_size = 8
            else:
                # CPU模式使用较小的批处理大小
                batch_size = 4

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 使用批处理大小: {batch_size}")
            features = extractor.batch_extract(image_paths, batch_size=batch_size)
        else:
            features = []
            for path in tqdm(image_paths, desc="Extracting Features"):
                features.append(extractor.extract(path))
                if args.detail:
                    print(extractor.extract(path))

        features = np.array(features)
        return features
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] 特征提取失败!! {str(e)}")
        logging.error(f"特征提取失败: {str(e)}")
        raise e


class AnnoyIndexer:
    def __init__(self, num_trees=10,
                 metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"] = "angular"):
        # 确保metric是合法的选项之一
        self.num_trees = num_trees
        self.metric = metric  # 直接传递合法的metric值
        self.index = None

    def build_index(self, data):
        dim = data.shape[1]
        self.index = AnnoyIndex(dim, metric=self.metric)
        # 向Annoy索引中添加数据
        for i, vec in enumerate(data):
            self.index.add_item(i, vec)
        # 构建树
        self.index.build(self.num_trees)

    def query(self, query_vec, k=5):
        return self.index.get_nns_by_vector(query_vec, k)


# ----------------------
# 模块4：评估系统
# ----------------------
class Evaluator:
    @staticmethod
    def evaluate_batch(num_samples, image_paths, image_names, labels, features, indexer, annoy_indexer, top_k=10,
                       show=False):
        """
        对随机抽取的图片进行批量评估
        Args:
            num_samples: 要评估的样本数量
            image_paths: 图片路径列表
            image_names: 图片名称列表
            labels: 图片标签列表
            features: 预先提取的特征
            indexer: 自定义LSH索引器
            annoy_indexer: Annoy索引器
            top_k: 检索的相似图片数量
            show: 是否在终端显示进度
        Returns:
            (custom_lsh_hit_rate, annoy_hit_rate): 平均命中率
        """
        # 随机抽样
        total_images = len(image_paths)
        if num_samples > total_images:
            num_samples = total_images
            if show:
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 警告: 请求的样本数大于可用图片数。使用全部 {total_images} 张图片。")
        sample_indices = np.random.choice(total_images, num_samples, replace=False)
        # 初始化命中计数
        custom_lsh_hits = 0
        annoy_hits = 0
        # 评估每个抽样图片
        for i, idx in enumerate(tqdm(sample_indices, desc="评估图片中")):
            query_path = image_paths[idx]
            query_name = image_names[idx]
            query_label = labels[idx]
            # 使用预先提取的特征
            query_feature = features[idx]
            # 使用自定义LSH查询
            retrieved_indices = indexer.query(query_feature, k=top_k)
            # 不过滤掉查询图片自身
            retrieved_labels = [labels[i] for i in retrieved_indices]
            # 计算命中率
            if retrieved_labels:
                custom_hits = sum(1 for label in retrieved_labels if label == query_label)
                custom_lsh_hits += custom_hits / len(retrieved_labels)
            # 使用Annoy查询
            retrieved_indices_annoy = annoy_indexer.query(query_feature, k=top_k)
            # 不过滤掉查询图片自身
            retrieved_labels_annoy = [labels[i] for i in retrieved_indices_annoy]
            # 计算命中率
            if retrieved_labels_annoy:
                annoy_hits_count = sum(1 for label in retrieved_labels_annoy if label == query_label)
                annoy_hits += annoy_hits_count / len(retrieved_labels_annoy)
        # 计算平均命中率
        custom_lsh_hit_rate = custom_lsh_hits / num_samples
        annoy_hit_rate = annoy_hits / num_samples
        return custom_lsh_hit_rate, annoy_hit_rate


def display_results(query_img_path, result_indices, database_image_paths, database_image_names, top_k=5, name="ICERAY"):
    """
    展示查询图片和检索结果
    """
    query_img = Image.open(query_img_path)
    filename = os.path.basename(query_img_path)
    plt.figure(figsize=(15, 3))
    # 显示查询图片
    plt.subplot(1, top_k + 1, 1)
    plt.imshow(query_img)
    plt.title(f"{name} · {filename}")
    plt.axis('off')
    # 显示检索结果及其文件名
    for i, idx in enumerate(result_indices[:top_k]):
        img_path = database_image_paths[idx]
        img_name = database_image_names[idx]
        img = Image.open(img_path)
        plt.subplot(1, top_k + 1, i + 2)
        plt.imshow(img)
        plt.title(f"Top {i + 1}\n{img_name}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def process_query(query_image, top_k, extractor, indexer, annoy_indexer, image_paths, image_names, labels, show=False):
    """处理单张查询图片"""
    try:
        # 检查查询图片是否存在
        if not os.path.exists(query_image):
            raise FileNotFoundError(f"Query image {query_image} not found!")
        filename = os.path.basename(query_image)
        query_label = str(filename.split('_')[0])
        print(f"查询图片标签：{query_label}")
        # 提取查询图片特征
        query_feature = extractor.extract(query_image)
        # 使用自定义索引进行检索
        retrieved_indices = indexer.query(query_feature, k=top_k)
        print(f"自定义检索结果: {retrieved_indices}")
        # 收集检索到的文件名
        retrieved_filenames = [image_names[idx] for idx in retrieved_indices]
        retrieved_labels = [labels[idx] for idx in retrieved_indices]
        print(f"成功检索到 {top_k} 张，由 ICERAY · LSH 索引进行相似图片")
        print(f"自定义·检索到的文件名：{retrieved_filenames}")  # 打印所有检索到的文件名
        print(f"自定义·检索到的标签：{retrieved_labels}")
        iceray_hit_rate = np.sum(np.array(retrieved_labels) == query_label) / len(retrieved_labels)
        print(f"自定义·命中率：{100 * iceray_hit_rate:.2f}%")
        # 可视化结果
        display_results(query_image, retrieved_indices, image_paths, image_names, top_k=top_k, name="ICERAY")
        # 使用Annoy索引进行检索
        retrieved_indices_annoy = annoy_indexer.query(query_feature, k=top_k)
        print(f"成功检索到 {top_k} 张，由 Annoy · LSH 索引进行相似图片")
        # 收集检索到的文件名
        retrieved_filenames_annoy = [image_names[idx] for idx in retrieved_indices_annoy]
        retrieved_labels_annoy = [labels[idx] for idx in retrieved_indices_annoy]
        print(f"Annoy·检索到的文件名：{retrieved_filenames_annoy}")
        print(f"Annoy·检索到的标签：{retrieved_labels_annoy}")
        annoy_hit_rate = np.sum(np.array(retrieved_labels_annoy) == query_label) / len(retrieved_labels_annoy)
        print(f"Annoy·命中率：{100 * annoy_hit_rate:.2f}%")
        # 可视化结果
        display_results(query_image, retrieved_indices_annoy, image_paths, image_names, top_k=top_k, name="Annoy")
        return True
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] 查询处理失败: {str(e)}")
        return False


def interactive_mode(image_paths, image_names, labels, features, indexer, annoy_indexer, extractor, query_image=None, show=False):
    """交互式模式，允许用户调整参数并进行查询"""
    print("\n" + "=" * 80)
    print("欢迎进入交互式查询模式")
    print("=" * 80)
    while True:
        print("\n请选择操作：")
        print("1. 进行批量评估 (指定 n 和 k)")
        print("2. 处理查询图像 (指定 k)")
        print("3. 退出")
        choice = input("\n请输入选项 (1/2/3): ").strip()
        if choice == '1':
            try:
                n = int(input("请输入评估样本数量 (-n): ").strip())
                k = int(input("请输入Top-K值 (-k): ").strip())
                print(f"\n开始对随机 {n} 张图片进行批量评估，Top-K={k}")
                custom_hit_rate, annoy_hit_rate = Evaluator.evaluate_batch(
                    n, image_paths, image_names, labels,
                    features, indexer, annoy_indexer, top_k=k, show=show
                )
                print(f"自定义LSH平均命中率: {100 * custom_hit_rate:.2f}%")
                print(f"Annoy LSH平均命中率: {100 * annoy_hit_rate:.2f}%")
            except ValueError:
                print("输入无效，请确保输入的是整数。")
            except Exception as e:
                print(f"批量评估失败: {str(e)}")
        elif choice == '2':
            if query_image:
                try:
                    k = int(input("请输入Top-K值 (-k): ").strip())
                    process_query(query_image, k, extractor, indexer, annoy_indexer,
                                  image_paths, image_names, labels, show)
                except ValueError:
                    print("输入无效，请确保输入的是整数。")
            else:
                query_path = input("请输入查询图像路径: ").strip()
                if os.path.exists(query_path):
                    try:
                        k = int(input("请输入Top-K值 (-k): ").strip())
                        process_query(query_path, k, extractor, indexer, annoy_indexer,
                                      image_paths, image_names, labels, show)
                    except ValueError:
                        print("输入无效，请确保输入的是整数。")
                else:
                    print(f"图像路径不存在: {query_path}")
        elif choice == '3':
            print("退出交互式模式。")
            break
        else:
            print("无效选项，请输入1、2或3。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于LSH的图像检索系统")
    parser.add_argument("--path", type=str, default="dataset_simple", help="Path to the image files")
    parser.add_argument("-s", "--show", action='store_true', default=False, help='Show in Terminal. Default=\'False\'')
    parser.add_argument("-m", "--feature_method", type=str, default="hog",
                        help="Feature method. Please input: \'hog\', \'hog_p\', \'lbp\', \'lbp_p\', \'cnn\'. Default=\'hog\'")
    parser.add_argument("-l", "--lsh_method", type=str, default="stable", help="LSH method")
    parser.add_argument("-nt", "--num_tables", type=int, default=15, help="Number of hash tables. Default=\'15\'")
    parser.add_argument("-hs", "--hash_size", type=int, default=9, help="Hash size. Default=\'9\'")
    parser.add_argument("-ntree", "--num_trees", type=int, default=15, help="Number of Annoy trees. Default=\'15\'")
    parser.add_argument("-d", "--detail", action='store_true', default=False,
                        help='Show EACH feature. Default=\'False\'')
    parser.add_argument("-q", "--query_image", type=str, help="Path to the query image for search")
    parser.add_argument("-k", "--top_k", type=int, default=10,
                        help="Number of top similar images to retrieve. Default=10")
    # 添加参数指定的评估（如果有）
    parser.add_argument("-n", "--num_eval_samples", type=int, help="Number of random images to sample for evaluation")
    parser.add_argument("-i", "--interactive", action='store_true', default=False,
                        help='Enable interactive mode after preprocessing')
    # 添加GPU相关参数
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Batch size for CNN feature extraction (0 for auto-detection)")
    parser.add_argument("--no_gpu", action='store_true', default=False,
                        help='Disable GPU even if available')
    args = parser.parse_args()

    # 如果用户指定了不使用GPU，则设置环境变量
    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 已禁用GPU，将使用CPU进行计算")

    # 配置日志
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    log_filename = f"logs/{current_time}.log"
    log_folder = os.path.dirname(log_filename)  # 提取日志文件夹路径
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)  # 检查文件夹是否存在，如果不存在则创建
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 打印GPU信息
    print_gpu_info()
    logging.info("检查GPU可用性")
    if torch.cuda.is_available() and not args.no_gpu:
        gpu_info = f"GPU可用: {torch.cuda.get_device_name(0)}"
        logging.info(gpu_info)
    else:
        logging.info("GPU不可用或已禁用，将使用CPU")

    # /* =================================== 参数配置 =================================== */
    logging.info("Step0. === Step0.=== 参数配置如下")
    if args.show:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Step0.=== 参数配置如下")
    DATASET_PATH = args.path
    FEATURE_METHOD = args.feature_method
    NUM_TABLES = args.num_tables
    HASH_SIZE = args.hash_size
    NUM_TREES = args.num_trees
    logging.info(f"DATASET_PATH : {DATASET_PATH}")
    logging.info(f"特征提取方法   : {FEATURE_METHOD}")
    logging.info(f"LSH表数量     : {NUM_TABLES}")
    logging.info(f"LSH表大小     : {HASH_SIZE}")
    logging.info(f"Top_K        : {args.top_k}")
    logging.info(f"Annoy 树数量  : {NUM_TREES}")
    logging.info(f"是否存在查询   : {args.query_image}")
    logging.info(f"评估样本数量   : {args.num_eval_samples}")
    logging.info(f"交互式模式     : {args.interactive}")
    if args.show:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DATASET_PATH : {DATASET_PATH}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 特征提取方法   : {FEATURE_METHOD}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LSH表数量     : {NUM_TABLES}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LSH表大小     : {HASH_SIZE}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Top_K        : {args.top_k}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Annoy 树数量  : {NUM_TREES}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 是否存在查询   : {args.query_image}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 评估样本数量   : {args.num_eval_samples}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 交互式模式     : {args.interactive}")
    # /* =================================== 数据加载 =================================== */
    logging.info("Step1. 开始进行数据加载")
    if args.show:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Step1.=== 开始进行数据加载")
    # 1. 数据加载
    loader = DataLoader(DATASET_PATH)
    image_paths, image_names, labels = loader.load_data()
    logging.info(f"已加载 {len(image_paths)} 张图像")

    # /* =================================== 特征提取 =================================== */
    logging.info("Step2. 开始进行特征提取")
    if args.show:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Step2.=== 开始进行特征提取")
    try:
        extractor = FeatureExtractor(FEATURE_METHOD)
        # 使用优化的特征提取函数
        features = extract_features_optimized(extractor, image_paths, args)
        logging.info(f"特征提取完成，特征维度: {features.shape}")
    except Exception as e:
        if args.show:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] 特征提取失败!!")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] {e}")
        logging.info("[ERROR] 特征提取失败!!")
        logging.error(e)
        exit(1)

    # /* =================================== LSH 索引构建 =================================== */
    logging.info("Step3. 开始进行 ICERAY · LSH 索引构建")
    if args.show:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Step3.=== 开始进行 ICERAY · LSH 索引构建")
    try:
        # 自实现LSH
        indexer = LSHIndexer(NUM_TABLES, HASH_SIZE)
        indexer.build_index(features, args.lsh_method)
        logging.info("ICERAY LSH 索引构建完成")
    except Exception as e:
        if args.show:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] ICERAY · LSH 索引构建失败!!")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] {e}")
        logging.info("[ERROR] ICERAY · LSH 索引构建失败!!")
        logging.error(e)
        exit(1)

    logging.info("Step4. 开始进行 Annoy-LSH 索引构建")
    if args.show:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Step4.=== 开始进行 Annoy-LSH 索引构建")
    try:
        # 使用Annoy进行LSH索引
        annoy_indexer = AnnoyIndexer(NUM_TREES, metric='angular')
        annoy_indexer.build_index(features)
        logging.info("Annoy-LSH 索引构建完成")
    except Exception as e:
        if args.show:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Annoy-LSH 索引构建失败!!")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] {e}")
        logging.info("[ERROR] Annoy-LSH 索引构建失败!!")
        logging.error(e)
        exit(1)

    # 首先执行参数指定的评估（如果有）
    if args.num_eval_samples:
        logging.info(f"Step5. 开始对随机 {args.num_eval_samples} 张图片进行批量评估")
        if args.show:
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Step5.=== 开始对随机 {args.num_eval_samples} 张图片进行批量评估")
        try:
            custom_hit_rate, annoy_hit_rate = Evaluator.evaluate_batch(
                args.num_eval_samples, image_paths, image_names, labels,
                features, indexer, annoy_indexer, top_k=args.top_k, show=args.show
            )
            logging.info(f"自定义LSH平均命中率: {100 * custom_hit_rate:.2f}%")
            logging.info(f"Annoy LSH平均命中率: {100 * annoy_hit_rate:.2f}%")
            # if args.show:
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 自定义LSH平均命中率: {100 * custom_hit_rate:.2f}%")
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Annoy LSH平均命中率: {100 * annoy_hit_rate:.2f}%")
        except Exception as e:
            if args.show:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] 批量评估失败!!")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] {e}")
            logging.info("[ERROR] 批量评估失败!!")
            logging.error(e)

    # 首先处理参数指定的查询图片（如果有）
    if args.query_image and not args.interactive:
        process_query(args.query_image, args.top_k, extractor, indexer, annoy_indexer,
                      image_paths, image_names, labels, args.show)

    # 进入交互式模式
    if args.interactive:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 预处理和索引构建已完成，进入交互式模式")
        logging.info("预处理和索引构建已完成，进入交互式模式")
        interactive_mode(image_paths, image_names, labels, features, indexer,
                         annoy_indexer, extractor, args.query_image, args.show)

    if args.show:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 程序执行完毕")
    logging.info("程序执行完毕")



