#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BBB肽预测系统 - 基于ESM2特征提取和多种机器学习方法（特征独立存储版）
支持data文件夹统一管理和ESM2特征文件独立存储
Author: AI Assistant
Date: 2025-01-27
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
import joblib
import json
import hashlib
from datetime import datetime
from collections import Counter
import multiprocessing as mp
from Bio import SeqIO
import torch
import torch.nn.functional as F

# ESM2模型相关
try:
    import esm
    ESM_AVAILABLE = True
    print("✅ ESM模块加载成功")
except ImportError:
    ESM_AVAILABLE = False
    print("⚠️ ESM模块未安装，请使用: pip install fair-esm")

# 机器学习相关
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# 模型算法
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier

# 参数分布
from scipy.stats import randint, uniform

# 可视化
try:
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.font_manager import FontProperties
    import seaborn as sns
    VISUALIZATION_ENABLED = True
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.family'] = ['SimHei'] if os.name == 'nt' else ['Arial Unicode MS']
    
except ImportError:
    print("⚠️ matplotlib未安装，将跳过可视化功能")
    VISUALIZATION_ENABLED = False

# 高级模型库
try:
    import xgboost as xgb
    xgb_installed = True
except ImportError:
    print("⚠️ XGBoost未安装")
    xgb_installed = False

try:
    import lightgbm as lgb
    lgbm_installed = True
except ImportError:
    print("⚠️ LightGBM未安装")
    lgbm_installed = False

try:
    import catboost as cb
    catboost_installed = True
except ImportError:
    print("⚠️ CatBoost未安装")
    catboost_installed = False

# 忽略警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# 配置参数
# =============================================================================

# 📁 文件路径配置 - 统一放在data文件夹
DATA_DIR = "data"
POS_TEST_FILE = os.path.join(DATA_DIR, "pos_test.fasta")
TRAIN_POS_ORG_FILE = os.path.join(DATA_DIR, "train_pos_org.fasta")
NEG_TRAIN_FILE = os.path.join(DATA_DIR, "neg_train.fasta")
NEG_TEST_FILE = os.path.join(DATA_DIR, "neg_test.fasta")

# 伪正样本文件配置 - 支持多种格式
PSEUDO_POS_FILES = [
    os.path.join(DATA_DIR, "df_all_final_1000.csv"),  # CSV格式
    os.path.join(DATA_DIR, "pseudo_positive.fasta"),   # FASTA格式（备选）
]

# 🗂️ ESM2特征存储目录配置
FEATURES_DIR = "esm2_features"
TRAIN_FEATURES_DIR = os.path.join(FEATURES_DIR, "train")
TEST_FEATURES_DIR = os.path.join(FEATURES_DIR, "test")
FEATURE_METADATA_FILE = os.path.join(FEATURES_DIR, "feature_metadata.json")

# 🔮 ESM2配置
ESM2_MODEL_NAME = "esm2_t30_150M_UR50D"  # 可选: esm2_t33_650M_UR50D, esm2_t36_3B_UR50D
ESM2_REPRESENTATION_LAYER = 30  # 对应模型的最后一层
ESM2_BATCH_SIZE = 4
ESM2_MAX_LENGTH = 1022  # ESM2最大序列长度

# 📊 训练配置
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
OPTIMIZATION_TARGET = 'roc_auc'

# 🔍 参数搜索配置
USE_RANDOMIZED_SEARCH = True
RANDOMIZED_SEARCH_ITERATIONS = 50
EXTENDED_SEARCH_ENABLED = True

# 📈 可视化配置
PLOT_DPI = 300
SAVE_PLOTS = True

# ⚡ 性能配置
CPU_COUNT = min(4, mp.cpu_count())

# 🎨 输出配置
OUTPUT_DIR = "bbb_peptide_results"

# =============================================================================
# 特征目录管理
# =============================================================================

def create_feature_directories():
    """创建特征存储目录结构"""
    directories = [FEATURES_DIR, TRAIN_FEATURES_DIR, TEST_FEATURES_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 创建/确认目录: {directory}")

def generate_feature_hash(sequences, dataset_type):
    """为序列集合生成唯一的哈希标识"""
    sequences_str = ''.join(sequences) + dataset_type
    hash_obj = hashlib.md5(sequences_str.encode())
    return hash_obj.hexdigest()[:12]

def get_feature_filename(dataset_type, sequences, model_name=ESM2_MODEL_NAME):
    """生成特征文件名"""
    hash_id = generate_feature_hash(sequences, dataset_type)
    return f"{dataset_type}_{model_name}_{len(sequences)}seq_{hash_id}.npy"

def save_feature_metadata(metadata):
    """保存特征元数据"""
    with open(FEATURE_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    print(f"💾 特征元数据已保存: {FEATURE_METADATA_FILE}")

def load_feature_metadata():
    """加载特征元数据"""
    if os.path.exists(FEATURE_METADATA_FILE):
        with open(FEATURE_METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# =============================================================================
# 数据加载器类
# =============================================================================

class BBBDataLoader:
    """BBB肽数据加载器，支持多种文件格式"""
    
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.ensure_data_dir()
    
    def ensure_data_dir(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"📁 创建数据目录: {self.data_dir}")
    
    def read_fasta_file(self, file_path):
        """读取FASTA文件"""
        sequences = []
        sequence_ids = []
        
        try:
            print(f"📖 读取FASTA文件: {file_path}")
            
            if not os.path.exists(file_path):
                print(f"❌ 文件不存在: {file_path}")
                return [], []
            
            with open(file_path, 'r') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    sequences.append(str(record.seq))
                    sequence_ids.append(record.id)
            
            print(f"✅ 读取完成，共 {len(sequences)} 个序列")
            return sequences, sequence_ids
            
        except Exception as e:
            print(f"❌ 读取FASTA文件失败: {e}")
            return [], []
    
    def read_csv_peptides(self, file_path, peptide_column="peptide"):
        """读取CSV文件中的肽段序列"""
        sequences = []
        sequence_ids = []
        
        try:
            print(f"📖 读取CSV文件: {file_path}")
            
            if not os.path.exists(file_path):
                print(f"❌ 文件不存在: {file_path}")
                return [], []
            
            df = pd.read_csv(file_path)
            print(f"📊 CSV文件形状: {df.shape}")
            print(f"📋 可用列名: {list(df.columns)}")
            
            # 智能查找肽段列
            possible_columns = [peptide_column, 'peptide', 'Peptide', 'sequence', 'Sequence', 'seq']
            peptide_col = None
            
            for col in possible_columns:
                if col in df.columns:
                    peptide_col = col
                    break
            
            if peptide_col is None:
                print(f"❌ 未找到肽段列，尝试过的列名: {possible_columns}")
                print(f"📋 实际列名: {list(df.columns)}")
                return [], []
            
            print(f"✅ 使用列名: {peptide_col}")
            
            # 提取序列并过滤空值
            sequences_raw = df[peptide_col].dropna().tolist()
            sequences = [str(seq).strip() for seq in sequences_raw if str(seq).strip()]
            sequence_ids = [f"csv_seq_{i+1}" for i in range(len(sequences))]
            
            print(f"✅ 读取完成，共 {len(sequences)} 个有效序列")
            return sequences, sequence_ids
            
        except Exception as e:
            print(f"❌ 读取CSV文件失败: {e}")
            return [], []
    
    def load_pseudo_positive_samples(self):
        """智能加载伪正样本数据，支持多种文件格式"""
        all_sequences = []
        all_ids = []
        
        print("🔍 搜索伪正样本文件...")
        
        for file_path in PSEUDO_POS_FILES:
            if not os.path.exists(file_path):
                print(f"⚠️ 文件不存在: {file_path}")
                continue
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                sequences, seq_ids = self.read_csv_peptides(file_path)
            elif file_ext in ['.fasta', '.fa']:
                sequences, seq_ids = self.read_fasta_file(file_path)
            else:
                print(f"⚠️ 不支持的文件格式: {file_ext}")
                continue
            
            if sequences:
                all_sequences.extend(sequences)
                all_ids.extend(seq_ids)
                print(f"✅ 从 {os.path.basename(file_path)} 加载 {len(sequences)} 个序列")
                break  # 找到第一个有效文件就停止
        
        if not all_sequences:
            print("❌ 没有找到有效的伪正样本文件")
            # 尝试查找任何包含"pseudo"或数字的文件
            print("🔍 尝试在data目录中查找其他可能的伪正样本文件...")
            for filename in os.listdir(self.data_dir):
                if any(keyword in filename.lower() for keyword in ['pseudo', 'final', '1000']):
                    file_path = os.path.join(self.data_dir, filename)
                    print(f"🔍 尝试文件: {file_path}")
                    
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext == '.csv':
                        sequences, seq_ids = self.read_csv_peptides(file_path)
                    elif file_ext in ['.fasta', '.fa']:
                        sequences, seq_ids = self.read_fasta_file(file_path)
                    else:
                        continue
                    
                    if sequences:
                        all_sequences.extend(sequences)
                        all_ids.extend(seq_ids)
                        print(f"✅ 成功从 {filename} 加载 {len(sequences)} 个序列")
                        break
        
        print(f"📊 伪正样本数据汇总: {len(all_sequences)} 个序列")
        return all_sequences, all_ids
    
    def load_all_training_data(self):
        """加载所有训练数据"""
        print("\n" + "="*60)
        print("📊 加载训练数据")
        print("="*60)
        
        all_sequences = []
        all_labels = []
        all_ids = []
        all_sources = []
        
        # 1. 加载正样本训练数据
        pos_train_sequences, pos_train_ids = self.read_fasta_file(TRAIN_POS_ORG_FILE)
        if pos_train_sequences:
            all_sequences.extend(pos_train_sequences)
            all_labels.extend([1] * len(pos_train_sequences))
            all_ids.extend(pos_train_ids)
            all_sources.extend(['pos_train'] * len(pos_train_sequences))
            print(f"✅ 正样本训练数据: {len(pos_train_sequences)} 个")
        
        # 2. 加载伪正样本数据
        pseudo_pos_sequences, pseudo_pos_ids = self.load_pseudo_positive_samples()
        if pseudo_pos_sequences:
            all_sequences.extend(pseudo_pos_sequences)
            all_labels.extend([1] * len(pseudo_pos_sequences))
            all_ids.extend(pseudo_pos_ids)
            all_sources.extend(['pseudo_pos'] * len(pseudo_pos_sequences))
            print(f"✅ 伪正样本数据: {len(pseudo_pos_sequences)} 个")
        
        # 3. 加载负样本训练数据
        neg_train_sequences, neg_train_ids = self.read_fasta_file(NEG_TRAIN_FILE)
        if neg_train_sequences:
            all_sequences.extend(neg_train_sequences)
            all_labels.extend([0] * len(neg_train_sequences))
            all_ids.extend(neg_train_ids)
            all_sources.extend(['neg_train'] * len(neg_train_sequences))
            print(f"✅ 负样本训练数据: {len(neg_train_sequences)} 个")
        
        # 数据统计
        print(f"\n📊 训练数据统计:")
        print(f"  总序列数: {len(all_sequences)}")
        if all_labels:
            pos_count = sum(all_labels)
            neg_count = len(all_labels) - pos_count
            print(f"  正样本数: {pos_count} ({pos_count/len(all_labels)*100:.1f}%)")
            print(f"  负样本数: {neg_count} ({neg_count/len(all_labels)*100:.1f}%)")
        
        return all_sequences, all_labels, all_ids, all_sources
    
    def load_all_test_data(self):
        """加载所有测试数据"""
        print("\n" + "="*30)
        print("📊 加载测试数据")
        print("="*30)
        
        test_sequences = []
        test_labels = []
        test_ids = []
        test_sources = []
        
        # 1. 加载正样本测试数据
        pos_test_sequences, pos_test_ids = self.read_fasta_file(POS_TEST_FILE)
        if pos_test_sequences:
            test_sequences.extend(pos_test_sequences)
            test_labels.extend([1] * len(pos_test_sequences))
            test_ids.extend(pos_test_ids)
            test_sources.extend(['pos_test'] * len(pos_test_sequences))
            print(f"✅ 正样本测试数据: {len(pos_test_sequences)} 个")
        
        # 2. 加载负样本测试数据
        neg_test_sequences, neg_test_ids = self.read_fasta_file(NEG_TEST_FILE)
        if neg_test_sequences:
            test_sequences.extend(neg_test_sequences)
            test_labels.extend([0] * len(neg_test_sequences))
            test_ids.extend(neg_test_ids)
            test_sources.extend(['neg_test'] * len(neg_test_sequences))
            print(f"✅ 负样本测试数据: {len(neg_test_sequences)} 个")
        
        # 数据统计
        print(f"\n📊 测试数据统计:")
        print(f"  总序列数: {len(test_sequences)}")
        if test_labels:
            pos_count = sum(test_labels)
            neg_count = len(test_labels) - pos_count
            print(f"  正样本数: {pos_count} ({pos_count/len(test_labels)*100:.1f}%)")
            print(f"  负样本数: {neg_count} ({neg_count/len(test_labels)*100:.1f}%)")
        
        return test_sequences, test_labels, test_ids, test_sources

# =============================================================================
# ESM2特征提取器 - 支持特征缓存
# =============================================================================

class ESM2FeatureExtractor:
    """ESM2特征提取器 - 支持特征缓存和独立存储"""
    
    def __init__(self, model_name=ESM2_MODEL_NAME, representation_layer=ESM2_REPRESENTATION_LAYER):
        self.model_name = model_name
        self.representation_layer = representation_layer
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """加载ESM2模型"""
        if not ESM_AVAILABLE:
            raise ImportError("ESM模块未安装，请使用: pip install fair-esm")
        
        print(f"🔬 加载ESM2模型: {self.model_name}")
        print(f"🎯 使用设备: {self.device}")
        
        try:
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.batch_converter = self.alphabet.get_batch_converter()
            print(f"✅ ESM2模型加载成功")
            return True
        except Exception as e:
            print(f"❌ ESM2模型加载失败: {e}")
            return False
    
    def check_cached_features(self, sequences, dataset_type):
        """检查是否存在缓存的特征文件"""
        feature_filename = get_feature_filename(dataset_type, sequences, self.model_name)
        
        if dataset_type == 'train':
            feature_path = os.path.join(TRAIN_FEATURES_DIR, feature_filename)
        else:
            feature_path = os.path.join(TEST_FEATURES_DIR, feature_filename)
            
        if os.path.exists(feature_path):
            print(f"🎯 发现缓存特征文件: {feature_path}")
            try:
                features = np.load(feature_path)
                print(f"✅ 成功加载缓存特征，形状: {features.shape}")
                return features
            except Exception as e:
                print(f"⚠️ 加载缓存特征失败: {e}")
                return None
        
        return None
    
    def save_features(self, features, sequences, dataset_type, labels=None, seq_ids=None):
        """保存特征文件到对应目录"""
        feature_filename = get_feature_filename(dataset_type, sequences, self.model_name)
        
        if dataset_type == 'train':
            feature_path = os.path.join(TRAIN_FEATURES_DIR, feature_filename)
        else:
            feature_path = os.path.join(TEST_FEATURES_DIR, feature_filename)
        
        try:
            # 保存特征文件
            np.save(feature_path, features)
            print(f"💾 特征文件已保存: {feature_path}")
            
            # 更新元数据
            metadata = load_feature_metadata()
            feature_info = {
                'filename': feature_filename,
                'dataset_type': dataset_type,
                'model_name': self.model_name,
                'representation_layer': self.representation_layer,
                'sequence_count': len(sequences),
                'feature_dim': features.shape[1],
                'creation_time': datetime.now().isoformat(),
                'file_size_mb': os.path.getsize(feature_path) / (1024 * 1024)
            }
            
            if labels is not None:
                feature_info['label_distribution'] = {
                    'positive': int(sum(labels)),
                    'negative': int(len(labels) - sum(labels))
                }
            
            metadata[feature_filename] = feature_info
            save_feature_metadata(metadata)
            
            return feature_path
            
        except Exception as e:
            print(f"❌ 保存特征文件失败: {e}")
            return None
    
    def extract_features(self, sequences, dataset_type='unknown', labels=None, seq_ids=None, batch_size=ESM2_BATCH_SIZE):
        """提取ESM2特征，支持缓存机制"""
        print(f"🧬 处理 {dataset_type} 数据集的ESM2特征...")
        
        # 检查缓存
        cached_features = self.check_cached_features(sequences, dataset_type)
        if cached_features is not None:
            return cached_features
        
        # 加载模型
        if self.model is None:
            if not self.load_model():
                return None
        
        print(f"🔬 开始提取 {len(sequences)} 个序列的ESM2特征...")
        print(f"📦 批处理大小: {batch_size}")
        
        all_features = []
        
        try:
            with torch.no_grad():
                for i in range(0, len(sequences), batch_size):
                    batch_sequences = sequences[i:i + batch_size]
                    
                    # 准备批处理数据
                    batch_labels, batch_strs, batch_tokens = self.batch_converter([
                        (f"seq_{j}", seq[:ESM2_MAX_LENGTH]) 
                        for j, seq in enumerate(batch_sequences)
                    ])
                    
                    batch_tokens = batch_tokens.to(self.device)
                    
                    # 前向传播
                    results = self.model(batch_tokens, repr_layers=[self.representation_layer])
                    
                    # 提取特征 (使用序列级别的平均池化)
                    representations = results["representations"][self.representation_layer]
                    
                    # 对每个序列计算平均特征 (排除特殊token)
                    for j, seq in enumerate(batch_sequences):
                        seq_len = min(len(seq), ESM2_MAX_LENGTH)
                        # 取中间部分，排除 [CLS] 和 [SEP] token
                        seq_repr = representations[j, 1:seq_len+1].mean(dim=0)
                        all_features.append(seq_repr.cpu().numpy())
                    
                    if (i // batch_size + 1) % 10 == 0:
                        print(f"  已处理: {min(i + batch_size, len(sequences))}/{len(sequences)}")
            
            features_array = np.array(all_features)
            print(f"✅ 特征提取完成，特征维度: {features_array.shape}")
            
            # 保存特征文件
            self.save_features(features_array, sequences, dataset_type, labels, seq_ids)
            
            return features_array
            
        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            return None

# =============================================================================
# 机器学习模型训练 (保持原有功能)
# =============================================================================

def get_extended_param_grids():
    """获取扩展的参数网格"""
    param_grids = {}
    
    # SVM
    if USE_RANDOMIZED_SEARCH:
        param_grids['SVM'] = {
            'C': uniform(0.01, 99.99),
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'probability': [True]
        }
    else:
        param_grids['SVM'] = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear'],
            'probability': [True]
        }
    
    # 随机森林
    if USE_RANDOMIZED_SEARCH:
        param_grids['随机森林'] = {
            'n_estimators': randint(50, 500),
            'max_depth': [None, 5, 10, 15, 20, 25, 30],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None]
        }
    else:
        param_grids['随机森林'] = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
    
    # 梯度提升
    if USE_RANDOMIZED_SEARCH:
        param_grids['梯度提升'] = {
            'n_estimators': randint(50, 300),
            'learning_rate': uniform(0.01, 0.29),
            'max_depth': randint(3, 12),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        }
    else:
        param_grids['梯度提升'] = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9]
        }
    
    # 逻辑回归
    if USE_RANDOMIZED_SEARCH:
        param_grids['逻辑回归'] = {
            'C': uniform(0.01, 99.99),
            'penalty': ['l1', 'l2', None],
            'solver': ['liblinear', 'lbfgs', 'saga'],
            'max_iter': [1000, 2000, 3000, 4000, 5000]
        }
    else:
        param_grids['逻辑回归'] = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000]
        }
    
    # K近邻
    if USE_RANDOMIZED_SEARCH:
        param_grids['K近邻'] = {
            'n_neighbors': randint(3, 30),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
            'p': [1, 2, 3, 4, 5]
        }
    else:
        param_grids['K近邻'] = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    
    # 极端随机树
    if USE_RANDOMIZED_SEARCH:
        param_grids['极端随机树'] = {
            'n_estimators': randint(50, 500),
            'max_depth': [None, 5, 10, 15, 20, 25, 30],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None]
        }
    else:
        param_grids['极端随机树'] = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    
    # 朴素贝叶斯
    param_grids['朴素贝叶斯'] = {
        'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]
    }
    
    # XGBoost
    if xgb_installed:
        if USE_RANDOMIZED_SEARCH:
            param_grids['XGBoost'] = {
                'n_estimators': randint(50, 500),
                'learning_rate': uniform(0.01, 0.29),
                'max_depth': randint(3, 12),
                'min_child_weight': randint(1, 10),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'reg_alpha': uniform(0, 0.1),
                'reg_lambda': uniform(0, 0.1)
            }
        else:
            param_grids['XGBoost'] = {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9]
            }
    
    # LightGBM
    if lgbm_installed:
        if USE_RANDOMIZED_SEARCH:
            param_grids['LightGBM'] = {
                'n_estimators': randint(50, 500),
                'learning_rate': uniform(0.01, 0.29),
                'max_depth': randint(3, 12),
                'num_leaves': randint(10, 100),
                'min_child_samples': randint(5, 50),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'reg_alpha': uniform(0, 0.1),
                'reg_lambda': uniform(0, 0.1)
            }
        else:
            param_grids['LightGBM'] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [10, 31, 50]
            }
    
    # CatBoost
    if catboost_installed:
        if USE_RANDOMIZED_SEARCH:
            param_grids['CatBoost'] = {
                'iterations': randint(50, 500),
                'learning_rate': uniform(0.01, 0.29),
                'depth': randint(3, 10),
                'l2_leaf_reg': uniform(0.1, 9.9),
                'border_count': [32, 64, 128, 200, 254],
                'verbose': [False]
            }
        else:
            param_grids['CatBoost'] = {
                'iterations': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'depth': [3, 5, 7],
                'verbose': [False]
            }
    
    return param_grids

def get_model_instance(model_name):
    """根据模型名称获取模型实例"""
    if model_name == 'SVM':
        return SVC(random_state=RANDOM_STATE)
    elif model_name == '随机森林':
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == '梯度提升':
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    elif model_name == '逻辑回归':
        return LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    elif model_name == 'K近邻':
        return KNeighborsClassifier(n_jobs=-1)
    elif model_name == '极端随机树':
        return ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == '朴素贝叶斯':
        return GaussianNB()
    elif model_name == 'XGBoost' and xgb_installed:
        return xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1)
    elif model_name == 'LightGBM' and lgbm_installed:
        return lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    elif model_name == 'CatBoost' and catboost_installed:
        return cb.CatBoostClassifier(random_state=RANDOM_STATE, verbose=False, thread_count=-1)
    else:
        raise ValueError(f"不支持的模型: {model_name}")

def train_and_evaluate_model(model_name, param_grid, X_train, y_train, X_test, y_test, output_dir):
    """训练和评估单个模型"""
    try:
        print(f"\n🤖 开始训练: {model_name}")
        
        model = get_model_instance(model_name)
        
        # 参数搜索
        if USE_RANDOMIZED_SEARCH:
            search = RandomizedSearchCV(
                model, param_grid, 
                n_iter=RANDOMIZED_SEARCH_ITERATIONS,
                cv=CV_FOLDS, 
                scoring=OPTIMIZATION_TARGET,
                n_jobs=CPU_COUNT,
                random_state=RANDOM_STATE,
                verbose=0
            )
        else:
            search = GridSearchCV(
                model, param_grid,
                cv=CV_FOLDS,
                scoring=OPTIMIZATION_TARGET,
                n_jobs=CPU_COUNT,
                verbose=0
            )
        
        # 训练
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        
        print(f"✅ {model_name} 参数搜索完成")
        print(f"🎯 最优参数: {best_params}")
        
        # 预测
        y_pred = best_model.predict(X_test)
        
        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        elif hasattr(best_model, 'decision_function'):
            y_pred_proba = best_model.decision_function(X_test)
        else:
            y_pred_proba = y_pred.astype(float)
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'aupr': average_precision_score(y_test, y_pred_proba),
            'best_params': best_params
        }
        
        print(f"📊 {model_name} 性能指标:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        print(f"  AUPR: {metrics['aupr']:.4f}")
        
        # 保存模型
        model_path = os.path.join(output_dir, f'{model_name}_model.pkl')
        joblib.dump(best_model, model_path)
        print(f"💾 模型已保存: {model_path}")
        
        # 生成可视化
        if VISUALIZATION_ENABLED and SAVE_PLOTS:
            generate_model_visualizations(y_test, y_pred, y_pred_proba, model_name, output_dir)
        
        return model_name, best_model, metrics
        
    except Exception as e:
        print(f"❌ 训练 {model_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_model_visualizations(y_test, y_pred, y_pred_proba, model_name, output_dir):
    """生成模型可视化图表"""
    try:
        # 1. 混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['非BBB肽', 'BBB肽'], yticklabels=['非BBB肽', 'BBB肽'])
        plt.title(f'{model_name} - 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'), 
                   dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        # 2. ROC曲线
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率')
        plt.ylabel('真正例率')
        plt.title(f'{model_name} - ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curve.png'), 
                   dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        # 3. PR曲线
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        aupr = average_precision_score(y_test, y_pred_proba)
        plt.plot(recall, precision, label=f'{model_name} (AUPR = {aupr:.3f})', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title(f'{model_name} - PR曲线')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_pr_curve.png'), 
                   dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"📈 {model_name} 可视化图表已保存")
        
    except Exception as e:
        print(f"⚠️ 生成 {model_name} 可视化失败: {e}")

def generate_model_comparison_report(results, output_dir):
    """生成模型对比报告"""
    if not results:
        print("❌ 没有结果可供比较")
        return

    try:
        # 创建对比数据
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                '模型': model_name,
                'AUC': f"{metrics['auc']:.4f}",
                '准确率': f"{metrics['accuracy']:.4f}",
                '精确率': f"{metrics['precision']:.4f}",
                '召回率': f"{metrics['recall']:.4f}",
                'F1分数': f"{metrics['f1']:.4f}",
                'AUPR': f"{metrics['aupr']:.4f}"
            })

        # 按AUC排序
        comparison_data.sort(key=lambda x: float(x['AUC']), reverse=True)
        
        # 保存CSV
        comparison_df = pd.DataFrame(comparison_data)
        csv_path = os.path.join(output_dir, 'model_comparison.csv')
        comparison_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"📋 模型对比表已保存: {csv_path}")
        
        # 保存详细结果
        detailed_results_path = os.path.join(output_dir, 'detailed_results.json')
        with open(detailed_results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"📋 详细结果已保存: {detailed_results_path}")

        # 打印结果
        print("\n" + "="*80)
        print("🏆 BBB肽预测模型性能对比 (按AUC排序)")
        print("="*80)
        print(f"{'模型':<15}{'AUC':<8}{'准确率':<8}{'精确率':<8}{'召回率':<8}{'F1分数':<8}{'AUPR':<8}")
        print("-"*80)
        
        for item in comparison_data:
            print(f"{item['模型']:<15}{item['AUC']:<8}{item['准确率']:<8}{item['精确率']:<8}"
                  f"{item['召回率']:<8}{item['F1分数']:<8}{item['AUPR']:<8}")

        # 输出最佳模型
        best_model_name = comparison_data[0]['模型']
        best_auc = comparison_data[0]['AUC']
        print(f"\n🥇 最佳模型: {best_model_name} (AUC: {best_auc})")
        print("="*80)

    except Exception as e:
        print(f"❌ 生成模型对比报告失败: {e}")

# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    print("🧬 BBB肽预测系统 (特征独立存储版)")
    print("基于ESM2特征提取和多种机器学习方法")
    print("支持data文件夹统一管理和ESM2特征文件独立存储")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 检查ESM可用性
    if not ESM_AVAILABLE:
        print("❌ ESM模块未安装，请先安装:")
        print("pip install fair-esm")
        return False
    
    # 创建输出目录和特征目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    create_feature_directories()
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"📁 特征存储目录: {FEATURES_DIR}")
    
    try:
        # 1. 初始化数据加载器
        data_loader = BBBDataLoader()
        
        # 2. 准备数据
        print("\n" + "="*60)
        print("第一步: 数据准备")
        print("="*60)
        
        train_sequences, train_labels, train_ids, train_sources = data_loader.load_all_training_data()
        test_sequences, test_labels, test_ids, test_sources = data_loader.load_all_test_data()
        
        if not train_sequences or not test_sequences:
            print("❌ 数据加载失败，请检查data文件夹中的文件")
            print("📋 需要的文件:")
            print(f"  - {POS_TEST_FILE}")
            print(f"  - {TRAIN_POS_ORG_FILE}")
            print(f"  - {NEG_TRAIN_FILE}")
            print(f"  - {NEG_TEST_FILE}")
            print(f"  - 伪正样本文件 (CSV或FASTA格式)")
            return False
        
        # 3. ESM2特征提取 (支持缓存)
        print("\n" + "="*60)
        print("第二步: ESM2特征提取 (支持缓存)")
        print("="*60)
        
        extractor = ESM2FeatureExtractor()
        
        # 提取训练集特征
        print("🔬 处理训练集特征...")
        X_train = extractor.extract_features(
            train_sequences, 
            dataset_type='train', 
            labels=train_labels, 
            seq_ids=train_ids
        )
        if X_train is None:
            print("❌ 训练集特征提取失败")
            return False
        
        # 提取测试集特征
        print("🔬 处理测试集特征...")
        X_test = extractor.extract_features(
            test_sequences, 
            dataset_type='test', 
            labels=test_labels, 
            seq_ids=test_ids
        )
        if X_test is None:
            print("❌ 测试集特征提取失败")
            return False
        
        # 保存传统格式的特征文件（兼容性）
        np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
        np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
        np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), train_labels)
        np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), test_labels)
        print(f"💾 兼容性特征数据已保存到 {OUTPUT_DIR}")
        
        # 4. 特征标准化
        print("\n" + "="*30)
        print("第三步: 特征标准化")
        print("="*30)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 保存标准化器
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'feature_scaler.pkl'))
        print("✅ 特征标准化完成")
        
        # 5. 模型训练
        print("\n" + "="*60)
        print("第四步: 多模型训练与评估")
        print("="*60)
        
        param_grids = get_extended_param_grids()
        trained_models = {}
        all_results = {}
        
        # 训练所有模型
        for model_name, param_grid in param_grids.items():
            result = train_and_evaluate_model(
                model_name, param_grid, 
                X_train_scaled, train_labels, 
                X_test_scaled, test_labels, 
                OUTPUT_DIR
            )
            
            if result:
                name, model, metrics = result
                trained_models[name] = model
                all_results[name] = metrics
        
        # 6. 集成学习
        print("\n" + "="*40)
        print("第五步: 集成学习")
        print("="*40)
        
        if len(trained_models) >= 2:
            # 投票分类器
            try:
                print("🤝 训练投票分类器...")
                voting_models = list(trained_models.items())[:5]  # 最多5个模型
                voting_classifier = VotingClassifier(
                    estimators=voting_models, 
                    voting='soft'
                )
                voting_classifier.fit(X_train_scaled, train_labels)
                
                # 手动评估投票分类器
                y_pred = voting_classifier.predict(X_test_scaled)
                y_pred_proba = voting_classifier.predict_proba(X_test_scaled)[:, 1]
                
                voting_metrics = {
                    'accuracy': accuracy_score(test_labels, y_pred),
                    'precision': precision_score(test_labels, y_pred, zero_division=0),
                    'recall': recall_score(test_labels, y_pred, zero_division=0),
                    'f1': f1_score(test_labels, y_pred, zero_division=0),
                    'auc': roc_auc_score(test_labels, y_pred_proba),
                    'aupr': average_precision_score(test_labels, y_pred_proba),
                    'best_params': 'ensemble_voting'
                }
                
                all_results['投票分类器'] = voting_metrics
                joblib.dump(voting_classifier, os.path.join(OUTPUT_DIR, '投票分类器_model.pkl'))
                
                if VISUALIZATION_ENABLED:
                    generate_model_visualizations(test_labels, y_pred, y_pred_proba, '投票分类器', OUTPUT_DIR)
                
                print("✅ 投票分类器训练完成")
                
            except Exception as e:
                print(f"⚠️ 投票分类器训练失败: {e}")
            
            # 堆叠分类器
            if len(trained_models) >= 3:
                try:
                    print("🏗️ 训练堆叠分类器...")
                    base_models = list(trained_models.items())[:4]  # 最多4个基模型
                    meta_model = LogisticRegression(random_state=RANDOM_STATE)
                    stacking_classifier = StackingClassifier(
                        estimators=base_models,
                        final_estimator=meta_model,
                        cv=3
                    )
                    stacking_classifier.fit(X_train_scaled, train_labels)
                    
                    # 手动评估堆叠分类器
                    y_pred = stacking_classifier.predict(X_test_scaled)
                    y_pred_proba = stacking_classifier.predict_proba(X_test_scaled)[:, 1]
                    
                    stacking_metrics = {
                        'accuracy': accuracy_score(test_labels, y_pred),
                        'precision': precision_score(test_labels, y_pred, zero_division=0),
                        'recall': recall_score(test_labels, y_pred, zero_division=0),
                        'f1': f1_score(test_labels, y_pred, zero_division=0),
                        'auc': roc_auc_score(test_labels, y_pred_proba),
                        'aupr': average_precision_score(test_labels, y_pred_proba),
                        'best_params': 'ensemble_stacking'
                    }
                    
                    all_results['堆叠分类器'] = stacking_metrics
                    joblib.dump(stacking_classifier, os.path.join(OUTPUT_DIR, '堆叠分类器_model.pkl'))
                    
                    if VISUALIZATION_ENABLED:
                        generate_model_visualizations(test_labels, y_pred, y_pred_proba, '堆叠分类器', OUTPUT_DIR)
                    
                    print("✅ 堆叠分类器训练完成")
                    
                except Exception as e:
                    print(f"⚠️ 堆叠分类器训练失败: {e}")
        
        # 7. 生成报告
        print("\n" + "="*60)
        print("第六步: 生成结果报告")
        print("="*60)
        
        generate_model_comparison_report(all_results, OUTPUT_DIR)
        
        # 保存数据信息
        data_info = {
            'train_samples': len(train_sequences),
            'test_samples': len(test_sequences),
            'train_positive': sum(train_labels),
            'test_positive': sum(test_labels),
            'feature_dim': X_train.shape[1],
            'esm2_model': ESM2_MODEL_NAME,
            'data_directory': DATA_DIR,
            'features_directory': FEATURES_DIR,
            'training_time': datetime.now().isoformat()
        }
        
        with open(os.path.join(OUTPUT_DIR, 'data_info.json'), 'w', encoding='utf-8') as f:
            json.dump(data_info, f, ensure_ascii=False, indent=2)
        
        # 显示特征存储信息
        print(f"\n📂 ESM2特征文件存储信息:")
        print(f"  特征根目录: {FEATURES_DIR}")
        print(f"  训练特征目录: {TRAIN_FEATURES_DIR}")
        print(f"  测试特征目录: {TEST_FEATURES_DIR}")
        print(f"  特征元数据: {FEATURE_METADATA_FILE}")
        
        # 显示特征文件
        metadata = load_feature_metadata()
        if metadata:
            print(f"\n📋 已保存的特征文件:")
            for filename, info in metadata.items():
                print(f"  🗂️ {filename}")
                print(f"     数据集: {info['dataset_type']}")
                print(f"     序列数: {info['sequence_count']}")
                print(f"     特征维度: {info['feature_dim']}")
                print(f"     文件大小: {info['file_size_mb']:.2f} MB")
                print(f"     创建时间: {info['creation_time'][:19]}")
        
        print(f"\n🎉 BBB肽预测模型训练完成!")
        print(f"📁 所有结果保存在: {OUTPUT_DIR}")
        print(f"🗂️ 特征文件保存在: {FEATURES_DIR}")
        print(f"📊 共训练了 {len(all_results)} 个模型")
        print(f"📂 数据来源: {DATA_DIR} 文件夹")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    
    # 运行主程序
    success = main()
    
    if success:
        print("\n✅ 程序执行成功!")
    else:
        print("\n❌ 程序执行失败!")
    
    # 在Windows下暂停，方便查看结果
    if os.name == 'nt':
        input("\n按回车键退出...")