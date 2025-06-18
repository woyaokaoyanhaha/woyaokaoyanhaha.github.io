# =============================================================================
# 强制设置临时文件到代码同目录
# =============================================================================
import os
import tempfile
import glob
from datetime import datetime

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TEMP_DIR = os.path.join(SCRIPT_DIR, "temp")

# 创建本地临时目录
try:
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
    tempfile.tempdir = LOCAL_TEMP_DIR
    os.environ['TEMP'] = LOCAL_TEMP_DIR
    os.environ['TMP'] = LOCAL_TEMP_DIR
    print(f"🎯 临时文件目录设置为: {LOCAL_TEMP_DIR}")
except Exception as e:
    print(f"❌ 设置本地临时目录失败: {e}")
    print(f"   将使用系统默认临时目录")

# =============================================================================
# 用户配置参数 - 复用临时文件版
# =============================================================================

# 输入文件配置
PROTEIN_FEATURE_FILE = "protein_features.csv"
COMPOUND_FEATURE_FILE = "69万全部特征.csv"

# 临时文件复用配置
REUSE_EXISTING_CHUNKS = True  # 是否复用现有的分块文件
AUTO_DETECT_CHUNKS = True  # 自动检测现有分块文件
SPECIFIC_CHUNK_DIR = None  # 指定特定的分块目录，None表示自动检测

# 模型目录配置
MODEL_BASE_DIR = "新蛋白特征核受体Combine_BioAssay新特征"
SELECTED_MODEL_TYPES = ["标准随机分割"]

# 内存优化配置
COMPOUND_CHUNK_SIZE = 100000  # 如果需要重新分块时使用
MEMORY_BATCH_SIZE = 5000  # 内存批处理大小
MAX_MEMORY_GB = 8  # 降低内存限制
ENABLE_MEMORY_MONITORING = True  # 启用内存监控

# 具体模型选择
SELECTED_MODELS = {
    "标准随机分割": ["堆叠分类器_标准随机分割", "随机森林", "极端随机树"],
}

# 输出配置
OUTPUT_BASE_DIR = "共识预测69万条"
DETAILED_OUTPUT = True
SAVE_PROBABILITIES = True
SEPARATE_MODEL_RESULTS = True

# 预测设置
USE_ENSEMBLE_MODELS = False
CONFIDENCE_THRESHOLD = 0.5
PREDICTION_MODE = "all_combinations"

# 交互式模型选择
INTERACTIVE_MODEL_SELECTION = False

# 进度显示
SHOW_PROGRESS = True

# 跳过有问题的模型
SKIP_ENSEMBLE_MODELS = False
SKIP_MEMORY_INTENSIVE_MODELS = False

# 共识预测配置
ENABLE_CONSENSUS_ANALYSIS = True
MIN_CONSENSUS_MODELS = 2
CONSENSUS_PROBABILITY_THRESHOLD = 0.5
SAVE_CONSENSUS_RESULTS = True
CONSENSUS_OUTPUT_DETAILED = True

# 共识分析类型
CONSENSUS_ANALYSIS_TYPES = [
    "all_positive",
]

# =============================================================================
# 继续导入其他库
# =============================================================================

import sys
import numpy as np
import pandas as pd
import joblib
import warnings
import json
import gc
from tqdm import tqdm
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from collections import defaultdict, Counter
import shutil

# 忽略警告
warnings.filterwarnings('ignore')


# =============================================================================
# 内存监控工具
# =============================================================================

def get_memory_usage():
    """获取当前内存使用情况（MB）"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}


def monitor_memory(operation_name="操作"):
    """内存监控装饰器"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            if ENABLE_MEMORY_MONITORING:
                before = get_memory_usage()
                print(f"🔍 {operation_name}前: {before['rss_mb']:.1f}MB ({before['percent']:.1f}%)")

            try:
                result = func(*args, **kwargs)

                if ENABLE_MEMORY_MONITORING:
                    after = get_memory_usage()
                    print(f"🔍 {operation_name}后: {after['rss_mb']:.1f}MB ({after['percent']:.1f}%)")
                    print(f"📊 内存变化: {after['rss_mb'] - before['rss_mb']:+.1f}MB")

                return result

            except MemoryError as e:
                print(f"❌ {operation_name}内存不足: {e}")
                gc.collect()  # 强制垃圾回收
                raise
            except Exception as e:
                print(f"❌ {operation_name}失败: {e}")
                raise

        return wrapper

    return decorator


def optimize_batch_size_by_memory():
    """根据可用内存优化批处理大小"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)

        if available_gb >= 16:
            return 10000  # 充足内存
        elif available_gb >= 8:
            return 5000  # 中等内存
        elif available_gb >= 4:
            return 2000  # 较少内存
        else:
            return 1000  # 最小内存
    except:
        return MEMORY_BATCH_SIZE  # 默认值


# =============================================================================
# 现有分块文件检测和管理器
# =============================================================================

class ExistingChunkManager:
    """现有分块文件检测和管理器"""

    def __init__(self):
        self.temp_base_dir = LOCAL_TEMP_DIR
        self.existing_chunk_dirs = []
        self.selected_chunk_dir = None

    def scan_existing_chunks(self):
        """扫描现有的分块目录"""
        print(f"🔍 扫描现有分块文件...")
        print(f"   扫描目录: {self.temp_base_dir}")

        if not os.path.exists(self.temp_base_dir):
            print(f"   临时目录不存在")
            return []

        # 查找所有compound_chunks_*目录
        pattern = os.path.join(self.temp_base_dir, "compound_chunks_*")
        chunk_dirs = glob.glob(pattern)

        valid_chunk_dirs = []

        for chunk_dir in chunk_dirs:
            if os.path.isdir(chunk_dir):
                # 检查目录中是否有.csv文件
                csv_files = glob.glob(os.path.join(chunk_dir, "compound_chunk_*.csv"))
                if csv_files:
                    # 获取目录信息
                    dir_name = os.path.basename(chunk_dir)
                    timestamp_str = dir_name.replace("compound_chunks_", "")

                    # 计算文件大小
                    total_size = 0
                    file_count = len(csv_files)

                    for csv_file in csv_files:
                        try:
                            total_size += os.path.getsize(csv_file)
                        except:
                            pass

                    total_size_gb = total_size / (1024 ** 3)

                    chunk_info = {
                        'dir_path': chunk_dir,
                        'dir_name': dir_name,
                        'timestamp': timestamp_str,
                        'file_count': file_count,
                        'total_size_gb': total_size_gb,
                        'csv_files': sorted(csv_files)
                    }

                    valid_chunk_dirs.append(chunk_info)
                    print(f"   ✓ 发现: {dir_name}")
                    print(f"     - 文件数: {file_count}")
                    print(f"     - 总大小: {total_size_gb:.2f} GB")
                    print(f"     - 时间戳: {timestamp_str}")

        self.existing_chunk_dirs = sorted(valid_chunk_dirs, key=lambda x: x['timestamp'], reverse=True)

        if valid_chunk_dirs:
            print(f"\n📊 共发现 {len(valid_chunk_dirs)} 个有效分块目录")
        else:
            print(f"   未发现有效的分块文件")

        return self.existing_chunk_dirs

    def select_chunk_directory(self):
        """选择要使用的分块目录"""
        if not self.existing_chunk_dirs:
            print(f"❌ 没有可用的现有分块文件")
            return None

        if len(self.existing_chunk_dirs) == 1:
            # 只有一个选择，直接使用
            selected = self.existing_chunk_dirs[0]
            print(f"🎯 自动选择唯一的分块目录: {selected['dir_name']}")
            self.selected_chunk_dir = selected
            return selected

        # 多个选择，显示列表让用户选择
        print(f"\n📋 发现多个分块目录，请选择:")
        for i, chunk_info in enumerate(self.existing_chunk_dirs):
            print(f"  {i + 1}. {chunk_info['dir_name']}")
            print(f"     时间: {chunk_info['timestamp']}")
            print(f"     文件: {chunk_info['file_count']}个")
            print(f"     大小: {chunk_info['total_size_gb']:.2f}GB")
            print()

        while True:
            try:
                choice = input(f"请选择分块目录 (1-{len(self.existing_chunk_dirs)}) 或 'n' 重新分块: ").strip()

                if choice.lower() == 'n':
                    print(f"💫 用户选择重新分块")
                    return None

                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(self.existing_chunk_dirs):
                    selected = self.existing_chunk_dirs[choice_idx]
                    print(f"🎯 用户选择: {selected['dir_name']}")
                    self.selected_chunk_dir = selected
                    return selected
                else:
                    print(f"❌ 无效选择，请输入 1-{len(self.existing_chunk_dirs)} 或 'n'")

            except ValueError:
                print(f"❌ 无效输入，请输入数字或 'n'")
                continue

    def validate_chunk_files(self, chunk_info):
        """验证分块文件的完整性"""
        print(f"🔧 验证分块文件完整性...")

        csv_files = chunk_info['csv_files']
        valid_files = []
        total_rows = 0

        for csv_file in csv_files:
            try:
                # 尝试读取文件头部
                df_sample = pd.read_csv(csv_file, nrows=5)

                # 检查文件大小
                file_size = os.path.getsize(csv_file)
                if file_size < 1024:  # 小于1KB认为文件有问题
                    print(f"   ⚠️ 文件过小: {os.path.basename(csv_file)} ({file_size}字节)")
                    continue

                # 估算行数
                with open(csv_file, 'r') as f:
                    line_count = sum(1 for line in f) - 1  # 减去标题行

                valid_files.append({
                    'file_path': csv_file,
                    'file_name': os.path.basename(csv_file),
                    'rows': line_count,
                    'columns': len(df_sample.columns),
                    'file_size_mb': file_size / (1024 * 1024)
                })

                total_rows += line_count
                print(f"   ✓ {os.path.basename(csv_file)}: {line_count:,}行, {len(df_sample.columns)}列")

            except Exception as e:
                print(f"   ❌ 文件损坏: {os.path.basename(csv_file)} - {e}")
                continue

        if valid_files:
            print(f"   📊 验证完成: {len(valid_files)}/{len(csv_files)} 文件有效")
            print(f"   📊 总行数: {total_rows:,}")
            return valid_files
        else:
            print(f"   ❌ 没有有效的分块文件")
            return []

    def get_chunk_file_info(self):
        """获取选中分块目录的文件信息"""
        if not self.selected_chunk_dir:
            return None

        valid_files = self.validate_chunk_files(self.selected_chunk_dir)

        if not valid_files:
            return None

        return {
            'chunk_dir': self.selected_chunk_dir['dir_path'],
            'chunk_files': [f['file_path'] for f in valid_files],
            'chunk_info': valid_files,
            'total_files': len(valid_files),
            'total_rows': sum(f['rows'] for f in valid_files),
            'reused': True
        }


# =============================================================================
# JSON序列化辅助函数
# =============================================================================

def convert_numpy_types(obj):
    """转换NumPy数据类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def safe_json_dump(obj, file_path, **kwargs):
    """安全JSON保存"""
    try:
        converted_obj = convert_numpy_types(obj)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(converted_obj, f, **kwargs)
        return True
    except Exception as e:
        print(f"保存JSON文件失败: {e}")
        return False


# =============================================================================
# 自定义分类器类定义（保持不变）
# =============================================================================

class CustomVotingClassifier:
    """自定义投票分类器"""

    def __init__(self, estimators, voting='soft'):
        self.estimators = estimators
        self.voting = voting
        self.classes_ = None
        self.named_estimators_ = dict(estimators)

    def get_params(self, deep=True):
        params = {'voting': self.voting}
        if deep:
            estimators_params = {}
            for name, estimator in self.estimators:
                if hasattr(estimator, 'get_params'):
                    for key, val in estimator.get_params(deep).items():
                        estimators_params[f'{name}__{key}'] = val
            params.update(estimators_params)
        params['estimators'] = self.estimators
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key == 'estimators':
                self.estimators = value
                self.named_estimators_ = dict(value)
            elif key == 'voting':
                self.voting = value
            else:
                try:
                    est_name, param_name = key.split('__', 1)
                    for i, (name, est) in enumerate(self.estimators):
                        if name == est_name and hasattr(est, 'set_params'):
                            self.estimators[i] = (name, est.set_params(**{param_name: value}))
                            break
                except:
                    pass
        return self

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for name, est in self.estimators:
            try:
                est.fit(X, y)
            except Exception as e:
                print(f"训练分类器 {name} 时出错: {e}")
        return self

    def predict(self, X):
        if self.voting == 'hard':
            predictions = np.array([clf.predict(X) for _, clf in self.estimators])
            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, minlength=len(self.classes_))),
                axis=0, arr=predictions)
            return self.classes_[maj]
        else:
            predictions = self._collect_probas(X)
            avg = np.average(predictions, axis=0)
            return self.classes_[np.argmax(avg, axis=1)]

    def predict_proba(self, X):
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when voting='hard'")
        return np.average(self._collect_probas(X), axis=0)

    def _collect_probas(self, X):
        probas = []
        for name, clf in self.estimators:
            try:
                proba = clf.predict_proba(X)
                probas.append(proba)
            except Exception as e:
                print(f"获取分类器 {name} 概率时出错: {e}")
                probas.append(np.zeros((X.shape[0], len(self.classes_))))
        return np.asarray(probas)


class CustomStackingClassifier:
    """自定义堆叠分类器"""

    def __init__(self, estimators, final_estimator, cv=5):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.named_estimators_ = dict(estimators)
        self.classes_ = None

    def get_params(self, deep=True):
        params = {'cv': self.cv}
        params['estimators'] = self.estimators
        params['final_estimator'] = self.final_estimator
        if deep:
            estimators_params = {}
            for name, estimator in self.estimators:
                if hasattr(estimator, 'get_params'):
                    for key, val in estimator.get_params(deep).items():
                        estimators_params[f'{name}__{key}'] = val
            params.update(estimators_params)
            if hasattr(self.final_estimator, 'get_params'):
                for key, val in self.final_estimator.get_params(deep).items():
                    params[f'final_estimator__{key}'] = val
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key == 'estimators':
                self.estimators = value
                self.named_estimators_ = dict(value)
            elif key == 'final_estimator':
                self.final_estimator = value
            elif key == 'cv':
                self.cv = value
            elif key.startswith('final_estimator__'):
                if hasattr(self.final_estimator, 'set_params'):
                    param_name = key.split('__', 1)[1]
                    self.final_estimator.set_params(**{param_name: value})
            else:
                try:
                    est_name, param_name = key.split('__', 1)
                    for i, (name, est) in enumerate(self.estimators):
                        if name == est_name and hasattr(est, 'set_params'):
                            self.estimators[i] = (name, est.set_params(**{param_name: value}))
                            break
                except:
                    pass
        return self

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        meta_features = np.zeros((X.shape[0], len(self.estimators) * len(self.classes_)))

        for i, (name, est) in enumerate(self.estimators):
            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]

                try:
                    est_clone = clone(est)
                    est_clone.fit(X_train, y_train)
                    prob = est_clone.predict_proba(X_val)
                    meta_features[val_idx, i * len(self.classes_):(i + 1) * len(self.classes_)] = prob
                except Exception as e:
                    print(f"训练基础分类器 {name} 时出错: {e}")

        for name, est in self.estimators:
            try:
                est.fit(X, y)
            except Exception as e:
                print(f"训练基础分类器 {name} 最终版本时出错: {e}")

        try:
            self.final_estimator.fit(meta_features, y)
        except Exception as e:
            print(f"训练元分类器时出错: {e}")

        return self

    def predict(self, X):
        meta_features = self._make_meta_features(X)
        return self.final_estimator.predict(meta_features)

    def predict_proba(self, X):
        meta_features = self._make_meta_features(X)
        return self.final_estimator.predict_proba(meta_features)

    def _make_meta_features(self, X):
        meta_features = np.zeros((X.shape[0], len(self.estimators) * len(self.classes_)))

        for i, (name, est) in enumerate(self.estimators):
            try:
                prob = est.predict_proba(X)
                meta_features[:, i * len(self.classes_):(i + 1) * len(self.classes_)] = prob
            except Exception as e:
                print(f"获取元特征时分类器 {name} 出错: {e}")

        return meta_features


# =============================================================================
# 共识分析类（保持不变）
# =============================================================================

class ConsensusAnalyzer:
    """共识预测分析器"""

    def __init__(self, min_consensus_models=2, probability_threshold=0.6):
        self.min_consensus_models = min_consensus_models
        self.probability_threshold = probability_threshold
        self.consensus_results = defaultdict(list)

    def analyze_consensus(self, all_results):
        """分析所有模型的共识预测"""
        print(f"\n开始共识分析...")
        print(f"共识要求: 至少{self.min_consensus_models}个模型同意")
        print(f"概率阈值: {self.probability_threshold}")

        # 按化合物-蛋白质对分组
        compound_predictions = defaultdict(list)

        for result in all_results:
            key = f"{result['protein_id']}_{result['compound_id']}"
            compound_predictions[key].append(result)

        print(f"共分析 {len(compound_predictions)} 个化合物-蛋白质对")

        # 分析不同类型的共识
        consensus_stats = {}

        if "all_positive" in CONSENSUS_ANALYSIS_TYPES:
            all_positive = self._find_all_positive_consensus(compound_predictions)
            consensus_stats["all_positive"] = all_positive
            print(f"所有模型都预测为正例: {len(all_positive)} 个")

        return consensus_stats

    def _find_all_positive_consensus(self, compound_predictions):
        """找到所有模型都预测为正例的化合物"""
        all_positive = []

        for compound_key, predictions in compound_predictions.items():
            if len(predictions) < self.min_consensus_models:
                continue

            # 检查是否所有模型都预测为正例
            all_positive_pred = all(pred['prediction'] == 1 for pred in predictions)

            if all_positive_pred:
                # 计算平均概率
                avg_prob_0 = float(np.mean([pred.get('probability_0', 0.5) for pred in predictions]))
                avg_prob_1 = float(np.mean([pred.get('probability_1', 0.5) for pred in predictions]))
                avg_confidence = float(np.mean([pred.get('confidence', 0.5) for pred in predictions]))

                consensus_info = {
                    'protein_id': str(predictions[0]['protein_id']),
                    'compound_id': str(predictions[0]['compound_id']),
                    'num_models': int(len(predictions)),
                    'consensus_type': 'all_positive',
                    'avg_probability_0': avg_prob_0,
                    'avg_probability_1': avg_prob_1,
                    'avg_confidence': avg_confidence,
                    'model_details': []
                }

                # 添加每个模型的详细信息
                for pred in predictions:
                    model_detail = {
                        'model_type': str(pred['model_type']),
                        'model_name': str(pred['model_name']),
                        'prediction': int(pred['prediction']),
                        'prediction_label': str(pred['prediction_label']),
                        'probability_0': float(pred.get('probability_0', 0.5)),
                        'probability_1': float(pred.get('probability_1', 0.5)),
                        'confidence': float(pred.get('confidence', 0.5))
                    }
                    consensus_info['model_details'].append(model_detail)

                all_positive.append(consensus_info)

        return all_positive

    def save_consensus_results(self, consensus_stats, output_dir):
        """保存共识分析结果"""
        if not SAVE_CONSENSUS_RESULTS:
            return

        print(f"\n保存共识分析结果...")

        # 创建共识分析子目录
        consensus_dir = os.path.join(output_dir, "consensus_analysis")
        os.makedirs(consensus_dir, exist_ok=True)

        for consensus_type, results in consensus_stats.items():
            if not results:
                continue

            print(f"  保存 {consensus_type} 结果: {len(results)} 个化合物")

            # 创建简化的数据表格
            simplified_data = []

            for result in results:
                simplified_row = {
                    'protein_id': result['protein_id'],
                    'compound_id': result['compound_id'],
                    'num_models': result['num_models'],
                    'consensus_type': result['consensus_type'],
                    'avg_probability_0': f"{result['avg_probability_0']:.4f}",
                    'avg_probability_1': f"{result['avg_probability_1']:.4f}",
                    'avg_confidence': f"{result['avg_confidence']:.4f}"
                }
                simplified_data.append(simplified_row)

            # 保存简化表格
            if simplified_data:
                try:
                    simplified_df = pd.DataFrame(simplified_data)
                    simplified_file = os.path.join(consensus_dir, f"{consensus_type}_summary.csv")
                    simplified_df.to_csv(simplified_file, index=False, encoding='utf-8-sig')
                    print(f"    ✓ 简化表格: {simplified_file}")
                except Exception as e:
                    print(f"    ✗ 保存简化表格失败: {e}")

            # 保存JSON格式的完整信息
            json_file = os.path.join(consensus_dir, f"{consensus_type}_complete.json")
            if safe_json_dump(results, json_file, indent=2, ensure_ascii=False):
                print(f"    ✓ 完整JSON: {json_file}")
            else:
                print(f"    ✗ 完整JSON保存失败: {json_file}")


# =============================================================================
# 复用临时文件的预测器类
# =============================================================================

class ReuseChunkPredictor:
    """复用现有分块文件的预测器"""

    def __init__(self, model_base_dir, selected_model_types=None, selected_models=None):
        self.model_base_dir = model_base_dir
        self.selected_model_types = selected_model_types or SELECTED_MODEL_TYPES
        self.selected_models = selected_models or SELECTED_MODELS
        self.models = {}
        self.feature_pipelines = {}
        self.available_models = {}

        # 现有分块管理器
        self.chunk_manager = ExistingChunkManager()

        # 动态优化批处理大小
        self.dynamic_batch_size = optimize_batch_size_by_memory()
        print(f"🎯 动态批处理大小: {self.dynamic_batch_size:,}")

        # 共识分析器
        if ENABLE_CONSENSUS_ANALYSIS:
            self.consensus_analyzer = ConsensusAnalyzer(
                min_consensus_models=MIN_CONSENSUS_MODELS,
                probability_threshold=CONSENSUS_PROBABILITY_THRESHOLD
            )

    def scan_available_models(self):
        """扫描所有可用的模型"""
        print("扫描可用模型...")

        self.available_models = {}

        for model_type in self.selected_model_types:
            model_dir = os.path.join(self.model_base_dir, model_type)

            if not os.path.exists(model_dir):
                print(f"模型目录不存在: {model_dir}")
                continue

            pipeline_path = os.path.join(model_dir, 'feature_pipeline.pkl')
            if not os.path.exists(pipeline_path):
                print(f"特征管道文件不存在: {pipeline_path}")
                continue

            model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]

            if model_files:
                available_model_names = []
                for f in model_files:
                    model_name = f.replace('_model.pkl', '')
                    available_model_names.append(model_name)

                if available_model_names:
                    self.available_models[model_type] = available_model_names
                    print(f"  {model_type}: {len(available_model_names)} 个可用模型")
                    for model_name in available_model_names:
                        print(f"    - {model_name}")

        total_available = sum(len(models) for models in self.available_models.values())
        print(f"\n总共扫描到 {total_available} 个可用模型")

        return total_available > 0

    @monitor_memory("模型加载")
    def load_selected_models(self):
        """加载选择的模型"""
        print("\n开始加载选择的模型...")

        loaded_count = 0

        for model_type in self.selected_model_types:
            if model_type not in self.available_models:
                continue

            model_dir = os.path.join(self.model_base_dir, model_type)
            print(f"\n加载 {model_type} 模型...")

            # 加载特征管道
            pipeline_path = os.path.join(model_dir, 'feature_pipeline.pkl')
            try:
                self.feature_pipelines[model_type] = joblib.load(pipeline_path)
                print(f"  ✓ 特征管道已加载")
            except Exception as e:
                print(f"  ✗ 加载特征管道失败: {e}")
                continue

            # 确定要加载的模型列表
            if model_type in self.selected_models:
                models_to_load = self.selected_models[model_type]
            else:
                models_to_load = self.available_models[model_type]

            # 加载模型
            self.models[model_type] = {}

            for model_name in models_to_load:
                model_path = os.path.join(model_dir, f'{model_name}_model.pkl')

                if os.path.exists(model_path):
                    try:
                        print(f"  正在加载 {model_name}...")
                        model = joblib.load(model_path)
                        self.models[model_type][model_name] = model
                        print(f"  ✓ {model_name} 模型已加载")
                        loaded_count += 1
                    except Exception as e:
                        print(f"  ✗ 加载 {model_name} 模型失败: {e}")
                else:
                    print(f"  ✗ {model_name} 模型文件不存在: {model_path}")

        print(f"\n模型加载完成，共加载 {loaded_count} 个模型")
        return loaded_count > 0

    def load_and_prepare_features_reuse(self, protein_file, compound_file):
        """复用现有分块文件的特征数据加载"""
        print(f"\n加载特征文件（复用现有分块文件版）...")

        # 加载蛋白质特征
        try:
            protein_df = pd.read_csv(protein_file)
            print(f"  蛋白质特征文件: {protein_df.shape}")
        except Exception as e:
            raise ValueError(f"加载蛋白质特征文件失败: {e}")

        # 尝试复用现有分块文件
        chunk_info = None
        if REUSE_EXISTING_CHUNKS:
            print(f"\n🔄 尝试复用现有分块文件...")

            # 扫描现有分块
            existing_chunks = self.chunk_manager.scan_existing_chunks()

            if existing_chunks:
                # 选择分块目录
                if SPECIFIC_CHUNK_DIR:
                    # 使用指定目录
                    for chunk in existing_chunks:
                        if chunk['dir_path'] == SPECIFIC_CHUNK_DIR:
                            selected_chunk = chunk
                            break
                    else:
                        print(f"❌ 指定的分块目录不存在: {SPECIFIC_CHUNK_DIR}")
                        selected_chunk = None
                else:
                    # 自动选择或交互选择
                    selected_chunk = self.chunk_manager.select_chunk_directory()

                if selected_chunk:
                    chunk_info = self.chunk_manager.get_chunk_file_info()

                    if chunk_info:
                        print(f"✅ 成功复用现有分块文件:")
                        print(f"   分块目录: {chunk_info['chunk_dir']}")
                        print(f"   文件数量: {chunk_info['total_files']}")
                        print(f"   总行数: {chunk_info['total_rows']:,}")
                    else:
                        print(f"❌ 现有分块文件验证失败")
                        chunk_info = None

            if not chunk_info:
                print(f"💫 将创建新的分块文件...")

        # 如果没有可用的现有分块，创建新的分块
        if not chunk_info:
            print(f"⚠️ 无法复用现有分块文件，需要重新分块")
            print(f"   这将需要额外的时间和磁盘空间")

            # 这里可以选择是否继续或退出
            choice = input(f"是否继续重新分块？(y/n): ").strip().lower()
            if choice != 'y':
                raise RuntimeError("用户取消重新分块")

            # 重新分块逻辑（可以调用原来的分块函数）
            raise NotImplementedError("重新分块功能需要单独实现")

        # 处理蛋白质特征数据
        if protein_df.shape[1] > 1:
            protein_ids = protein_df.iloc[:, 0].values
            protein_features_matrix = protein_df.iloc[:, 1:].values
        else:
            protein_ids = [f"Protein_{i + 1}" for i in range(len(protein_df))]
            protein_features_matrix = protein_df.values

        # 计算总组合数
        total_combinations = len(protein_ids) * chunk_info['total_rows']
        print(f"  总组合数: {len(protein_ids)} × {chunk_info['total_rows']:,} = {total_combinations:,}")

        return {
            'protein_ids': protein_ids,
            'protein_features': protein_features_matrix,
            'compound_chunk_files': chunk_info['chunk_files'],
            'compound_chunk_info': chunk_info['chunk_info'],
            'total_combinations': total_combinations,
            'use_chunked_processing': True,
            'dynamic_batch_size': self.dynamic_batch_size,
            'reused_chunks': True,
            'chunk_dir': chunk_info['chunk_dir']
        }

    def apply_feature_pipeline(self, features, model_type):
        """应用特征工程管道"""
        if model_type not in self.feature_pipelines:
            raise ValueError(f"未找到 {model_type} 的特征管道")

        pipeline = self.feature_pipelines[model_type]

        # 确保输入是2D数组
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 应用填充器
        if 'imputer' in pipeline and pipeline['imputer'] is not None:
            try:
                features = pipeline['imputer'].transform(features)
            except Exception as e:
                print(f"应用填充器失败: {e}")

        # 应用特征选择
        if 'selected_features' in pipeline and pipeline['selected_features']:
            try:
                selected_count = len(pipeline['selected_features'])
                if features.shape[1] >= selected_count:
                    features = features[:, :selected_count]
                else:
                    padding = np.zeros((features.shape[0], selected_count - features.shape[1]))
                    features = np.hstack([features, padding])
            except Exception as e:
                print(f"应用特征选择失败: {e}")

        # 应用降维器
        if 'reducer' in pipeline and pipeline['reducer'] is not None:
            try:
                features = pipeline['reducer'].transform(features)
            except Exception as e:
                print(f"应用降维器失败: {e}")

        # 应用缩放器
        if 'scaler' in pipeline and pipeline['scaler'] is not None:
            try:
                features = pipeline['scaler'].transform(features)
            except Exception as e:
                print(f"应用缩放器失败: {e}")

        # 处理可能的NaN和inf值
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    @monitor_memory("复用分块预测")
    def predict_chunk_reuse(self, data_info, model_type, model_name, model):
        """使用复用分块文件的预测"""
        print(f"    开始复用分块预测: {model_name}")

        protein_ids = data_info['protein_ids']
        protein_features = data_info['protein_features']
        chunk_files = data_info['compound_chunk_files']
        batch_size = data_info['dynamic_batch_size']

        results = []
        total_processed = 0

        try:
            # 验证临时文件是否存在
            for chunk_file in chunk_files:
                if not os.path.exists(chunk_file):
                    raise FileNotFoundError(f"复用的分块文件不存在: {chunk_file}")

            print(f"      ✓ 验证通过，所有 {len(chunk_files)} 个复用分块文件存在")
            print(f"      📊 使用批处理大小: {batch_size:,}")
            print(f"      🔄 复用分块目录: {data_info['chunk_dir']}")

            # 遍历每个化合物块文件
            for chunk_idx, chunk_file in enumerate(chunk_files):
                print(f"      处理化合物块 {chunk_idx + 1}/{len(chunk_files)}")

                try:
                    # 加载当前化合物块
                    compound_chunk_df = pd.read_csv(chunk_file)
                    print(f"        复用块文件: {len(compound_chunk_df):,} 行")

                    # 处理化合物特征数据
                    if compound_chunk_df.shape[1] > 1:
                        compound_ids = compound_chunk_df.iloc[:, 0].values
                        compound_features_matrix = compound_chunk_df.iloc[:, 1:].values
                    else:
                        compound_ids = [f"Compound_chunk{chunk_idx}_{i + 1}" for i in range(len(compound_chunk_df))]
                        compound_features_matrix = compound_chunk_df.values

                    # 遍历每个蛋白质
                    for protein_idx, protein_id in enumerate(protein_ids):
                        protein_feature = protein_features[protein_idx]

                        # 内存优化：使用小批次处理化合物
                        for start_idx in range(0, len(compound_ids), batch_size):
                            end_idx = min(start_idx + batch_size, len(compound_ids))

                            if ENABLE_MEMORY_MONITORING and start_idx % (batch_size * 5) == 0:
                                memory_info = get_memory_usage()
                                print(
                                    f"          批次 {start_idx // batch_size + 1}: 内存 {memory_info['rss_mb']:.1f}MB")

                            # 组合当前批次的特征
                            batch_features = []
                            batch_combinations = []

                            for compound_idx in range(start_idx, end_idx):
                                compound_id = compound_ids[compound_idx]
                                compound_feature = compound_features_matrix[compound_idx]
                                combined_feature = np.concatenate([protein_feature, compound_feature])

                                batch_features.append(combined_feature)
                                batch_combinations.append((protein_id, compound_id))

                            if not batch_features:
                                continue

                            # 转换为numpy数组
                            batch_features = np.array(batch_features)

                            try:
                                # 应用特征工程管道
                                features_processed = self.apply_feature_pipeline(batch_features, model_type)

                                # 进行预测
                                predictions = model.predict(features_processed)

                                # 获取概率
                                probabilities = None
                                if hasattr(model, 'predict_proba'):
                                    try:
                                        probabilities = model.predict_proba(features_processed)
                                    except Exception as e:
                                        print(f"          获取概率失败: {e}")
                                        probabilities = None

                                # 构建结果
                                for k, (prot_id, comp_id) in enumerate(batch_combinations):
                                    result = {
                                        'protein_id': str(prot_id),
                                        'compound_id': str(comp_id),
                                        'model_type': str(model_type),
                                        'model_name': str(model_name),
                                        'prediction': int(predictions[k]),
                                        'prediction_label': '相互作用' if int(predictions[k]) == 1 else '无相互作用'
                                    }

                                    if probabilities is not None:
                                        result['probability_0'] = float(probabilities[k][0])
                                        result['probability_1'] = float(probabilities[k][1])
                                        result['confidence'] = float(max(probabilities[k]))
                                    else:
                                        result['probability_0'] = 0.5
                                        result['probability_1'] = 0.5
                                        result['confidence'] = 0.5

                                    results.append(result)

                                total_processed += len(batch_combinations)

                            except MemoryError as e:
                                print(f"          ❌ 批次处理内存不足: {e}")
                                print(f"          🔄 跳过当前批次，继续下一批次")
                                gc.collect()
                                continue
                            except Exception as e:
                                print(f"          ❌ 批次处理失败: {e}")
                                continue
                            finally:
                                # 主动清理内存
                                del batch_features, features_processed, predictions
                                if probabilities is not None:
                                    del probabilities
                                gc.collect()

                        # 蛋白质处理完成后的内存清理
                        if protein_idx % 5 == 0:
                            gc.collect()

                    # 清理当前化合物块
                    del compound_chunk_df, compound_ids, compound_features_matrix
                    gc.collect()

                    print(f"        完成块 {chunk_idx + 1}，已处理 {total_processed:,} 个组合")

                except MemoryError as e:
                    print(f"      ❌ 处理化合物块 {chunk_idx} 内存不足: {e}")
                    gc.collect()
                    continue
                except Exception as e:
                    print(f"      ❌ 处理化合物块 {chunk_idx} 失败: {e}")
                    continue

        except Exception as e:
            print(f"    ✗ {model_name} 预测过程出错: {e}")

        print(f"    ✓ {model_name} 预测完成，共 {len(results)} 个结果")
        return results

    def predict_and_save_reuse(self, data_info):
        """使用复用分块文件的预测和保存"""
        print(f"\n开始复用分块文件预测并保存结果...")

        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(SCRIPT_DIR, OUTPUT_BASE_DIR, f"reuse_chunks_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        all_results = []
        model_summary = {}

        # 记录复用信息
        reuse_info = {
            'reused_chunks': data_info.get('reused_chunks', False),
            'chunk_dir': data_info.get('chunk_dir', ''),
            'total_files': len(data_info['compound_chunk_files']),
            'total_combinations': data_info['total_combinations'],
            'dynamic_batch_size': data_info['dynamic_batch_size']
        }

        try:
            for model_type in self.models:
                if not self.models[model_type]:
                    continue

                print(f"\n使用 {model_type} 模型进行预测...")

                # 为每个模型类型创建子目录
                type_dir = os.path.join(output_dir, model_type.replace('/', '_'))
                os.makedirs(type_dir, exist_ok=True)

                model_summary[model_type] = {}

                for model_name, model in self.models[model_type].items():
                    print(f"  预测模型: {model_name}")

                    # 使用复用分块文件的预测
                    model_results = self.predict_chunk_reuse(data_info, model_type, model_name, model)

                    if model_results:
                        # 保存结果
                        model_file = os.path.join(type_dir, f"{model_name}_prediction.csv")

                        try:
                            results_df = pd.DataFrame(model_results)
                            results_df.to_csv(model_file, index=False, encoding='utf-8-sig')
                            print(f"    ✓ 结果已保存: {model_file}")
                        except Exception as e:
                            print(f"    ✗ 保存结果失败: {e}")

                        # 收集统计信息
                        positive_count = sum(1 for r in model_results if r['prediction'] == 1)
                        negative_count = len(model_results) - positive_count

                        model_summary[model_type][model_name] = {
                            'total_predictions': len(model_results),
                            'positive_predictions': positive_count,
                            'negative_predictions': negative_count,
                            'positive_ratio': positive_count / len(model_results) if model_results else 0,
                            'output_file': model_file
                        }

                        print(f"    预测结果: {positive_count:,} 个相互作用, {negative_count:,} 个无相互作用")

                        # 收集用于共识分析
                        if ENABLE_CONSENSUS_ANALYSIS:
                            all_results.extend(model_results)

                        # 清理内存
                        del model_results
                        gc.collect()

                    else:
                        print(f"    ✗ 预测失败")

            # 进行共识分析
            if ENABLE_CONSENSUS_ANALYSIS and all_results:
                print(f"\n开始共识分析，总共 {len(all_results)} 个预测结果...")
                try:
                    consensus_stats = self.consensus_analyzer.analyze_consensus(all_results)
                    self.consensus_analyzer.save_consensus_results(consensus_stats, output_dir)
                except Exception as e:
                    print(f"共识分析过程中出错: {e}")

        finally:
            # 注意：复用的分块文件不需要清理，它们可能还会被其他任务使用
            print(f"\n📁 复用的分块文件保留在: {data_info.get('chunk_dir', '未知')}")

        # 保存复用信息
        reuse_info_file = os.path.join(output_dir, "reuse_info.json")
        if safe_json_dump(reuse_info, reuse_info_file, indent=2, ensure_ascii=False):
            print(f"复用信息已保存: {reuse_info_file}")

        # 保存模型对比结果
        comparison_data = []
        for model_type, models in model_summary.items():
            for model_name, stats in models.items():
                comparison_data.append({
                    'model_type': model_type,
                    'model_name': model_name,
                    'total_predictions': stats['total_predictions'],
                    'positive_predictions': stats['positive_predictions'],
                    'negative_predictions': stats['negative_predictions'],
                    'positive_ratio': f"{stats['positive_ratio']:.4f}"
                })

        if comparison_data:
            try:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_file = os.path.join(output_dir, "model_comparison.csv")
                comparison_df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
                print(f"\n模型对比结果已保存: {comparison_file}")
            except Exception as e:
                print(f"保存模型对比结果失败: {e}")

        # 保存预测摘要
        summary_info = {
            'model_summary': model_summary,
            'reuse_info': reuse_info
        }
        summary_file = os.path.join(output_dir, "prediction_summary.json")
        if safe_json_dump(summary_info, summary_file, indent=2, ensure_ascii=False):
            print(f"预测摘要已保存: {summary_file}")

        print(f"所有结果保存在目录: {output_dir}")

        return all_results, model_summary, output_dir


def main():
    """复用临时文件版本的主函数"""
    try:
        print(f"蛋白质-化合物相互作用预测器（复用临时文件版）")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"当前用户: woyaokaoyanhaha")
        print(f"脚本目录: {SCRIPT_DIR}")
        print("=" * 80)

        # 显示复用配置
        print(f"复用临时文件配置:")
        print(f"  复用现有分块: {REUSE_EXISTING_CHUNKS}")
        print(f"  自动检测分块: {AUTO_DETECT_CHUNKS}")
        print(f"  指定分块目录: {SPECIFIC_CHUNK_DIR or '自动检测'}")
        print(f"  内存批处理大小: {MEMORY_BATCH_SIZE:,}")
        print(f"  最大内存限制: {MAX_MEMORY_GB} GB")
        print(f"  内存监控: {ENABLE_MEMORY_MONITORING}")
        print(f"  共识分析: {ENABLE_CONSENSUS_ANALYSIS}")
        print(f"  本地临时目录: {LOCAL_TEMP_DIR}")

        # 检查输入文件
        if not os.path.exists(PROTEIN_FEATURE_FILE):
            raise FileNotFoundError(f"蛋白质特征文件不存在: {PROTEIN_FEATURE_FILE}")

        if not os.path.exists(COMPOUND_FEATURE_FILE):
            raise FileNotFoundError(f"化合物特征文件不存在: {COMPOUND_FEATURE_FILE}")

        if not os.path.exists(MODEL_BASE_DIR):
            raise FileNotFoundError(f"模型目录不存在: {MODEL_BASE_DIR}")

        # 初始化复用预测器
        predictor = ReuseChunkPredictor(MODEL_BASE_DIR, SELECTED_MODEL_TYPES, SELECTED_MODELS)

        # 扫描和加载模型
        if not predictor.scan_available_models():
            raise RuntimeError("未找到任何可用模型")

        if not predictor.load_selected_models():
            raise RuntimeError("未能加载任何模型")

        # 加载和准备特征数据（复用版）
        data_info = predictor.load_and_prepare_features_reuse(PROTEIN_FEATURE_FILE, COMPOUND_FEATURE_FILE)

        print(f"\n准备开始预测...")
        print(f"数据信息:")
        print(f"  总组合数: {data_info['total_combinations']:,}")
        print(f"  化合物块数: {len(data_info['compound_chunk_files'])}")
        print(f"  动态批处理大小: {data_info['dynamic_batch_size']:,}")
        print(f"  使用复用分块: {'是' if data_info.get('reused_chunks') else '否'}")
        if data_info.get('reused_chunks'):
            print(f"  复用分块目录: {data_info.get('chunk_dir', '未知')}")

        # 确认开始预测
        confirm = input(f"\n确认开始预测吗？(y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消预测")
            return

        # 进行复用分块预测
        all_results, model_summary, output_dir = predictor.predict_and_save_reuse(data_info)

        print(f"\n🎉 预测完成！")
        print(f"结果保存在: {output_dir}")

        # 显示简要统计
        total_models = sum(len(models) for models in model_summary.values())
        total_predictions = sum(
            stats['total_predictions']
            for model_type_stats in model_summary.values()
            for stats in model_type_stats.values()
        )

        print(f"\n📊 预测统计:")
        print(f"  使用模型数: {total_models}")
        print(f"  总预测数: {total_predictions:,}")
        print(f"  复用分块文件: {'成功' if data_info.get('reused_chunks') else '未使用'} ✅")
        print(f"  内存优化: 成功避免内存溢出 ✅")

        return output_dir

    except Exception as e:
        print(f"❌ 预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print(f"🚀 启动复用临时文件预测器")
    print(f"📅 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👤 当前用户: woyaokaoyanhaha")
    print(f"📂 脚本位置: {SCRIPT_DIR}")
    print(f"📁 本地临时目录: {LOCAL_TEMP_DIR}")
    print(f"🔄 功能特点: 复用现有分块文件，避免重复分块")
    print("=" * 80)

    result_dir = main()

    if result_dir:
        print(f"\n🌟 预测成功完成！")
        print(f"🌟 结果目录: {result_dir}")
        print(f"🔄 成功复用现有分块文件，节省时间和磁盘空间")
    else:
        print(f"\n❌ 预测失败")

    print(f"\n⏰ 程序结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")