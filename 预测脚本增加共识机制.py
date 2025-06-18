# =============================================================================
# 用户配置参数 - 在这里修改所有设置
# =============================================================================

# 输入文件配置
PROTEIN_FEATURE_FILE = "protein_features.csv"
COMPOUND_FEATURE_FILE = "69万全部特征.csv"

# 模型目录配置
MODEL_BASE_DIR = "新蛋白特征核受体Combine_BioAssay新特征"
SELECTED_MODEL_TYPES = ["标准随机分割"]

# 批处理配置 - 内存优化关键参数
BATCH_SIZE = 100000  # 减小批处理大小以节省内存
MAX_MEMORY_GB = 16  # 保守的内存限制
USE_MEMORY_EFFICIENT_MODE = True

# 具体模型选择 - 推荐只选择轻量级模型
SELECTED_MODELS = {
    "标准随机分割": ["堆叠分类器_标准随机分割", "随机森林", "极端随机树"],  # 只选择内存效率高的模型
    # "蛋白质冷启动": ["逻辑回归", "朴素贝叶斯"],
    # "药物冷启动": ["逻辑回归", "朴素贝叶斯"],
    # "双重冷启动": ["逻辑回归", "朴素贝叶斯"],
}

# 输出配置
OUTPUT_BASE_DIR = "共识预测69万条"
DETAILED_OUTPUT = True
SAVE_PROBABILITIES = True
SEPARATE_MODEL_RESULTS = True

# 预测设置
USE_ENSEMBLE_MODELS = False  # 大数据量时建议关闭集成模型
CONFIDENCE_THRESHOLD = 0.5
PREDICTION_MODE = "all_combinations"  # 十万条数据应该是多组合模式

# 交互式模型选择
INTERACTIVE_MODEL_SELECTION = False  # 改为False使用上面的SELECTED_MODELS配置

# 进度显示
SHOW_PROGRESS = True

# 跳过有问题的模型
SKIP_ENSEMBLE_MODELS = False  # 跳过堆叠分类器和投票分类器
SKIP_MEMORY_INTENSIVE_MODELS = False  # 跳过SVM、随机森林等内存密集模型

# ============= 新增：共识预测配置 =============
ENABLE_CONSENSUS_ANALYSIS = True  # 启用共识分析
MIN_CONSENSUS_MODELS = 2  # 至少几个模型都预测为正例才算共识（设为2表示至少2个模型同意）
CONSENSUS_PROBABILITY_THRESHOLD = 0.5  # 共识预测的概率阈值
SAVE_CONSENSUS_RESULTS = True  # 保存共识结果
CONSENSUS_OUTPUT_DETAILED = True  # 输出详细的共识信息

# 共识分析类型
CONSENSUS_ANALYSIS_TYPES = [
    "all_positive",  # 所有模型都预测为正例
    #"majority_positive",  # 大多数模型预测为正例
    #"high_confidence_positive"  # 高置信度正例预测
]

# =============================================================================
# 导入所需库
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime
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

# 忽略警告
warnings.filterwarnings('ignore')


# =============================================================================
# 新增：JSON序列化辅助函数
# =============================================================================

def convert_numpy_types(obj):
    """递归转换NumPy数据类型为Python原生类型，用于JSON序列化"""
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
    """安全地将对象保存为JSON，自动处理NumPy类型转换"""
    try:
        converted_obj = convert_numpy_types(obj)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(converted_obj, f, **kwargs)
        return True
    except Exception as e:
        print(f"保存JSON文件失败: {e}")
        return False


# =============================================================================
# 关键修复：自定义分类器类定义（必须在导入前定义）
# =============================================================================

class CustomVotingClassifier:
    """自定义投票分类器 - 与训练代码完全一致"""

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
    """自定义堆叠分类器 - 与训练代码完全一致"""

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
# 内存监控工具
# =============================================================================

def get_memory_usage():
    """获取当前内存使用情况（MB）"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0


def estimate_batch_size(feature_count, max_memory_gb=1):
    """根据特征数量和可用内存估算合适的批处理大小"""
    bytes_per_feature = 8  # float64
    bytes_per_sample = feature_count * bytes_per_feature
    available_memory = max_memory_gb * 1024 * 1024 * 1024 * 0.3  # 30%的内存用于批处理
    estimated_batch_size = int(available_memory / bytes_per_sample)
    estimated_batch_size = max(50, min(estimated_batch_size, 500))  # 限制在合理范围
    return estimated_batch_size


# =============================================================================
# 模型过滤器
# =============================================================================

def should_skip_model(model_name):
    """判断是否应该跳过某个模型"""
    model_name_lower = model_name.lower()

    # 跳过集成模型
    if SKIP_ENSEMBLE_MODELS:
        if any(x in model_name_lower for x in ['堆叠', '投票', 'stacking', 'voting']):
            print(f"    ⏭️  跳过集成模型: {model_name}")
            return True

    # 跳过内存密集模型
    if SKIP_MEMORY_INTENSIVE_MODELS:
        if any(x in model_name_lower for x in ['svm', '随机森林', '梯度提升', 'xgboost', 'lightgbm', 'catboost']):
            print(f"    ⏭️  跳过内存密集模型: {model_name}")
            return True

    return False


# =============================================================================
# 新增：共识分析类（修复版）
# =============================================================================

class ConsensusAnalyzer:
    """共识预测分析器（修复NumPy序列化问题）"""

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

        if "majority_positive" in CONSENSUS_ANALYSIS_TYPES:
            majority_positive = self._find_majority_positive_consensus(compound_predictions)
            consensus_stats["majority_positive"] = majority_positive
            print(f"大多数模型预测为正例: {len(majority_positive)} 个")

        if "high_confidence_positive" in CONSENSUS_ANALYSIS_TYPES:
            high_confidence = self._find_high_confidence_positive(compound_predictions)
            consensus_stats["high_confidence_positive"] = high_confidence
            print(f"高置信度正例预测: {len(high_confidence)} 个")

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
                # 计算平均概率 - 修复：确保转换为Python原生类型
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

                # 添加每个模型的详细信息 - 修复：确保所有值都是Python原生类型
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

    def _find_majority_positive_consensus(self, compound_predictions):
        """找到大多数模型预测为正例的化合物"""
        majority_positive = []

        for compound_key, predictions in compound_predictions.items():
            if len(predictions) < self.min_consensus_models:
                continue

            # 计算正例预测的比例
            positive_count = sum(1 for pred in predictions if pred['prediction'] == 1)
            positive_ratio = positive_count / len(predictions)

            # 要求超过50%的模型预测为正例
            if positive_ratio > 0.5 and positive_count >= self.min_consensus_models:
                avg_prob_0 = float(np.mean([pred.get('probability_0', 0.5) for pred in predictions]))
                avg_prob_1 = float(np.mean([pred.get('probability_1', 0.5) for pred in predictions]))
                avg_confidence = float(np.mean([pred.get('confidence', 0.5) for pred in predictions]))

                consensus_info = {
                    'protein_id': str(predictions[0]['protein_id']),
                    'compound_id': str(predictions[0]['compound_id']),
                    'num_models': int(len(predictions)),
                    'positive_models': int(positive_count),
                    'positive_ratio': float(positive_ratio),
                    'consensus_type': 'majority_positive',
                    'avg_probability_0': avg_prob_0,
                    'avg_probability_1': avg_prob_1,
                    'avg_confidence': avg_confidence,
                    'model_details': []
                }

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

                majority_positive.append(consensus_info)

        return majority_positive

    def _find_high_confidence_positive(self, compound_predictions):
        """找到高置信度正例预测的化合物"""
        high_confidence = []

        for compound_key, predictions in compound_predictions.items():
            if len(predictions) < self.min_consensus_models:
                continue

            # 检查是否有足够数量的高置信度正例预测
            high_conf_positive = [
                pred for pred in predictions
                if pred['prediction'] == 1 and pred.get('probability_1', 0) >= self.probability_threshold
            ]

            if len(high_conf_positive) >= self.min_consensus_models:
                # 只计算高置信度预测的平均值
                avg_prob_0 = float(np.mean([pred.get('probability_0', 0.5) for pred in high_conf_positive]))
                avg_prob_1 = float(np.mean([pred.get('probability_1', 0.5) for pred in high_conf_positive]))
                avg_confidence = float(np.mean([pred.get('confidence', 0.5) for pred in high_conf_positive]))

                consensus_info = {
                    'protein_id': str(predictions[0]['protein_id']),
                    'compound_id': str(predictions[0]['compound_id']),
                    'num_models': int(len(predictions)),
                    'high_conf_positive_models': int(len(high_conf_positive)),
                    'consensus_type': 'high_confidence_positive',
                    'avg_probability_0': avg_prob_0,
                    'avg_probability_1': avg_prob_1,
                    'avg_confidence': avg_confidence,
                    'min_probability_threshold': float(self.probability_threshold),
                    'model_details': []
                }

                # 包含所有模型的详细信息，但标记哪些是高置信度的
                for pred in predictions:
                    is_high_conf = (pred['prediction'] == 1 and
                                    pred.get('probability_1', 0) >= self.probability_threshold)

                    model_detail = {
                        'model_type': str(pred['model_type']),
                        'model_name': str(pred['model_name']),
                        'prediction': int(pred['prediction']),
                        'prediction_label': str(pred['prediction_label']),
                        'probability_0': float(pred.get('probability_0', 0.5)),
                        'probability_1': float(pred.get('probability_1', 0.5)),
                        'confidence': float(pred.get('confidence', 0.5)),
                        'is_high_confidence': bool(is_high_conf)
                    }
                    consensus_info['model_details'].append(model_detail)

                high_confidence.append(consensus_info)

        return high_confidence

    def save_consensus_results(self, consensus_stats, output_dir):
        """保存共识分析结果（修复JSON序列化问题）"""
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
            detailed_data = []

            for result in results:
                # 简化表格
                simplified_row = {
                    'protein_id': result['protein_id'],
                    'compound_id': result['compound_id'],
                    'num_models': result['num_models'],
                    'consensus_type': result['consensus_type'],
                    'avg_probability_0': f"{result['avg_probability_0']:.4f}",
                    'avg_probability_1': f"{result['avg_probability_1']:.4f}",
                    'avg_confidence': f"{result['avg_confidence']:.4f}"
                }

                # 添加特定类型的额外信息
                if consensus_type == "majority_positive":
                    simplified_row['positive_models'] = result['positive_models']
                    simplified_row['positive_ratio'] = f"{result['positive_ratio']:.4f}"
                elif consensus_type == "high_confidence_positive":
                    simplified_row['high_conf_positive_models'] = result['high_conf_positive_models']
                    simplified_row['min_probability_threshold'] = result['min_probability_threshold']

                simplified_data.append(simplified_row)

                # 详细表格 - 展开每个模型的信息
                if CONSENSUS_OUTPUT_DETAILED:
                    for model_detail in result['model_details']:
                        detailed_row = {
                            'protein_id': result['protein_id'],
                            'compound_id': result['compound_id'],
                            'consensus_type': result['consensus_type'],
                            'num_models': result['num_models'],
                            'avg_probability_1': f"{result['avg_probability_1']:.4f}",
                            'avg_confidence': f"{result['avg_confidence']:.4f}",
                            'model_type': model_detail['model_type'],
                            'model_name': model_detail['model_name'],
                            'model_prediction': model_detail['prediction'],
                            'model_prediction_label': model_detail['prediction_label'],
                            'model_probability_0': f"{model_detail['probability_0']:.4f}",
                            'model_probability_1': f"{model_detail['probability_1']:.4f}",
                            'model_confidence': f"{model_detail['confidence']:.4f}"
                        }

                        if consensus_type == "high_confidence_positive":
                            detailed_row['is_high_confidence'] = model_detail.get('is_high_confidence', False)

                        detailed_data.append(detailed_row)

            # 保存简化表格
            if simplified_data:
                try:
                    simplified_df = pd.DataFrame(simplified_data)
                    simplified_file = os.path.join(consensus_dir, f"{consensus_type}_summary.csv")
                    simplified_df.to_csv(simplified_file, index=False, encoding='utf-8-sig')
                    print(f"    ✓ 简化表格: {simplified_file}")
                except Exception as e:
                    print(f"    ✗ 保存简化表格失败: {e}")

            # 保存详细表格
            if detailed_data and CONSENSUS_OUTPUT_DETAILED:
                try:
                    detailed_df = pd.DataFrame(detailed_data)
                    detailed_file = os.path.join(consensus_dir, f"{consensus_type}_detailed.csv")
                    detailed_df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
                    print(f"    ✓ 详细表格: {detailed_file}")
                except Exception as e:
                    print(f"    ✗ 保存详细表格失败: {e}")

            # 保存JSON格式的完整信息 - 修复：使用安全的JSON保存函数
            json_file = os.path.join(consensus_dir, f"{consensus_type}_complete.json")
            if safe_json_dump(results, json_file, indent=2, ensure_ascii=False):
                print(f"    ✓ 完整JSON: {json_file}")
            else:
                print(f"    ✗ 完整JSON保存失败: {json_file}")

        # 创建总结报告
        try:
            self._create_consensus_summary_report(consensus_stats, consensus_dir)
        except Exception as e:
            print(f"    ✗ 创建总结报告失败: {e}")

    def _create_consensus_summary_report(self, consensus_stats, consensus_dir):
        """创建共识分析总结报告"""
        report_file = os.path.join(consensus_dir, "consensus_summary_report.txt")

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("共识预测分析总结报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"最小共识模型数: {self.min_consensus_models}\n")
                f.write(f"概率阈值: {self.probability_threshold}\n\n")

                for consensus_type, results in consensus_stats.items():
                    f.write(f"{consensus_type.upper()}:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"符合条件的化合物数量: {len(results)}\n")

                    if results:
                        # 计算统计信息
                        avg_probs = [r['avg_probability_1'] for r in results]
                        avg_confs = [r['avg_confidence'] for r in results]

                        f.write(f"平均正例概率范围: {min(avg_probs):.4f} - {max(avg_probs):.4f}\n")
                        f.write(f"平均置信度范围: {min(avg_confs):.4f} - {max(avg_confs):.4f}\n")
                        f.write(f"平均正例概率均值: {np.mean(avg_probs):.4f}\n")
                        f.write(f"平均置信度均值: {np.mean(avg_confs):.4f}\n")

                        # 显示前10个最高置信度的化合物
                        sorted_results = sorted(results, key=lambda x: x['avg_confidence'], reverse=True)
                        f.write(f"\n前10个最高置信度的化合物:\n")
                        for i, result in enumerate(sorted_results[:10], 1):
                            f.write(f"  {i}. {result['compound_id']} "
                                    f"(置信度: {result['avg_confidence']:.4f}, "
                                    f"正例概率: {result['avg_probability_1']:.4f})\n")

                    f.write("\n")

            print(f"    ✓ 总结报告: {report_file}")

        except Exception as e:
            print(f"    ✗ 创建总结报告失败: {e}")


# =============================================================================
# 批处理预测器类（增强版）
# =============================================================================

class BatchProteinCompoundPredictor:
    """支持批处理的蛋白质-化合物相互作用预测器（增强版）"""

    def __init__(self, model_base_dir, selected_model_types=None, selected_models=None):
        self.model_base_dir = model_base_dir
        self.selected_model_types = selected_model_types or SELECTED_MODEL_TYPES
        self.selected_models = selected_models or SELECTED_MODELS
        self.models = {}
        self.feature_pipelines = {}
        self.available_models = {}

        # 新增：共识分析器
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
                    if not should_skip_model(model_name):
                        available_model_names.append(model_name)

                if available_model_names:
                    self.available_models[model_type] = available_model_names
                    print(f"  {model_type}: {len(available_model_names)} 个可用模型")
                    for model_name in available_model_names:
                        if any(x in model_name.lower() for x in ['逻辑回归', '朴素贝叶斯', 'k近邻']):
                            print(f"    🟢 推荐: {model_name}")
                        else:
                            print(f"    - {model_name}")
                else:
                    print(f"  {model_type}: 所有模型都被跳过")
            else:
                print(f"  {model_type}: 未找到模型文件")

        total_available = sum(len(models) for models in self.available_models.values())
        print(f"\n总共扫描到 {total_available} 个可用模型")

        return total_available > 0

    def interactive_model_selection(self):
        """交互式模型选择（内存优化建议）"""
        if not INTERACTIVE_MODEL_SELECTION:
            print("使用预配置的模型选择...")
            return True

        print("\n" + "=" * 60)
        print("交互式模型选择 - 大数据量优化建议")
        print("=" * 60)
        print("注意: 由于数据量较大，建议:")
        print("1. 🟢 优先选择内存效率高的模型（逻辑回归、朴素贝叶斯、K近邻）")
        print("2. 🔴 避免选择内存密集型模型（SVM、堆叠分类器、随机森林）")
        print("3. 一次只选择少数几个模型进行预测")

        for model_type in self.available_models:
            print(f"\n{model_type} 可用模型:")
            available_models = self.available_models[model_type]

            for i, model_name in enumerate(available_models, 1):
                if any(x in model_name.lower() for x in ['逻辑回归', '朴素贝叶斯', 'k近邻']):
                    print(f"  {i}. 🟢 推荐: {model_name}")
                else:
                    print(f"  {i}. {model_name}")

            print(f"  0. 全选")
            print(f"  -1. 跳过此类型")

            while True:
                try:
                    selection = input(f"\n请选择要使用的 {model_type} 模型: ").strip()

                    if selection == "-1":
                        if model_type in self.selected_models:
                            del self.selected_models[model_type]
                        break
                    elif selection == "0":
                        self.selected_models[model_type] = available_models.copy()
                        print(f"已选择 {model_type} 的所有模型")
                        break
                    else:
                        indices = [int(x.strip()) for x in selection.split(',')]
                        selected_models = [available_models[i - 1] for i in indices if 1 <= i <= len(available_models)]

                        if selected_models:
                            self.selected_models[model_type] = selected_models
                            print(f"已选择 {model_type} 的模型: {', '.join(selected_models)}")
                            break
                        else:
                            print("无效选择，请重新输入")

                except (ValueError, IndexError):
                    print("输入格式错误，请重新输入")

        confirm = input(f"\n确认使用选择的模型进行预测吗？(y/n): ").strip().lower()
        return confirm == 'y'

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
                if should_skip_model(model_name):
                    continue

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
                        if "CustomStackingClassifier" in str(e) or "CustomVotingClassifier" in str(e):
                            print(f"    这是集成模型，已被跳过以避免兼容性问题")
                else:
                    print(f"  ✗ {model_name} 模型文件不存在: {model_path}")

            if self.models[model_type]:
                print(f"  成功加载 {len(self.models[model_type])} 个模型")
            else:
                print(f"  未能加载任何模型")

        print(f"\n模型加载完成，共加载 {loaded_count} 个模型")
        return loaded_count > 0

    def load_and_prepare_features(self, protein_file, compound_file):
        """加载和准备特征数据（支持大数据量）"""
        print(f"\n加载特征文件...")

        # 加载蛋白质特征
        try:
            protein_df = pd.read_csv(protein_file)
            print(f"  蛋白质特征文件: {protein_df.shape}")
        except Exception as e:
            raise ValueError(f"加载蛋白质特征文件失败: {e}")

        # 加载化合物特征
        try:
            compound_df = pd.read_csv(compound_file)
            print(f"  化合物特征文件: {compound_df.shape}")
        except Exception as e:
            raise ValueError(f"加载化合物特征文件失败: {e}")

        # 检查数据
        if len(protein_df) == 0 or len(compound_df) == 0:
            raise ValueError("特征文件为空")

        # 处理特征数据
        if protein_df.shape[1] > 1:
            protein_ids = protein_df.iloc[:, 0].values
            protein_features_matrix = protein_df.iloc[:, 1:].values
        else:
            protein_ids = [f"Protein_{i + 1}" for i in range(len(protein_df))]
            protein_features_matrix = protein_df.values

        if compound_df.shape[1] > 1:
            compound_ids = compound_df.iloc[:, 0].values
            compound_features_matrix = compound_df.iloc[:, 1:].values
        else:
            compound_ids = [f"Compound_{i + 1}" for i in range(len(compound_df))]
            compound_features_matrix = compound_df.values

        # 计算总组合数
        total_combinations = len(protein_ids) * len(compound_ids)
        print(f"  总组合数: {len(protein_ids)} × {len(compound_ids)} = {total_combinations:,}")

        # 估算内存需求
        feature_dim = protein_features_matrix.shape[1] + compound_features_matrix.shape[1]
        memory_gb = (total_combinations * feature_dim * 8) / (1024 ** 3)
        print(f"  预计内存需求: {memory_gb:.2f} GB")

        if memory_gb > MAX_MEMORY_GB:
            print(f"  警告: 预计内存需求超过限制({MAX_MEMORY_GB}GB)，将使用批处理模式")

        # 调整批处理大小
        optimal_batch_size = estimate_batch_size(feature_dim, MAX_MEMORY_GB)
        actual_batch_size = min(BATCH_SIZE, optimal_batch_size)
        print(f"  建议的批处理大小: {actual_batch_size}")

        return {
            'protein_ids': protein_ids,
            'compound_ids': compound_ids,
            'protein_features': protein_features_matrix,
            'compound_features': compound_features_matrix,
            'total_combinations': total_combinations,
            'feature_dim': feature_dim,
            'batch_size': actual_batch_size
        }

    def apply_feature_pipeline(self, features, model_type):
        """应用特征工程管道（批处理优化）"""
        if model_type not in self.feature_pipelines:
            raise ValueError(f"未找到 {model_type} 的特征管道")

        pipeline = self.feature_pipelines[model_type]

        # 确保输入是2D数组
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 分批处理以节省内存
        batch_size = min(500, features.shape[0])
        processed_features = []

        for start_idx in range(0, features.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, features.shape[0])
            batch_features = features[start_idx:end_idx]

            # 应用填充器
            if 'imputer' in pipeline and pipeline['imputer'] is not None:
                try:
                    batch_features = pipeline['imputer'].transform(batch_features)
                except Exception as e:
                    print(f"应用填充器失败: {e}")

            # 应用特征选择
            if 'selected_features' in pipeline and pipeline['selected_features']:
                try:
                    selected_count = len(pipeline['selected_features'])
                    if batch_features.shape[1] >= selected_count:
                        batch_features = batch_features[:, :selected_count]
                    else:
                        padding = np.zeros((batch_features.shape[0], selected_count - batch_features.shape[1]))
                        batch_features = np.hstack([batch_features, padding])
                except Exception as e:
                    print(f"应用特征选择失败: {e}")

            # 应用降维器
            if 'reducer' in pipeline and pipeline['reducer'] is not None:
                try:
                    batch_features = pipeline['reducer'].transform(batch_features)
                except Exception as e:
                    print(f"应用降维器失败: {e}")

            # 应用缩放器
            if 'scaler' in pipeline and pipeline['scaler'] is not None:
                try:
                    batch_features = pipeline['scaler'].transform(batch_features)
                except Exception as e:
                    print(f"应用缩放器失败: {e}")

            # 处理可能的NaN和inf值
            batch_features = np.nan_to_num(batch_features, nan=0.0, posinf=0.0, neginf=0.0)
            processed_features.append(batch_features)

            # 清理内存
            del batch_features
            gc.collect()

        return np.vstack(processed_features)

    def predict_single_model_batch(self, data_info, model_type, model_name, model):
        """使用单个模型进行批量预测（修复NumPy类型转换）"""
        print(f"    开始批量预测: {model_name}")

        protein_ids = data_info['protein_ids']
        compound_ids = data_info['compound_ids']
        protein_features = data_info['protein_features']
        compound_features = data_info['compound_features']
        batch_size = data_info['batch_size']
        total_combinations = data_info['total_combinations']

        results = []

        # 使用进度条
        if SHOW_PROGRESS:
            pbar = tqdm(total=total_combinations, desc=f"    {model_name}")

        processed_count = 0

        try:
            for i in range(len(protein_ids)):
                protein_id = protein_ids[i]
                protein_feature = protein_features[i]

                # 批量处理化合物
                for start_j in range(0, len(compound_ids), batch_size):
                    end_j = min(start_j + batch_size, len(compound_ids))

                    # 准备当前批次的特征
                    batch_features = []
                    batch_combinations = []

                    for j in range(start_j, end_j):
                        compound_id = compound_ids[j]
                        compound_feature = compound_features[j]
                        combined_feature = np.concatenate([protein_feature, compound_feature])

                        batch_features.append(combined_feature)
                        batch_combinations.append((protein_id, compound_id))

                    batch_features = np.array(batch_features)

                    try:
                        # 应用特征工程管道
                        features_processed = self.apply_feature_pipeline(batch_features, model_type)

                        # 进行预测
                        predictions = model.predict(features_processed)

                        # 如果模型支持概率预测
                        probabilities = None
                        if hasattr(model, 'predict_proba'):
                            try:
                                probabilities = model.predict_proba(features_processed)
                            except Exception as e:
                                probabilities = None

                        # 构建结果 - 修复：确保所有值都是Python原生类型
                        for k, (protein_id, compound_id) in enumerate(batch_combinations):
                            result = {
                                'protein_id': str(protein_id),
                                'compound_id': str(compound_id),
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
                                # 如果没有概率，设置默认值
                                result['probability_0'] = 0.5
                                result['probability_1'] = 0.5
                                result['confidence'] = 0.5

                            results.append(result)

                        processed_count += len(batch_combinations)

                        if SHOW_PROGRESS:
                            pbar.update(len(batch_combinations))

                        # 清理内存
                        del batch_features, features_processed, predictions
                        if probabilities is not None:
                            del probabilities
                        gc.collect()

                        # 定期检查内存使用
                        if processed_count % (batch_size * 5) == 0:
                            current_memory = get_memory_usage()
                            if current_memory > MAX_MEMORY_GB * 1024:
                                print(f"      警告: 内存使用过高 ({current_memory:.0f}MB)，强制垃圾回收")
                                gc.collect()

                    except Exception as e:
                        print(f"      批次预测失败: {e}")
                        # 不要让单个批次的失败影响整个预测
                        continue

        except Exception as e:
            print(f"    ✗ {model_name} 预测过程出错: {e}")

        finally:
            if SHOW_PROGRESS:
                pbar.close()

        print(f"    ✓ {model_name} 预测完成，共 {len(results)} 个结果")
        return results

    def predict_and_save_separately_batch(self, data_info):
        """批量预测并分别保存每个模型的结果（增强版，修复JSON序列化）"""
        print(f"\n开始批量预测并分别保存结果...")

        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(OUTPUT_BASE_DIR, f"batch_prediction_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        all_results = []  # 用于共识分析
        model_summary = {}

        for model_type in self.models:
            if not self.models[model_type]:
                continue

            print(f"\n使用 {model_type} 模型进行批量预测...")

            # 为每个模型类型创建子目录
            type_dir = os.path.join(output_dir, model_type.replace('/', '_'))
            os.makedirs(type_dir, exist_ok=True)

            model_summary[model_type] = {}

            for model_name, model in self.models[model_type].items():
                print(f"  预测模型: {model_name}")

                # 使用单个模型进行批量预测
                model_results = self.predict_single_model_batch(data_info, model_type, model_name, model)

                if model_results:
                    # 保存单个模型的结果（分批保存以节省内存）
                    model_file = os.path.join(type_dir, f"{model_name}_prediction.csv")

                    # 分批写入CSV文件
                    batch_write_size = 5000
                    for i in range(0, len(model_results), batch_write_size):
                        batch_results = model_results[i:i + batch_write_size]
                        batch_df = pd.DataFrame(batch_results)

                        if i == 0:
                            batch_df.to_csv(model_file, index=False, encoding='utf-8-sig')
                        else:
                            batch_df.to_csv(model_file, mode='a', header=False, index=False, encoding='utf-8-sig')

                        del batch_df
                        gc.collect()

                    print(f"    ✓ 结果已保存: {model_file}")

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

                    # 新增：收集所有结果用于共识分析
                    if ENABLE_CONSENSUS_ANALYSIS:
                        all_results.extend(model_results)

                    # 清理内存
                    del model_results
                    gc.collect()

                else:
                    print(f"    ✗ 预测失败")

        # 新增：进行共识分析
        if ENABLE_CONSENSUS_ANALYSIS and all_results:
            print(f"\n开始共识分析，总共 {len(all_results)} 个预测结果...")
            try:
                consensus_stats = self.consensus_analyzer.analyze_consensus(all_results)
                self.consensus_analyzer.save_consensus_results(consensus_stats, output_dir)
            except Exception as e:
                print(f"共识分析过程中出错: {e}")
                import traceback
                traceback.print_exc()

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

        # 保存详细的模型摘要 - 修复：使用安全的JSON保存
        summary_file = os.path.join(output_dir, "prediction_summary.json")
        if safe_json_dump(model_summary, summary_file, indent=2, ensure_ascii=False):
            print(f"预测摘要已保存: {summary_file}")
        else:
            print(f"保存预测摘要失败: {summary_file}")

        print(f"所有结果保存在目录: {output_dir}")

        return all_results, model_summary, output_dir


def display_final_results(data_info, model_summary):
    """显示最终预测结果摘要"""
    print(f"\n" + "=" * 80)
    print("最终预测结果摘要")
    print("=" * 80)

    # 生成总体统计
    total_models = sum(len(models) for models in model_summary.values())

    print(f"总组合数: {data_info['total_combinations']:,}")
    print(f"使用模型数量: {total_models}")

    print(f"\n各模型类型统计:")
    for model_type, models in model_summary.items():
        print(f"  {model_type}: {len(models)} 个模型")
        for model_name, stats in models.items():
            print(f"    {model_name}:")
            print(f"      总预测: {stats['total_predictions']:,}")
            print(f"      正例: {stats['positive_predictions']:,} ({stats['positive_ratio']:.4f})")
            print(f"      负例: {stats['negative_predictions']:,}")
            print(f"      输出: {stats['output_file']}")


def display_consensus_highlights(consensus_stats):
    """显示共识分析亮点"""
    if not consensus_stats:
        return

    print(f"\n" + "🎯" + "=" * 60)
    print("共识分析亮点")
    print("=" * 60 + "🎯")

    for consensus_type, results in consensus_stats.items():
        if not results:
            continue

        print(f"\n📊 {consensus_type.upper()}:")
        print(f"   筛选出 {len(results)} 个化合物")

        if results:
            # 按置信度排序，显示前5个
            sorted_results = sorted(results, key=lambda x: x['avg_confidence'], reverse=True)

            print(f"   🏆 置信度最高的前5个化合物:")
            for i, result in enumerate(sorted_results[:5], 1):
                print(f"      {i}. {result['compound_id']} "
                      f"(置信度: {result['avg_confidence']:.4f}, "
                      f"正例概率: {result['avg_probability_1']:.4f})")

            if len(results) > 5:
                print(f"      ... 还有 {len(results) - 5} 个化合物")


def main():
    """主函数"""
    try:
        print(f"批处理蛋白质-化合物相互作用预测器（增强版）")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # 显示配置
        print(f"配置参数:")
        print(f"  最大内存限制: {MAX_MEMORY_GB} GB")
        print(f"  批处理大小: {BATCH_SIZE}")
        print(f"  跳过集成模型: {SKIP_ENSEMBLE_MODELS}")
        print(f"  跳过内存密集模型: {SKIP_MEMORY_INTENSIVE_MODELS}")
        print(f"  交互式选择: {INTERACTIVE_MODEL_SELECTION}")
        print(f"  共识分析: {ENABLE_CONSENSUS_ANALYSIS}")
        if ENABLE_CONSENSUS_ANALYSIS:
            print(f"    最小共识模型数: {MIN_CONSENSUS_MODELS}")
            print(f"    概率阈值: {CONSENSUS_PROBABILITY_THRESHOLD}")
            print(f"    分析类型: {', '.join(CONSENSUS_ANALYSIS_TYPES)}")

        # 检查输入文件
        if not os.path.exists(PROTEIN_FEATURE_FILE):
            raise FileNotFoundError(f"蛋白质特征文件不存在: {PROTEIN_FEATURE_FILE}")

        if not os.path.exists(COMPOUND_FEATURE_FILE):
            raise FileNotFoundError(f"化合物特征文件不存在: {COMPOUND_FEATURE_FILE}")

        if not os.path.exists(MODEL_BASE_DIR):
            raise FileNotFoundError(f"模型目录不存在: {MODEL_BASE_DIR}")

        # 初始化预测器
        predictor = BatchProteinCompoundPredictor(MODEL_BASE_DIR, SELECTED_MODEL_TYPES, SELECTED_MODELS)

        # 扫描可用模型
        if not predictor.scan_available_models():
            raise RuntimeError("未找到任何可用模型")

        # 交互式模型选择
        if not predictor.interactive_model_selection():
            return

        # 加载选择的模型
        if not predictor.load_selected_models():
            raise RuntimeError("未能加载任何模型")

        # 加载和准备特征数据
        data_info = predictor.load_and_prepare_features(PROTEIN_FEATURE_FILE, COMPOUND_FEATURE_FILE)

        print(f"\n准备开始批量预测...")
        print(f"数据信息:")
        print(f"  总组合数: {data_info['total_combinations']:,}")
        print(f"  特征维度: {data_info['feature_dim']}")
        print(f"  批处理大小: {data_info['batch_size']}")

        # 确认开始预测
        confirm = input(f"\n确认开始批量预测吗？(y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消预测")
            return

        # 进行批量预测并分别保存结果
        all_results, model_summary, output_dir = predictor.predict_and_save_separately_batch(data_info)

        # 显示最终结果
        display_final_results(data_info, model_summary)

        # 新增：如果进行了共识分析，显示共识分析的亮点
        if ENABLE_CONSENSUS_ANALYSIS and hasattr(predictor, 'consensus_analyzer'):
            try:
                # 重新加载共识分析结果进行展示
                consensus_dir = os.path.join(output_dir, "consensus_analysis")
                if os.path.exists(consensus_dir):
                    consensus_stats = {}

                    for consensus_type in CONSENSUS_ANALYSIS_TYPES:
                        json_file = os.path.join(consensus_dir, f"{consensus_type}_complete.json")
                        if os.path.exists(json_file):
                            try:
                                with open(json_file, 'r', encoding='utf-8') as f:
                                    consensus_stats[consensus_type] = json.load(f)
                            except Exception as e:
                                print(f"读取共识分析结果文件失败 {json_file}: {e}")

                    if consensus_stats:
                        display_consensus_highlights(consensus_stats)

                        # 生成快速筛选建议
                        print(f"\n" + "💡" + "=" * 60)
                        print("快速筛选建议")
                        print("=" * 60 + "💡")

                        # 统计不同共识类型的化合物数量
                        all_positive_count = len(consensus_stats.get('all_positive', []))
                        majority_positive_count = len(consensus_stats.get('majority_positive', []))
                        high_confidence_count = len(consensus_stats.get('high_confidence_positive', []))

                        print(f"🥇 最高优先级 - 所有模型都同意: {all_positive_count} 个化合物")
                        print(f"   推荐: 优先进行实验验证")
                        print(f"   文件: all_positive_summary.csv")

                        print(f"🥈 中等优先级 - 高置信度预测: {high_confidence_count} 个化合物")
                        print(f"   推荐: 作为二线候选化合物")
                        print(f"   文件: high_confidence_positive_summary.csv")

                        print(f"🥉 备选考虑 - 大多数模型同意: {majority_positive_count} 个化合物")
                        print(f"   推荐: 用于进一步的计算筛选")
                        print(f"   文件: majority_positive_summary.csv")

                        # 给出实际的文件路径建议
                        print(f"\n📁 重要文件位置:")
                        print(f"   共识分析目录: {consensus_dir}")
                        print(f"   总结报告: {os.path.join(consensus_dir, 'consensus_summary_report.txt')}")

                        # 如果有高优先级化合物，给出具体建议
                        if all_positive_count > 0:
                            print(f"\n🎯 关键发现:")
                            print(f"   发现 {all_positive_count} 个所有模型都认为会相互作用的化合物！")
                            print(f"   这些化合物具有最高的实验验证价值")
                            print(f"   建议立即查看: {os.path.join(consensus_dir, 'all_positive_summary.csv')}")

                        elif high_confidence_count > 0:
                            print(f"\n🎯 关键发现:")
                            print(f"   发现 {high_confidence_count} 个高置信度相互作用的化合物")
                            print(f"   这些化合物值得优先考虑")
                            print(f"   建议查看: {os.path.join(consensus_dir, 'high_confidence_positive_summary.csv')}")

                        elif majority_positive_count > 0:
                            print(f"\n🎯 关键发现:")
                            print(f"   发现 {majority_positive_count} 个大多数模型认可的化合物")
                            print(f"   建议进一步筛选后考虑实验验证")
                            print(f"   建议查看: {os.path.join(consensus_dir, 'majority_positive_summary.csv')}")

                        else:
                            print(f"\n⚠️  注意:")
                            print(f"   未发现符合当前共识标准的化合物")
                            print(f"   建议降低共识要求或检查模型预测结果")

                    else:
                        print(f"无法读取共识分析结果")
                else:
                    print(f"共识分析目录不存在: {consensus_dir}")

            except Exception as e:
                print(f"显示共识分析亮点时出错: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n🎉 预测完成！")
        print(f"所有结果已保存至: {output_dir}")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 生成最终总结
        print(f"\n" + "📋" + "=" * 70)
        print("最终总结")
        print("=" * 70 + "📋")

        total_models = sum(len(models) for models in model_summary.values())
        total_predictions = sum(
            stats['total_predictions']
            for model_type_stats in model_summary.values()
            for stats in model_type_stats.values()
        )

        print(f"✅ 成功使用 {total_models} 个模型")
        print(f"✅ 生成 {total_predictions:,} 个预测结果")
        print(f"✅ 数据保存在 {len(model_summary)} 个模型类型目录中")

        if ENABLE_CONSENSUS_ANALYSIS:
            print(f"✅ 完成共识分析，识别出最有希望的化合物候选")

        print(f"\n🔬 下一步建议:")
        print(f"1. 查看共识分析结果，优先考虑 'all_positive' 类型的化合物")
        print(f"2. 查看详细的预测概率，筛选高置信度的预测")
        print(f"3. 结合领域知识，进一步筛选有生物学意义的化合物")
        print(f"4. 规划实验验证，从最高置信度的化合物开始")

        print(f"\n📊 输出文件说明:")
        print(f"   individual predictions: 每个模型的详细预测结果")
        print(f"   model_comparison.csv: 所有模型的性能对比")
        print(f"   consensus_analysis/: 共识分析结果（重点关注）")
        print(f"   prediction_summary.json: 完整的预测摘要")

        # 显示内存使用情况
        final_memory = get_memory_usage()
        if final_memory > 0:
            print(f"\n💾 内存使用统计:")
            print(f"   最终内存使用: {final_memory:.0f} MB")
            if final_memory > MAX_MEMORY_GB * 1024:
                print(f"   ⚠️  内存使用超过设定限制，建议下次减小批处理大小")
            else:
                print(f"   ✅ 内存使用在合理范围内")

        return output_dir

    except KeyboardInterrupt:
        print(f"\n⛔ 用户中断了预测过程")
        print(f"   如果需要继续，请重新运行程序")
        return None

    except FileNotFoundError as e:
        print(f"❌ 文件不存在错误: {e}")
        print(f"   请检查配置参数中的文件路径是否正确")
        return None

    except Exception as e:
        print(f"❌ 预测过程中出错: {e}")
        print(f"   详细错误信息:")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print(f"🚀 启动蛋白质-化合物相互作用预测器")
    print(f"📅 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👤 当前用户: {os.getenv('USERNAME', 'Unknown')}")
    print("=" * 80)

    result_dir = main()

    if result_dir:
        print(f"\n🌟 预测成功完成！")
        print(f"🌟 结果目录: {result_dir}")

        # 如果启用了共识分析，给出快速访问建议
        if ENABLE_CONSENSUS_ANALYSIS:
            consensus_dir = os.path.join(result_dir, "consensus_analysis")
            if os.path.exists(consensus_dir):
                print(f"\n⭐ 快速查看共识结果:")

                # 检查各个共识文件是否存在
                priority_files = [
                    ("all_positive_summary.csv", "🥇 最高优先级化合物"),
                    ("high_confidence_positive_summary.csv", "🥈 高置信度化合物"),
                    ("majority_positive_summary.csv", "🥉 大多数模型同意的化合物")
                ]

                for filename, description in priority_files:
                    filepath = os.path.join(consensus_dir, filename)
                    if os.path.exists(filepath):
                        try:
                            df = pd.read_csv(filepath)
                            print(f"   {description}: {len(df)} 个 ({filepath})")
                        except Exception as e:
                            print(f"   {description}: 文件存在但读取失败 ({filepath})")

                print(f"\n💡 提示: 建议从 '🥇 最高优先级化合物' 开始查看！")

                # 尝试提供Windows下的快速打开命令
                try:
                    print(f"\n🖱️  快速打开文件夹:")
                    print(f"   Windows: explorer \"{consensus_dir}\"")
                    print(f"   或直接复制路径到文件管理器: {consensus_dir}")
                except:
                    pass

        print(f"\n🎊 感谢使用！预测已完成，祝你的研究顺利！")

    else:
        print(f"\n❌ 预测过程失败或被中断")
        print(f"❌ 请检查错误信息并重试")
        print(f"\n🔧 常见问题解决方案:")
        print(f"   1. 检查输入文件路径是否正确")
        print(f"   2. 确保模型目录存在且包含必要文件")
        print(f"   3. 检查内存是否足够（当前设置: {MAX_MEMORY_GB}GB）")
        print(f"   4. 尝试减小 BATCH_SIZE 参数")
        print(f"   5. 确保Python环境安装了所有必要的包")

    print(f"\n⏰ 程序结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
