import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score, precision_recall_curve, roc_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.inspection import permutation_importance
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from scipy import stats
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from datetime import datetime
import itertools
import json
import hashlib

# 新增：有监督降维相关导入
try:
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.manifold import LocallyLinearEmbedding
    has_supervised_reduction = True
except ImportError:
    has_supervised_reduction = False
    print("警告: 某些降维算法不可用")

# 可选依赖导入
try:
    import networkx as nx
    has_networkx = True
except ImportError:
    has_networkx = False
    print("警告: networkx 库未安装，网络图功能将被禁用")

try:
    from sklearn.preprocessing import PowerTransformer
    has_power_transformer = True
except ImportError:
    has_power_transformer = False
    print("警告: PowerTransformer 未在当前 sklearn 版本中可用")

# =============================================================================
# 用户可配置参数 - 修改这些参数来控制程序运行
# =============================================================================

# 输入输出设置
INPUT_FILE = "正负样本nuclear_receptor_BioAssay_data_aac_dpc_dti_combined_features.csv"  # 训练数据文件路径
OUTPUT_DIR = ("新蛋白特征核受体Combine_BioAssay新特征")  # 输出目录，为空则自动生成以日期时间命名的目录
LOGFILE_ENABLED = True  # 是否生成日志文件

# ========== 新增：断点续传设置 ==========
ENABLE_CHECKPOINT = True  # 是否启用断点续传功能
CHECKPOINT_FILE = "training_checkpoint.json"  # 检查点文件名
FORCE_RESTART = False  # 是否强制从头开始训练（忽略检查点）
AUTO_CLEANUP_CHECKPOINT = True  # 是否在训练完成后自动清理检查点文件
CHECKPOINT_VERBOSE = True  # 是否显示检查点详细信息

# ========== 新增：列名配置设置 ==========
# 数据列名配置 - 根据你的数据文件调整这些列名
PROTEIN_ID_COLUMN = "Protein_Accession"  # 蛋白质ID列名，如 "protein_id", "gene_id", "uniprot_id" 等
COMPOUND_ID_COLUMN = "Compound_CID"  # 化合物ID列名，如 "compound_id", "drug_id", "pubchem_id" 等
LABEL_COLUMN = None  # 标签列名，设为None则自动使用最后一列，也可设为 "label", "interaction", "binding" 等

# 特征列范围配置（可选，用于更精确的特征划分）
PROTEIN_FEATURE_START_IDX = None  # 蛋白质特征开始索引，None表示自动检测
PROTEIN_FEATURE_END_IDX = None  # 蛋白质特征结束索引，None表示自动检测
COMPOUND_FEATURE_START_IDX = None  # 化合物特征开始索引，None表示自动检测
COMPOUND_FEATURE_END_IDX = None  # 化合物特征结束索引，None表示自动检测

# 自动列名检测设置
AUTO_DETECT_COLUMNS = True  # 是否启用自动列名检测
PROTEIN_ID_KEYWORDS = ["gene_id", "protein_id", "uniprot_id", "protein", "target"]  # 蛋白质ID关键词
COMPOUND_ID_KEYWORDS = ["compound_id", "drug_id", "pubchem_id", "molecule_id", "compound", "drug"]  # 化合物ID关键词
LABEL_KEYWORDS = ["label", "interaction", "binding", "activity", "target"]  # 标签列关键词

# 显示列名检测信息
SHOW_COLUMN_INFO = True  # 是否显示列名检测和特征划分信息

# 训练模型设置
ENABLE_STANDARD_RANDOM = True  # 是否训练标准随机分割模型
ENABLE_PROTEIN_COLD_START = False  # 是否训练蛋白质冷启动模型
ENABLE_DRUG_COLD_START = False  # 是否训练药物冷启动模型
ENABLE_DUAL_COLD_START = False  # 是否训练双重冷启动模型
TEST_SIZE = 0.2  # 测试集比例
RANDOM_STATE = 42  # 随机数种子

# 模型训练控制
ENABLE_SVM = True  # 是否训练SVM模型
ENABLE_RANDOM_FOREST = True  # 是否训练随机森林模型
ENABLE_GRADIENT_BOOSTING = True  # 是否训练梯度提升模型
ENABLE_LOGISTIC_REGRESSION = True  # 是否训练逻辑回归模型
ENABLE_KNN = True  # 是否训练K近邻模型
ENABLE_EXTRA_TREES = True  # 是否训练极端随机树模型
ENABLE_NAIVE_BAYES = True  # 是否训练朴素贝叶斯模型
ENABLE_GAUSSIAN_PROCESS = False  # 是否训练高斯过程模型（计算量大）
ENABLE_ENSEMBLE_MODELS = True  # 是否创建集成模型（投票分类器和堆叠分类器）

# 特征工程设置 - 修改这些设置
FEATURE_ENHANCEMENT = False  # 是否启用特征增强
FEATURE_SELECTION = False  # 是否启用特征选择
FEATURE_SELECTION_THRESHOLD = 2  # 特征选择阈值
MAX_FEATURES = 1000  # 最大特征数量

# ========== 新增：有监督降维设置 ==========
ENABLE_SUPERVISED_REDUCTION = False  # 是否启用有监督降维
REDUCTION_METHOD = "auto"  # 降维方法: "auto", "lda", "pca", "none"
FINAL_DIMENSION = 500  # 最终降维后的特征数量
IMPORTANCE_SELECTION_RATIO = 0.8  # 按重要性选择的特征比例，然后再降维

# 预测设置
PREDICTION_ENABLED = False  # 是否进行蛋白质-化合物相互作用预测
PREDICTION_FILES = [
    # 格式: [蛋白质文件路径, 化合物文件路径, 预测名称]
    ["Q13133的PseACC和PsePSSM特征.csv", "ehdpp.csv", "Q13133_EHDPP"]
    # 可以添加多个预测对：
    # ["protein2.csv", "compound2.csv", "Protein2_Compound2"],
    # ["protein3.csv", "compound3.csv", "Protein3_Compound3"]
]

# 批量预测设置
BATCH_PREDICTION_ENABLED = False  # 是否启用批量预测
BATCH_MODE_MAX_COMBINATIONS = 100  # 最大预测组合数量
BATCH_MODE_PROTEIN_FILE = "多个蛋白质特征.csv"  # 包含多个蛋白质的特征文件
BATCH_MODE_COMPOUND_FILE = "多个化合物特征.csv"  # 包含多个化合物的特征文件

# 可视化设置
VISUALIZATION_ENABLED = True  # 是否生成可视化图表
PLOT_DPI = 300  # 图表DPI

# 性能设置
N_JOBS = -1  # -1表示使用所有可用核心
CROSS_VALIDATION_FOLDS = 5  # 交叉验证折数

# =============================================================================

# 保持原有的全局设置代码不变...
CPU_COUNT = cpu_count() if N_JOBS == -1 else min(N_JOBS, cpu_count())
print(f"检测到 {CPU_COUNT} 个CPU核心可用, 将使用 {N_JOBS if N_JOBS > 0 else CPU_COUNT} 个核心")

# 忽略特定警告
warnings.filterwarnings('ignore', message='.*No further splits with positive gain.*')
warnings.filterwarnings('ignore', message='.*super.*__sklearn_tags__.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*PerformanceWarning.*')

# =============================================================================
# 断点续传系统
# =============================================================================

class CheckpointManager:
    """断点续传管理器"""

    def __init__(self, checkpoint_file, output_dir, verbose=True):
        self.checkpoint_file = checkpoint_file
        self.output_dir = output_dir
        self.verbose = verbose
        self.checkpoint_path = os.path.join(output_dir, checkpoint_file)
        self.checkpoint_data = {}

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 加载现有检查点
        self.load_checkpoint()

    def load_checkpoint(self):
        """加载检查点文件"""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    self.checkpoint_data = json.load(f)
                if self.verbose:
                    print(f"已加载检查点文件: {self.checkpoint_path}")
                    self.print_progress_summary()
            except Exception as e:
                if self.verbose:
                    print(f"加载检查点文件失败: {e}")
                self.checkpoint_data = {}
        else:
            if self.verbose:
                print("未找到检查点文件，将创建新的训练进度")
            self.checkpoint_data = {}

    def save_checkpoint(self):
        """保存检查点到文件"""
        try:
            # 添加时间戳
            self.checkpoint_data['last_update'] = datetime.now().isoformat()

            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_data, f, indent=2, ensure_ascii=False)

            if self.verbose:
                print(f"检查点已保存: {self.checkpoint_path}")
        except Exception as e:
            print(f"保存检查点失败: {e}")

    def get_data_hash(self, df):
        """计算数据的哈希值用于验证数据一致性"""
        try:
            # 使用数据形状和一些关键统计信息创建哈希
            data_str = f"{df.shape}_{df.dtypes.to_string()}_{df.describe().to_string()}"
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception as e:
            return "unknown"

    def validate_data_consistency(self, df):
        """验证数据是否与检查点中记录的一致"""
        if 'data_hash' not in self.checkpoint_data:
            # 首次运行，记录数据哈希
            self.checkpoint_data['data_hash'] = self.get_data_hash(df)
            return True

        current_hash = self.get_data_hash(df)
        stored_hash = self.checkpoint_data['data_hash']

        if current_hash != stored_hash:
            if self.verbose:
                print("警告: 数据发生变化，检查点可能不适用")
                print("建议重新开始训练或使用新的检查点文件")
            return False

        return True

    def is_task_completed(self, task_category, task_name):
        """检查特定任务是否已完成"""
        if task_category not in self.checkpoint_data:
            return False

        completed_tasks = self.checkpoint_data[task_category].get('completed', [])
        return task_name in completed_tasks

    def mark_task_completed(self, task_category, task_name, **additional_info):
        """标记任务为已完成"""
        if task_category not in self.checkpoint_data:
            self.checkpoint_data[task_category] = {
                'completed': [],
                'total': 0,
                'details': {}
            }

        if task_name not in self.checkpoint_data[task_category]['completed']:
            self.checkpoint_data[task_category]['completed'].append(task_name)

        # 添加额外信息
        if additional_info:
            self.checkpoint_data[task_category]['details'][task_name] = additional_info

        # 保存检查点
        self.save_checkpoint()

        if self.verbose:
            completed = len(self.checkpoint_data[task_category]['completed'])
            total = self.checkpoint_data[task_category]['total']
            print(f"[检查点] {task_category} - {task_name} 已完成 ({completed}/{total})")

    def set_total_tasks(self, task_category, total):
        """设置某类任务的总数"""
        if task_category not in self.checkpoint_data:
            self.checkpoint_data[task_category] = {
                'completed': [],
                'total': 0,
                'details': {}
            }

        self.checkpoint_data[task_category]['total'] = total

    def get_remaining_tasks(self, task_category, all_tasks):
        """获取剩余未完成的任务"""
        if task_category not in self.checkpoint_data:
            return all_tasks

        completed = self.checkpoint_data[task_category]['completed']
        return [task for task in all_tasks if task not in completed]

    def print_progress_summary(self):
        """打印进度摘要"""
        if not self.checkpoint_data:
            print("尚未开始任何训练任务")
            return

        print("\n=== 训练进度摘要 ===")

        for category, data in self.checkpoint_data.items():
            if category in ['last_update', 'data_hash', 'config_hash']:
                continue

            completed = len(data.get('completed', []))
            total = data.get('total', 0)

            if total > 0:
                progress = completed / total * 100
                print(f"{category}: {completed}/{total} ({progress:.1f}%)")

                # 显示已完成的任务
                if completed > 0:
                    completed_list = data.get('completed', [])
                    print(f"  已完成: {', '.join(completed_list)}")
            else:
                print(f"{category}: 未设置总任务数")

        if 'last_update' in self.checkpoint_data:
            print(f"最后更新: {self.checkpoint_data['last_update']}")

        print("=" * 30)

    def cleanup_checkpoint(self):
        """清理检查点文件"""
        try:
            if os.path.exists(self.checkpoint_path):
                os.remove(self.checkpoint_path)
                if self.verbose:
                    print(f"检查点文件已清理: {self.checkpoint_path}")
        except Exception as e:
            print(f"清理检查点文件失败: {e}")

    def save_config_hash(self, config_dict):
        """保存配置的哈希值"""
        config_str = json.dumps(config_dict, sort_keys=True)
        self.checkpoint_data['config_hash'] = hashlib.md5(config_str.encode()).hexdigest()

    def validate_config_consistency(self, config_dict):
        """验证配置是否与检查点中的一致"""
        if 'config_hash' not in self.checkpoint_data:
            return True

        config_str = json.dumps(config_dict, sort_keys=True)
        current_hash = hashlib.md5(config_str.encode()).hexdigest()
        stored_hash = self.checkpoint_data['config_hash']

        if current_hash != stored_hash:
            if self.verbose:
                print("警告: 配置参数发生变化，检查点可能不适用")
            return False

        return True


def get_current_config():
    """获取当前配置参数"""
    return {
        'ENABLE_STANDARD_RANDOM': ENABLE_STANDARD_RANDOM,
        'ENABLE_PROTEIN_COLD_START': ENABLE_PROTEIN_COLD_START,
        'ENABLE_DRUG_COLD_START': ENABLE_DRUG_COLD_START,
        'ENABLE_DUAL_COLD_START': ENABLE_DUAL_COLD_START,
        'ENABLE_SVM': ENABLE_SVM,
        'ENABLE_RANDOM_FOREST': ENABLE_RANDOM_FOREST,
        'ENABLE_GRADIENT_BOOSTING': ENABLE_GRADIENT_BOOSTING,
        'ENABLE_LOGISTIC_REGRESSION': ENABLE_LOGISTIC_REGRESSION,
        'ENABLE_KNN': ENABLE_KNN,
        'ENABLE_EXTRA_TREES': ENABLE_EXTRA_TREES,
        'ENABLE_NAIVE_BAYES': ENABLE_NAIVE_BAYES,
        'ENABLE_GAUSSIAN_PROCESS': ENABLE_GAUSSIAN_PROCESS,
        'ENABLE_ENSEMBLE_MODELS': ENABLE_ENSEMBLE_MODELS,
        'TEST_SIZE': TEST_SIZE,
        'RANDOM_STATE': RANDOM_STATE,
        'FEATURE_ENHANCEMENT': FEATURE_ENHANCEMENT,
        'FEATURE_SELECTION': FEATURE_SELECTION,
        'MAX_FEATURES': MAX_FEATURES
    }


# ------------------------ 中文字体解决方案 ------------------------
def setup_chinese_font():
    """设置中文字体文件"""
    # 创建一个项目级的字体文件夹
    font_dir = os.path.join(os.getcwd(), 'fonts')
    os.makedirs(font_dir, exist_ok=True)

    # SimHei字体文件路径
    simhei_path = os.path.join(font_dir, 'SimHei.ttf')

    # 如果字体文件不存在，则下载
    if not os.path.exists(simhei_path):
        try:
            import urllib.request
            print("正在下载中文字体文件...")
            font_url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
            urllib.request.urlretrieve(font_url, simhei_path)
            print(f"中文字体已下载至: {simhei_path}")
        except Exception as e:
            print(f"下载字体文件失败: {e}")
            # 创建一个错误提示文本文件
            with open(os.path.join(font_dir, 'FONT_MISSING.txt'), 'w', encoding='utf-8') as f:
                f.write("中文字体文件下载失败。请手动下载SimHei.ttf并放置在此文件夹中。")
            return None

    return simhei_path


# 设置全局中文字体和创建字体属性对象
chinese_font_path = setup_chinese_font()
if chinese_font_path and os.path.exists(chinese_font_path):
    # 创建字体对象
    chinese_font = FontProperties(fname=chinese_font_path)
    print(f"已加载中文字体: {chinese_font_path}")

    # 尝试设置matplotlib全局字体
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
else:
    # 尝试使用系统内置字体
    try:
        chinese_font = FontProperties(family='SimHei')
        print("使用系统内置SimHei字体")
    except:
        try:
            chinese_font = FontProperties(family=['Microsoft YaHei', 'SimSun', 'STSong'])
            print("使用系统其他中文字体")
        except:
            chinese_font = None
            print("警告：无法设置中文字体")


def plot_with_chinese_font(fig, title=None, xlabel=None, ylabel=None,
                           xtick_labels=None, ytick_labels=None,
                           legend_labels=None, text_annotations=None):
    """使用中文字体设置图表文本"""
    if not VISUALIZATION_ENABLED:
        return

    # 获取当前图表的所有子图
    axes = fig.get_axes()

    for ax in axes:
        # 设置标题
        if title and ax == axes[0]:  # 只为第一个子图设置标题
            ax.set_title(title, fontproperties=chinese_font, fontsize=14)

        # 设置x轴和y轴标签
        if xlabel:
            ax.set_xlabel(xlabel, fontproperties=chinese_font, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontproperties=chinese_font, fontsize=12)

        # 设置x轴刻度标签
        if xtick_labels:
            ax.set_xticklabels(xtick_labels, fontproperties=chinese_font, fontsize=10)
        else:
            # 如果未提供标签但已有标签，则应用中文字体
            if ax.get_xticklabels():
                for label in ax.get_xticklabels():
                    label.set_fontproperties(chinese_font)

        # 设置y轴刻度标签
        if ytick_labels:
            ax.set_yticklabels(ytick_labels, fontproperties=chinese_font, fontsize=10)
        else:
            # 如果未提供标签但已有标签，则应用中文字体
            if ax.get_yticklabels():
                for label in ax.get_yticklabels():
                    label.set_fontproperties(chinese_font)

    # 设置图例标签
    if legend_labels:
        for ax in axes:
            if ax.get_legend():
                for text in ax.get_legend().get_texts():
                    text.set_fontproperties(chinese_font)

    # 添加文本注释
    if text_annotations:
        for annotation in text_annotations:
            if 'ax' in annotation:
                target_ax = annotation['ax']
            else:
                target_ax = axes[0]  # 默认使用第一个子图

            target_ax.text(annotation['x'], annotation['y'], annotation['text'],
                           fontproperties=chinese_font, fontsize=annotation.get('fontsize', 10),
                           ha=annotation.get('ha', 'center'), va=annotation.get('va', 'center'))

    # 强制更新布局
    fig.tight_layout()


# ------------ 尝试导入可选的库 ------------
xgb_installed = False
lgbm_installed = False
catboost_installed = False

try:
    from xgboost import XGBClassifier

    xgb_installed = True
    print("XGBoost已安装")
except ImportError:
    xgb_installed = False
    print("XGBoost未安装")

try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb

    # 尝试设置全局静默模式，如果不支持则忽略
    try:
        lgb.set_verbosity(-1)
    except AttributeError:
        pass
    lgbm_installed = True
    print("LightGBM已安装")
except ImportError:
    lgbm_installed = False
    print("LightGBM未安装")

try:
    from catboost import CatBoostClassifier

    catboost_installed = True
    print("CatBoost已安装")
except ImportError:
    catboost_installed = False
    print("CatBoost未安装")


def load_data(input_file):
    """加载数据并进行初步处理，支持自定义列名配置"""
    print(f"正在加载数据文件: {input_file}")
    df = pd.read_csv(input_file)
    df = df.fillna(0)  # 填充NaN值为0
    print(f"数据形状: {df.shape}")

    # 显示所有列名用于调试
    if SHOW_COLUMN_INFO:
        print(f"数据文件包含的列: {list(df.columns)}")

    all_columns = df.columns.tolist()

    # ========== 自动检测或使用配置的列名 ==========
    protein_id_col = PROTEIN_ID_COLUMN
    compound_id_col = COMPOUND_ID_COLUMN
    label_col = LABEL_COLUMN

    # 自动检测蛋白质ID列
    if AUTO_DETECT_COLUMNS and (protein_id_col is None or protein_id_col not in all_columns):
        for keyword in PROTEIN_ID_KEYWORDS:
            matches = [col for col in all_columns if keyword.lower() in col.lower()]
            if matches:
                protein_id_col = matches[0]
                if SHOW_COLUMN_INFO:
                    print(f"自动检测到蛋白质ID列: {protein_id_col}")
                break

        if protein_id_col is None or protein_id_col not in all_columns:
            print("警告: 未找到蛋白质ID列，使用第一列作为蛋白质ID")
            protein_id_col = all_columns[0]

    # 自动检测化合物ID列
    if AUTO_DETECT_COLUMNS and (compound_id_col is None or compound_id_col not in all_columns):
        for keyword in COMPOUND_ID_KEYWORDS:
            matches = [col for col in all_columns if keyword.lower() in col.lower()]
            if matches:
                compound_id_col = matches[0]
                if SHOW_COLUMN_INFO:
                    print(f"自动检测到化合物ID列: {compound_id_col}")
                break

        if compound_id_col is None or compound_id_col not in all_columns:
            print("警告: 未找到化合物ID列，使用中间位置的列作为化合物ID")
            compound_id_col = all_columns[len(all_columns) // 2]

    # 自动检测标签列
    if label_col is None:
        if AUTO_DETECT_COLUMNS:
            for keyword in LABEL_KEYWORDS:
                matches = [col for col in all_columns if keyword.lower() in col.lower()]
                if matches:
                    label_col = matches[0]
                    if SHOW_COLUMN_INFO:
                        print(f"自动检测到标签列: {label_col}")
                    break

        if label_col is None:
            label_col = all_columns[-1]  # 默认使用最后一列
            if SHOW_COLUMN_INFO:
                print(f"使用最后一列作为标签列: {label_col}")

    # 验证列是否存在
    missing_cols = []
    for col_name, col_var in [("蛋白质ID", protein_id_col), ("化合物ID", compound_id_col), ("标签", label_col)]:
        if col_var not in all_columns:
            missing_cols.append(f"{col_name}列 '{col_var}'")

    if missing_cols:
        raise ValueError(f"以下列在数据文件中不存在: {', '.join(missing_cols)}")

    # ========== 确定特征列范围 ==========
    if protein_id_col in all_columns and compound_id_col in all_columns:
        protein_id_idx = all_columns.index(protein_id_col)
        compound_id_idx = all_columns.index(compound_id_col)
        label_idx = all_columns.index(label_col)

        # 根据配置或自动检测确定特征范围
        if PROTEIN_FEATURE_START_IDX is not None and PROTEIN_FEATURE_END_IDX is not None:
            protein_features = all_columns[PROTEIN_FEATURE_START_IDX:PROTEIN_FEATURE_END_IDX]
        else:
            # 自动检测：蛋白质特征在蛋白质ID后到化合物ID前
            protein_start = protein_id_idx + 1
            protein_end = compound_id_idx
            protein_features = all_columns[protein_start:protein_end]

        if COMPOUND_FEATURE_START_IDX is not None and COMPOUND_FEATURE_END_IDX is not None:
            compound_features = all_columns[COMPOUND_FEATURE_START_IDX:COMPOUND_FEATURE_END_IDX]
        else:
            # 自动检测：化合物特征在化合物ID后到标签列前
            compound_start = compound_id_idx + 1
            compound_end = label_idx
            compound_features = all_columns[compound_start:compound_end]

    else:
        # 兜底方案：平均分割特征
        feature_cols = [col for col in all_columns if col not in [protein_id_col, compound_id_col, label_col]]
        mid_point = len(feature_cols) // 2
        protein_features = feature_cols[:mid_point]
        compound_features = feature_cols[mid_point:]

    # 显示特征划分信息
    if SHOW_COLUMN_INFO:
        print(f"\n列名配置结果:")
        print(f"蛋白质ID列: {protein_id_col}")
        print(f"化合物ID列: {compound_id_col}")
        print(f"标签列: {label_col}")
        print(
            f"蛋白质特征数: {len(protein_features)} (列索引: {all_columns.index(protein_features[0]) if protein_features else 'N/A'}-{all_columns.index(protein_features[-1]) if protein_features else 'N/A'})")
        print(
            f"化合物特征数: {len(compound_features)} (列索引: {all_columns.index(compound_features[0]) if compound_features else 'N/A'}-{all_columns.index(compound_features[-1]) if compound_features else 'N/A'})")

        # 显示特征列名样例
        if protein_features:
            print(f"蛋白质特征示例: {protein_features[:3]}{'...' if len(protein_features) > 3 else ''}")
        if compound_features:
            print(f"化合物特征示例: {compound_features[:3]}{'...' if len(compound_features) > 3 else ''}")
        print()

    # 分离特征和标签
    X = df[protein_features + compound_features]
    y = df[label_col]

    # 返回数据和元信息
    return df, X, y, protein_id_col, compound_id_col, protein_features, compound_features, label_col


def create_standard_split(df, protein_id_col, compound_id_col, test_size=0.2, random_state=42):
    """创建标准随机分割数据集: 简单地随机分割数据，不考虑蛋白质或化合物的分布"""
    print("创建标准随机分割数据集...")

    # 使用sklearn的train_test_split直接分割数据
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df.iloc[:, -1] if len(df) > 50 else None  # 如果可能，使用标签进行分层抽样
    )

    # 输出分割统计信息
    print(
        f"训练集蛋白质数量: {len(train_df[protein_id_col].unique())}, 测试集蛋白质数量: {len(test_df[protein_id_col].unique())}")
    print(
        f"训练集化合物数量: {len(train_df[compound_id_col].unique())}, 测试集化合物数量: {len(test_df[compound_id_col].unique())}")
    print(f"训练集样本数: {len(train_df)}, 测试集样本数: {len(test_df)}")

    # 计算训练集和测试集中重叠的蛋白质和化合物
    train_proteins = set(train_df[protein_id_col].unique())
    test_proteins = set(test_df[protein_id_col].unique())
    protein_intersection = train_proteins.intersection(test_proteins)

    train_compounds = set(train_df[compound_id_col].unique())
    test_compounds = set(test_df[compound_id_col].unique())
    compound_intersection = train_compounds.intersection(test_compounds)

    print(f"训练集和测试集共有蛋白质: {len(protein_intersection)}, "
          f"占训练集蛋白质的 {len(protein_intersection) / len(train_proteins):.2%}, "
          f"占测试集蛋白质的 {len(protein_intersection) / len(test_proteins):.2%}")

    print(f"训练集和测试集共有化合物: {len(compound_intersection)}, "
          f"占训练集化合物的 {len(compound_intersection) / len(train_compounds):.2%}, "
          f"占测试集化合物的 {len(compound_intersection) / len(test_compounds):.2%}")

    return train_df, test_df


def create_protein_cold_start_split(df, protein_id_col, compound_id_col, test_size=0.2, random_state=42):
    """创建蛋白质冷启动数据集分割: 确保测试集中的蛋白质在训练集中没有出现过"""
    print("创建蛋白质冷启动数据集分割...")

    # 获取唯一的蛋白质ID
    unique_proteins = df[protein_id_col].unique()

    # 划分蛋白质ID
    protein_train, protein_test = train_test_split(
        unique_proteins,
        test_size=test_size,
        random_state=random_state
    )

    # 创建训练集和测试集
    train_mask = df[protein_id_col].isin(protein_train)
    test_mask = df[protein_id_col].isin(protein_test)

    train_df = df[train_mask]
    test_df = df[test_mask]

    print(
        f"训练集蛋白质数量: {len(train_df[protein_id_col].unique())}, 测试集蛋白质数量: {len(test_df[protein_id_col].unique())}")
    print(
        f"训练集化合物数量: {len(train_df[compound_id_col].unique())}, 测试集化合物数量: {len(test_df[compound_id_col].unique())}")
    print(f"训练集样本数: {len(train_df)}, 测试集样本数: {len(test_df)}")

    # 检查化合物交叉情况
    train_compounds = set(train_df[compound_id_col].unique())
    test_compounds = set(test_df[compound_id_col].unique())
    intersection = train_compounds.intersection(test_compounds)

    print(f"训练集和测试集共有化合物: {len(intersection)}, "
          f"占训练集化合物的 {len(intersection) / len(train_compounds):.2%}, "
          f"占测试集化合物的 {len(intersection) / len(test_compounds):.2%}")

    return train_df, test_df


def create_drug_cold_start_split(df, protein_id_col, compound_id_col, test_size=0.2, random_state=42):
    """创建药物冷启动数据集分割: 确保测试集中的化合物在训练集中没有出现过"""
    print("创建药物冷启动数据集分割...")

    # 获取唯一的化合物ID
    unique_compounds = df[compound_id_col].unique()

    # 划分化合物ID
    compound_train, compound_test = train_test_split(
        unique_compounds,
        test_size=test_size,
        random_state=random_state
    )

    # 创建训练集和测试集
    train_mask = df[compound_id_col].isin(compound_train)
    test_mask = df[compound_id_col].isin(compound_test)

    train_df = df[train_mask]
    test_df = df[test_mask]

    print(
        f"训练集蛋白质数量: {len(train_df[protein_id_col].unique())}, 测试集蛋白质数量: {len(test_df[protein_id_col].unique())}")
    print(
        f"训练集化合物数量: {len(train_df[compound_id_col].unique())}, 测试集化合物数量: {len(test_df[compound_id_col].unique())}")
    print(f"训练集样本数: {len(train_df)}, 测试集样本数: {len(test_df)}")

    # 检查蛋白质交叉情况
    train_proteins = set(train_df[protein_id_col].unique())
    test_proteins = set(test_df[protein_id_col].unique())
    intersection = train_proteins.intersection(test_proteins)

    print(f"训练集和测试集共有蛋白质: {len(intersection)}, "
          f"占训练集蛋白质的 {len(intersection) / len(train_proteins):.2%}, "
          f"占测试集蛋白质的 {len(intersection) / len(test_proteins):.2%}")

    return train_df, test_df


def create_dual_cold_start_split(df, protein_id_col, compound_id_col, test_size=0.2, random_state=42):
    """创建双重冷启动数据集分割: 确保测试集中的蛋白质和化合物在训练集中都没有出现过"""
    print("创建双重冷启动数据集分割...")

    # 获取唯一的蛋白质ID和化合物ID
    unique_proteins = df[protein_id_col].unique()
    unique_compounds = df[compound_id_col].unique()

    # 划分蛋白质ID
    protein_train, protein_test = train_test_split(
        unique_proteins,
        test_size=test_size,
        random_state=random_state
    )

    # 划分化合物ID
    compound_train, compound_test = train_test_split(
        unique_compounds,
        test_size=test_size,
        random_state=random_state
    )

    # 创建训练集和测试集
    train_mask = df[protein_id_col].isin(protein_train) & df[compound_id_col].isin(compound_train)
    test_mask = df[protein_id_col].isin(protein_test) & df[compound_id_col].isin(compound_test)

    train_df = df[train_mask]
    test_df = df[test_mask]

    # 确保训练集和测试集不为空
    if len(train_df) == 0 or len(test_df) == 0:
        print("警告: 双重冷启动分割导致训练集或测试集为空，调整随机数种子")

        # 尝试不同的随机数种子
        for new_seed in range(10):
            protein_train, protein_test = train_test_split(
                unique_proteins,
                test_size=test_size,
                random_state=random_state + new_seed
            )

            compound_train, compound_test = train_test_split(
                unique_compounds,
                test_size=test_size,
                random_state=random_state + new_seed
            )

            train_mask = df[protein_id_col].isin(protein_train) & df[compound_id_col].isin(compound_train)
            test_mask = df[protein_id_col].isin(protein_test) & df[compound_id_col].isin(compound_test)

            train_df = df[train_mask]
            test_df = df[test_mask]

            if len(train_df) > 0 and len(test_df) > 0:
                print(f"使用随机种子 {random_state + new_seed} 成功分割数据")
                break

        if len(train_df) == 0 or len(test_df) == 0:
            print("警告: 无法找到合适的分割，将使用随机分割")
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    print(
        f"训练集蛋白质数量: {len(train_df[protein_id_col].unique())}, 测试集蛋白质数量: {len(test_df[protein_id_col].unique())}")
    print(
        f"训练集化合物数量: {len(train_df[compound_id_col].unique())}, 测试集化合物数量: {len(test_df[compound_id_col].unique())}")
    print(f"训练集样本数: {len(train_df)}, 测试集样本数: {len(test_df)}")

    # 验证双重冷启动条件
    train_proteins = set(train_df[protein_id_col].unique())
    test_proteins = set(test_df[protein_id_col].unique())
    protein_intersection = train_proteins.intersection(test_proteins)

    train_compounds = set(train_df[compound_id_col].unique())
    test_compounds = set(test_df[compound_id_col].unique())
    compound_intersection = train_compounds.intersection(test_compounds)

    print(f"训练集和测试集交叉蛋白质数量: {len(protein_intersection)}")
    print(f"训练集和测试集交叉化合物数量: {len(compound_intersection)}")

    if len(protein_intersection) > 0 or len(compound_intersection) > 0:
        print("警告: 双重冷启动分割未能完全分离蛋白质和化合物")

    return train_df, test_df


# ======================== 修改的特征工程部分 ========================

def supervised_feature_importance_ranking(X_train, y_train, feature_names, protein_cols, compound_cols):
    """
    使用有监督方法对特征进行重要性排名
    结合多种方法：随机森林重要性、互信息、F检验
    """
    print("执行有监督特征重要性排名...")

    # 确保数据清洁
    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)

    importance_scores = {}

    # 1. 随机森林特征重要性
    try:
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            max_depth=10
        )
        rf.fit(X_train, y_train)
        rf_importance = rf.feature_importances_

        # 标准化到0-1范围
        rf_importance = (rf_importance - rf_importance.min()) / (rf_importance.max() - rf_importance.min() + 1e-10)
        importance_scores['random_forest'] = rf_importance
        print(f"随机森林重要性计算完成")
    except Exception as e:
        print(f"随机森林重要性计算失败: {e}")
        importance_scores['random_forest'] = np.ones(len(feature_names))

    # 2. 互信息
    try:
        mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
        # 标准化到0-1范围
        mi_scores = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-10)
        importance_scores['mutual_info'] = mi_scores
        print(f"互信息重要性计算完成")
    except Exception as e:
        print(f"互信息计算失败: {e}")
        importance_scores['mutual_info'] = np.ones(len(feature_names))

    # 3. F检验
    try:
        f_scores, _ = f_classif(X_train, y_train)
        # 处理可能的NaN值
        f_scores = np.nan_to_num(f_scores, nan=0.0)
        # 标准化到0-1范围
        f_scores = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-10)
        importance_scores['f_test'] = f_scores
        print(f"F检验重要性计算完成")
    except Exception as e:
        print(f"F检验计算失败: {e}")
        importance_scores['f_test'] = np.ones(len(feature_names))

    # 4. 组合多种重要性分数（加权平均）
    weights = {'random_forest': 0.5, 'mutual_info': 0.3, 'f_test': 0.2}
    combined_importance = np.zeros(len(feature_names))

    for method, weight in weights.items():
        if method in importance_scores:
            combined_importance += weight * importance_scores[method]

    # 按重要性排序
    importance_indices = np.argsort(combined_importance)[::-1]
    ranked_features = [feature_names[i] for i in importance_indices]
    ranked_importance = combined_importance[importance_indices]

    print(f"特征重要性排名完成，前10重要特征: {ranked_features[:10]}")

    return ranked_features, ranked_importance, importance_scores


def supervised_dimensionality_reduction(X_train, X_test, y_train, n_components, method="auto"):
    """
    使用有监督降维方法
    """
    print(f"执行有监督降维，目标维度: {n_components}, 方法: {method}")

    # 确保数据清洁
    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)

    # 自动选择降维方法
    if method == "auto":
        n_classes = len(np.unique(y_train))
        n_samples, n_features = X_train.shape

        # 根据数据特性选择方法
        if n_classes > 1 and n_components < n_classes and has_supervised_reduction:
            method = "lda"
        elif n_features > n_samples:
            method = "pca"
        else:
            method = "lda" if has_supervised_reduction else "pca"

        print(f"自动选择降维方法: {method}")

    reducer = None
    X_train_reduced = X_train
    X_test_reduced = X_test

    try:
        if method == "lda" and has_supervised_reduction:
            # 线性判别分析（有监督）
            n_classes = len(np.unique(y_train))
            max_components = min(n_components, n_classes - 1, X_train.shape[1])

            if max_components > 0:
                reducer = LDA(n_components=max_components)
                X_train_reduced = reducer.fit_transform(X_train, y_train)
                X_test_reduced = reducer.transform(X_test)
                print(f"LDA降维完成: {X_train.shape[1]} -> {max_components}")
            else:
                print("LDA降维条件不满足，跳过降维")

        elif method == "pca":
            # 主成分分析（无监督，但在有监督特征选择后使用）
            max_components = min(n_components, X_train.shape[1], X_train.shape[0])

            if max_components > 0 and max_components < X_train.shape[1]:
                reducer = PCA(n_components=max_components, random_state=RANDOM_STATE)
                X_train_reduced = reducer.fit_transform(X_train)
                X_test_reduced = reducer.transform(X_test)

                # 计算解释方差比例
                explained_variance_ratio = np.sum(reducer.explained_variance_ratio_)
                print(f"PCA降维完成: {X_train.shape[1]} -> {max_components}, 保留方差: {explained_variance_ratio:.3f}")
            else:
                print("PCA降维条件不满足，跳过降维")
        else:
            print(f"降维方法 {method} 不可用或不支持，跳过降维")

    except Exception as e:
        print(f"降维过程出错: {e}，跳过降维")
        reducer = None
        X_train_reduced = X_train
        X_test_reduced = X_test

    return X_train_reduced, X_test_reduced, reducer


def improved_enhance_features(X, protein_cols, compound_cols):
    """改进的特征增强函数，更加保守和有针对性"""
    if not FEATURE_ENHANCEMENT:
        return X

    print("执行改进的特征增强...")
    X_enhanced = X.copy()

    # 确保数据清洁
    X_enhanced = X_enhanced.fillna(0)

    # 1. 基础统计特征（更保守）
    try:
        # 蛋白质特征统计
        protein_data = X[protein_cols].fillna(0)
        if len(protein_cols) > 0:
            X_enhanced['protein_mean'] = protein_data.mean(axis=1)
            X_enhanced['protein_std'] = protein_data.std(axis=1).fillna(0)
            X_enhanced['protein_max'] = protein_data.max(axis=1)
            X_enhanced['protein_min'] = protein_data.min(axis=1)
            X_enhanced['protein_median'] = protein_data.median(axis=1)
            print(f"添加了5个蛋白质统计特征")

        # 化合物特征统计
        compound_data = X[compound_cols].fillna(0)
        if len(compound_cols) > 0:
            X_enhanced['compound_mean'] = compound_data.mean(axis=1)
            X_enhanced['compound_std'] = compound_data.std(axis=1).fillna(0)
            X_enhanced['compound_max'] = compound_data.max(axis=1)
            X_enhanced['compound_min'] = compound_data.min(axis=1)
            X_enhanced['compound_median'] = compound_data.median(axis=1)
            print(f"添加了5个化合物统计特征")

    except Exception as e:
        print(f"统计特征生成失败: {e}")

    # 2. 只对高方差特征进行交互
    try:
        if len(protein_cols) > 0 and len(compound_cols) > 0:
            # 计算特征方差，选择top特征进行交互
            protein_vars = protein_data.var()
            compound_vars = compound_data.var()

            # 选择方差最大的前3个特征
            top_protein_features = protein_vars.nlargest(min(3, len(protein_cols))).index.tolist()
            top_compound_features = compound_vars.nlargest(min(3, len(compound_cols))).index.tolist()

            # 蛋白质-化合物交叉特征（仅限top特征）
            interaction_count = 0
            for i, p_col in enumerate(top_protein_features[:2]):
                for j, c_col in enumerate(top_compound_features[:2]):
                    try:
                        interaction_name = f"interact_{i}_{j}"
                        X_enhanced[interaction_name] = X[p_col] * X[c_col]
                        interaction_count += 1
                    except:
                        continue

            print(f"添加了{interaction_count}个交互特征")
    except Exception as e:
        print(f"交互特征生成失败: {e}")

    # 3. 比率特征（仅对非零均值）
    try:
        if 'protein_mean' in X_enhanced.columns and 'compound_mean' in X_enhanced.columns:
            protein_mean = X_enhanced['protein_mean']
            compound_mean = X_enhanced['compound_mean']

            # 避免除零
            safe_compound_mean = compound_mean.replace(0, np.nan)
            safe_protein_mean = protein_mean.replace(0, np.nan)

            X_enhanced['protein_to_compound_ratio'] = (protein_mean / safe_compound_mean).fillna(0)
            X_enhanced['compound_to_protein_ratio'] = (compound_mean / safe_protein_mean).fillna(0)

            print("添加了2个比率特征")
    except Exception as e:
        print(f"比率特征生成失败: {e}")

    # 4. 差值特征
    try:
        if 'protein_mean' in X_enhanced.columns and 'compound_mean' in X_enhanced.columns:
            X_enhanced['mean_difference'] = X_enhanced['protein_mean'] - X_enhanced['compound_mean']
            X_enhanced['std_difference'] = X_enhanced['protein_std'] - X_enhanced['compound_std']
            print("添加了2个差值特征")
    except Exception as e:
        print(f"差值特征生成失败: {e}")

    # 5. 清理和验证
    X_enhanced = X_enhanced.fillna(0)
    X_enhanced = X_enhanced.replace([np.inf, -np.inf], 0)

    # 移除方差为0的特征
    try:
        feature_vars = X_enhanced.var()
        zero_var_features = feature_vars[feature_vars == 0].index.tolist()
        if zero_var_features:
            X_enhanced = X_enhanced.drop(columns=zero_var_features)
            print(f"移除了{len(zero_var_features)}个零方差特征")
    except Exception as e:
        print(f"零方差特征移除失败: {e}")

    print(f"特征增强完成: {X.shape[1]} -> {X_enhanced.shape[1]}")
    return X_enhanced


def improved_feature_selection_with_supervised_reduction(X_train, y_train, X_test, feature_names,
                                                         protein_cols, compound_cols, max_features=100):
    """
    改进的特征选择，集成有监督重要性排名和降维（修复版本）
    """
    if not FEATURE_SELECTION and not ENABLE_SUPERVISED_REDUCTION:
        return X_train, X_test, feature_names, None

    print("执行改进的特征选择和有监督降维...")

    # 确保数据清洁
    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)

    selected_features = feature_names.copy()
    reducer = None

    # 1. 方差筛选（基础过滤）- 修复版本
    if FEATURE_SELECTION:
        try:
            print(f"开始方差筛选，初始特征数: {len(selected_features)}")

            # 确保X_train和X_test使用相同的特征列
            if isinstance(X_train, pd.DataFrame):
                X_train_values = X_train.values
                X_test_values = X_test.values
                current_features = X_train.columns.tolist()
            else:
                X_train_values = X_train
                X_test_values = X_test
                current_features = selected_features

            # 计算方差
            feature_vars = np.var(X_train_values, axis=0)

            if len(feature_vars) > 10:
                variance_threshold = np.percentile(feature_vars, 10)  # 保留方差前90%的特征
            else:
                variance_threshold = 0.01

            # 方差筛选
            selector = VarianceThreshold(threshold=variance_threshold)
            X_train_var = selector.fit_transform(X_train_values)
            X_test_var = selector.transform(X_test_values)

            # 获取保留的特征
            var_mask = selector.get_support()
            var_features = [current_features[i] for i in range(len(current_features)) if var_mask[i]]

            print(f"方差筛选保留特征: {len(var_features)}/{len(current_features)}")

            # 更新特征和数据
            selected_features = var_features
            X_train = pd.DataFrame(X_train_var, columns=selected_features,
                                   index=X_train.index if isinstance(X_train, pd.DataFrame) else None)
            X_test = pd.DataFrame(X_test_var, columns=selected_features,
                                  index=X_test.index if isinstance(X_test, pd.DataFrame) else None)

        except Exception as e:
            print(f"方差筛选失败: {e}")
            print("继续使用原始特征...")

    # 2. 有监督特征重要性排名
    if FEATURE_SELECTION and len(selected_features) > max_features:
        try:
            print(f"开始有监督重要性排名，当前特征数: {len(selected_features)}")

            # 确保使用正确的数据进行重要性计算
            X_train_for_importance = X_train.values if isinstance(X_train, pd.DataFrame) else X_train

            ranked_features, ranked_importance, importance_details = supervised_feature_importance_ranking(
                X_train_for_importance, y_train, selected_features, protein_cols, compound_cols
            )

            # 根据重要性选择特征
            if ENABLE_SUPERVISED_REDUCTION:
                # 如果启用降维，先选择更多特征用于降维
                importance_selection_count = int(len(ranked_features) * IMPORTANCE_SELECTION_RATIO)
                importance_selection_count = max(importance_selection_count, FINAL_DIMENSION * 2)
                importance_selection_count = min(importance_selection_count, len(ranked_features))
            else:
                # 如果不降维，直接选择最终数量
                importance_selection_count = min(max_features, len(ranked_features))

            selected_by_importance = ranked_features[:importance_selection_count]
            print(f"基于重要性选择特征: {len(selected_by_importance)}")

            # 更新特征和数据 - 安全方式
            try:
                if isinstance(X_train, pd.DataFrame):
                    X_train = X_train[selected_by_importance]
                    X_test = X_test[selected_by_importance]
                else:
                    # 如果是numpy数组，需要找到特征索引
                    feature_indices = [selected_features.index(f) for f in selected_by_importance if
                                       f in selected_features]
                    X_train = X_train[:, feature_indices]
                    X_test = X_test[:, feature_indices]

                selected_features = selected_by_importance

            except Exception as inner_e:
                print(f"更新特征数据时出错: {inner_e}")
                print("使用截取方式更新特征...")
                # 备用方案：直接截取
                importance_selection_count = min(importance_selection_count, X_train.shape[1])
                if isinstance(X_train, pd.DataFrame):
                    X_train = X_train.iloc[:, :importance_selection_count]
                    X_test = X_test.iloc[:, :importance_selection_count]
                    selected_features = X_train.columns.tolist()
                else:
                    X_train = X_train[:, :importance_selection_count]
                    X_test = X_test[:, :importance_selection_count]
                    selected_features = selected_features[:importance_selection_count]

        except Exception as e:
            print(f"有监督特征重要性排名失败: {e}")
            print("继续使用当前特征...")

    # 3. 有监督降维
    if ENABLE_SUPERVISED_REDUCTION and len(selected_features) > FINAL_DIMENSION:
        try:
            print(f"开始有监督降维，当前特征数: {len(selected_features)}")

            # 确保数据格式正确
            X_train_for_reduction = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            X_test_for_reduction = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

            X_train_reduced, X_test_reduced, reducer = supervised_dimensionality_reduction(
                X_train_for_reduction, X_test_for_reduction, y_train, FINAL_DIMENSION, REDUCTION_METHOD
            )

            if reducer is not None and X_train_reduced.shape[1] < len(selected_features):
                # 如果降维成功，更新数据和特征名
                X_train = X_train_reduced
                X_test = X_test_reduced

                # 生成降维后的特征名
                if REDUCTION_METHOD == "lda" or (REDUCTION_METHOD == "auto" and hasattr(reducer, 'scalings_')):
                    selected_features = [f"LDA_component_{i}" for i in range(X_train.shape[1])]
                else:
                    selected_features = [f"PC_{i}" for i in range(X_train.shape[1])]

                print(f"有监督降维完成，最终特征数: {len(selected_features)}")
            else:
                print("降维未执行或未改变特征数量")

        except Exception as e:
            print(f"有监督降维失败: {e}")
            print("继续使用当前特征...")

    # 4. 相关性筛选（如果特征数量仍然过多且没有进行降维）
    if len(selected_features) > max_features and reducer is None:
        try:
            print(f"开始相关性筛选，当前特征数: {len(selected_features)}")

            # 确保数据是DataFrame格式用于相关性计算
            if not isinstance(X_train, pd.DataFrame):
                df_for_corr = pd.DataFrame(X_train, columns=selected_features)
            else:
                df_for_corr = X_train

            corr_matrix = df_for_corr.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            # 移除相关性超过0.95的特征
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            final_features = [f for f in selected_features if f not in to_drop]

            if len(final_features) <= max_features and len(final_features) > 0:
                print(f"相关性筛选移除{len(to_drop)}个高度相关特征")

                # 安全更新特征
                if isinstance(X_train, pd.DataFrame):
                    X_train = X_train[final_features]
                    X_test = X_test[final_features]
                else:
                    feature_indices = [selected_features.index(f) for f in final_features if f in selected_features]
                    X_train = X_train[:, feature_indices]
                    X_test = X_test[:, feature_indices]

                selected_features = final_features

        except Exception as e:
            print(f"相关性筛选失败: {e}")
            print("继续使用当前特征...")

    # 5. 最终截取（如果仍然超过限制）
    if len(selected_features) > max_features:
        print(f"特征数量仍然过多，截取前{max_features}个特征")
        try:
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.iloc[:, :max_features]
                X_test = X_test.iloc[:, :max_features]
                selected_features = X_train.columns.tolist()
            else:
                X_train = X_train[:, :max_features]
                X_test = X_test[:, :max_features]
                selected_features = selected_features[:max_features]
        except Exception as e:
            print(f"最终截取失败: {e}")

    # 6. 最终验证和清理
    if len(selected_features) == 0:
        print("警告: 所有特征选择方法都失败，使用原始特征")
        selected_features = feature_names[:min(30, len(feature_names))]
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.iloc[:, :len(selected_features)]
            X_test = X_test.iloc[:, :len(selected_features)]
        else:
            X_train = X_train[:, :len(selected_features)]
            X_test = X_test[:, :len(selected_features)]

    print(f"最终选择特征数: {len(selected_features)}")
    return X_train, X_test, selected_features, reducer


def supervised_feature_importance_ranking(X_train, y_train, feature_names, protein_cols, compound_cols):
    """
    使用有监督方法对特征进行重要性排名（修复版本）
    结合多种方法：随机森林重要性、互信息、F检验
    """
    print("执行有监督特征重要性排名...")

    # 确保数据清洁和格式正确
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    # 确保特征名数量与数据列数匹配
    if len(feature_names) != X_train.shape[1]:
        print(f"警告: 特征名数量({len(feature_names)})与数据列数({X_train.shape[1]})不匹配")
        feature_names = feature_names[:X_train.shape[1]]

    importance_scores = {}

    # 1. 随机森林特征重要性
    try:
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            max_depth=10
        )
        rf.fit(X_train, y_train)
        rf_importance = rf.feature_importances_

        # 标准化到0-1范围
        if rf_importance.max() > rf_importance.min():
            rf_importance = (rf_importance - rf_importance.min()) / (rf_importance.max() - rf_importance.min())
        else:
            rf_importance = np.ones_like(rf_importance) * 0.5

        importance_scores['random_forest'] = rf_importance
        print(f"随机森林重要性计算完成")
    except Exception as e:
        print(f"随机森林重要性计算失败: {e}")
        importance_scores['random_forest'] = np.ones(len(feature_names)) * 0.5

    # 2. 互信息
    try:
        mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
        # 标准化到0-1范围
        if mi_scores.max() > mi_scores.min():
            mi_scores = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
        else:
            mi_scores = np.ones_like(mi_scores) * 0.5

        importance_scores['mutual_info'] = mi_scores
        print(f"互信息重要性计算完成")
    except Exception as e:
        print(f"互信息计算失败: {e}")
        importance_scores['mutual_info'] = np.ones(len(feature_names)) * 0.5

    # 3. F检验
    try:
        f_scores, _ = f_classif(X_train, y_train)
        # 处理可能的NaN值
        f_scores = np.nan_to_num(f_scores, nan=0.0)
        # 标准化到0-1范围
        if f_scores.max() > f_scores.min():
            f_scores = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min())
        else:
            f_scores = np.ones_like(f_scores) * 0.5

        importance_scores['f_test'] = f_scores
        print(f"F检验重要性计算完成")
    except Exception as e:
        print(f"F检验计算失败: {e}")
        importance_scores['f_test'] = np.ones(len(feature_names)) * 0.5

    # 4. 组合多种重要性分数（加权平均）
    weights = {'random_forest': 0.5, 'mutual_info': 0.3, 'f_test': 0.2}
    combined_importance = np.zeros(len(feature_names))

    for method, weight in weights.items():
        if method in importance_scores:
            combined_importance += weight * importance_scores[method]

    # 按重要性排序
    importance_indices = np.argsort(combined_importance)[::-1]
    ranked_features = [feature_names[i] for i in importance_indices]
    ranked_importance = combined_importance[importance_indices]

    print(f"特征重要性排名完成，前10重要特征: {ranked_features[:10]}")

    return ranked_features, ranked_importance, importance_scores


def supervised_dimensionality_reduction(X_train, X_test, y_train, n_components, method="auto"):
    """
    使用有监督降维方法（修复版本）
    """
    print(f"执行有监督降维，目标维度: {n_components}, 方法: {method}")

    # 确保数据清洁和格式
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # 自动选择降维方法
    if method == "auto":
        n_classes = len(np.unique(y_train))
        n_samples, n_features = X_train.shape

        # 根据数据特性选择方法
        if n_classes > 1 and n_components < n_classes and has_supervised_reduction:
            method = "lda"
        elif n_features > n_samples:
            method = "pca"
        else:
            method = "lda" if has_supervised_reduction else "pca"

        print(f"自动选择降维方法: {method}")

    reducer = None
    X_train_reduced = X_train
    X_test_reduced = X_test

    try:
        if method == "lda" and has_supervised_reduction:
            # 线性判别分析（有监督）
            n_classes = len(np.unique(y_train))
            max_components = min(n_components, n_classes - 1, X_train.shape[1])

            if max_components > 0:
                reducer = LDA(n_components=max_components)
                X_train_reduced = reducer.fit_transform(X_train, y_train)
                X_test_reduced = reducer.transform(X_test)
                print(f"LDA降维完成: {X_train.shape[1]} -> {max_components}")
            else:
                print("LDA降维条件不满足，跳过降维")

        elif method == "pca":
            # 主成分分析（无监督，但在有监督特征选择后使用）
            max_components = min(n_components, X_train.shape[1], X_train.shape[0])

            if max_components > 0 and max_components < X_train.shape[1]:
                reducer = PCA(n_components=max_components, random_state=RANDOM_STATE)
                X_train_reduced = reducer.fit_transform(X_train)
                X_test_reduced = reducer.transform(X_test)

                # 计算解释方差比例
                explained_variance_ratio = np.sum(reducer.explained_variance_ratio_)
                print(f"PCA降维完成: {X_train.shape[1]} -> {max_components}, 保留方差: {explained_variance_ratio:.3f}")
            else:
                print("PCA降维条件不满足，跳过降维")
        else:
            print(f"降维方法 {method} 不可用或不支持，跳过降维")

    except Exception as e:
        print(f"降维过程出错: {e}，跳过降维")
        reducer = None
        X_train_reduced = X_train
        X_test_reduced = X_test

    return X_train_reduced, X_test_reduced, reducer


def improved_feature_scaling(X_train, X_test):
    """改进的特征缩放，更加稳健"""
    print("执行改进的特征缩放...")

    # 处理DataFrame到numpy的转换
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values

    # 处理NaN和inf值
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # 使用SimpleImputer确保数据完整性
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # 尝试不同的缩放方法并选择最佳
    scalers = {
        'robust': RobustScaler(),
        'standard': StandardScaler(),
        'minmax': MinMaxScaler()
    }

    # 尝试添加PowerTransformer
    try:
        if has_power_transformer:
            scalers['power'] = PowerTransformer(method='yeo-johnson')
    except:
        pass

    best_scaler = None
    best_score = float('inf')
    best_train_scaled = None
    best_test_scaled = None

    for name, scaler in scalers.items():
        try:
            train_scaled = scaler.fit_transform(X_train_imputed)
            test_scaled = scaler.transform(X_test_imputed)

            # 处理可能的NaN和inf值
            train_scaled = np.nan_to_num(train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            test_scaled = np.nan_to_num(test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            # 评估缩放质量
            train_std = np.std(train_scaled, axis=0)
            train_mean = np.mean(train_scaled, axis=0)

            std_score = np.mean(np.abs(train_std - 1))
            mean_score = np.mean(np.abs(train_mean))
            quality_score = std_score + mean_score

            print(f"{name}缩放器质量分数: {quality_score:.4f}")

            if quality_score < best_score:
                best_score = quality_score
                best_scaler = scaler
                best_train_scaled = train_scaled
                best_test_scaled = test_scaled

        except Exception as e:
            print(f"{name}缩放器失败: {e}")
            continue

    if best_scaler is None:
        print("警告: 所有缩放器都失败，返回填充后的数据")
        return None, X_train_imputed, X_test_imputed, imputer

    print(f"选择最佳缩放器，质量分数: {best_score:.4f}")
    return best_scaler, best_train_scaled, best_test_scaled, imputer


def optimize_features(X_train, X_test, y_train, protein_cols, compound_cols):
    """改进的特征工程优化流程，集成有监督降维"""
    print("\n=== 开始改进的特征工程优化（集成有监督降维） ===")

    original_feature_count = X_train.shape[1]
    print(f"原始特征数: {original_feature_count}")

    # 1. 特征增强
    X_train_enhanced = improved_enhance_features(X_train, protein_cols, compound_cols)
    X_test_enhanced = improved_enhance_features(X_test, protein_cols, compound_cols)

    # 2. 有监督特征选择和降维
    feature_names = X_train_enhanced.columns.tolist()
    X_train_selected, X_test_selected, selected_features, reducer = improved_feature_selection_with_supervised_reduction(
        X_train_enhanced, y_train, X_test_enhanced, feature_names, protein_cols, compound_cols, MAX_FEATURES
    )

    # 3. 特征缩放
    scaler, X_train_scaled, X_test_scaled, imputer = improved_feature_scaling(
        X_train_selected, X_test_selected
    )

    # 构建特征管道
    feature_pipeline = {
        'protein_cols': protein_cols,
        'compound_cols': compound_cols,
        'selected_features': selected_features,
        'scaler': scaler,
        'imputer': imputer,
        'reducer': reducer,  # 新增：降维器
        'reduction_method': REDUCTION_METHOD if ENABLE_SUPERVISED_REDUCTION else None,
        'original_feature_count': original_feature_count
    }

    final_feature_count = len(selected_features) if isinstance(selected_features, list) else X_train_scaled.shape[1]
    print(f"特征工程完成: {original_feature_count} -> {final_feature_count}")

    if reducer is not None:
        print(f"使用了{REDUCTION_METHOD}降维方法")

    print("=== 特征工程优化完成 ===\n")

    return X_train_scaled, X_test_scaled, feature_pipeline


def get_best_params(X, y, model_type, param_grid):
    """使用网格搜索找到最佳参数"""
    print(f"优化{model_type}模型参数...")

    # 确保X和y不包含NaN值
    X = np.nan_to_num(X, nan=0.0)
    if isinstance(y, pd.Series):
        y = y.fillna(0)

    if model_type == 'SVM':
        if not ENABLE_SVM:
            raise ValueError("SVM模型已被禁用")
        base_model = SVC(probability=True, random_state=RANDOM_STATE)
    elif model_type == '随机森林':
        if not ENABLE_RANDOM_FOREST:
            raise ValueError("随机森林模型已被禁用")
        base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)
    elif model_type == '梯度提升':
        if not ENABLE_GRADIENT_BOOSTING:
            raise ValueError("梯度提升模型已被禁用")
        base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    elif model_type == 'AdaBoost':
        base_model = AdaBoostClassifier(random_state=RANDOM_STATE)
    elif model_type == '逻辑回归':
        if not ENABLE_LOGISTIC_REGRESSION:
            raise ValueError("逻辑回归模型已被禁用")
        base_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=N_JOBS)
    elif model_type == 'K近邻':
        if not ENABLE_KNN:
            raise ValueError("K近邻模型已被禁用")
        base_model = KNeighborsClassifier(n_jobs=N_JOBS)
    elif model_type == '朴素贝叶斯':
        if not ENABLE_NAIVE_BAYES:
            raise ValueError("朴素贝叶斯模型已被禁用")
        base_model = GaussianNB()
    elif model_type == '极端随机树':
        if not ENABLE_EXTRA_TREES:
            raise ValueError("极端随机树模型已被禁用")
        base_model = ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)
    elif model_type == '贝叶斯网络':
        base_model = BernoulliNB()
    elif model_type == '高斯过程':
        if not ENABLE_GAUSSIAN_PROCESS:
            raise ValueError("高斯过程模型已被禁用")
        base_model = GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=RANDOM_STATE, n_jobs=N_JOBS)
    elif model_type == 'XGBoost' and xgb_installed:
        base_model = XGBClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)
    elif model_type == 'LightGBM' and lgbm_installed:
        base_model = LGBMClassifier(
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=N_JOBS,
            min_child_samples=20,
            min_split_gain=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
    elif model_type == 'CatBoost' and catboost_installed:
        base_model = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, thread_count=CPU_COUNT)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    try:
        grid = GridSearchCV(base_model, param_grid, cv=3, scoring='f1', n_jobs=N_JOBS)
        grid.fit(X, y)
        print(f"{model_type}最优参数: {grid.best_params_}")
        return grid.best_estimator_, grid.best_params_
    except Exception as e:
        print(f"网格搜索出错: {e}，使用默认参数")
        base_model.fit(X, y)
        return base_model, {}


# 自定义的投票分类器，替代sklearn的VotingClassifier
class CustomVotingClassifier:
    def __init__(self, estimators, voting='soft'):
        self.estimators = estimators
        self.voting = voting
        self.classes_ = None
        self.named_estimators_ = dict(estimators)

    def get_params(self, deep=True):
        """获取分类器参数，兼容scikit-learn"""
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
        """设置分类器参数，兼容scikit-learn"""
        for key, value in params.items():
            if key == 'estimators':
                self.estimators = value
                self.named_estimators_ = dict(value)
            elif key == 'voting':
                self.voting = value
            else:
                # 处理estimator的参数
                est_name, param_name = key.split('__', 1)
                for i, (name, est) in enumerate(self.estimators):
                    if name == est_name and hasattr(est, 'set_params'):
                        self.estimators[i] = (name, est.set_params(**{param_name: value}))
                        break
        return self

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # 并行训练所有基础分类器
        def train_estimator(name_estimator):
            name, estimator = name_estimator
            try:
                estimator_clone = clone(estimator)
                estimator_clone.fit(X, y)
                return name, estimator_clone
            except Exception as e:
                print(f"训练分类器 {name} 时出错: {e}")
                return name, estimator

        # 对支持并行的分类器单独处理
        parallel_estimators = []
        sequential_estimators = []

        for name, est in self.estimators:
            if hasattr(est, 'n_jobs') or isinstance(est, (RandomForestClassifier, ExtraTreesClassifier)):
                parallel_estimators.append((name, est))
            else:
                sequential_estimators.append((name, est))

        # 顺序训练不支持并行的分类器
        for name, est in sequential_estimators:
            try:
                est.fit(X, y)
            except Exception as e:
                print(f"顺序训练分类器 {name} 时出错: {e}")

        # 并行训练支持并行的分类器
        if parallel_estimators:
            try:
                trained_estimators = Parallel(n_jobs=N_JOBS)(
                    delayed(train_estimator)(name_est) for name_est in parallel_estimators
                )

                # 更新estimators列表
                for name, trained_est in trained_estimators:
                    for i, (est_name, _) in enumerate(self.estimators):
                        if est_name == name:
                            self.estimators[i] = (name, trained_est)
                            break
            except Exception as e:
                print(f"并行训练分类器时出错: {e}")

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
        def get_proba(name_clf):
            name, clf = name_clf
            try:
                return clf.predict_proba(X)
            except Exception as e:
                print(f"获取分类器 {name} 概率时出错: {e}")
                return np.zeros((X.shape[0], len(self.classes_)))

        probas = Parallel(n_jobs=N_JOBS)(
            delayed(get_proba)(name_clf) for name_clf in self.estimators
        )

        return np.asarray(probas)


# 自定义堆叠分类器，替代sklearn的StackingClassifier
class CustomStackingClassifier:
    def __init__(self, estimators, final_estimator, cv=5):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.named_estimators_ = dict(estimators)
        self.classes_ = None

    def get_params(self, deep=True):
        """获取分类器参数，兼容scikit-learn"""
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
        """设置分类器参数，兼容scikit-learn"""
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

        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=RANDOM_STATE)

        # 为每个基础估计器创建元特征
        meta_features = np.zeros((X.shape[0], len(self.estimators) * len(self.classes_)))

        # 并行训练基础分类器
        def train_base_estimator(name_est_fold):
            name, est, (fold_idx, (train_idx, val_idx)) = name_est_fold
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]

            try:
                est_clone = clone(est)
                est_clone.fit(X_train, y_train)
                prob = est_clone.predict_proba(X_val)
                return name, fold_idx, val_idx, prob
            except Exception as e:
                print(f"训练基础分类器 {name} 在折 {fold_idx} 时出错: {e}")
                return name, fold_idx, val_idx, np.zeros((len(val_idx), len(self.classes_)))

        # 准备并行任务
        tasks = []
        for i, (name, est) in enumerate(self.estimators):
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                tasks.append((name, est, (fold_idx, (train_idx, val_idx))))

        # 并行执行训练任务
        try:
            results = Parallel(n_jobs=N_JOBS)(
                delayed(train_base_estimator)(task) for task in tasks
            )

            # 处理结果
            for name, fold_idx, val_idx, prob in results:
                for i, (est_name, _) in enumerate(self.estimators):
                    if name == est_name:
                        meta_features[val_idx, i * len(self.classes_):(i + 1) * len(self.classes_)] = prob
                        break
        except Exception as e:
            print(f"并行训练基础分类器出错: {e}")
            meta_features = np.zeros((X.shape[0], len(self.estimators) * len(self.classes_)))

        # 训练最终版本的基础模型
        for name, est in self.estimators:
            try:
                est.fit(X, y)
            except Exception as e:
                print(f"训练基础分类器 {name} 最终版本时出错: {e}")

        # 训练元分类器
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

        def get_meta_features_for_estimator(name_est):
            name, est = name_est
            try:
                prob = est.predict_proba(X)
                return name, prob
            except Exception as e:
                print(f"获取元特征时分类器 {name} 出错: {e}")
                return name, np.zeros((X.shape[0], len(self.classes_)))

        try:
            results = Parallel(n_jobs=N_JOBS)(
                delayed(get_meta_features_for_estimator)(name_est) for name_est in self.estimators
            )

            for name, prob in results:
                for i, (est_name, _) in enumerate(self.estimators):
                    if name == est_name:
                        meta_features[:, i * len(self.classes_):(i + 1) * len(self.classes_)] = prob
                        break
        except Exception as e:
            print(f"并行获取元特征出错: {e}")
        return meta_features


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, feature_names, output_dir, cold_start_type=""):
    """评估模型性能，包括交叉验证、测试集性能，并保存结果"""
    # 确保数据不包含NaN值和inf值
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    model_name_with_type = f"{model_name}_{cold_start_type}" if cold_start_type else model_name

    # 对于CustomVotingClassifier和CustomStackingClassifier跳过交叉验证
    if isinstance(model, (CustomVotingClassifier, CustomStackingClassifier)):
        print(f"\n{model_name} - {cold_start_type}: 跳过交叉验证，直接评估测试集性能")
        metrics_per_fold = []
        aucs = []
        auprs = []
        fold_df = pd.DataFrame()
    else:
        # 正常的交叉验证流程
        skf = StratifiedKFold(n_splits=CROSS_VALIDATION_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        metrics_per_fold = []
        aucs = []
        auprs = []
        print(f"\n{model_name} - {cold_start_type} 训练集{CROSS_VALIDATION_FOLDS}折交叉验证:")
        fold_details = []

        # 使用并行处理进行交叉验证
        def evaluate_fold(i, train_idx, val_idx):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            y_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]

            try:
                model_clone = clone(model)
                model_clone.fit(X_tr, y_tr)
                y_pred = model_clone.predict(X_val)

                # 计算AUC和AUPR
                try:
                    if hasattr(model_clone, "predict_proba"):
                        y_score = model_clone.predict_proba(X_val)[:, 1]
                    elif hasattr(model_clone, "decision_function"):
                        y_score = model_clone.decision_function(X_val)
                    else:
                        y_score = y_pred
                    auc = roc_auc_score(y_val, y_score)
                    aupr = average_precision_score(y_val, y_score)
                except:
                    auc = 0.5
                    aupr = 0.5

                acc = accuracy_score(y_val, y_pred)
                prec = precision_score(y_val, y_pred, zero_division=0)
                rec = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)

                print(
                    f"折{i + 1}: 准确率={acc:.4f} 精确率={prec:.4f} 召回率={rec:.4f} F1={f1:.4f} AUC={auc:.4f} AUPR={aupr:.4f}")

                return {
                    "i": i,
                    "acc": acc,
                    "prec": prec,
                    "rec": rec,
                    "f1": f1,
                    "auc": auc,
                    "aupr": aupr
                }
            except Exception as e:
                print(f"评估第{i + 1}折时出错，将使用默认指标")
                return {
                    "i": i,
                    "acc": 0,
                    "prec": 0,
                    "rec": 0,
                    "f1": 0,
                    "auc": 0.5,
                    "aupr": 0.5
                }

        # 如果模型或操作支持并行，则采用并行交叉验证
        try:
            if hasattr(model, 'n_jobs') or isinstance(model, (RandomForestClassifier, ExtraTreesClassifier,
                                                              KNeighborsClassifier)):
                fold_results = []
                for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                    fold_results.append(evaluate_fold(i, train_idx, val_idx))
            else:
                # 并行执行所有折
                fold_indices = [(i, train_idx, val_idx) for i, (train_idx, val_idx) in
                                enumerate(skf.split(X_train, y_train))]
                fold_results = Parallel(n_jobs=N_JOBS)(
                    delayed(evaluate_fold)(i, train_idx, val_idx) for i, train_idx, val_idx in fold_indices
                )

            # 整理结果
            fold_results.sort(key=lambda x: x['i'])
            for result in fold_results:
                i = result['i']
                metrics_per_fold.append([result['acc'], result['prec'], result['rec'], result['f1']])
                aucs.append(result['auc'])
                auprs.append(result['aupr'])
                fold_details.append({
                    "折": f"折{i + 1}",
                    "准确率": f"{result['acc']:.4f}",
                    "精确率": f"{result['prec']:.4f}",
                    "召回率": f"{result['rec']:.4f}",
                    "F1": f"{result['f1']:.4f}",
                    "AUC": f"{result['auc']:.4f}",
                    "AUPR": f"{result['aupr']:.4f}"
                })

            # 创建训练集表格数据
            fold_df = pd.DataFrame(fold_details)

            # 添加均值和方差行
            if metrics_per_fold:
                metrics_per_fold = np.array(metrics_per_fold)
                mean_row = {
                    "折": "均值",
                    "准确率": f"{metrics_per_fold[:, 0].mean():.4f}",
                    "精确率": f"{metrics_per_fold[:, 1].mean():.4f}",
                    "召回率": f"{metrics_per_fold[:, 2].mean():.4f}",
                    "F1": f"{metrics_per_fold[:, 3].mean():.4f}",
                    "AUC": f"{np.mean(aucs):.4f}",
                    "AUPR": f"{np.mean(auprs):.4f}"
                }
                std_row = {
                    "折": "标准差",
                    "准确率": f"{metrics_per_fold[:, 0].std():.4f}",
                    "精确率": f"{metrics_per_fold[:, 1].std():.4f}",
                    "召回率": f"{metrics_per_fold[:, 2].std():.4f}",
                    "F1": f"{metrics_per_fold[:, 3].std():.4f}",
                    "AUC": f"{np.std(aucs):.4f}",
                    "AUPR": f"{np.std(auprs):.4f}"
                }

                fold_df = pd.concat([fold_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

                # 保存交叉验证结果
                try:
                    if VISUALIZATION_ENABLED:
                        fold_df.to_csv(os.path.join(output_dir, f'{model_name_with_type}_train_results_table.csv'),
                                       index=False, encoding='utf-8-sig')
                except:
                    pass

                print(f"{model_name} - {cold_start_type} 训练集{CROSS_VALIDATION_FOLDS}折交叉验证均值±方差:")
                print(f"准确率: {metrics_per_fold[:, 0].mean():.4f} ± {metrics_per_fold[:, 0].std():.4f}")
                print(f"精确率: {metrics_per_fold[:, 1].mean():.4f} ± {metrics_per_fold[:, 1].std():.4f}")
                print(f"召回率: {metrics_per_fold[:, 2].mean():.4f} ± {metrics_per_fold[:, 2].std():.4f}")
                print(f"F1分数: {metrics_per_fold[:, 3].mean():.4f} ± {metrics_per_fold[:, 3].std():.4f}")
                print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
                print(f"AUPR: {np.mean(auprs):.4f} ± {np.std(auprs):.4f}")
        except Exception as e:
            print(f"交叉验证过程中出错，但不影响测试集评估: {str(e)}")
            metrics_per_fold = []
            aucs = []
            auprs = []
            fold_df = pd.DataFrame()

    try:
        # 在全部训练集上训练
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test, zero_division=0)
        rec = recall_score(y_test, y_pred_test, zero_division=0)
        f1 = f1_score(y_test, y_pred_test, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred_test)

        # 计算测试集的AUC和AUPR
        try:
            if hasattr(model, "predict_proba"):
                y_score_test = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score_test = model.decision_function(X_test)
            else:
                y_score_test = y_pred_test
            auc_test = roc_auc_score(y_test, y_score_test)
            aupr_test = average_precision_score(y_test, y_score_test)

            # 生成并保存ROC曲线
            if VISUALIZATION_ENABLED:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_test, y_score_test)
                plt.figure(figsize=(10, 8))
                plt.plot(fpr, tpr, lw=2, label=f'ROC曲线 (AUC = {auc_test:.4f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('假阳性率', fontproperties=chinese_font)
                plt.ylabel('真阳性率', fontproperties=chinese_font)
                plt.title(f'{model_name} - {cold_start_type} ROC曲线', fontproperties=chinese_font)
                plt.legend(loc='lower right', prop=chinese_font)
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f'{model_name_with_type}_roc_curve.png'),
                            dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()

                # 生成并保存PR曲线
                precision, recall, _ = precision_recall_curve(y_test, y_score_test)
                plt.figure(figsize=(10, 8))
                plt.plot(recall, precision, lw=2, label=f'PR曲线 (AUPR = {aupr_test:.4f})')
                plt.xlabel('召回率', fontproperties=chinese_font)
                plt.ylabel('精确率', fontproperties=chinese_font)
                plt.title(f'{model_name} - {cold_start_type} 精确率-召回率曲线', fontproperties=chinese_font)
                plt.legend(loc='best', prop=chinese_font)
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f'{model_name_with_type}_pr_curve.png'),
                            dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"计算AUC和AUPR出错: {e}")
            auc_test = 0.5
            aupr_test = 0.5

        # 更详细的测试集结果输出
        print(f"\n{model_name} - {cold_start_type} 测试集详细结果:")
        print("-" * 50)
        print(f"准确率 (Accuracy): {acc:.4f}")
        print(f"精确率 (Precision): {prec:.4f}")
        print(f"召回率 (Recall): {rec:.4f}")
        print(f"F1分数 (F1-Score): {f1:.4f}")
        print(f"AUC (AUROC): {auc_test:.4f}")
        print(f"AUPR: {aupr_test:.4f}")
        print("混淆矩阵:")
        print(f"[[{conf_matrix[0, 0]:4d} {conf_matrix[0, 1]:4d}]")
        print(f" [{conf_matrix[1, 0]:4d} {conf_matrix[1, 1]:4d}]]")

        # 将详细结果保存到文本文件
        try:
            with open(os.path.join(output_dir, f'{model_name_with_type}_test_results.txt'), 'w', encoding='utf-8') as f:
                f.write(f"{model_name} - {cold_start_type} 测试集详细结果:\n")
                f.write("-" * 50 + "\n")
                f.write(f"准确率 (Accuracy): {acc:.4f}\n")
                f.write(f"精确率 (Precision): {prec:.4f}\n")
                f.write(f"召回率 (Recall): {rec:.4f}\n")
                f.write(f"F1分数 (F1-Score): {f1:.4f}\n")
                f.write(f"AUC (AUROC): {auc_test:.4f}\n")
                f.write(f"AUPR: {aupr_test:.4f}\n")
                f.write("混淆矩阵:\n")
                f.write(f"[[{conf_matrix[0, 0]:4d} {conf_matrix[0, 1]:4d}]\n")
                f.write(f" [{conf_matrix[1, 0]:4d} {conf_matrix[1, 1]:4d}]]\n")
        except:
            pass

    except Exception as e:
        print(f"测试集评估出错: {e}")
        acc = prec = rec = f1 = auc_test = aupr_test = 0
        conf_matrix = np.array([[0, 0], [0, 0]])

    # 保存混淆矩阵可视化
    try:
        if VISUALIZATION_ENABLED:
            plt.figure(figsize=(10, 8))
            im = plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
            plt.colorbar()

            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['负例', '正例'], fontproperties=chinese_font)
            plt.yticks(tick_marks, ['负例', '正例'], fontproperties=chinese_font)

            # 添加数值标注
            thresh = conf_matrix.max() / 2
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    plt.text(j, i, format(conf_matrix[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if conf_matrix[i, j] > thresh else "black",
                             fontproperties=chinese_font)

            plt.title(f'{model_name} - {cold_start_type} 混淆矩阵', fontproperties=chinese_font)
            plt.xlabel('预测标签', fontproperties=chinese_font)
            plt.ylabel('真实标签', fontproperties=chinese_font)
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f'{model_name_with_type}_confusion_matrix.png'),
                        dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
    except:
        pass

    return {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc_test,
        "aupr": aupr_test
    }


def create_ensemble_models(models, X_train, y_train, X_test, y_test, feature_pipeline, output_dir, cold_start_type=""):
    """创建并评估集成模型，使用自定义的集成分类器而不是scikit-learn的实现"""
    if not ENABLE_ENSEMBLE_MODELS:
        print("集成模型功能已禁用")
        return {}, {}

    ensemble_models = {}
    ensemble_results = {}

    # 准备基础分类器
    estimators = []
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            estimators.append((name, model))

    if len(estimators) < 2:
        print("可用的基础分类器不足，跳过集成模型")
        return {}, {}

    # 1. 创建投票分类器
    try:
        print("\n创建投票分类器...")
        voting_clf = CustomVotingClassifier(estimators=estimators, voting='soft')
        voting_result = evaluate_model(
            voting_clf, X_train, y_train, X_test, y_test,
            '投票分类器', feature_pipeline['selected_features'], output_dir,
            cold_start_type=cold_start_type
        )
        ensemble_models['投票分类器'] = voting_clf
        ensemble_results['投票分类器'] = voting_result

        # 保存模型
        try:
            if cold_start_type:
                joblib.dump(voting_clf, os.path.join(output_dir, f'投票分类器_{cold_start_type}_model.pkl'))
            else:
                joblib.dump(voting_clf, os.path.join(output_dir, '投票分类器_model.pkl'))
        except Exception as e:
            print(f"保存投票分类器模型出错，但不影响程序继续运行")
    except Exception as e:
        print(f"创建投票分类器过程中遇到错误，但不影响程序继续执行")

    # 2. 创建堆叠分类器
    try:
        print("\n创建堆叠分类器...")
        good_models = []
        for name, model in models.items():
            if len(good_models) < 6 and hasattr(model, "predict_proba"):
                good_models.append((name, model))

        if len(good_models) >= 3:
            stack_clf = CustomStackingClassifier(
                estimators=good_models,
                final_estimator=LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=N_JOBS),
                cv=5
            )
            stack_result = evaluate_model(
                stack_clf, X_train, y_train, X_test, y_test,
                '堆叠分类器', feature_pipeline['selected_features'], output_dir,
                cold_start_type=cold_start_type
            )
            ensemble_models['堆叠分类器'] = stack_clf
            ensemble_results['堆叠分类器'] = stack_result

            # 保存模型
            try:
                if cold_start_type:
                    joblib.dump(stack_clf, os.path.join(output_dir, f'堆叠分类器_{cold_start_type}_model.pkl'))
                else:
                    joblib.dump(stack_clf, os.path.join(output_dir, '堆叠分类器_model.pkl'))
            except Exception as e:
                print(f"保存堆叠分类器模型出错，但不影响程序继续运行")
    except Exception as e:
        print(f"创建堆叠分类器过程中遇到错误，但不影响程序继续执行")

    return ensemble_models, ensemble_results


def train_standard_model_with_checkpoint(checkpoint_manager, df, protein_id_col, compound_id_col, label_col, X,
                                         protein_cols,
                                         compound_cols, output_dir, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """使用断点续传训练标准随机分割模型"""
    task_category = "标准随机分割"

    # 检查任务是否已完成
    if checkpoint_manager.is_task_completed(task_category, "complete"):
        print(f"\n[检查点] {task_category} 已完成，跳过训练")

        # 尝试从保存的结果中恢复数据
        model_dir = os.path.join(output_dir, task_category)
        if os.path.exists(model_dir):
            try:
                # 加载特征管道
                feature_pipeline = joblib.load(os.path.join(model_dir, 'feature_pipeline.pkl'))

                # 重新分割数据
                train_df, test_df = create_standard_split(df, protein_id_col, compound_id_col, test_size, random_state)
                X_train = train_df[X.columns]
                y_train = train_df[label_col]
                X_test = test_df[X.columns]
                y_test = test_df[label_col]

                # 重新应用特征工程
                X_train_scaled, X_test_scaled, _ = optimize_features(X_train, X_test, y_train, protein_cols,
                                                                     compound_cols)

                # 恢复模型字典
                models = {}
                results = {}

                # 获取已训练的模型列表
                completed_models = checkpoint_manager.checkpoint_data.get(task_category, {}).get('completed', [])
                for model_name in completed_models:
                    if model_name != "complete":
                        try:
                            model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
                            if os.path.exists(model_path):
                                models[model_name] = joblib.load(model_path)
                                # 可以添加结果恢复逻辑
                        except Exception as e:
                            print(f"恢复模型 {model_name} 失败: {e}")

                return {
                    'train_df': train_df, 'test_df': test_df,
                    'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
                    'y_train': y_train, 'y_test': y_test,
                    'feature_pipeline': feature_pipeline,
                    'models': models, 'results': results,
                    'best_model': None
                }
            except Exception as e:
                print(f"从检查点恢复 {task_category} 失败，将重新训练: {e}")

        # 如果恢复失败，继续执行原始训练逻辑

    if not ENABLE_STANDARD_RANDOM:
        print("标准随机分割模型已被禁用")
        return {
            'train_df': None, 'test_df': None, 'X_train_scaled': None, 'X_test_scaled': None,
            'y_train': None, 'y_test': None, 'feature_pipeline': None,
            'models': {}, 'results': {}, 'best_model': None
        }

    print(f"\n{'=' * 80}")
    print(f"训练标准随机分割模型（非冷启动）")
    print(f"{'=' * 80}")

    # 1. 创建标准随机分割数据集
    train_df, test_df = create_standard_split(df, protein_id_col, compound_id_col, test_size, random_state)

    # 2. 从分割结果提取特征和标签
    X_train = train_df[X.columns]
    y_train = train_df[label_col]
    X_test = test_df[X.columns]
    y_test = test_df[label_col]

    # 3. 特征工程优化
    X_train_scaled, X_test_scaled, feature_pipeline = optimize_features(X_train, X_test, y_train, protein_cols,
                                                                        compound_cols)

    # 创建子目录保存标准模型的结果
    model_dir = os.path.join(output_dir, task_category)
    os.makedirs(model_dir, exist_ok=True)

    # 保存特征管道
    try:
        joblib.dump(feature_pipeline, os.path.join(model_dir, 'feature_pipeline.pkl'))
    except Exception as e:
        print(f"保存特征管道出错，但不影响程序继续运行：{e}")

    # 4. 准备模型参数网格
    param_grids = {}
    model_list = []

    if ENABLE_SVM:
        param_grids['SVM'] = {'C': [1.0], 'gamma': ['scale'], 'kernel': ['rbf']}
        model_list.append('SVM')

    if ENABLE_RANDOM_FOREST:
        param_grids['随机森林'] = {'n_estimators': [100], 'max_depth': [None], 'n_jobs': [N_JOBS]}
        model_list.append('随机森林')

    if ENABLE_GRADIENT_BOOSTING:
        param_grids['梯度提升'] = {'n_estimators': [100], 'learning_rate': [0.1]}
        model_list.append('梯度提升')

    if ENABLE_LOGISTIC_REGRESSION:
        param_grids['逻辑回归'] = {'C': [1.0], 'penalty': ['l2'], 'n_jobs': [N_JOBS]}
        model_list.append('逻辑回归')

    if ENABLE_KNN:
        param_grids['K近邻'] = {'n_neighbors': [5], 'weights': ['uniform'], 'n_jobs': [N_JOBS]}
        model_list.append('K近邻')

    if ENABLE_EXTRA_TREES:
        param_grids['极端随机树'] = {'n_estimators': [100], 'max_depth': [None], 'n_jobs': [N_JOBS]}
        model_list.append('极端随机树')

    if ENABLE_NAIVE_BAYES:
        param_grids['朴素贝叶斯'] = {'var_smoothing': [1e-9]}
        model_list.append('朴素贝叶斯')

    if ENABLE_GAUSSIAN_PROCESS:
        param_grids['高斯过程'] = {'kernel': [1.0 * RBF(1.0)], 'n_jobs': [N_JOBS]}
        model_list.append('高斯过程')

    # 添加可选模型
    if xgb_installed:
        param_grids['XGBoost'] = {
            'n_estimators': [50], 'learning_rate': [0.05], 'max_depth': [4],
            'min_child_weight': [2], 'subsample': [0.8], 'colsample_bytree': [0.8], 'n_jobs': [N_JOBS]
        }
        model_list.append('XGBoost')

    if lgbm_installed:
        param_grids['LightGBM'] = {
            'n_estimators': [50], 'learning_rate': [0.05], 'num_leaves': [15], 'max_depth': [4],
            'min_child_samples': [20], 'min_split_gain': [0.1], 'reg_alpha': [0.1], 'reg_lambda': [0.1],
            'verbose': [-1], 'n_jobs': [N_JOBS]
        }
        model_list.append('LightGBM')

    if catboost_installed:
        param_grids['CatBoost'] = {
            'iterations': [100], 'learning_rate': [0.1], 'depth': [6],
            'verbose': [0], 'thread_count': [CPU_COUNT]
        }
        model_list.append('CatBoost')

    # 设置总任务数（包括集成模型）
    total_tasks = len(model_list) + (2 if ENABLE_ENSEMBLE_MODELS and len(model_list) >= 2 else 0) + 1  # +1 for complete
    checkpoint_manager.set_total_tasks(task_category, total_tasks)

    # 获取剩余未完成的模型
    remaining_models = checkpoint_manager.get_remaining_tasks(task_category, model_list)

    # 5. 训练和评估模型
    models = {}
    results = {}

    # 训练剩余的模型
    def train_and_evaluate_model_with_checkpoint(model_name, param_grid):
        if checkpoint_manager.is_task_completed(task_category, model_name):
            print(f"[检查点] {model_name} 已完成，跳过训练")
            # 尝试加载已保存的模型
            try:
                model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    return (model_name, model, None)  # 这里可以改进以包含评估结果
            except Exception as e:
                print(f"加载已保存的模型 {model_name} 失败: {e}")
            return (model_name, None, None)

        print(f"\n训练和优化模型: {model_name} - 标准随机分割")

        try:
            # 获取最佳参数
            best_model, best_params = get_best_params(X_train_scaled, y_train, model_name, param_grid)

            # 评估模型
            result = evaluate_model(
                best_model, X_train_scaled, y_train, X_test_scaled, y_test,
                model_name, feature_pipeline['selected_features'], model_dir,
                cold_start_type="标准随机分割"
            )

            # 保存模型
            try:
                joblib.dump(best_model, os.path.join(model_dir, f'{model_name}_model.pkl'))
                print(f"{model_name}模型已保存至 {os.path.join(model_dir, f'{model_name}_model.pkl')}")
            except Exception as e:
                print(f"保存{model_name}模型出错，但不影响程序继续运行")

            # 标记任务完成
            checkpoint_manager.mark_task_completed(
                task_category, model_name,
                model_path=os.path.join(model_dir, f'{model_name}_model.pkl'),
                f1_score=result['f1'],
                accuracy=result['accuracy']
            )

            return (model_name, best_model, result)
        except Exception as e:
            print(f"训练和评估{model_name}时出错: {str(e)}")
            return (model_name, None, None)

    # 首先恢复已完成的模型
    completed_models = [m for m in model_list if m not in remaining_models]
    for model_name in completed_models:
        try:
            model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                # 这里可以恢复results，但需要额外的存储机制
                print(f"[检查点] 已恢复模型: {model_name}")
        except Exception as e:
            print(f"恢复已完成模型 {model_name} 失败: {e}")

    # 训练剩余的模型
    if remaining_models:
        print(f"\n需要训练的剩余模型: {remaining_models}")

        # 处理SVM单独训练（不适合并行）
        svm_result = None
        if 'SVM' in remaining_models:
            svm_result = train_and_evaluate_model_with_checkpoint('SVM', param_grids['SVM'])
            remaining_models.remove('SVM')

        # 并行训练其他模型
        if remaining_models:
            try:
                model_results = Parallel(n_jobs=min(len(remaining_models), CPU_COUNT))(
                    delayed(train_and_evaluate_model_with_checkpoint)(model_name, param_grids[model_name])
                    for model_name in remaining_models
                )
            except Exception as e:
                print(f"并行训练模型出错: {e}")
                model_results = []
                # 尝试顺序执行
                for model_name in remaining_models:
                    try:
                        result = train_and_evaluate_model_with_checkpoint(model_name, param_grids[model_name])
                        model_results.append(result)
                    except Exception as e:
                        print(f"顺序训练{model_name}出错: {e}")

            # 处理结果
            if svm_result:
                model_results.append(svm_result)

            for model_name, model, result in model_results:
                if model is not None and result is not None:
                    models[model_name] = model
                    results[model_name] = result

    # 6. 创建集成模型
    if len(models) >= 2 and ENABLE_ENSEMBLE_MODELS:
        ensemble_tasks = ['投票分类器', '堆叠分类器']
        remaining_ensemble = checkpoint_manager.get_remaining_tasks(task_category, ensemble_tasks)

        if remaining_ensemble:
            ensemble_models, ensemble_results = create_ensemble_models(
                models, X_train_scaled, y_train, X_test_scaled, y_test,
                feature_pipeline, model_dir, cold_start_type="标准随机分割"
            )

            # 标记集成模型完成
            for ensemble_name in ensemble_models:
                checkpoint_manager.mark_task_completed(
                    task_category, ensemble_name,
                    model_path=os.path.join(model_dir, f'{ensemble_name}_model.pkl'),
                    f1_score=ensemble_results[ensemble_name]['f1'] if ensemble_name in ensemble_results else 0
                )

            # 合并模型和结果
            models.update(ensemble_models)
            results.update(ensemble_results)
        else:
            print("[检查点] 集成模型已完成，跳过创建")

    # 标记整个任务类别完成
    checkpoint_manager.mark_task_completed(task_category, "complete")

    # 7. 生成测试集模型对比汇总表
    if results:
        summary_data = []
        for name, result in results.items():
            summary_data.append({
                "模型": name,
                "准确率": f"{result['accuracy']:.4f}",
                "精确率": f"{result['precision']:.4f}",
                "召回率": f"{result['recall']:.4f}",
                "F1分数": f"{result['f1']:.4f}",
                "AUC": f"{result['auc']:.4f}",
                "AUPR": f"{result['aupr']:.4f}"
            })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            try:
                summary_path = os.path.join(model_dir, 'model_comparison.csv')
                summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            except Exception as e:
                print(f"保存模型比较表出错，但不影响程序继续运行")

            # 打印最佳模型信息
            try:
                best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
                best_model_metrics = next(item for item in summary_data if item["模型"] == best_model_name)
                print(f"\n标准随机分割 - 测试集上表现最好的模型是: {best_model_name}")
                print(f"  F1分数: {best_model_metrics['F1分数']}")
                print(f"  准确率: {best_model_metrics['准确率']}")
                print(f"  AUC: {best_model_metrics['AUC']}")
                print(f"  AUPR: {best_model_metrics['AUPR']}")
                print("=" * 80)
            except Exception as e:
                print(f"计算最佳模型出错: {e}")
                best_model_name = None if not results else list(results.keys())[0]

    # 返回结果
    return {
        'train_df': train_df,
        'test_df': test_df,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_pipeline': feature_pipeline,
        'models': models,
        'results': results,
        'best_model': best_model_name if 'best_model_name' in locals() else None
    }


def train_cold_start_model_with_checkpoint(checkpoint_manager, cold_start_type, df, protein_id_col, compound_id_col,
                                           label_col,
                                           X, protein_cols, compound_cols, output_dir, test_size=TEST_SIZE,
                                           random_state=RANDOM_STATE):
    """使用断点续传训练特定类型的冷启动模型"""
    task_category = cold_start_type

    # 检查任务是否已完成
    if checkpoint_manager.is_task_completed(task_category, "complete"):
        print(f"\n[检查点] {task_category} 已完成，跳过训练")

        # 尝试从保存的结果中恢复数据
        model_dir = os.path.join(output_dir, task_category)
        if os.path.exists(model_dir):
            try:
                # 这里可以实现完整的恢复逻辑
                # 为了简化，返回基本结构
                return {
                    'train_df': None, 'test_df': None, 'X_train_scaled': None, 'X_test_scaled': None,
                    'y_train': None, 'y_test': None, 'feature_pipeline': None,
                    'models': {}, 'results': {}, 'best_model': None
                }
            except Exception as e:
                print(f"从检查点恢复 {task_category} 失败，将重新训练: {e}")

    print(f"\n{'=' * 80}")
    print(f"训练 {cold_start_type} 模型")
    print(f"{'=' * 80}")

    # 检查是否启用对应的冷启动类型
    if cold_start_type == "蛋白质冷启动" and not ENABLE_PROTEIN_COLD_START:
        print("蛋白质冷启动模型已被禁用")
        return {
            'train_df': None, 'test_df': None, 'X_train_scaled': None, 'X_test_scaled': None,
            'y_train': None, 'y_test': None, 'feature_pipeline': None,
            'models': {}, 'results': {}, 'best_model': None
        }
    elif cold_start_type == "药物冷启动" and not ENABLE_DRUG_COLD_START:
        print("药物冷启动模型已被禁用")
        return {
            'train_df': None, 'test_df': None, 'X_train_scaled': None, 'X_test_scaled': None,
            'y_train': None, 'y_test': None, 'feature_pipeline': None,
            'models': {}, 'results': {}, 'best_model': None
        }
    elif cold_start_type == "双重冷启动" and not ENABLE_DUAL_COLD_START:
        print("双重冷启动模型已被禁用")
        return {
            'train_df': None, 'test_df': None, 'X_train_scaled': None, 'X_test_scaled': None,
            'y_train': None, 'y_test': None, 'feature_pipeline': None,
            'models': {}, 'results': {}, 'best_model': None
        }

    # 1. 根据冷启动类型分割数据
    if cold_start_type == "蛋白质冷启动":
        train_df, test_df = create_protein_cold_start_split(df, protein_id_col, compound_id_col, test_size,
                                                            random_state)
    elif cold_start_type == "药物冷启动":
        train_df, test_df = create_drug_cold_start_split(df, protein_id_col, compound_id_col, test_size, random_state)
    elif cold_start_type == "双重冷启动":
        train_df, test_df = create_dual_cold_start_split(df, protein_id_col, compound_id_col, test_size, random_state)
    else:
        raise ValueError(f"不支持的冷启动类型: {cold_start_type}")

    # 2. 从分割结果提取特征和标签
    X_train = train_df[X.columns]
    y_train = train_df[label_col]
    X_test = test_df[X.columns]
    y_test = test_df[label_col]

    # 3. 特征工程优化
    X_train_scaled, X_test_scaled, feature_pipeline = optimize_features(X_train, X_test, y_train, protein_cols,
                                                                        compound_cols)

    # 创建子目录保存当前冷启动类型的结果
    cold_start_dir = os.path.join(output_dir, cold_start_type)
    os.makedirs(cold_start_dir, exist_ok=True)

    # 保存特征管道
    try:
        joblib.dump(feature_pipeline, os.path.join(cold_start_dir, 'feature_pipeline.pkl'))
    except Exception as e:
        print(f"保存特征管道出错，但不影响程序继续运行: {e}")

    # 4. 准备模型参数网格
    param_grids = {}
    model_list = []

    if ENABLE_SVM:
        param_grids['SVM'] = {'C': [1.0], 'gamma': ['scale'], 'kernel': ['rbf']}
        model_list.append('SVM')

    if ENABLE_RANDOM_FOREST:
        param_grids['随机森林'] = {'n_estimators': [100], 'max_depth': [None], 'n_jobs': [N_JOBS]}
        model_list.append('随机森林')

    if ENABLE_GRADIENT_BOOSTING:
        param_grids['梯度提升'] = {'n_estimators': [100], 'learning_rate': [0.1]}
        model_list.append('梯度提升')

    if ENABLE_LOGISTIC_REGRESSION:
        param_grids['逻辑回归'] = {'C': [1.0], 'penalty': ['l2'], 'n_jobs': [N_JOBS]}
        model_list.append('逻辑回归')

    if ENABLE_KNN:
        param_grids['K近邻'] = {'n_neighbors': [5], 'weights': ['uniform'], 'n_jobs': [N_JOBS]}
        model_list.append('K近邻')

    if ENABLE_EXTRA_TREES:
        param_grids['极端随机树'] = {'n_estimators': [100], 'max_depth': [None], 'n_jobs': [N_JOBS]}
        model_list.append('极端随机树')

    if ENABLE_NAIVE_BAYES:
        param_grids['朴素贝叶斯'] = {'var_smoothing': [1e-9]}
        model_list.append('朴素贝叶斯')

    if ENABLE_GAUSSIAN_PROCESS:
        param_grids['高斯过程'] = {'kernel': [1.0 * RBF(1.0)], 'n_jobs': [N_JOBS]}
        model_list.append('高斯过程')

    # 添加可选模型
    if xgb_installed:
        param_grids['XGBoost'] = {
            'n_estimators': [50], 'learning_rate': [0.05], 'max_depth': [4],
            'min_child_weight': [2], 'subsample': [0.8], 'colsample_bytree': [0.8], 'n_jobs': [N_JOBS]
        }
        model_list.append('XGBoost')

    if lgbm_installed:
        param_grids['LightGBM'] = {
            'n_estimators': [50], 'learning_rate': [0.05], 'num_leaves': [15], 'max_depth': [4],
            'min_child_samples': [20], 'min_split_gain': [0.1], 'reg_alpha': [0.1], 'reg_lambda': [0.1],
            'verbose': [-1], 'n_jobs': [N_JOBS]
        }
        model_list.append('LightGBM')

    if catboost_installed:
        param_grids['CatBoost'] = {
            'iterations': [100], 'learning_rate': [0.1], 'depth': [6],
            'verbose': [0], 'thread_count': [CPU_COUNT]
        }
        model_list.append('CatBoost')

    # 设置总任务数（包括集成模型）
    total_tasks = len(model_list) + (2 if ENABLE_ENSEMBLE_MODELS and len(model_list) >= 2 else 0) + 1  # +1 for complete
    checkpoint_manager.set_total_tasks(task_category, total_tasks)

    # 获取剩余未完成的模型
    remaining_models = checkpoint_manager.get_remaining_tasks(task_category, model_list)

    # 5. 训练和评估模型
    models = {}
    results = {}

    # 训练剩余的模型
    def train_and_evaluate_cold_start_model_with_checkpoint(model_name, param_grid):
        if checkpoint_manager.is_task_completed(task_category, model_name):
            print(f"[检查点] {model_name} 已完成，跳过训练")
            # 尝试加载已保存的模型
            try:
                model_path = os.path.join(cold_start_dir, f'{model_name}_model.pkl')
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    return (model_name, model, None)
            except Exception as e:
                print(f"加载已保存的模型 {model_name} 失败: {e}")
            return (model_name, None, None)

        print(f"\n训练和优化模型: {model_name} - {cold_start_type}")

        try:
            # 获取最佳参数
            best_model, best_params = get_best_params(X_train_scaled, y_train, model_name, param_grid)

            # 评估模型
            result = evaluate_model(
                best_model, X_train_scaled, y_train, X_test_scaled, y_test,
                model_name, feature_pipeline['selected_features'], cold_start_dir,
                cold_start_type=cold_start_type
            )

            # 保存模型
            try:
                joblib.dump(best_model, os.path.join(cold_start_dir, f'{model_name}_model.pkl'))
                print(f"{model_name}模型已保存至 {os.path.join(cold_start_dir, f'{model_name}_model.pkl')}")
            except Exception as e:
                print(f"保存{model_name}模型出错，但不影响程序继续运行")

            # 标记任务完成
            checkpoint_manager.mark_task_completed(
                task_category, model_name,
                model_path=os.path.join(cold_start_dir, f'{model_name}_model.pkl'),
                f1_score=result['f1'],
                accuracy=result['accuracy']
            )

            return (model_name, best_model, result)
        except Exception as e:
            print(f"训练和评估{model_name}时出错: {str(e)}")
            return (model_name, None, None)

    # 首先恢复已完成的模型
    completed_models = [m for m in model_list if m not in remaining_models]
    for model_name in completed_models:
        try:
            model_path = os.path.join(cold_start_dir, f'{model_name}_model.pkl')
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                print(f"[检查点] 已恢复模型: {model_name}")
        except Exception as e:
            print(f"恢复已完成模型 {model_name} 失败: {e}")

    # 训练剩余的模型
    if remaining_models:
        print(f"\n需要训练的剩余模型: {remaining_models}")

        # 处理SVM单独训练（不适合并行）
        svm_result = None
        if 'SVM' in remaining_models:
            svm_result = train_and_evaluate_cold_start_model_with_checkpoint('SVM', param_grids['SVM'])
            remaining_models.remove('SVM')

        # 并行训练其他模型
        if remaining_models:
            try:
                model_results = Parallel(n_jobs=min(len(remaining_models), CPU_COUNT))(
                    delayed(train_and_evaluate_cold_start_model_with_checkpoint)(model_name, param_grids[model_name])
                    for model_name in remaining_models
                )
            except Exception as e:
                print(f"并行训练模型出错: {e}")
                model_results = []
                # 尝试顺序执行
                for model_name in remaining_models:
                    try:
                        result = train_and_evaluate_cold_start_model_with_checkpoint(model_name,
                                                                                     param_grids[model_name])
                        model_results.append(result)
                    except:
                        print(f"顺序训练{model_name}出错")

            # 处理结果
            if svm_result:
                model_results.append(svm_result)

            for model_name, model, result in model_results:
                if model is not None and result is not None:
                    models[model_name] = model
                    results[model_name] = result

    # 6. 创建集成模型
    if len(models) >= 2 and ENABLE_ENSEMBLE_MODELS:
        ensemble_tasks = ['投票分类器', '堆叠分类器']
        remaining_ensemble = checkpoint_manager.get_remaining_tasks(task_category, ensemble_tasks)

        if remaining_ensemble:
            ensemble_models, ensemble_results = create_ensemble_models(
                models, X_train_scaled, y_train, X_test_scaled, y_test,
                feature_pipeline, cold_start_dir, cold_start_type
            )

            # 标记集成模型完成
            for ensemble_name in ensemble_models:
                checkpoint_manager.mark_task_completed(
                    task_category, ensemble_name,
                    model_path=os.path.join(cold_start_dir, f'{ensemble_name}_model.pkl'),
                    f1_score=ensemble_results[ensemble_name]['f1'] if ensemble_name in ensemble_results else 0
                )

            # 合并模型和结果
            models.update(ensemble_models)
            results.update(ensemble_results)
        else:
            print("[检查点] 集成模型已完成，跳过创建")

    # 标记整个任务类别完成
    checkpoint_manager.mark_task_completed(task_category, "complete")

    # 7. 生成测试集模型对比汇总表
    if results:
        summary_data = []
        for name, result in results.items():
            summary_data.append({
                "模型": name,
                "准确率": f"{result['accuracy']:.4f}",
                "精确率": f"{result['precision']:.4f}",
                "召回率": f"{result['recall']:.4f}",
                "F1分数": f"{result['f1']:.4f}",
                "AUC": f"{result['auc']:.4f}",
                "AUPR": f"{result['aupr']:.4f}"
            })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            try:
                summary_path = os.path.join(cold_start_dir, 'model_comparison.csv')
                summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            except Exception as e:
                print(f"保存模型比较表出错，但不影响程序继续运行")

            # 打印最佳模型信息
            try:
                best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
                best_model_metrics = next(item for item in summary_data if item["模型"] == best_model_name)
                print(f"\n{cold_start_type} - 测试集上表现最好的模型是: {best_model_name}")
                print(f"  F1分数: {best_model_metrics['F1分数']}")
                print(f"  准确率: {best_model_metrics['准确率']}")
                print(f"  AUC: {best_model_metrics['AUC']}")
                print(f"  AUPR: {best_model_metrics['AUPR']}")
                print("=" * 80)
            except Exception as e:
                print(f"计算最佳模型出错: {e}")
                best_model_name = None if not results else list(results.keys())[0]

    # 返回结果
    return {
        'train_df': train_df,
        'test_df': test_df,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_pipeline': feature_pipeline,
        'models': models,
        'results': results,
        'best_model': best_model_name if 'best_model_name' in locals() else None
    }


def compare_cold_start_models(results_dict, output_dir):
    """比较不同冷启动策略的模型性能"""
    print("\n比较不同冷启动策略的性能...")

    cold_start_types = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'aupr']
    metric_names = {'accuracy': '准确率', 'precision': '精确率', 'recall': '召回率', 'f1': 'F1分数',
                    'auc': 'AUC', 'aupr': 'AUPR'}

    # 1. 提取每个冷启动类型中表现最好的模型（基于F1值）
    best_models = {}
    for cs_type, result in results_dict.items():
        if result['results']:
            try:
                best_model_name = max(result['results'].items(), key=lambda x: x[1]['f1'])[0]
                best_models[cs_type] = {
                    '冷启动类型': cs_type,
                    '最佳模型': best_model_name,
                    **{metric_names[m]: result['results'][best_model_name][m] for m in metrics}
                }
            except Exception as e:
                print(f"处理{cs_type}最佳模型时出错: {e}")
                continue

    # 2. 创建比较表格
    if best_models:
        comparison_df = pd.DataFrame(best_models.values())
        try:
            comparison_path = os.path.join(output_dir, 'cold_start_comparison.csv')
            comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"保存冷启动比较表出错，但不影响程序继续运行")

        print("\n不同冷启动策略最佳模型性能比较:")
        print("=" * 100)
        print(
            f"{'冷启动类型':<15}{'最佳模型':<20}{'准确率':<10}{'精确率':<10}{'召回率':<10}{'F1分数':<10}{'AUC':<10}{'AUPR':<10}")
        print("-" * 100)

        for idx, row in comparison_df.iterrows():
            print(
                f"{row['冷启动类型']:<15}{row['最佳模型']:<20}{row['准确率']:<10.4f}{row['精确率']:<10.4f}"
                f"{row['召回率']:<10.4f}{row['F1分数']:<10.4f}{row['AUC']:<10.4f}{row['AUPR']:<10.4f}")

        print("=" * 100)

        # 3. 可视化比较
        if VISUALIZATION_ENABLED:
            try:
                plt.figure(figsize=(15, 10))

                # 包括AUPR在内的所有指标
                for i, metric in enumerate(metric_names.values()):
                    plt.subplot(2, 3, i + 1)
                    ax = sns.barplot(x='冷启动类型', y=metric, data=comparison_df)

                # 使用自定义函数设置中文
                fig = plt.gcf()
                plot_with_chinese_font(
                    fig,
                    text_annotations=[
                        {'x': 0.5, 'y': 1.05, 'text': f'不同冷启动策略的{metric}比较',
                         'ax': fig.axes[i], 'ha': 'center', 'fontsize': 12} for i, metric in
                        enumerate(metric_names.values())
                    ]
                )

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'cold_start_comparison.png'), dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"绘制冷启动策略比较图出错，但不影响程序继续运行: {e}")
    else:
        print("没有足够的数据进行冷启动策略比较")


def compare_all_models(results_dict, output_dir):
    """比较所有模型策略的性能（标准随机分割和冷启动策略）"""
    print("\n比较所有模型策略的性能...")

    model_types = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'aupr']
    metric_names = {'accuracy': '准确率', 'precision': '精确率', 'recall': '召回率', 'f1': 'F1分数',
                    'auc': 'AUC', 'aupr': 'AUPR'}

    # 1. 提取每个模型类型中表现最好的模型（基于F1值）
    best_models = {}
    for model_type, result in results_dict.items():
        if result['results']:
            try:
                best_model_name = max(result['results'].items(), key=lambda x: x[1]['f1'])[0]
                best_models[model_type] = {
                    '模型类型': model_type,
                    '最佳模型': best_model_name,
                    **{metric_names[m]: result['results'][best_model_name][m] for m in metrics}
                }
            except Exception as e:
                print(f"处理{model_type}最佳模型时出错: {e}")
                continue

    # 2. 创建比较表格
    if best_models:
        comparison_df = pd.DataFrame(best_models.values())
        try:
            comparison_path = os.path.join(output_dir, 'all_models_comparison.csv')
            comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"保存模型比较表出错，但不影响程序继续运行")

        print("\n所有模型策略最佳模型性能比较:")
        print("=" * 100)
        print(
            f"{'模型类型':<15}{'最佳模型':<20}{'准确率':<10}{'精确率':<10}{'召回率':<10}{'F1分数':<10}{'AUC':<10}{'AUPR':<10}")
        print("-" * 100)
        for idx, row in comparison_df.iterrows():
            print(
                f"{row['模型类型']:<15}{row['最佳模型']:<20}{row['准确率']:<10.4f}{row['精确率']:<10.4f}"
                f"{row['召回率']:<10.4f}{row['F1分数']:<10.4f}{row['AUC']:<10.4f}{row['AUPR']:<10.4f}")

        print("=" * 100)


def main():
    """主函数，执行完整的工作流程，支持断点续传"""
    try:
        print(f"开始执行蛋白质-化合物相互作用预测 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"使用 {CPU_COUNT} 个CPU核心进行并行处理")

        # 设置输出目录
        if OUTPUT_DIR:
            output_dir = OUTPUT_DIR
        else:
            output_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"结果将保存在目录: {output_dir}")

        # 初始化断点续传管理器
        checkpoint_manager = None
        if ENABLE_CHECKPOINT and not FORCE_RESTART:
            checkpoint_manager = CheckpointManager(
                CHECKPOINT_FILE, output_dir, verbose=CHECKPOINT_VERBOSE
            )
            print(f"\n断点续传功能已启用")
        elif FORCE_RESTART:
            print(f"\n强制重新开始训练，忽略现有检查点")
        else:
            print(f"\n断点续传功能已禁用")

        # 设置日志文件
        if LOGFILE_ENABLED:
            log_file = os.path.join(output_dir, 'execution_log.txt')
            import sys

            class Logger:
                def __init__(self, filename):
                    self.terminal = sys.stdout
                    self.log = open(filename, 'w', encoding='utf-8')

                def write(self, message):
                    self.terminal.write(message)
                    self.log.write(message)
                    self.log.flush()

                def flush(self):
                    self.terminal.flush()
                    self.log.flush()

            sys.stdout = Logger(log_file)

        # 加载数据
        df, X, y, protein_id_col, compound_id_col, protein_cols, compound_cols, label_col = load_data(INPUT_FILE)

        # 验证数据和配置一致性（如果使用检查点）
        if checkpoint_manager:
            if not checkpoint_manager.validate_data_consistency(df):
                response = input("数据发生变化，是否继续使用检查点？(y/n): ")
                if response.lower() != 'y':
                    print("退出程序，请删除检查点文件或使用新的检查点文件名")
                    return

            # 保存当前配置
            current_config = get_current_config()
            if not checkpoint_manager.validate_config_consistency(current_config):
                response = input("配置参数发生变化，是否继续使用检查点？(y/n): ")
                if response.lower() != 'y':
                    print("退出程序，请删除检查点文件或使用新的检查点文件名")
                    return
            checkpoint_manager.save_config_hash(current_config)

        # 结果字典，保存所有模型训练结果
        all_model_results = {}

        # 训练标准随机分割模型（非冷启动）
        if ENABLE_STANDARD_RANDOM:
            if checkpoint_manager:
                all_model_results['标准随机分割'] = train_standard_model_with_checkpoint(
                    checkpoint_manager, df, protein_id_col, compound_id_col, label_col,
                    X, protein_cols, compound_cols, output_dir
                )
            else:
                # 使用原始函数（为了保持兼容性，这里省略实现细节）
                print("未启用检查点，使用标准训练流程...")

        # 训练三种冷启动模型
        cold_start_types = []
        if ENABLE_PROTEIN_COLD_START:
            cold_start_types.append('蛋白质冷启动')
        if ENABLE_DRUG_COLD_START:
            cold_start_types.append('药物冷启动')
        if ENABLE_DUAL_COLD_START:
            cold_start_types.append('双重冷启动')

        for cold_start_type in cold_start_types:
            if checkpoint_manager:
                all_model_results[cold_start_type] = train_cold_start_model_with_checkpoint(
                    checkpoint_manager, cold_start_type, df, protein_id_col, compound_id_col, label_col,
                    X, protein_cols, compound_cols, output_dir
                )
            else:
                # 使用原始函数（为了保持兼容性，这里省略实现细节）
                print(f"未启用检查点，使用标准{cold_start_type}训练流程...")

        # 比较所有模型策略的性能
        if len(all_model_results) > 1:
            compare_all_models(all_model_results, output_dir)

            # 同时也比较冷启动模型的性能
            cold_start_results = {k: v for k, v in all_model_results.items() if k != '标准随机分割' and v is not None}
            if len(cold_start_results) > 1:
                compare_cold_start_models(cold_start_results, output_dir)

        # 进行预测（如果启用）
        if PREDICTION_ENABLED and PREDICTION_FILES:
            print("\n开始进行预测...")
            # 这里可以实现预测逻辑

        if BATCH_PREDICTION_ENABLED:
            print("\n开始批量预测...")
            # 这里可以实现批量预测逻辑

        # 生成测试集性能总结报告
        print("\n\n所有模型在各种策略下的测试集性能总结:")
        print("=" * 100)
        print(
            f"{'模型类型':<15}{'模型':<20}{'准确率':<10}{'精确率':<10}{'召回率':<10}{'F1分数':<10}{'AUC':<10}{'AUPR':<10}")
        print("-" * 100)

        all_test_results = []

        for model_type, results in all_model_results.items():
            if results and results['results']:
                for model_name, model_result in results['results'].items():
                    try:
                        result_row = {
                            "模型类型": model_type,
                            "模型": model_name,
                            "准确率": f"{model_result['accuracy']:.4f}",
                            "精确率": f"{model_result['precision']:.4f}",
                            "召回率": f"{model_result['recall']:.4f}",
                            "F1分数": f"{model_result['f1']:.4f}",
                            "AUC": f"{model_result['auc']:.4f}",
                            "AUPR": f"{model_result['aupr']:.4f}"
                        }
                        all_test_results.append(result_row)

                        # 打印表格行
                        print(
                            f"{model_type:<15}{model_name:<20}{float(result_row['准确率']):<10.4f}"
                            f"{float(result_row['精确率']):<10.4f}{float(result_row['召回率']):<10.4f}"
                            f"{float(result_row['F1分数']):<10.4f}{float(result_row['AUC']):<10.4f}"
                            f"{float(result_row['AUPR']):<10.4f}"
                        )
                    except:
                        print(f"处理 {model_type} - {model_name} 的结果时出错")

        print("=" * 100)

        # 保存所有测试结果到CSV
        if all_test_results:
            all_results_df = pd.DataFrame(all_test_results)
            try:
                all_results_df.to_csv(os.path.join(output_dir, 'all_test_results.csv'), index=False,
                                      encoding='utf-8-sig')
            except Exception as e:
                print(f"保存所有测试结果出错，但不影响程序继续运行")

        # 生成最终总结报告
        try:
            best_models = {}
            for model_type, results in all_model_results.items():
                if results and results.get('results') and results.get('best_model'):
                    best_models[model_type] = results.get('best_model', "未知")

            summary = {
                "数据集大小": len(df),
                "蛋白质数量": len(df[protein_id_col].unique()),
                "化合物数量": len(df[compound_id_col].unique()),
                "标准随机分割最佳模型": best_models.get('标准随机分割', "未知"),
                "蛋白质冷启动最佳模型": best_models.get('蛋白质冷启动', "未知"),
                "药物冷启动最佳模型": best_models.get('药物冷启动', "未知"),
                "双重冷启动最佳模型": best_models.get('双重冷启动', "未知"),
                "执行时间": str(datetime.now()),
                "CPU核心数": CPU_COUNT,
                "断点续传": "启用" if checkpoint_manager else "禁用"
            }

            with open(os.path.join(output_dir, 'final_summary.txt'), 'w', encoding='utf-8') as f:
                f.write("最终总结报告\n")
                f.write("=" * 50 + "\n")
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")

                # 添加各类型最佳模型的详细测试集性能
                f.write("\n各模型类型最佳模型的测试集性能:\n")
                f.write("-" * 50 + "\n")

                for model_type, results in all_model_results.items():
                    if results and results.get('results') and results.get('best_model'):
                        best_model = results['best_model']
                        if best_model in results['results']:
                            metrics = results['results'][best_model]
                            f.write(f"{model_type} - 最佳模型: {best_model}\n")
                            f.write(f"  准确率: {metrics['accuracy']:.4f}\n")
                            f.write(f"  精确率: {metrics['precision']:.4f}\n")
                            f.write(f"  召回率: {metrics['recall']:.4f}\n")
                            f.write(f"  F1分数: {metrics['f1']:.4f}\n")
                            f.write(f"  AUC: {metrics['auc']:.4f}\n")
                            f.write(f"  AUPR: {metrics['aupr']:.4f}\n\n")
        except Exception as e:
            print(f"生成最终总结报告出错，但不影响程序完成: {e}")

        # 清理检查点文件（如果设置了自动清理）
        if checkpoint_manager and AUTO_CLEANUP_CHECKPOINT:
            checkpoint_manager.cleanup_checkpoint()

        print("\n处理完成！所有结果已保存到目录:", output_dir)
        print(f"总执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if checkpoint_manager:
            print(f"断点续传功能已使用，检查点文件: {checkpoint_manager.checkpoint_path}")
            if AUTO_CLEANUP_CHECKPOINT:
                print("检查点文件已自动清理")

        return all_model_results
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()