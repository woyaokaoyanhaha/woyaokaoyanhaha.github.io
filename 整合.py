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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
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
INPUT_FILE = "核受体PsePSSM和PseACC特征.csv"  # 训练数据文件路径
OUTPUT_DIR = ""  # 输出目录，为空则自动生成以日期时间命名的目录
LOGFILE_ENABLED = True  # 是否生成日志文件

# 训练模型设置
ENABLE_STANDARD_RANDOM = True  # 是否训练标准随机分割模型
ENABLE_PROTEIN_COLD_START = True  # 是否训练蛋白质冷启动模型
ENABLE_DRUG_COLD_START = True  # 是否训练药物冷启动模型
ENABLE_DUAL_COLD_START = True  # 是否训练双重冷启动模型
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
ENABLE_GAUSSIAN_PROCESS = True  # 是否训练高斯过程模型（计算量大）
ENABLE_ENSEMBLE_MODELS = True  # 是否创建集成模型（投票分类器和堆叠分类器）

# 特征工程设置
FEATURE_ENHANCEMENT = True  # 是否启用特征增强
FEATURE_SELECTION = True  # 是否启用特征选择
FEATURE_SELECTION_THRESHOLD = 2  # 特征选择阈值

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

# 全局设置并行任务核心数
CPU_COUNT = cpu_count() if N_JOBS == -1 else min(N_JOBS, cpu_count())
print(f"检测到 {CPU_COUNT} 个CPU核心可用, 将使用 {N_JOBS if N_JOBS > 0 else CPU_COUNT} 个核心")

# 忽略特定警告
warnings.filterwarnings('ignore', message='.*No further splits with positive gain.*')
warnings.filterwarnings('ignore', message='.*super.*__sklearn_tags__.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


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
    """加载数据并进行初步处理"""
    print(f"正在加载数据文件: {input_file}")
    df = pd.read_csv(input_file)
    df = df.fillna(0)  # 填充NaN值为0
    print(f"数据形状: {df.shape}")

    # 提取蛋白质ID和化合物ID
    protein_id_col = 'gene_id'
    compound_id_col = 'compound_id'
    all_columns = df.columns.tolist()

    # 找到ID列的索引
    if protein_id_col in all_columns and compound_id_col in all_columns:
        protein_id_idx = all_columns.index(protein_id_col)
        compound_id_idx = all_columns.index(compound_id_col)

        # 提取特征列和标签列
        protein_features = all_columns[protein_id_idx + 1:compound_id_idx]
        compound_features = all_columns[compound_id_idx + 1:-1]
        label_col = all_columns[-1]
    else:
        # 默认划分
        protein_features = all_columns[1:int(len(all_columns) / 2)]
        compound_features = all_columns[int(len(all_columns) / 2):-1]
        label_col = all_columns[-1]

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


# 使用joblib并行处理特征增强
def process_feature_batch(X_batch, protein_cols, compound_cols):
    """处理一批次的特征增强"""
    if not FEATURE_ENHANCEMENT:
        return X_batch.copy()

    X_enhanced = X_batch.copy()
    X_enhanced = X_enhanced.fillna(0)

    # 1. 对数变换 - 用于右偏特征
    for col in X_batch.columns:
        if X_batch[col].dtype not in [np.float64, np.int64]:
            continue
        try:
            if (X_batch[col] > 0).any():
                min_positive = X_batch[col][X_batch[col] > 0].min()
                X_enhanced[f"{col}_log"] = np.log1p(X_batch[col] - min_positive + 1e-6)
        except:
            pass

    # 2. 平方和平方根变换 - 增强非线性关系
    for col in X_batch.columns:
        if X_batch[col].dtype not in [np.float64, np.int64]:
            continue
        try:
            X_enhanced[f"{col}_squared"] = X_batch[col] ** 2
            X_enhanced[f"{col}_sqrt"] = np.sqrt(np.abs(X_batch[col]) + 1e-10)
        except:
            pass

    # 3. 分箱处理 - 针对连续特征
    for col in X_batch.columns:
        if X_batch[col].dtype not in [np.float64, np.int64]:
            continue
        try:
            X_enhanced[f"{col}_bin"] = pd.qcut(X_batch[col], q=5, labels=False, duplicates='drop')
        except:
            pass

    # 4. 多项式特征 - 基于重要特征
    top_protein = protein_cols[:min(5, len(protein_cols))]
    top_compound = compound_cols[:min(5, len(compound_cols))]

    # 蛋白质特征内部交互
    for i, col1 in enumerate(top_protein):
        for col2 in top_protein[i + 1:]:
            try:
                X_enhanced[f"{col1}_{col2}_interact"] = X_batch[col1] * X_batch[col2]
            except:
                pass

    # 化合物特征内部交互
    for i, col1 in enumerate(top_compound):
        for col2 in top_compound[i + 1:]:
            try:
                X_enhanced[f"{col1}_{col2}_interact"] = X_batch[col1] * X_batch[col2]
            except:
                pass

    # 5. 蛋白质-化合物交叉特征
    for p_col in top_protein[:2]:
        for c_col in top_compound[:2]:
            try:
                X_enhanced[f"{p_col}_{c_col}_cross"] = X_batch[p_col] * X_batch[c_col]
            except:
                pass

    # 6. 特征比率
    for i, col1 in enumerate(top_protein[:2]):
        for col2 in top_compound[:2]:
            try:
                denominator = X_batch[col2].replace(0, np.nan)
                ratio = X_batch[col1] / denominator
                X_enhanced[f"{col1}_to_{col2}_ratio"] = ratio.fillna(0)
            except:
                pass

    # 7. 统计特征
    try:
        X_enhanced['protein_mean'] = X_batch[protein_cols].mean(axis=1)
        X_enhanced['protein_std'] = X_batch[protein_cols].std(axis=1)
        X_enhanced['compound_mean'] = X_batch[compound_cols].mean(axis=1)
        X_enhanced['compound_std'] = X_batch[compound_cols].std(axis=1)
        X_enhanced['protein_to_compound_ratio'] = X_enhanced['protein_mean'] / X_enhanced['compound_mean'].replace(0, 1)
    except:
        pass

    # 确保最终数据不含NaN值和无穷值
    X_enhanced = X_enhanced.fillna(0)
    X_enhanced = X_enhanced.replace([np.inf, -np.inf], np.nan)
    X_enhanced = X_enhanced.fillna(0)

    return X_enhanced


def enhance_features(X, protein_cols, compound_cols):
    """增强特征通过构造新特征和转换，使用并行处理"""
    if not FEATURE_ENHANCEMENT:
        return X

    print("增强特征（使用并行处理）...")

    # 确定分批处理的大小和数量
    batch_size = max(1, min(1000, len(X) // (CPU_COUNT * 2)))
    n_batches = (len(X) + batch_size - 1) // batch_size

    if n_batches <= 1:
        # 如果数据量小，直接处理
        X_enhanced = process_feature_batch(X, protein_cols, compound_cols)
    else:
        # 分批并行处理
        print(f"将数据分为 {n_batches} 批进行并行特征增强处理")

        # 创建批次
        batches = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))
            batches.append(X.iloc[start_idx:end_idx])

        # 并行处理批次
        results = Parallel(n_jobs=N_JOBS)(
            delayed(process_feature_batch)(batch, protein_cols, compound_cols)
            for batch in batches
        )

        # 合并结果
        X_enhanced = pd.concat(results, axis=0)

    print(f"原始特征数: {X.shape[1]}, 增强后特征数: {X_enhanced.shape[1]}")
    return X_enhanced


def select_features(X_train, y_train, X_test, feature_names, selection_threshold=FEATURE_SELECTION_THRESHOLD):
    """使用多种特征选择方法选择最佳特征"""
    if not FEATURE_SELECTION:
        # 不进行特征选择，返回原始特征
        return X_train, X_test, feature_names

    print("执行特征选择...")

    # 确保输入数据没有NaN值
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    results = {}

    # 1. 方差筛选 - 过滤低方差特征
    try:
        selector = VarianceThreshold(threshold=0.01)
        X_var = selector.fit_transform(X_train)
        var_support = selector.get_support()
        var_features = [feature_names[i] for i in range(len(feature_names)) if var_support[i]]
        print(f"方差筛选后保留特征数: {len(var_features)}")
        results['方差筛选'] = var_features
    except Exception as e:
        print(f"方差筛选出错: {e}")
        results['方差筛选'] = []

    # 2. 统计检验 - F检验
    try:
        selector_f = SelectKBest(f_classif, k=min(100, X_train.shape[1]))
        X_f = selector_f.fit_transform(X_train, y_train)
        f_support = selector_f.get_support()
        f_features = [feature_names[i] for i in range(len(feature_names)) if f_support[i]]
        print(f"F检验选择后保留特征数: {len(f_features)}")
        results['F检验'] = f_features
    except Exception as e:
        print(f"F检验出错: {e}")
        results['F检验'] = []

    # 3. 互信息
    try:
        selector_mi = SelectKBest(mutual_info_classif, k=min(100, X_train.shape[1]))
        X_mi = selector_mi.fit_transform(X_train, y_train)
        mi_support = selector_mi.get_support()
        mi_features = [feature_names[i] for i in range(len(feature_names)) if mi_support[i]]
        print(f"互信息选择后保留特征数: {len(mi_features)}")
        results['互信息'] = mi_features
    except Exception as e:
        print(f"互信息计算失败: {e}")
        results['互信息'] = []

    # 4. 递归特征消除(RFE) - 使用多核处理
    try:
        estimator = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=N_JOBS)
        rfe = RFE(estimator=estimator, n_features_to_select=min(100, X_train.shape[1]), step=5)
        X_rfe = rfe.fit_transform(X_train, y_train)
        rfe_support = rfe.get_support()
        rfe_features = [feature_names[i] for i in range(len(feature_names)) if rfe_support[i]]
        print(f"RFE选择后保留特征数: {len(rfe_features)}")
        results['RFE'] = rfe_features
    except Exception as e:
        print(f"RFE计算失败: {e}")
        results['RFE'] = []

    # 5. 基于模型的特征重要性 - 使用多核
    try:
        model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=N_JOBS)
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        model_features = [feature_names[i] for i in indices[:100]]
        print(f"模型特征重要性选择后保留特征数: {len(model_features)}")
        results['模型重要性'] = model_features
    except Exception as e:
        print(f"模型特征重要性计算失败: {e}")
        results['模型重要性'] = []

    # 6. 综合特征选择 - 取被多种方法选择的特征
    all_selected_features = set()
    feature_vote = {f: 0 for f in feature_names}

    for method, features in results.items():
        for f in features:
            feature_vote[f] += 1
            all_selected_features.add(f)

    # 被至少selection_threshold种方法选中的特征
    consensus_features = [f for f, votes in feature_vote.items() if votes >= selection_threshold]
    print(f"综合特征选择后保留特征数: {len(consensus_features)}")

    # 如果综合特征太少，则使用所有方法的并集
    if len(consensus_features) < 30:
        consensus_features = list(all_selected_features)
        print(f"综合特征数量太少，使用所有方法选择的特征并集, 特征数: {len(consensus_features)}")

    # 如果所有方法都失败，则使用原始特征的前100个
    if len(consensus_features) == 0:
        consensus_features = feature_names[:min(100, len(feature_names))]
        print(f"所有特征选择方法都失败，使用原始特征的前{len(consensus_features)}个")

    # 应用特征选择
    X_train_selected = X_train[consensus_features]
    X_test_selected = X_test[consensus_features]

    return X_train_selected, X_test_selected, consensus_features


def scale_features(X_train, X_test):
    """应用多种缩放方法并选择最合适的"""
    print("优化特征缩放...")

    # 确保没有NaN和inf值
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)

    # 首先处理 NaN 值
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # 存储不同缩放器的结果
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }

    # 尝试添加PowerTransformer
    try:
        from sklearn.preprocessing import PowerTransformer
        scalers['power'] = PowerTransformer(method='yeo-johnson')
    except:
        pass

    scaled_data = {}

    # 并行执行缩放
    def scale_with_method(name, scaler):
        try:
            # 对训练集拟合并转换
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            # 对测试集转换
            X_test_scaled = scaler.transform(X_test_imputed)

            # 计算统计量，安全处理可能的异常值
            train_mean = np.nanmean(np.mean(X_train_scaled, axis=0))
            train_std = np.nanmean(np.std(X_train_scaled, axis=0))

            # 计算偏度
            try:
                train_skew_values = []
                for i in range(X_train_scaled.shape[1]):
                    col_data = X_train_scaled[:, i]
                    if np.std(col_data) > 1e-10:
                        skew_value = stats.skew(col_data)
                        if not np.isnan(skew_value) and not np.isinf(skew_value):
                            train_skew_values.append(abs(skew_value))
                train_skew = np.mean(train_skew_values) if train_skew_values else 0
            except:
                train_skew = 0

            # 计算质量分数
            quality_score = abs(train_mean) + abs(train_std - 1) + train_skew

            print(
                f"{name} 缩放器: 平均差异={train_mean:.3f}, 标准差差异={abs(train_std - 1):.3f}, 偏度={train_skew:.3f}")

            return {
                'scaler': scaler,
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
                'quality_score': quality_score,
                'name': name
            }
        except Exception as e:
            print(f"{name} 缩放器出错: {str(e)}")
            return {
                'scaler': scaler,
                'X_train_scaled': X_train_imputed,
                'X_test_scaled': X_test_imputed,
                'quality_score': float('inf'),
                'name': name
            }

    # 并行执行不同的缩放方法
    scale_results = Parallel(n_jobs=N_JOBS)(
        delayed(scale_with_method)(name, scaler) for name, scaler in scalers.items()
    )

    # 将结果转换为字典
    for result in scale_results:
        name = result.pop('name')
        scaled_data[name] = result

    # 选择质量分数最低的缩放器(分布差异最小)
    best_scaler_name = min(scaled_data.keys(), key=lambda x: scaled_data[x]['quality_score'])
    print(f"最佳缩放方法: {best_scaler_name}")

    # 检查结果中是否还有NaN值或inf值
    best_result = scaled_data[best_scaler_name]
    X_train_scaled = best_result['X_train_scaled']
    X_test_scaled = best_result['X_test_scaled']

    if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any() or \
            np.isinf(X_train_scaled).any() or np.isinf(X_test_scaled).any():
        print("警告：缩放后的数据仍包含NaN或inf值，将进行填充")
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        best_result['scaler'],
        X_train_scaled,
        X_test_scaled,
        imputer
    )


def optimize_features(X_train, X_test, y_train, protein_cols, compound_cols):
    """特征工程优化流程"""
    print("\n开始特征工程优化...")

    # 记录原始特征数
    original_feature_count = X_train.shape[1]

    # 步骤1：特征增强
    print("增强特征...")
    X_train_enhanced = enhance_features(X_train, protein_cols, compound_cols)
    X_test_enhanced = enhance_features(X_test, protein_cols, compound_cols)

    # 步骤2：特征选择
    print("选择最优特征...")
    feature_names = X_train_enhanced.columns.tolist()
    X_train_selected, X_test_selected, selected_features = select_features(
        X_train_enhanced, y_train, X_test_enhanced, feature_names
    )

    # 确保没有无穷值和NaN值
    X_train_selected = X_train_selected.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_selected = X_test_selected.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 步骤3：优化特征缩放
    print("优化特征缩放...")
    scaler, X_train_scaled, X_test_scaled, imputer = scale_features(X_train_selected, X_test_selected)

    # 构建特征管道
    feature_pipeline = {
        'protein_cols': protein_cols,
        'compound_cols': compound_cols,
        'selected_features': selected_features,
        'scaler': scaler,
        'imputer': imputer
    }

    print(f"特征工程完成. 原始特征数: {original_feature_count}, 优化后特征数: {len(selected_features)}")

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
        # 修改LightGBM参数，避免过拟合和警告
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

    # 添加scikit-learn兼容的方法
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
                return name, estimator  # 返回原始分类器作为备用

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
        else:  # 'soft' voting
            predictions = self._collect_probas(X)
            avg = np.average(predictions, axis=0)
            return self.classes_[np.argmax(avg, axis=1)]

    def predict_proba(self, X):
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when voting='hard'")
        return np.average(self._collect_probas(X), axis=0)

    def _collect_probas(self, X):
        # 并行收集概率
        def get_proba(name_clf):
            name, clf = name_clf
            try:
                return clf.predict_proba(X)
            except Exception as e:
                print(f"获取分类器 {name} 概率时出错: {e}")
                return np.zeros((X.shape[0], len(self.classes_)))

        # 并行获取所有分类器的概率
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

    # 添加scikit-learn兼容的方法
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
                # 处理estimator的参数
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
        # 训练所有基础分类器
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

        # 并行获取元特征
        def get_meta_features_for_estimator(name_est):
            name, est = name_est
            try:
                prob = est.predict_proba(X)
                return name, prob
            except Exception as e:
                print(f"获取元特征时分类器 {name} 出错: {e}")
                return name, np.zeros((X.shape[0], len(self.classes_)))

        # 并行执行
        try:
            results = Parallel(n_jobs=N_JOBS)(
                delayed(get_meta_features_for_estimator)(name_est) for name_est in self.estimators
            )

            # 处理结果
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
        auprs = []  # 添加AUPR存储
        fold_df = pd.DataFrame()
    else:
        # 正常的交叉验证流程
        skf = StratifiedKFold(n_splits=CROSS_VALIDATION_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        metrics_per_fold = []
        aucs = []
        auprs = []  # 添加AUPR存储列表
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
                    aupr = average_precision_score(y_val, y_score)  # 计算AUPR
                except:
                    auc = 0.5  # 默认AUC
                    aupr = 0.5  # 默认AUPR

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
                    "aupr": aupr  # 添加AUPR到结果
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
                    "aupr": 0.5  # 默认AUPR
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
            fold_results.sort(key=lambda x: x['i'])  # 按折号排序
            for result in fold_results:
                i = result['i']
                metrics_per_fold.append([result['acc'], result['prec'], result['rec'], result['f1']])
                aucs.append(result['auc'])
                auprs.append(result['aupr'])  # 存储AUPR值
                fold_details.append({
                    "折": f"折{i + 1}",
                    "准确率": f"{result['acc']:.4f}",
                    "精确率": f"{result['prec']:.4f}",
                    "召回率": f"{result['rec']:.4f}",
                    "F1": f"{result['f1']:.4f}",
                    "AUC": f"{result['auc']:.4f}",
                    "AUPR": f"{result['aupr']:.4f}"  # 添加AUPR到折详情
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
                    "AUPR": f"{np.mean(auprs):.4f}"  # 添加AUPR均值
                }
                std_row = {
                    "折": "标准差",
                    "准确率": f"{metrics_per_fold[:, 0].std():.4f}",
                    "精确率": f"{metrics_per_fold[:, 1].std():.4f}",
                    "召回率": f"{metrics_per_fold[:, 2].std():.4f}",
                    "F1": f"{metrics_per_fold[:, 3].std():.4f}",
                    "AUC": f"{np.std(aucs):.4f}",
                    "AUPR": f"{np.std(auprs):.4f}"  # 添加AUPR标准差
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
                print(f"AUPR: {np.mean(auprs):.4f} ± {np.std(auprs):.4f}")  # 打印AUPR均值和标准差
        except Exception as e:
            print(f"交叉验证过程中出错，但不影响测试集评估: {str(e)}")
            metrics_per_fold = []
            aucs = []
            auprs = []  # 添加空的AUPR列表
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
            aupr_test = average_precision_score(y_test, y_score_test)  # 计算测试集AUPR

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
        print(f"AUPR: {aupr_test:.4f}")  # 打印AUPR结果
        print("混淆矩阵:")
        print(f"[[{conf_matrix[0, 0]:4d} {conf_matrix[0, 1]:4d}]")
        print(f" [{conf_matrix[1, 0]:4d} {conf_matrix[1, 1]:4d}]]")
        print(f"  真实负例 正例")
        print(f"预测负例 {conf_matrix[0, 0]:4d}  {conf_matrix[0, 1]:4d}")
        print(f"  正例 {conf_matrix[1, 0]:4d}  {conf_matrix[1, 1]:4d}")

        # 额外计算和输出指标
        try:
            tn, fp, fn, tp = conf_matrix.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (rec + specificity) / 2
            mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator > 0 else 0

            print(f"特异性 (Specificity): {specificity:.4f}")
            print(f"平衡准确率 (Balanced Accuracy): {balanced_acc:.4f}")
            print(f"马修相关系数 (MCC): {mcc:.4f}")
        except Exception as e:
            print(f"计算额外指标出错，但不影响主要结果")

        print("-" * 50)

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
                f.write(f"AUPR: {aupr_test:.4f}\n")  # 添加AUPR到文件
                f.write("混淆矩阵:\n")
                f.write(f"[[{conf_matrix[0, 0]:4d} {conf_matrix[0, 1]:4d}]\n")
                f.write(f" [{conf_matrix[1, 0]:4d} {conf_matrix[1, 1]:4d}]]\n")
                try:
                    tn, fp, fn, tp = conf_matrix.ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    balanced_acc = (rec + specificity) / 2
                    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                    mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator > 0 else 0

                    f.write(f"特异性 (Specificity): {specificity:.4f}\n")
                    f.write(f"平衡准确率 (Balanced Accuracy): {balanced_acc:.4f}\n")
                    f.write(f"马修相关系数 (MCC): {mcc:.4f}\n")
                except Exception as e:
                    f.write(f"计算额外指标出错，但不影响主要结果\n")
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
            # 明确使用matplotlib的imshow而不是seaborn的heatmap
            im = plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
            plt.colorbar()

            # 明确设置坐标位置
            tick_marks = np.arange(2)  # 二分类问题有2个类别
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
        # 不输出详细错误，只是跳过图形生成
        pass

    # 输出前十个最相关特征
    print(f"\n{model_name} - {cold_start_type} 前十个最相关特征（按权重绝对值排序）：")
    importances = None
    top10_idx = None

    # 对不同模型采用不同的特征重要性获取方法
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            sorted_idx = np.argsort(np.abs(importances))[::-1]
            top10_idx = sorted_idx[:10]
            feature_importance_type = "内置特征重要性"
        elif hasattr(model, "coef_"):
            importances = model.coef_.flatten()
            sorted_idx = np.argsort(np.abs(importances))[::-1]
            top10_idx = sorted_idx[:10]
            feature_importance_type = "系数绝对值"
        else:
            # 对所有模型统一使用排列重要性
            try:
                # 使用并行计算排列重要性
                perm_importance = permutation_importance(
                    model, X_test, y_test, n_repeats=5,
                    random_state=RANDOM_STATE, n_jobs=N_JOBS
                )
                importances = perm_importance.importances_mean
                sorted_idx = np.argsort(np.abs(importances))[::-1]
                top10_idx = sorted_idx[:10]
                feature_importance_type = "排列重要性"
            except Exception as e:
                print(f"计算排列重要性出错，但不影响主要结果")
                top10_idx = None
    except Exception as e:
        print(f"获取特征重要性出错，但不影响主要结果")

    if top10_idx is not None and len(feature_names) > 0:
        try:
            feature_importance_data = []
            for i, idx in enumerate(top10_idx):
                if idx < len(feature_names):
                    feature_importance_data.append({
                        "排名": i + 1,
                        "特征名": feature_names[idx],
                        "重要性": abs(importances[idx]),
                        "原始值": importances[idx]
                    })
                    print(f"{i + 1}. {feature_names[idx]}: {importances[idx]:.6f}")

            if feature_importance_data and VISUALIZATION_ENABLED:
                # 将特征重要性保存为CSV
                pd.DataFrame(feature_importance_data).to_csv(
                    os.path.join(output_dir, f'{model_name_with_type}_top10_features.csv'),
                    index=False, encoding='utf-8-sig'
                )

                # 可视化特征重要性
                plt.figure(figsize=(12, 8))
                importance_df = pd.DataFrame(feature_importance_data)
                ax = sns.barplot(x='重要性', y='特征名', data=importance_df)

                # 使用自定义函数设置中文
                fig = plt.gcf()
                plot_with_chinese_font(
                    fig,
                    title=f'{model_name} - {cold_start_type} 特征重要性',
                    xlabel='重要性',
                    ylabel='特征名'
                )

                plt.savefig(os.path.join(output_dir, f'{model_name_with_type}_feature_importance.png'),
                            dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"保存特征重要性出错，但不影响主要结果")

    return {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc_test,
        "aupr": aupr_test,
        "top_features": top10_idx,
        "feature_importances": importances
    }


def load_protein_compound_data(protein_file, compound_file):
    """加载蛋白质和化合物数据，返回两个DataFrame"""
    print(f"加载蛋白质和化合物数据: {protein_file}, {compound_file}")

    try:
        # 加载蛋白质和化合物数据
        protein_df = pd.read_csv(protein_file)
        compound_df = pd.read_csv(compound_file)

        print(f"加载了 {len(protein_df)} 个蛋白质样本和 {len(compound_df)} 个化合物样本")
        return protein_df, compound_df
    except Exception as e:
        print(f"加载蛋白质和化合物数据出错: {e}")
        return pd.DataFrame(), pd.DataFrame()


def create_feature_for_pair(protein_row, compound_row, feature_pipeline):
    """为单个蛋白质-化合物对创建特征向量"""
    features = {}

    # 添加蛋白质特征
    for col in feature_pipeline['protein_cols']:
        if col in protein_row:
            features[col] = protein_row[col]
        else:
            features[col] = 0

    # 添加化合物特征
    for col in feature_pipeline['compound_cols']:
        if col in compound_row:
            features[col] = compound_row[col]
        else:
            features[col] = 0

    # 创建特征DataFrame
    return pd.DataFrame([features])


def process_specific_sample(sample_df, feature_pipeline):
    """使用特征工程管道处理样本"""
    print("处理特定样本...")

    try:
        # 1. 特征增强
        sample_enhanced = enhance_features(sample_df, feature_pipeline['protein_cols'],
                                           feature_pipeline['compound_cols'])

        # 2. 选择指定特征
        for col in feature_pipeline['selected_features']:
            if col not in sample_enhanced.columns:
                sample_enhanced[col] = 0

        sample_selected = sample_enhanced[feature_pipeline['selected_features']]

        # 3. 确保没有无穷值和NaN值
        sample_selected = sample_selected.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 4. 特征缩放
        sample_imputed = feature_pipeline['imputer'].transform(sample_selected)
        sample_scaled = feature_pipeline['scaler'].transform(sample_imputed)

        # 5. 最后检查确保没有NaN和inf值
        sample_scaled = np.nan_to_num(sample_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        return sample_scaled
    except Exception as e:
        print(f"处理特定样本出错: {e}")
        # 创建一个全零的样本数据
        n_features = len(feature_pipeline['selected_features'])
        return np.zeros((1, n_features))


def predict_interaction_single(models, sample_scaled, protein_id, compound_id, output_dir=None, cold_start_type=""):
    """使用训练好的模型预测单个蛋白质-化合物相互作用"""
    suffix = f"_{cold_start_type}" if cold_start_type else ""
    print(f"预测蛋白质({protein_id})-化合物({compound_id})相互作用 - {cold_start_type}...")

    predictions = []
    probs = []

    # 预测
    for name, model in models.items():
        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(sample_scaled)[0, 1]
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(sample_scaled)[0]
                prob = 1 / (1 + np.exp(-decision))
            else:
                prob = model.predict(sample_scaled)[0]

            pred = 1 if prob >= 0.5 else 0
            print(f"{name} 预测结果: {'相互作用' if pred == 1 else '无相互作用'}, 概率: {prob:.4f}")
            predictions.append({"模型": name, "预测": pred, "概率": prob})
            if prob > 0:
                probs.append(prob)
        except Exception as e:
            print(f"{name}模型预测出错，但不影响其他模型: {e}")
            predictions.append({"模型": name, "预测": "错误", "概率": 0.0})

    # 计算集成结果
    if probs:
        avg_prob = sum(probs) / len(probs)
        ensemble_pred = 1 if avg_prob >= 0.5 else 0
        predictions.append({
            "模型": "集成模型",
            "预测": ensemble_pred,
            "概率": avg_prob
        })
        print(f"集成模型 预测结果: {'相互作用' if ensemble_pred == 1 else '无相互作用'}, 概率: {avg_prob:.4f}")
    else:
        print("所有模型预测都失败，无法计算集成结果")
        predictions.append({"模型": "集成模型", "预测": "错误", "概率": 0.0})

    # 添加蛋白质和化合物ID
    for pred in predictions:
        pred["蛋白质ID"] = protein_id
        pred["化合物ID"] = compound_id

    return predictions


def predict_batch_interactions(models, protein_df, compound_df, feature_pipeline, output_dir, cold_start_type=""):
    """批量预测多个蛋白质-化合物对的相互作用"""
    suffix = f"_{cold_start_type}" if cold_start_type else ""
    print(f"\n批量预测蛋白质-化合物相互作用 - {cold_start_type}...")

    # 获取蛋白质和化合物ID列的名称
    protein_id_col = None
    compound_id_col = None

    # 尝试自动检测ID列
    common_id_cols = ['id', 'ID', 'gene_id', 'protein_id', 'compound_id', 'drug_id', 'name', 'Name', 'NAME']

    # 检测蛋白质ID列
    for col in protein_df.columns:
        if col.lower() in [c.lower() for c in common_id_cols] or 'id' in col.lower():
            protein_id_col = col
            break

    # 检测化合物ID列
    for col in compound_df.columns:
        if col.lower() in [c.lower() for c in common_id_cols] or 'id' in col.lower():
            compound_id_col = col
            break

    # 如果没有找到ID列，使用索引作为ID
    if protein_id_col is None:
        protein_df['temp_protein_id'] = [f"Protein_{i}" for i in range(len(protein_df))]
        protein_id_col = 'temp_protein_id'
        print("警告: 未检测到蛋白质ID列，使用索引作为临时ID")

    if compound_id_col is None:
        compound_df['temp_compound_id'] = [f"Compound_{i}" for i in range(len(compound_df))]
        compound_id_col = 'temp_compound_id'
        print("警告: 未检测到化合物ID列，使用索引作为临时ID")

    # 创建蛋白质-化合物组合
    combinations = list(itertools.product(
        range(len(protein_df)),
        range(len(compound_df))
    ))

    # 限制组合数量，防止组合爆炸
    max_combinations = BATCH_MODE_MAX_COMBINATIONS
    if len(combinations) > max_combinations:
        print(
            f"警告: 蛋白质-化合物组合数 ({len(combinations)}) 超过最大限制 ({max_combinations})，将随机选择 {max_combinations} 个组合")
        np.random.seed(RANDOM_STATE)  # 设置随机种子确保结果可重复
        combinations = [combinations[i] for i in np.random.choice(len(combinations), max_combinations, replace=False)]

    print(f"共进行 {len(combinations)} 个蛋白质-化合物对的预测")

    # 批量预测结果
    all_predictions = []

    # 批量预测
    for i, (p_idx, c_idx) in enumerate(combinations):
        protein_row = protein_df.iloc[p_idx]
        compound_row = compound_df.iloc[c_idx]

        protein_id = protein_row[protein_id_col]
        compound_id = compound_row[compound_id_col]

        print(f"\n预测组合 {i + 1}/{len(combinations)}: 蛋白质 {protein_id} - 化合物 {compound_id}")

        # 创建特征
        sample_df = create_feature_for_pair(protein_row, compound_row, feature_pipeline)

        # 处理样本
        sample_scaled = process_specific_sample(sample_df, feature_pipeline)

        # 预测相互作用
        predictions = predict_interaction_single(models, sample_scaled, protein_id, compound_id,
                                                 output_dir, cold_start_type)

        # 添加到结果集
        all_predictions.extend(predictions)

    # 将结果转换为DataFrame并保存
    if all_predictions:
        result_df = pd.DataFrame(all_predictions)

        # 保存结果到文件
        os.makedirs(output_dir, exist_ok=True)
        result_df.to_csv(os.path.join(output_dir, f'batch_predictions{suffix}.csv'),
                         index=False, encoding='utf-8-sig')

        # 可视化预测结果
        if VISUALIZATION_ENABLED and len(all_predictions) > 0:
            try:
                # 筛选集成模型结果
                ensemble_results = result_df[result_df['模型'] == '集成模型'].copy()

                # 如果有足够多的数据，创建热图
                if len(ensemble_results) > 1:
                    # 将"预测"列转换为数值类型
                    ensemble_results['预测'] = pd.to_numeric(ensemble_results['预测'], errors='coerce').fillna(0)

                    # 创建交叉表以便绘制热图
                    pivot_table = ensemble_results.pivot(index='蛋白质ID', columns='化合物ID', values='概率')

                    plt.figure(figsize=(12, 10))
                    ax = sns.heatmap(pivot_table, annot=True, cmap='coolwarm', vmin=0, vmax=1,
                                     linewidths=.5, cbar_kws={'label': '相互作用概率'})

                    # 使用中文字体
                    plot_with_chinese_font(
                        plt.gcf(),
                        title=f'{cold_start_type} 蛋白质-化合物相互作用概率热图',
                        xlabel='化合物',
                        ylabel='蛋白质'
                    )

                    plt.savefig(os.path.join(output_dir, f'interaction_heatmap{suffix}.png'),
                                dpi=PLOT_DPI, bbox_inches='tight')
                    plt.close()

                    # 创建相互作用网络图
                    plt.figure(figsize=(12, 10))
                    # 筛选出预测为相互作用的蛋白质-化合物对
                    interactions = ensemble_results[ensemble_results['预测'] == 1]

                    if len(interactions) > 0:
                        try:
                            import networkx as nx
                            G = nx.Graph()

                            # 添加节点
                            unique_proteins = interactions['蛋白质ID'].unique()
                            unique_compounds = interactions['化合物ID'].unique()

                            # 添加蛋白质节点
                            for p in unique_proteins:
                                G.add_node(str(p), bipartite=0, node_type='蛋白质')

                            # 添加化合物节点
                            for c in unique_compounds:
                                G.add_node(str(c), bipartite=1, node_type='化合物')

                            # 添加边
                            for _, row in interactions.iterrows():
                                G.add_edge(
                                    str(row['蛋白质ID']),
                                    str(row['化合物ID']),
                                    weight=row['概率']
                                )

                            # 设置节点位置
                            pos = nx.spring_layout(G, seed=RANDOM_STATE)

                            # 绘制节点和边
                            protein_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == '蛋白质']
                            compound_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == '化合物']

                            # 绘制图形
                            nx.draw_networkx_nodes(G, pos, nodelist=protein_nodes, node_color='skyblue',
                                                   node_size=500, label='蛋白质')
                            nx.draw_networkx_nodes(G, pos, nodelist=compound_nodes, node_color='lightgreen',
                                                   node_size=500, label='化合物')

                            # 边的宽度与概率成比例
                            edge_widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]
                            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7)

                            # 标签
                            nx.draw_networkx_labels(G, pos, font_size=10)

                            plt.title(f'{cold_start_type} 蛋白质-化合物相互作用网络', fontproperties=chinese_font)
                            plt.legend(prop=chinese_font)
                            plt.axis('off')
                            plt.tight_layout()
                            plt.savefig(os.path.join(output_dir, f'interaction_network{suffix}.png'),
                                        dpi=PLOT_DPI, bbox_inches='tight')
                            plt.close()
                        except ImportError:
                            print("警告: 需要安装networkx库来创建网络图")
                        except Exception as e:
                            print(f"创建网络图出错: {e}")
            except Exception as e:
                print(f"创建热图或网络图出错: {e}")

        return result_df
    else:
        print("没有成功完成任何预测")
        return pd.DataFrame()


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
        if hasattr(model, "predict_proba"):  # 确保模型支持概率输出
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
        # 选择前6个最好的基础模型（基于F1分数）
        good_models = []
        for name, model in models.items():
            if len(good_models) < 6 and hasattr(model, "predict_proba"):  # 限制基础模型数量
                good_models.append((name, model))

        if len(good_models) >= 3:  # 至少需要3个分类器
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


def train_standard_model(df, protein_id_col, compound_id_col, label_col, X, protein_cols,
                         compound_cols, output_dir, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """训练标准随机分割模型（非冷启动模型）"""
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
    train_df, test_df = create_standard_split(
        df, protein_id_col, compound_id_col, test_size, random_state
    )

    # 2. 从分割结果提取特征和标签
    X_train = train_df[X.columns]
    y_train = train_df[label_col]
    X_test = test_df[X.columns]
    y_test = test_df[label_col]

    # 3. 特征工程优化
    X_train_scaled, X_test_scaled, feature_pipeline = optimize_features(
        X_train, X_test, y_train, protein_cols, compound_cols
    )

    # 创建子目录保存标准模型的结果
    model_dir = os.path.join(output_dir, "标准随机分割")
    os.makedirs(model_dir, exist_ok=True)

    # 保存特征管道
    try:
        joblib.dump(feature_pipeline, os.path.join(model_dir, 'feature_pipeline.pkl'))
    except Exception as e:
        print(f"保存特征管道出错，但不影响程序继续运行：{e}")

    # 4. 准备模型参数网格
    param_grids = {}

    # 根据选择添加模型
    if ENABLE_SVM:
        param_grids['SVM'] = {'C': [1.0], 'gamma': ['scale'], 'kernel': ['rbf']}

    if ENABLE_RANDOM_FOREST:
        param_grids['随机森林'] = {'n_estimators': [100], 'max_depth': [None], 'n_jobs': [N_JOBS]}

    if ENABLE_GRADIENT_BOOSTING:
        param_grids['梯度提升'] = {'n_estimators': [100], 'learning_rate': [0.1]}

    if ENABLE_LOGISTIC_REGRESSION:
        param_grids['逻辑回归'] = {'C': [1.0], 'penalty': ['l2'], 'n_jobs': [N_JOBS]}

    if ENABLE_KNN:
        param_grids['K近邻'] = {'n_neighbors': [5], 'weights': ['uniform'], 'n_jobs': [N_JOBS]}

    if ENABLE_EXTRA_TREES:
        param_grids['极端随机树'] = {'n_estimators': [100], 'max_depth': [None], 'n_jobs': [N_JOBS]}

    if ENABLE_NAIVE_BAYES:
        param_grids['朴素贝叶斯'] = {'var_smoothing': [1e-9]}

    if ENABLE_GAUSSIAN_PROCESS:
        param_grids['高斯过程'] = {'kernel': [1.0 * RBF(1.0)], 'n_jobs': [N_JOBS]}

    # 添加可选模型
    if xgb_installed:
        param_grids['XGBoost'] = {
            'n_estimators': [50],
            'learning_rate': [0.05],
            'max_depth': [4],
            'min_child_weight': [2],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'n_jobs': [N_JOBS]
        }

    if lgbm_installed:
        param_grids['LightGBM'] = {
            'n_estimators': [50],
            'learning_rate': [0.05],
            'num_leaves': [15],
            'max_depth': [4],
            'min_child_samples': [20],
            'min_split_gain': [0.1],
            'reg_alpha': [0.1],
            'reg_lambda': [0.1],
            'verbose': [-1],
            'n_jobs': [N_JOBS]
        }

    if catboost_installed:
        param_grids['CatBoost'] = {
            'iterations': [100],
            'learning_rate': [0.1],
            'depth': [6],
            'verbose': [0],
            'thread_count': [CPU_COUNT]
        }

    # 5. 训练和评估模型
    models = {}
    results = {}

    # 并行训练不同模型
    def train_and_evaluate_model(model_name, param_grid):
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

            return (model_name, best_model, result)
        except Exception as e:
            print(f"训练和评估{model_name}时出错: {str(e)}")
            return (model_name, None, None)

    # 处理SVM单独训练（不适合并行）
    svm_result = None
    if 'SVM' in param_grids:
        svm_result = train_and_evaluate_model('SVM', param_grids['SVM'])
        del param_grids['SVM']  # 从并行任务中移除

    # 并行训练其他模型
    try:
        model_results = Parallel(n_jobs=min(len(param_grids), CPU_COUNT))(
            delayed(train_and_evaluate_model)(model_name, param_grid)
            for model_name, param_grid in param_grids.items()
        )
    except Exception as e:
        print(f"并行训练模型出错: {e}")
        model_results = []
        # 尝试顺序执行
        for model_name, param_grid in param_grids.items():
            try:
                result = train_and_evaluate_model(model_name, param_grid)
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
    if len(models) >= 2:
        ensemble_models, ensemble_results = create_ensemble_models(
            models, X_train_scaled, y_train, X_test_scaled, y_test,
            feature_pipeline, model_dir, cold_start_type="标准随机分割"
        )

        # 合并模型和结果
        models.update(ensemble_models)
        results.update(ensemble_results)

    # 7. 生成测试集模型对比汇总表
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            "模型": name,
            "准确率": f"{result['accuracy']:.4f}",
            "精确率": f"{result['precision']:.4f}",
            "召回率": f"{result['recall']:.4f}",
            "F1分数": f"{result['f1']:.4f}",
            "AUC": f"{result['auc']:.4f}",
            "AUPR": f"{result['aupr']:.4f}"  # 添加AUPR
        })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        try:
            summary_path = os.path.join(model_dir, 'model_comparison.csv')
            summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"保存模型比较表出错，但不影响程序继续运行")

        # 输出测试集详细性能指标
        print(f"\n标准随机分割模型在测试集上的性能对比:")
        print("=" * 80)
        for row in summary_data:
            print(f"模型: {row['模型']}")
            print(f"  准确率: {row['准确率']}")
            print(f"  精确率: {row['精确率']}")
            print(f"  召回率: {row['召回率']}")
            print(f"  F1分数: {row['F1分数']}")
            print(f"  AUC: {row['AUC']}")
            print(f"  AUPR: {row['AUPR']}")  # 输出AUPR
            print("-" * 40)

        # 打印最佳模型信息
        try:
            best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
            best_model_metrics = next(item for item in summary_data if item["模型"] == best_model_name)
            print(f"\n标准随机分割 - 测试集上表现最好的模型是: {best_model_name}")
            print(f"  F1分数: {best_model_metrics['F1分数']}")
            print(f"  准确率: {best_model_metrics['准确率']}")
            print(f"  精确率: {best_model_metrics['精确率']}")
            print(f"  召回率: {best_model_metrics['召回率']}")
            print(f"  AUC: {best_model_metrics['AUC']}")
            print(f"  AUPR: {best_model_metrics['AUPR']}")  # 输出AUPR
            print("=" * 80)
        except Exception as e:
            print(f"计算最佳模型出错: {e}")
            best_model_name = None if not results else list(results.keys())[0]

        # 可视化模型性能对比
        if VISUALIZATION_ENABLED:
            try:
                plt.figure(figsize=(15, 10))

                # 添加AUPR到指标列表
                metrics = ['准确率', '精确率', '召回率', 'F1分数', 'AUC', 'AUPR']
                for i, metric in enumerate(metrics):
                    plt.subplot(2, 3, i + 1)
                    if len(summary_df) > 10:  # 如果模型太多，旋转标签
                        ax = sns.barplot(x='模型', y=metric, data=summary_df)
                        plt.xticks(rotation=90)
                    else:
                        ax = sns.barplot(x='模型', y=metric, data=summary_df)
                        plt.xticks(rotation=45)

                # 使用自定义函数设置中文
                fig = plt.gcf()
                plot_with_chinese_font(
                    fig,
                    # 单独为每个子图设置标题
                    text_annotations=[
                        {'x': 0.5, 'y': 1.05, 'text': f'标准随机分割 不同模型的{metrics[i]}对比',
                         'ax': fig.axes[i], 'ha': 'center', 'fontsize': 12} for i in range(len(metrics))
                    ]
                )

                plt.tight_layout()
                comparison_plot_path = os.path.join(model_dir, 'model_comparison.png')
                plt.savefig(comparison_plot_path, dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()

                # 单独创建AUPR对比图
                plt.figure(figsize=(10, 6))
                if len(summary_df) > 10:  # 如果模型太多，旋转标签
                    ax = sns.barplot(x='模型', y='AUPR', data=summary_df)
                    plt.xticks(rotation=90)
                else:
                    ax = sns.barplot(x='模型', y='AUPR', data=summary_df)
                    plt.xticks(rotation=45)

                plot_with_chinese_font(
                    fig=plt.gcf(),
                    title='标准随机分割 不同模型的AUPR对比',
                    xlabel='模型',
                    ylabel='AUPR'
                )

                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, 'aupr_comparison.png'), dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"绘制模型比较图出错，但不影响程序继续运行")

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


def train_cold_start_model(cold_start_type, df, protein_id_col, compound_id_col, label_col, X, protein_cols,
                           compound_cols, output_dir, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """训练特定类型的冷启动模型（蛋白质冷启动、药物冷启动或双重冷启动）"""
    print(f"\n{'=' * 80}")
    print(f"训练 {cold_start_type} 模型")
    print(f"{'=' * 80}")

    # 1. 根据冷启动类型分割数据
    if cold_start_type == "蛋白质冷启动":
        if not ENABLE_PROTEIN_COLD_START:
            print("蛋白质冷启动模型已被禁用")
            return {
                'train_df': None, 'test_df': None, 'X_train_scaled': None, 'X_test_scaled': None,
                'y_train': None, 'y_test': None, 'feature_pipeline': None,
                'models': {}, 'results': {}, 'best_model': None
            }
        train_df, test_df = create_protein_cold_start_split(
            df, protein_id_col, compound_id_col, test_size, random_state
        )
    elif cold_start_type == "药物冷启动":
        if not ENABLE_DRUG_COLD_START:
            print("药物冷启动模型已被禁用")
            return {
                'train_df': None, 'test_df': None, 'X_train_scaled': None, 'X_test_scaled': None,
                'y_train': None, 'y_test': None, 'feature_pipeline': None,
                'models': {}, 'results': {}, 'best_model': None
            }
        train_df, test_df = create_drug_cold_start_split(
            df, protein_id_col, compound_id_col, test_size, random_state
        )
    elif cold_start_type == "双重冷启动":
        if not ENABLE_DUAL_COLD_START:
            print("双重冷启动模型已被禁用")
            return {
                'train_df': None, 'test_df': None, 'X_train_scaled': None, 'X_test_scaled': None,
                'y_train': None, 'y_test': None, 'feature_pipeline': None,
                'models': {}, 'results': {}, 'best_model': None
            }
        train_df, test_df = create_dual_cold_start_split(
            df, protein_id_col, compound_id_col, test_size, random_state
        )
    else:
        raise ValueError(f"不支持的冷启动类型: {cold_start_type}")

    # 2. 从分割结果提取特征和标签
    X_train = train_df[X.columns]
    y_train = train_df[label_col]
    X_test = test_df[X.columns]
    y_test = test_df[label_col]

    # 3. 特征工程优化
    X_train_scaled, X_test_scaled, feature_pipeline = optimize_features(
        X_train, X_test, y_train, protein_cols, compound_cols
    )

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

    # 根据选择添加模型
    if ENABLE_SVM:
        param_grids['SVM'] = {'C': [1.0], 'gamma': ['scale'], 'kernel': ['rbf']}

    if ENABLE_RANDOM_FOREST:
        param_grids['随机森林'] = {'n_estimators': [100], 'max_depth': [None], 'n_jobs': [N_JOBS]}

    if ENABLE_GRADIENT_BOOSTING:
        param_grids['梯度提升'] = {'n_estimators': [100], 'learning_rate': [0.1]}

    if ENABLE_LOGISTIC_REGRESSION:
        param_grids['逻辑回归'] = {'C': [1.0], 'penalty': ['l2'], 'n_jobs': [N_JOBS]}

    if ENABLE_KNN:
        param_grids['K近邻'] = {'n_neighbors': [5], 'weights': ['uniform'], 'n_jobs': [N_JOBS]}

    if ENABLE_EXTRA_TREES:
        param_grids['极端随机树'] = {'n_estimators': [100], 'max_depth': [None], 'n_jobs': [N_JOBS]}

    if ENABLE_NAIVE_BAYES:
        param_grids['朴素贝叶斯'] = {'var_smoothing': [1e-9]}

    if ENABLE_GAUSSIAN_PROCESS:
        param_grids['高斯过程'] = {'kernel': [1.0 * RBF(1.0)], 'n_jobs': [N_JOBS]}

    # 添加可选模型
    if xgb_installed:
        param_grids['XGBoost'] = {
            'n_estimators': [50],
            'learning_rate': [0.05],
            'max_depth': [4],
            'min_child_weight': [2],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'n_jobs': [N_JOBS]
        }

    if lgbm_installed:
        param_grids['LightGBM'] = {
            'n_estimators': [50],
            'learning_rate': [0.05],
            'num_leaves': [15],
            'max_depth': [4],
            'min_child_samples': [20],
            'min_split_gain': [0.1],
            'reg_alpha': [0.1],
            'reg_lambda': [0.1],
            'verbose': [-1],
            'n_jobs': [N_JOBS]
        }

    if catboost_installed:
        param_grids['CatBoost'] = {
            'iterations': [100],
            'learning_rate': [0.1],
            'depth': [6],
            'verbose': [0],
            'thread_count': [CPU_COUNT]
        }

    # 5. 训练和评估模型
    models = {}
    results = {}

    # 并行训练不同模型
    def train_and_evaluate_model(model_name, param_grid):
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

            return (model_name, best_model, result)
        except Exception as e:
            print(f"训练和评估{model_name}时出错: {str(e)}")
            return (model_name, None, None)

    # 处理SVM单独训练（不适合并行）
    svm_result = None
    if 'SVM' in param_grids:
        svm_result = train_and_evaluate_model('SVM', param_grids['SVM'])
        del param_grids['SVM']  # 从并行任务中移除

    # 并行训练其他模型
    try:
        model_results = Parallel(n_jobs=min(len(param_grids), CPU_COUNT))(
            delayed(train_and_evaluate_model)(model_name, param_grid)
            for model_name, param_grid in param_grids.items()
        )
    except Exception as e:
        print(f"并行训练模型出错: {e}")
        model_results = []
        # 尝试顺序执行
        for model_name, param_grid in param_grids.items():
            try:
                result = train_and_evaluate_model(model_name, param_grid)
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
    if len(models) >= 2:
        ensemble_models, ensemble_results = create_ensemble_models(
            models, X_train_scaled, y_train, X_test_scaled, y_test,
            feature_pipeline, cold_start_dir, cold_start_type
        )

        # 合并模型和结果
        models.update(ensemble_models)
        results.update(ensemble_results)

    # 7. 生成测试集模型对比汇总表
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            "模型": name,
            "准确率": f"{result['accuracy']:.4f}",
            "精确率": f"{result['precision']:.4f}",
            "召回率": f"{result['recall']:.4f}",
            "F1分数": f"{result['f1']:.4f}",
            "AUC": f"{result['auc']:.4f}",
            "AUPR": f"{result['aupr']:.4f}"  # 添加AUPR
        })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        try:
            summary_path = os.path.join(cold_start_dir, 'model_comparison.csv')
            summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"保存模型比较表出错，但不影响程序继续运行")

        # 输出测试集详细性能指标
        print(f"\n{cold_start_type} 模型在测试集上的性能对比:")
        print("=" * 80)
        for row in summary_data:
            print(f"模型: {row['模型']}")
            print(f"  准确率: {row['准确率']}")
            print(f"  精确率: {row['精确率']}")
            print(f"  召回率: {row['召回率']}")
            print(f"  F1分数: {row['F1分数']}")
            print(f"  AUC: {row['AUC']}")
            print(f"  AUPR: {row['AUPR']}")  # 输出AUPR
            print("-" * 40)

        # 打印最佳模型信息
        try:
            best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
            best_model_metrics = next(item for item in summary_data if item["模型"] == best_model_name)
            print(f"\n{cold_start_type} - 测试集上表现最好的模型是: {best_model_name}")
            print(f"  F1分数: {best_model_metrics['F1分数']}")
            print(f"  准确率: {best_model_metrics['准确率']}")
            print(f"  精确率: {best_model_metrics['精确率']}")
            print(f"  召回率: {best_model_metrics['召回率']}")
            print(f"  AUC: {best_model_metrics['AUC']}")
            print(f"  AUPR: {best_model_metrics['AUPR']}")  # 输出AUPR
            print("=" * 80)
        except Exception as e:
            print(f"计算最佳模型出错: {e}")
            best_model_name = None if not results else list(results.keys())[0]

        # 可视化模型性能对比
        if VISUALIZATION_ENABLED:
            try:
                plt.figure(figsize=(15, 10))

                # 添加AUPR到指标列表
                metrics = ['准确率', '精确率', '召回率', 'F1分数', 'AUC', 'AUPR']
                for i, metric in enumerate(metrics):
                    plt.subplot(2, 3, i + 1)
                    if len(summary_df) > 10:  # 如果模型太多，旋转标签
                        ax = sns.barplot(x='模型', y=metric, data=summary_df)
                        plt.xticks(rotation=90)
                    else:
                        ax = sns.barplot(x='模型', y=metric, data=summary_df)
                        plt.xticks(rotation=45)

                # 使用自定义函数设置中文
                fig = plt.gcf()
                plot_with_chinese_font(
                    fig,
                    # 单独为每个子图设置标题
                    text_annotations=[
                        {'x': 0.5, 'y': 1.05, 'text': f'{cold_start_type} 不同模型的{metrics[i]}对比',
                         'ax': fig.axes[i], 'ha': 'center', 'fontsize': 12} for i in range(len(metrics))
                    ]
                )

                plt.tight_layout()
                comparison_plot_path = os.path.join(cold_start_dir, 'model_comparison.png')
                plt.savefig(comparison_plot_path, dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()

                # 单独创建AUPR对比图
                plt.figure(figsize=(10, 6))
                if len(summary_df) > 10:  # 如果模型太多，旋转标签
                    ax = sns.barplot(x='模型', y='AUPR', data=summary_df)
                    plt.xticks(rotation=90)
                else:
                    ax = sns.barplot(x='模型', y='AUPR', data=summary_df)
                    plt.xticks(rotation=45)

                plot_with_chinese_font(
                    fig=plt.gcf(),
                    title=f'{cold_start_type} 不同模型的AUPR对比',
                    xlabel='模型',
                    ylabel='AUPR'
                )

                plt.tight_layout()
                plt.savefig(os.path.join(cold_start_dir, 'aupr_comparison.png'), dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"绘制模型比较图出错，但不影响程序继续执行")

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
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'aupr']  # 添加aupr
    metric_names = {'accuracy': '准确率', 'precision': '精确率', 'recall': '召回率', 'f1': 'F1分数',
                    'auc': 'AUC', 'aupr': 'AUPR'}  # 添加AUPR到指标名称映射

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
                    # 单独为每个子图设置标题
                    text_annotations=[
                        {'x': 0.5, 'y': 1.05, 'text': f'不同冷启动策略的{metric}比较',
                         'ax': fig.axes[i], 'ha': 'center', 'fontsize': 12} for i, metric in
                        enumerate(metric_names.values())
                    ]
                )

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'cold_start_comparison.png'), dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()

                # 单独创建AUPR比较图
                plt.figure(figsize=(10, 6))
                ax = sns.barplot(x='冷启动类型', y='AUPR', data=comparison_df)

                plot_with_chinese_font(
                    fig=plt.gcf(),
                    title='不同冷启动策略的AUPR比较',
                    xlabel='冷启动类型',
                    ylabel='AUPR'
                )

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'cold_start_aupr_comparison.png'), dpi=PLOT_DPI,
                            bbox_inches='tight')
                plt.close()

                # 4. 雷达图比较
                try:


                    categories = list(metric_names.values())
                    N = len(categories)

                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111, polar=True)

                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]  # 闭合雷达图

                    for cs_type, model_data in best_models.items():
                        values = [model_data[m] for m in categories]
                        values += values[:1]  # 闭合雷达图

                        ax.plot(angles, values, linewidth=2, label=cs_type)
                        ax.fill(angles, values, alpha=0.25)

                    # 设置角度标签但使用自定义中文标签
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels([])  # 清除默认标签

                    # 手动添加中文标签
                    for i, angle in enumerate(angles[:-1]):
                        ha = 'center'
                        if angle == 0:
                            ha = 'center'
                        elif 0 < angle < np.pi:
                            ha = 'left'
                        elif angle > np.pi:
                            ha = 'right'

                        ax.text(angle, 1.15, categories[i], ha=ha, va='center', fontproperties=chinese_font,
                                fontsize=12)

                    ax.set_ylim(0, 1)
                    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                    plt.title('不同冷启动策略的性能比较', fontproperties=chinese_font, fontsize=14)

                    plt.savefig(os.path.join(output_dir, 'cold_start_radar_chart.png'), dpi=PLOT_DPI,
                                bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"创建雷达图出错：{e}")
            except Exception as e:
                print(f"绘制冷启动策略比较图出错，但不影响程序继续运行: {e}")
    else:
        print("没有足够的数据进行冷启动策略比较")


def compare_all_models(results_dict, output_dir):
    """比较所有模型策略的性能（标准随机分割和冷启动策略）"""
    print("\n比较所有模型策略的性能...")

    model_types = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'aupr']  # 添加aupr
    metric_names = {'accuracy': '准确率', 'precision': '精确率', 'recall': '召回率', 'f1': 'F1分数',
                    'auc': 'AUC', 'aupr': 'AUPR'}  # 添加AUPR到指标名称映射

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

        # 3. 可视化比较
        if VISUALIZATION_ENABLED:
            try:
                plt.figure(figsize=(15, 10))

                # 包括AUPR在内的所有指标
                for i, metric in enumerate(metric_names.values()):
                    plt.subplot(2, 3, i + 1)
                    ax = sns.barplot(x='模型类型', y=metric, data=comparison_df)

                # 使用自定义函数设置中文
                fig = plt.gcf()
                plot_with_chinese_font(
                    fig,
                    # 单独为每个子图设置标题
                    text_annotations=[
                        {'x': 0.5, 'y': 1.05, 'text': f'不同模型策略的{metric}比较',
                         'ax': fig.axes[i], 'ha': 'center', 'fontsize': 12} for i, metric in
                        enumerate(metric_names.values())
                    ]
                )

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'all_models_comparison.png'), dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()

                # 单独创建AUPR比较图
                plt.figure(figsize=(10, 6))
                ax = sns.barplot(x='模型类型', y='AUPR', data=comparison_df)

                plot_with_chinese_font(
                    fig=plt.gcf(),
                    title='不同模型策略的AUPR比较',
                    xlabel='模型类型',
                    ylabel='AUPR'
                )

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'all_models_aupr_comparison.png'), dpi=PLOT_DPI,
                            bbox_inches='tight')
                plt.close()

                # 4. 雷达图比较
                try:
                    categories = list(metric_names.values())
                    N = len(categories)

                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111, polar=True)

                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]  # 闭合雷达图

                    for model_type, model_data in best_models.items():
                        values = [model_data[m] for m in categories]
                        values += values[:1]  # 闭合雷达图

                        ax.plot(angles, values, linewidth=2, label=model_type)
                        ax.fill(angles, values, alpha=0.25)

                    # 设置角度标签但使用自定义中文标签
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels([])  # 清除默认标签

                    # 手动添加中文标签
                    for i, angle in enumerate(angles[:-1]):
                        ha = 'center'
                        if angle == 0:
                            ha = 'center'
                        elif 0 < angle < np.pi:
                            ha = 'left'
                        elif angle > np.pi:
                            ha = 'right'

                        ax.text(angle, 1.15, categories[i], ha=ha, va='center', fontproperties=chinese_font,
                                fontsize=12)

                    ax.set_ylim(0, 1)
                    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                    plt.title('所有模型策略的性能比较', fontproperties=chinese_font, fontsize=14)

                    plt.savefig(os.path.join(output_dir, 'all_models_radar_chart.png'), dpi=PLOT_DPI,
                                bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"创建雷达图出错：{e}")
            except Exception as e:
                print(f"绘制模型策略比较图出错，但不影响程序继续执行: {e}")
    else:
        print("没有足够的数据进行模型策略比较")


def get_model_files_in_directory(directory):
    """获取目录中所有可能的模型文件名（不包括路径和_model.pkl后缀）"""
    model_files = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith('_model.pkl'):
                model_name = filename[:-10]  # 去掉"_model.pkl"
                model_files.append(model_name)
    except Exception as e:
        print(f"读取目录 {directory} 中的模型文件出错: {e}")

    return model_files


def load_model_and_pipeline(model_dir, model_name):
    """加载模型和特征处理管道"""
    try:
        # 加载特征处理管道
        pipeline_path = os.path.join(model_dir, 'feature_pipeline.pkl')
        if not os.path.exists(pipeline_path):
            print(f"找不到特征管道文件: {pipeline_path}")
            return None, None

        feature_pipeline = joblib.load(pipeline_path)

        # 加载模型
        model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
        if not os.path.exists(model_path):
            print(f"找不到模型文件: {model_path}")
            return None, None

        model = joblib.load(model_path)

        return model, feature_pipeline
    except Exception as e:
        print(f"加载模型和特征管道出错: {e}")
        return None, None


def predict_files(model_dir, prediction_files, output_dir):
    """使用模型预测蛋白质-化合物对的交互作用"""
    if not PREDICTION_ENABLED or not prediction_files:
        print("预测功能已禁用或未指定预测文件")
        return

    print("\n开始预测蛋白质-化合物相互作用...")

    # 确认预测文件格式正确
    for protein_file, compound_file, pred_name in prediction_files:
        if not os.path.exists(protein_file):
            print(f"错误: 找不到蛋白质文件 {protein_file}")
            continue

        if not os.path.exists(compound_file):
            print(f"错误: 找不到化合物文件 {compound_file}")
            continue

        print(f"\n预测 {pred_name} 的相互作用:")

        # 为每种模型类型进行预测
        for model_type in ['标准随机分割', '蛋白质冷启动', '药物冷启动', '双重冷启动']:
            model_type_dir = os.path.join(model_dir, model_type)
            if not os.path.exists(model_type_dir):
                print(f"跳过 {model_type} 模型，目录不存在")
                continue

            print(f"\n使用 {model_type} 模型进行预测...")

            # 获取目录中的模型列表
            model_files = get_model_files_in_directory(model_type_dir)
            print(f"找到 {len(model_files)} 个模型文件: {', '.join(model_files)}")

            # 优先使用集成模型
            models = {}
            feature_pipeline = None

            # 首先尝试投票分类器
            voting_model, pipeline = load_model_and_pipeline(model_type_dir, '投票分类器')
            if voting_model:
                models['投票分类器'] = voting_model
                feature_pipeline = pipeline

            # 然后尝试堆叠分类器
            stacking_model, pipeline = load_model_and_pipeline(model_type_dir, '堆叠分类器')
            if stacking_model:
                models['堆叠分类器'] = stacking_model
                if not feature_pipeline:
                    feature_pipeline = pipeline

            # 如果没有集成模型，尝试使用其他模型
            if not models:
                # 常见模型名称
                common_models = ['随机森林', 'XGBoost', 'LightGBM', '梯度提升', 'SVM',
                                 '逻辑回归', 'K近邻', '极端随机树', 'CatBoost']

                # 首先加载常见模型
                for model_name in common_models:
                    if model_name in model_files:
                        model, curr_pipeline = load_model_and_pipeline(model_type_dir, model_name)
                        if model:
                            models[model_name] = model
                            if not feature_pipeline:
                                feature_pipeline = curr_pipeline

                # 如果还没有找到模型，加载所有模型
                if not models:
                    for model_name in model_files:
                        model, curr_pipeline = load_model_and_pipeline(model_type_dir, model_name)
                        if model:
                            models[model_name] = model
                            if not feature_pipeline:
                                feature_pipeline = curr_pipeline

            # 跳过此模型类型如果没有找到任何模型
            if not models or not feature_pipeline:
                print(f"没有在 {model_type} 中找到可用模型，跳过此模型类型")
                continue

            # 加载预测文件
            protein_df = pd.read_csv(protein_file)
            compound_df = pd.read_csv(compound_file)

            # 检查是否需要进行批量预测
            if len(protein_df) > 1 or len(compound_df) > 1:
                if BATCH_PREDICTION_ENABLED:
                    # 创建类型预测目录
                    prediction_dir = os.path.join(output_dir, f"{pred_name}_{model_type}")
                    os.makedirs(prediction_dir, exist_ok=True)

                    # 执行批量预测
                    batch_results = predict_batch_interactions(
                        models, protein_df, compound_df, feature_pipeline,
                        prediction_dir, model_type
                    )
                else:
                    print(f"检测到多个蛋白质或化合物，但批量预测功能已禁用。仅使用第一项进行预测。")
                    # 只使用第一行进行预测
                    protein_row = protein_df.iloc[0]
                    compound_row = compound_df.iloc[0]

                    # 尝试获取ID列
                    protein_id = "Unknown"
                    compound_id = "Unknown"
                    for col in protein_df.columns:
                        if col.lower() in ['id', 'gene_id', 'protein_id', 'name']:
                            protein_id = protein_row[col]
                            break
                    for col in compound_df.columns:
                        if col.lower() in ['id', 'compound_id', 'drug_id', 'name']:
                            compound_id = compound_row[col]
                            break

                    # 创建特征
                    sample_df = create_feature_for_pair(protein_row, compound_row, feature_pipeline)

                    # 处理样本
                    sample_scaled = process_specific_sample(sample_df, feature_pipeline)

                    # 创建预测目录
                    prediction_dir = os.path.join(output_dir, f"{pred_name}_{model_type}")
                    os.makedirs(prediction_dir, exist_ok=True)

                    # 预测相互作用
                    predictions = predict_interaction_single(
                        models, sample_scaled, protein_id, compound_id,
                        prediction_dir, model_type
                    )

                    # 保存预测结果
                    pd.DataFrame(predictions).to_csv(
                        os.path.join(prediction_dir, f'{protein_id}_{compound_id}_prediction.csv'),
                        index=False, encoding='utf-8-sig'
                    )
            else:
                # 单一预测
                protein_row = protein_df.iloc[0]
                compound_row = compound_df.iloc[0]

                # 尝试获取ID列
                protein_id = "Unknown"
                compound_id = "Unknown"
                for col in protein_df.columns:
                    if col.lower() in ['id', 'gene_id', 'protein_id', 'name']:
                        protein_id = protein_row[col]
                        break
                for col in compound_df.columns:
                    if col.lower() in ['id', 'compound_id', 'drug_id', 'name']:
                        compound_id = compound_row[col]
                        break

                # 创建特征
                sample_df = create_feature_for_pair(protein_row, compound_row, feature_pipeline)

                # 处理样本
                sample_scaled = process_specific_sample(sample_df, feature_pipeline)

                # 创建预测目录
                prediction_dir = os.path.join(output_dir, f"{pred_name}_{model_type}")
                os.makedirs(prediction_dir, exist_ok=True)

                # 预测相互作用
                predictions = predict_interaction_single(
                    models, sample_scaled, protein_id, compound_id,
                    prediction_dir, model_type
                )

                # 保存预测结果
                pd.DataFrame(predictions).to_csv(
                    os.path.join(prediction_dir, f'{protein_id}_{compound_id}_prediction.csv'),
                    index=False, encoding='utf-8-sig'
                )


def predict_batch_mode(model_dir, output_dir):
    """使用批量模式预测蛋白质-化合物对的交互作用"""
    if not BATCH_PREDICTION_ENABLED:
        print("批量预测功能已禁用")
        return

    protein_file = BATCH_MODE_PROTEIN_FILE
    compound_file = BATCH_MODE_COMPOUND_FILE

    if not os.path.exists(protein_file):
        print(f"错误: 找不到批量蛋白质文件 {protein_file}")
        return

    if not os.path.exists(compound_file):
        print(f"错误: 找不到批量化合物文件 {compound_file}")
        return

    print(f"\n执行批量预测: {protein_file} 和 {compound_file}")

    # 加载蛋白质和化合物数据
    protein_df, compound_df = load_protein_compound_data(protein_file, compound_file)

    if len(protein_df) == 0 or len(compound_df) == 0:
        print("错误: 蛋白质或化合物数据为空")
        return

    print(f"批量处理: 将预测 {len(protein_df)} 个蛋白质与 {len(compound_df)} 个化合物的组合")
    total_combinations = len(protein_df) * len(compound_df)
    if total_combinations > BATCH_MODE_MAX_COMBINATIONS:
        print(f"警告: 总组合数 {total_combinations} 超过限制 {BATCH_MODE_MAX_COMBINATIONS}，将随机选择部分组合")

    # 为每种模型类型进行预测
    for model_type in ['标准随机分割', '蛋白质冷启动', '药物冷启动', '双重冷启动']:
        model_type_dir = os.path.join(model_dir, model_type)
        if not os.path.exists(model_type_dir):
            print(f"跳过 {model_type} 模型，目录不存在")
            continue

        print(f"\n使用 {model_type} 模型进行批量预测...")

        # 获取目录中的模型列表
        model_files = get_model_files_in_directory(model_type_dir)
        print(f"找到 {len(model_files)} 个模型文件: {', '.join(model_files)}")

        # 优先使用集成模型
        models = {}
        feature_pipeline = None

        # 首先尝试投票分类器
        voting_model, pipeline = load_model_and_pipeline(model_type_dir, '投票分类器')
        if voting_model:
            models['投票分类器'] = voting_model
            feature_pipeline = pipeline

        # 然后尝试堆叠分类器
        stacking_model, pipeline = load_model_and_pipeline(model_type_dir, '堆叠分类器')
        if stacking_model:
            models['堆叠分类器'] = stacking_model
            if not feature_pipeline:
                feature_pipeline = pipeline

        # 如果没有集成模型，尝试使用其他模型
        if not models:
            # 常见模型名称
            common_models = ['随机森林', 'XGBoost', 'LightGBM', '梯度提升', 'SVM',
                             '逻辑回归', 'K近邻', '极端随机树', 'CatBoost']

            # 首先加载常见模型
            for model_name in common_models:
                if model_name in model_files:
                    model, curr_pipeline = load_model_and_pipeline(model_type_dir, model_name)
                    if model:
                        models[model_name] = model
                        if not feature_pipeline:
                            feature_pipeline = curr_pipeline

            # 如果还没有找到模型，加载所有模型
            if not models:
                for model_name in model_files:
                    model, curr_pipeline = load_model_and_pipeline(model_type_dir, model_name)
                    if model:
                        models[model_name] = model
                        if not feature_pipeline:
                            feature_pipeline = curr_pipeline

        # 跳过此模型类型如果没有找到任何模型
        if not models or not feature_pipeline:
            print(f"没有在 {model_type} 中找到可用模型，跳过此模型类型")
            continue

        # 创建批量预测目录
        batch_dir = os.path.join(output_dir, f"batch_{model_type}")
        os.makedirs(batch_dir, exist_ok=True)

        # 执行批量预测
        batch_results = predict_batch_interactions(
            models, protein_df, compound_df, feature_pipeline,
            batch_dir, model_type
        )


def main():
    """主函数，执行完整的工作流程"""
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

        # 设置日志文件
        if LOGFILE_ENABLED:
            log_file = os.path.join(output_dir, 'execution_log.txt')
            # 捕获并同时输出到控制台和日志文件
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

        # 结果字典，保存所有模型训练结果
        all_model_results = {}

        # 训练标准随机分割模型（非冷启动）
        if ENABLE_STANDARD_RANDOM:
            all_model_results['标准随机分割'] = train_standard_model(
                df, protein_id_col, compound_id_col, label_col,
                X, protein_cols, compound_cols, output_dir
            )

        # 训练三种冷启动模型
        # 1. 蛋白质冷启动
        if ENABLE_PROTEIN_COLD_START:
            all_model_results['蛋白质冷启动'] = train_cold_start_model(
                '蛋白质冷启动', df, protein_id_col, compound_id_col, label_col,
                X, protein_cols, compound_cols, output_dir
            )

        # 2. 药物冷启动
        if ENABLE_DRUG_COLD_START:
            all_model_results['药物冷启动'] = train_cold_start_model(
                '药物冷启动', df, protein_id_col, compound_id_col, label_col,
                X, protein_cols, compound_cols, output_dir
            )

        # 3. 双重冷启动
        if ENABLE_DUAL_COLD_START:
            all_model_results['双重冷启动'] = train_cold_start_model(
                '双重冷启动', df, protein_id_col, compound_id_col, label_col,
                X, protein_cols, compound_cols, output_dir
            )

        # 比较所有模型策略的性能
        if len(all_model_results) > 1:
            compare_all_models(all_model_results, output_dir)

            # 同时也比较冷启动模型的性能
            cold_start_results = {k: v for k, v in all_model_results.items() if k != '标准随机分割' and v is not None}
            if len(cold_start_results) > 1:
                compare_cold_start_models(cold_start_results, output_dir)

        # 进行普通预测
        if PREDICTION_ENABLED and PREDICTION_FILES:
            predict_files(output_dir, PREDICTION_FILES, output_dir)

        # 进行批量预测
        if BATCH_PREDICTION_ENABLED:
            predict_batch_mode(output_dir, output_dir)

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
                            "AUPR": f"{model_result['aupr']:.4f}"  # 添加AUPR
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
                "CPU核心数": CPU_COUNT
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
                            f.write(f"  AUPR: {metrics['aupr']:.4f}\n\n")  # 添加AUPR
        except Exception as e:
            print(f"生成最终总结报告出错，但不影响程序完成: {e}")

        print("\n处理完成！所有结果已保存到目录:", output_dir)
        print(f"总执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return all_model_results
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()