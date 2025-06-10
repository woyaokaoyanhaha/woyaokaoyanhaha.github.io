import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from datetime import datetime
import itertools

# =============================================================================
# 用户可配置参数 - 修改这些参数来运行不同的预测配置
# =============================================================================
# 必须参数
MODEL_DIR = "drug_coldstart_results_20250530_121806"  # 模型目录路径，包含模型文件和特征管道
PROTEIN_FILE = "molecular_descriptors_output - 副本.csv"  # 蛋白质特征文件路径
COMPOUND_FILE = "drugbank小分子化合物特征.csv"  # 化合物特征文件路径

# 可选参数
OUTPUT_DIR = ("")  # 输出目录，为空则自动生成 predictions_日期时间
MODEL_TYPE = "drug_cold"  # 选择: "standard", "protein_cold", "drug_cold", "dual_cold", 或 "all"

# 模型选择配置
PREFERRED_MODELS = ["catboost"]  # 优先使用的模型列表，例如 ["投票分类器", "随机森林"]，为空则尝试所有模型
EXCLUDED_MODELS = []  # 要排除的模型列表，例如 ["SVM", "K近邻"]，为空则不排除任何模型

# 批量预测配置
BATCH_MODE = True  # True: 处理所有蛋白质-化合物对的组合; False: 只处理第一个
MAX_COMBINATIONS = 999999999  # 最大处理的蛋白质-化合物组合数量，防止组合爆炸

NO_PLOT = True  # True: 不生成图形，仅输出数据; False: 生成图形和数据
# =============================================================================

# 全局设置并行任务核心数
N_JOBS = -1  # -1 表示使用所有可用核心
CPU_COUNT = cpu_count()
print(f"检测到 {CPU_COUNT} 个CPU核心可用")

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


# 设置全局中文字体
chinese_font_path = setup_chinese_font()
if chinese_font_path and os.path.exists(chinese_font_path):
    chinese_font = FontProperties(fname=chinese_font_path)
    print(f"已加载中文字体: {chinese_font_path}")
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
else:
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
    axes = fig.get_axes()
    for ax in axes:
        if title and ax == axes[0]:
            ax.set_title(title, fontproperties=chinese_font, fontsize=14)
        if xlabel:
            ax.set_xlabel(xlabel, fontproperties=chinese_font, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontproperties=chinese_font, fontsize=12)
        if xtick_labels:
            ax.set_xticklabels(xtick_labels, fontproperties=chinese_font, fontsize=10)
        else:
            if ax.get_xticklabels():
                for label in ax.get_xticklabels():
                    label.set_fontproperties(chinese_font)
        if ytick_labels:
            ax.set_yticklabels(ytick_labels, fontproperties=chinese_font, fontsize=10)
        else:
            if ax.get_yticklabels():
                for label in ax.get_yticklabels():
                    label.set_fontproperties(chinese_font)
    if legend_labels:
        for ax in axes:
            if ax.get_legend():
                for text in ax.get_legend().get_texts():
                    text.set_fontproperties(chinese_font)
    if text_annotations:
        for annotation in text_annotations:
            if 'ax' in annotation:
                target_ax = annotation['ax']
            else:
                target_ax = axes[0]
            target_ax.text(annotation['x'], annotation['y'], annotation['text'],
                           fontproperties=chinese_font, fontsize=annotation.get('fontsize', 10),
                           ha=annotation.get('ha', 'center'), va=annotation.get('va', 'center'))
    fig.tight_layout()


# 使用joblib并行处理特征增强
def process_feature_batch(X_batch, protein_cols, compound_cols):
    """处理一批次的特征增强"""
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
    """
    使用特征工程管道处理样本
    """
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
    """
    使用训练好的模型预测单个蛋白质-化合物相互作用
    """
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


def predict_interaction_batch(models, protein_df, compound_df, feature_pipeline, output_dir, cold_start_type=""):
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
    if len(combinations) > MAX_COMBINATIONS:
        print(
            f"警告: 蛋白质-化合物组合数 ({len(combinations)}) 超过最大限制 ({MAX_COMBINATIONS})，将随机选择 {MAX_COMBINATIONS} 个组合")
        np.random.seed(42)  # 设置随机种子确保结果可重复
        combinations = [combinations[i] for i in np.random.choice(len(combinations), MAX_COMBINATIONS, replace=False)]

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
        predictions = predict_interaction_single(models, sample_scaled, protein_id, compound_id, output_dir,
                                                 cold_start_type)

        # 添加到结果集
        all_predictions.extend(predictions)

    # 将结果转换为DataFrame并保存
    if all_predictions:
        result_df = pd.DataFrame(all_predictions)

        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        result_df.to_csv(os.path.join(output_dir, f'batch_predictions{suffix}.csv'),
                         index=False, encoding='utf-8-sig')

        # 筛选集成模型结果并绘制热图
        if not NO_PLOT:
            try:
                ensemble_results = result_df[result_df['模型'] == '集成模型'].copy()

                # 如果有足够多的数据，创建热图
                if len(ensemble_results) > 1:
                    # 数据诊断
                    print(f"集成模型结果数量: {len(ensemble_results)}")
                    print(f"唯一蛋白质ID数量: {ensemble_results['蛋白质ID'].nunique()}")
                    print(f"唯一化合物ID数量: {ensemble_results['化合物ID'].nunique()}")

                    # 检查是否存在多行只有一个蛋白质或化合物的情况
                    if ensemble_results['蛋白质ID'].nunique() == 1 or ensemble_results['化合物ID'].nunique() == 1:
                        print("警告: 只有一种蛋白质或化合物，热图将显示为单行或单列")

                    # 创建交叉表以便绘制热图（添加参数处理缺失值）
                    pivot_table = ensemble_results.pivot_table(
                        index='蛋白质ID',
                        columns='化合物ID',
                        values='概率',
                        fill_value=0  # 填充缺失值，避免NaN
                    )

                    # 打印诊断信息
                    print(f"透视表形状: {pivot_table.shape}")
                    print(f"透视表非空值比例: {pivot_table.notnull().mean().mean():.2%}")

                    # 调整图形大小，根据数据规模动态调整
                    fig_width = max(12, min(pivot_table.shape[1] * 0.5, 40))
                    fig_height = max(10, min(pivot_table.shape[0] * 0.5, 30))

                    plt.figure(figsize=(fig_width, fig_height))

                    # 确定是否显示注释（数据点太多时不显示数值）
                    show_annot = pivot_table.size <= 400  # 如果单元格总数超过400个，不显示注释

                    # 使用改进的热图参数
                    ax = sns.heatmap(
                        pivot_table,
                        annot=show_annot,  # 数据较少时显示具体数值
                        fmt=".2f",  # 保留两位小数
                        cmap='RdBu_r',  # 使用红蓝配色方案
                        vmin=0,
                        vmax=1,
                        linewidths=0.5 if pivot_table.size < 200 else 0,  # 数据较多时不显示网格线
                        cbar_kws={'label': '相互作用概率'},
                        square=True  # 保持单元格为正方形，提高辨识度
                    )

                    # 调整标签大小，避免重叠
                    label_fontsize = max(6, min(10, 500 / max(pivot_table.shape)))
                    plt.xticks(rotation=90, fontsize=label_fontsize)  # 更改为90度避免x轴标签重叠
                    plt.yticks(fontsize=label_fontsize)

                    # 使用中文字体 - 使用更加健壮的方式
                    try:
                        plot_with_chinese_font(
                            plt.gcf(),
                            title=f'{cold_start_type} 蛋白质-化合物相互作用概率热图',
                            xlabel='化合物',
                            ylabel='蛋白质'
                        )
                    except Exception as font_error:
                        # 如果自定义字体函数失败，使用基本方案
                        plt.title(f'{cold_start_type} 蛋白质-化合物相互作用概率热图')
                        plt.xlabel('化合物')
                        plt.ylabel('蛋白质')
                        print(f"中文字体应用失败，使用默认字体: {font_error}")

                    # 应用紧凑布局并保存
                    plt.tight_layout()
                    heatmap_path = os.path.join(output_dir, f'interaction_heatmap{suffix}.png')
                    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"热图已保存到: {heatmap_path}")

                    # 对于大型数据，创建分块热图
                    if pivot_table.size > 400:  # 如果数据点过多，生成分块热图
                        create_chunked_heatmaps(pivot_table, cold_start_type, output_dir, suffix)

            except Exception as e:
                print(f"创建热图出错: {e}")
                import traceback
                traceback.print_exc()  # 打印详细错误信息以便调试

        # 创建统计结果
        try:
            # 每个蛋白质与多少化合物有相互作用
            ensemble_results = result_df[result_df['模型'] == '集成模型'].copy()

            # 将"预测"列转换为数值，处理可能的错误值
            ensemble_results['预测'] = pd.to_numeric(ensemble_results['预测'], errors='coerce').fillna(0).astype(int)

            # 统计每个蛋白质有多少相互作用
            protein_interactions = ensemble_results.groupby('蛋白质ID')['预测'].sum().reset_index()
            protein_interactions.columns = ['蛋白质ID', '相互作用数']
            protein_interactions = protein_interactions.sort_values('相互作用数', ascending=False)

            # 统计每个化合物有多少相互作用
            compound_interactions = ensemble_results.groupby('化合物ID')['预测'].sum().reset_index()
            compound_interactions.columns = ['化合物ID', '相互作用数']
            compound_interactions = compound_interactions.sort_values('相互作用数', ascending=False)

            # 保存统计结果
            protein_interactions.to_csv(os.path.join(output_dir, f'protein_interaction_stats{suffix}.csv'),
                                        index=False, encoding='utf-8-sig')
            compound_interactions.to_csv(os.path.join(output_dir, f'compound_interaction_stats{suffix}.csv'),
                                         index=False, encoding='utf-8-sig')

            # 绘制蛋白质相互作用数量柱状图
            if not NO_PLOT and len(protein_interactions) > 1:
                plt.figure(figsize=(12, 8))
                sns.barplot(x='蛋白质ID', y='相互作用数', data=protein_interactions[:20])  # 只显示前20个

                plot_with_chinese_font(
                    plt.gcf(),
                    title=f'{cold_start_type} 蛋白质相互作用数量统计',
                    xlabel='蛋白质ID',
                    ylabel='相互作用数量'
                )

                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'protein_interaction_stats{suffix}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()

                # 绘制化合物相互作用数量柱状图
                plt.figure(figsize=(12, 8))
                sns.barplot(x='化合物ID', y='相互作用数', data=compound_interactions[:20])  # 只显示前20个

                plot_with_chinese_font(
                    plt.gcf(),
                    title=f'{cold_start_type} 化合物相互作用数量统计',
                    xlabel='化合物ID',
                    ylabel='相互作用数量'
                )

                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'compound_interaction_stats{suffix}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"创建统计结果出错，但不影响主要功能: {e}")

        return result_df
    else:
        print("没有成功完成任何预测")
        return pd.DataFrame()


def create_chunked_heatmaps(pivot_table, title_prefix, output_dir, suffix, chunk_size=20):
    """为大型数据集创建分块热图"""
    print(f"数据点过多 ({pivot_table.shape[0]}x{pivot_table.shape[1]}), 正在生成分块热图...")

    # 获取行和列标签
    proteins = pivot_table.index.tolist()
    compounds = pivot_table.columns.tolist()

    # 创建分块热图目录
    chunked_dir = os.path.join(output_dir, 'chunked_heatmaps')
    os.makedirs(chunked_dir, exist_ok=True)

    # 分块处理行（蛋白质）
    for i in range(0, len(proteins), chunk_size):
        p_chunk = proteins[i:i + chunk_size]

        # 分块处理列（化合物）
        for j in range(0, len(compounds), chunk_size):
            c_chunk = compounds[j:j + chunk_size]

            # 提取子矩阵
            sub_table = pivot_table.loc[p_chunk, c_chunk]

            # 创建子热图
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(
                sub_table,
                annot=True,  # 分块后可以显示数值
                fmt=".2f",
                cmap='RdBu_r',
                vmin=0,
                vmax=1,
                linewidths=0.5,
                cbar_kws={'label': '相互作用概率'},
                square=True
            )

            # 设置标题和标签
            chunk_title = f'{title_prefix} 相互作用热图 (块 {i // chunk_size + 1},{j // chunk_size + 1})'

            try:
                plot_with_chinese_font(
                    plt.gcf(),
                    title=chunk_title,
                    xlabel='化合物',
                    ylabel='蛋白质'
                )
            except:
                plt.title(chunk_title)
                plt.xlabel('化合物')
                plt.ylabel('蛋白质')

            # 旋转标签以避免重叠
            plt.xticks(rotation=45, ha='right')

            # 保存分块热图
            chunk_path = os.path.join(chunked_dir,
                                      f'heatmap_chunk_{i // chunk_size + 1}_{j // chunk_size + 1}{suffix}.png')
            plt.tight_layout()
            plt.savefig(chunk_path, dpi=300, bbox_inches='tight')
            plt.close()

    print(f"已生成分块热图，保存在: {chunked_dir}")




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


def main():
    """主函数，执行预测流程"""
    print("开始执行蛋白质-化合物相互作用预测...")

    # 处理输入参数
    model_dir = MODEL_DIR
    protein_file = PROTEIN_FILE
    compound_file = COMPOUND_FILE
    model_type = MODEL_TYPE.lower()
    preferred_models = PREFERRED_MODELS
    excluded_models = EXCLUDED_MODELS
    batch_mode = BATCH_MODE
    no_plot = NO_PLOT

    # 设置输出目录
    if OUTPUT_DIR:
        output_dir = OUTPUT_DIR
    else:
        output_dir = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存在目录: {output_dir}")

    # 验证输入文件
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录 '{model_dir}' 不存在")
        return

    if not os.path.exists(protein_file):
        print(f"错误: 蛋白质特征文件 '{protein_file}' 不存在")
        return

    if not os.path.exists(compound_file):
        print(f"错误: 化合物特征文件 '{compound_file}' 不存在")
        return

    # 加载蛋白质和化合物数据
    protein_df, compound_df = load_protein_compound_data(protein_file, compound_file)

    if len(protein_df) == 0 or len(compound_df) == 0:
        print("错误: 蛋白质或化合物数据为空")
        return

    # 输出批处理模式信息
    if batch_mode and (len(protein_df) > 1 or len(compound_df) > 1):
        print(f"批量处理模式: 将预测 {len(protein_df)} 个蛋白质与 {len(compound_df)} 个化合物的所有可能组合")
        total_combinations = len(protein_df) * len(compound_df)
        if total_combinations > MAX_COMBINATIONS:
            print(f"警告: 总组合数 {total_combinations} 超过限制 {MAX_COMBINATIONS}，将随机选择部分组合")
    else:
        print("单一预测模式: 只处理第一个蛋白质与第一个化合物的相互作用")
        # 确保只使用第一行数据
        protein_df = protein_df.iloc[[0]]
        compound_df = compound_df.iloc[[0]]

    # 选择模型类型
    model_types_map = {
        'standard': ['标准随机分割'],
        'protein_cold': ['蛋白质冷启动'],
        'drug_cold': ['药物冷启动'],
        'dual_cold': ['双重冷启动'],
        'all': ['标准随机分割', '蛋白质冷启动', '药物冷启动', '双重冷启动']
    }

    model_types = model_types_map.get(model_type, model_types_map['all'])

    # 创建日志文件
    log_file = os.path.join(output_dir, 'prediction_log.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"蛋白质-化合物相互作用预测 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型目录: {model_dir}\n")
        f.write(f"蛋白质文件: {protein_file} (包含 {len(protein_df)} 个蛋白质)\n")
        f.write(f"化合物文件: {compound_file} (包含 {len(compound_df)} 个化合物)\n")
        f.write(f"选择的模型类型: {model_type} ({', '.join(model_types)})\n")
        f.write(f"优先使用模型: {', '.join(preferred_models) if preferred_models else '无'}\n")
        f.write(f"排除的模型: {', '.join(excluded_models) if excluded_models else '无'}\n")
        f.write(f"批量处理模式: {'是' if batch_mode else '否'}\n\n")

    # 整合所有模型类型的预测结果
    all_model_type_results = []

    # 对每种模型类型进行预测
    for model_type in model_types:
        model_dir_path = os.path.join(model_dir, model_type)
        if not os.path.exists(model_dir_path):
            print(f"警告: 找不到模型目录 '{model_dir_path}'，跳过此模型类型")
            continue

        print(f"\n加载并使用 {model_type} 模型进行预测...")

        # 设置模型加载顺序
        models = {}
        feature_pipeline = None

        # 获取目录中所有可用的模型文件
        available_models = get_model_files_in_directory(model_dir_path)
        print(f"在目录 {model_dir_path} 中找到以下模型: {', '.join(available_models)}")

        # 过滤排除的模型
        if excluded_models:
            available_models = [m for m in available_models if m not in excluded_models]
            print(f"排除模型后的可用模型: {', '.join(available_models)}")

        # 如果指定了优先模型，首先尝试加载这些模型
        if preferred_models:
            for preferred_model in preferred_models:
                if preferred_model in available_models:
                    model, feature_pipeline = load_model_and_pipeline(model_dir_path, preferred_model)
                    if model is not None:
                        models[preferred_model] = model
                        print(f"成功加载优先指定的模型: {preferred_model}")

        # 如果没有找到优先模型或未指定，尝试加载集成模型
        if not models:
            for ensemble_name in ['投票分类器', '堆叠分类器']:
                if ensemble_name in available_models and ensemble_name not in excluded_models:
                    model, feature_pipeline = load_model_and_pipeline(model_dir_path, ensemble_name)
                    if model is not None:
                        models[ensemble_name] = model
                        print(f"成功加载集成模型: {ensemble_name}")
                        break

        # 如果仍没有找到模型，尝试加载所有可用模型
        if not models:
            # 常见模型名称
            common_models = ['随机森林', 'XGBoost', 'LightGBM', '梯度提升', 'SVM',
                             '逻辑回归', 'K近邻', '极端随机树', 'CatBoost']

            # 首先尝试常见的模型名称
            for model_name in common_models:
                if model_name in available_models and model_name not in excluded_models:
                    model, curr_pipeline = load_model_and_pipeline(model_dir_path, model_name)
                    if model is not None:
                        models[model_name] = model
                        if feature_pipeline is None:
                            feature_pipeline = curr_pipeline
                        print(f"成功加载模型: {model_name}")

            # 如果仍未找到模型，尝试目录中的其他模型文件
            if not models:
                for model_name in available_models:
                    if model_name not in excluded_models:
                        model, curr_pipeline = load_model_and_pipeline(model_dir_path, model_name)
                        if model is not None:
                            models[model_name] = model
                            if feature_pipeline is None:
                                feature_pipeline = curr_pipeline
                            print(f"成功加载模型: {model_name}")

        # 如果仍然没找到模型，继续下一个模型类型
        if not models:
            print(f"在 {model_dir_path} 中没有找到可用模型，跳过此模型类型")
            continue

        # 创建模型类型输出目录
        model_output_dir = os.path.join(output_dir, model_type)
        os.makedirs(model_output_dir, exist_ok=True)

        # 使用找到的feature_pipeline处理样本
        if feature_pipeline:
            if batch_mode and (len(protein_df) > 1 or len(compound_df) > 1):
                # 批量预测
                results_df = predict_interaction_batch(
                    models, protein_df, compound_df, feature_pipeline,
                    model_output_dir, cold_start_type=model_type
                )
            else:
                # 单一预测
                # 获取第一个蛋白质和化合物
                protein_row = protein_df.iloc[0]
                compound_row = compound_df.iloc[0]

                # 尝试获取ID列
                protein_id = protein_row.get('id', protein_row.get('ID', protein_row.get('gene_id', 'Protein_0')))
                compound_id = compound_row.get('id',
                                               compound_row.get('ID', compound_row.get('compound_id', 'Compound_0')))

                # 创建特征
                sample_df = create_feature_for_pair(protein_row, compound_row, feature_pipeline)

                # 处理样本
                sample_scaled = process_specific_sample(sample_df, feature_pipeline)

                # 预测相互作用
                predictions = predict_interaction_single(
                    models, sample_scaled, protein_id, compound_id,
                    model_output_dir, cold_start_type=model_type
                )

                # 保存预测结果
                results_df = pd.DataFrame(predictions)
                results_df.to_csv(
                    os.path.join(model_output_dir, 'prediction_result.csv'),
                    index=False, encoding='utf-8-sig'
                )

            # 提取集成模型的结果
            if not results_df.empty:
                ensemble_results = results_df[results_df['模型'] == '集成模型'].copy()

                if not ensemble_results.empty:
                    # 添加到汇总结果中
                    for _, row in ensemble_results.iterrows():
                        # 创建结果记录
                        result_record = {
                            '模型类型': model_type,
                            '预测': row['预测'],
                            '概率': row['概率']
                        }

                        # 添加蛋白质和化合物ID
                        if '蛋白质ID' in row and '化合物ID' in row:
                            result_record['蛋白质ID'] = row['蛋白质ID']
                            result_record['化合物ID'] = row['化合物ID']

                        all_model_type_results.append(result_record)

    # 汇总所有模型类型的预测结果
    if all_model_type_results:
        combined_df = pd.DataFrame(all_model_type_results)
        combined_df.to_csv(os.path.join(output_dir, 'combined_predictions.csv'),
                           index=False, encoding='utf-8-sig')

        print("\n所有模型的综合预测结果摘要:")
        print("=" * 80)

        # 显示结果摘要
        if '蛋白质ID' in combined_df.columns and '化合物ID' in combined_df.columns:
            # 批量预测结果摘要
            interaction_count = len(combined_df[combined_df['预测'] == 1])
            total_pairs = len(combined_df)

            print(f"总预测蛋白质-化合物对数: {total_pairs}")
            print(f"预测有相互作用的对数: {interaction_count} ({interaction_count / total_pairs:.2%})")
            print(f"预测相互作用概率平均值: {combined_df['概率'].mean():.4f}")

            # 获取最可能有相互作用的前5个蛋白质-化合物对
            if total_pairs > 1:
                top5 = combined_df.sort_values('概率', ascending=False).head(5)
                print("\n相互作用概率最高的5个蛋白质-化合物对:")
                for i, (_, row) in enumerate(top5.iterrows()):
                    pred_text = "相互作用" if row['预测'] == 1 else "无相互作用"
                    print(
                        f"{i + 1}. 蛋白质 {row['蛋白质ID']} - 化合物 {row['化合物ID']}: {pred_text} (概率: {row['概率']:.4f}, 模型: {row['模型类型']})")
        else:
            # 单一预测结果
            print(f"{'模型类型':<15}{'预测结果':<15}{'概率':<10}")
            print("-" * 80)

            for _, row in combined_df.iterrows():
                pred_text = "相互作用" if row['预测'] == 1 else "无相互作用" if row['预测'] == 0 else "错误"
                print(f"{row['模型类型']:<15}{pred_text:<15}{row['概率']:.4f}")

        print("=" * 80)

        # 可视化综合预测结果
        if not no_plot:
            try:
                if '蛋白质ID' not in combined_df.columns:
                    # 单一预测模型比较图
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='模型类型', y='概率', data=combined_df)
                    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)

                    plot_with_chinese_font(
                        plt.gcf(),
                        title='不同模型类型的相互作用概率预测',
                        xlabel='模型类型',
                        ylabel='概率'
                    )

                    plt.ylim(0, 1)
                    plt.savefig(os.path.join(output_dir, 'combined_predictions.png'), dpi=300, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                print(f"绘制综合预测结果图出错，但不影响程序继续运行: {e}")

        # 预测共识结果
        try:
            if '蛋白质ID' not in combined_df.columns:
                # 单一预测的共识结果
                valid_preds = combined_df[combined_df['预测'].isin([0, 1])]
                if not valid_preds.empty:
                    avg_prob = valid_preds['概率'].mean()
                    consensus_pred = 1 if avg_prob >= 0.5 else 0
                    consensus_text = "相互作用" if consensus_pred == 1 else "无相互作用"

                    print(f"\n最终共识预测结果: {consensus_text} (平均概率: {avg_prob:.4f})")

                    with open(os.path.join(output_dir, 'final_prediction.txt'), 'w', encoding='utf-8') as f:
                        if len(protein_df) > 0:
                            protein_id = protein_df.iloc[0].get('id', protein_df.iloc[0].get('ID',
                                                                                             protein_df.iloc[0].get(
                                                                                                 'gene_id', 'Unknown')))
                            f.write(f"预测的蛋白质: {protein_id}\n")

                        if len(compound_df) > 0:
                            compound_id = compound_df.iloc[0].get('id', compound_df.iloc[0].get('ID',
                                                                                                compound_df.iloc[0].get(
                                                                                                    'compound_id',
                                                                                                    'Unknown')))
                            f.write(f"预测的化合物: {compound_id}\n")

                        f.write(f"最终共识预测结果: {consensus_text}\n")
                        f.write(f"平均相互作用概率: {avg_prob:.4f}\n\n")

                        f.write("模型细节:\n")
                        for _, row in combined_df.iterrows():
                            pred_text = "相互作用" if row['预测'] == 1 else "无相互作用" if row['预测'] == 0 else "错误"
                            f.write(f"{row['模型类型']}: {pred_text} (概率: {row['概率']:.4f})\n")
                else:
                    print("\n所有模型预测均失败，无法给出共识结果")
        except Exception as e:
            print(f"计算共识预测出错: {e}")
    else:
        print("\n没有成功完成任何预测，请检查输入文件和模型文件夹")

    print(f"\n预测完成! 所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()