import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import stats
import warnings
from matplotlib.font_manager import FontProperties
import argparse
from datetime import datetime

# 忽略特定警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ======================== 参数设置 ========================
# 默认参数，可通过命令行参数修改
INPUT_FILE = "核受体PsePSSM和PseACC特征.csv"  # 输入文件路径
OUTPUT_DIR = f"feature_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # 输出目录
RANDOM_STATE = 42  # 随机数种子
TOP_FEATURES_DEFAULT = 20  # 显示前N个最具区分能力的特征
PLOT_DPI_DEFAULT = 300  # 图形DPI


# ====================== 中文字体设置 ======================
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


def plot_with_chinese_font(title=None, xlabel=None, ylabel=None):
    """使用中文字体设置图表文本"""
    if title and chinese_font:
        plt.title(title, fontproperties=chinese_font, fontsize=14)
    if xlabel and chinese_font:
        plt.xlabel(xlabel, fontproperties=chinese_font, fontsize=12)
    if ylabel and chinese_font:
        plt.ylabel(ylabel, fontproperties=chinese_font, fontsize=12)
    plt.tight_layout()


# ================== 数据加载和预处理函数 ==================
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


# ================== 特征区分能力测试函数 ==================
def test_feature_discrimination(X, y, protein_features, compound_features, output_dir, top_features=20):
    """测试特征的区分能力并生成报告"""
    os.makedirs(output_dir, exist_ok=True)

    # 划分正负样本
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    print(f"正样本数量: {len(X_pos)}, 负样本数量: {len(X_neg)}")

    # 生成报告文件
    report_file = os.path.join(output_dir, "feature_discrimination_report.txt")
    csv_file = os.path.join(output_dir, "feature_discrimination_scores.csv")

    # 1. 计算每个特征在正负样本间的差异统计量
    feature_stats = {}
    feature_pvalues = {}

    print("计算特征在正负样本间的统计差异...")
    for feature in X.columns:
        # t检验
        t_stat, p_value = stats.ttest_ind(X_pos[feature], X_neg[feature], equal_var=False)
        feature_stats[feature] = abs(t_stat)
        feature_pvalues[feature] = p_value

    # 将结果排序
    sorted_features = sorted(feature_stats.items(), key=lambda x: x[1], reverse=True)

    # 2. 计算特征的互信息
    print("计算特征与标签的互信息...")
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    mi_features = [(X.columns[i], mi_scores[i]) for i in range(len(mi_scores))]
    mi_features.sort(key=lambda x: x[1], reverse=True)

    # 3. 使用随机森林计算特征重要性
    print("计算随机森林特征重要性...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X, y)
    importances = rf.feature_importances_
    rf_features = [(X.columns[i], importances[i]) for i in range(len(importances))]
    rf_features.sort(key=lambda x: x[1], reverse=True)

    # 4. 使用ANOVA计算F值
    print("使用ANOVA计算特征F值...")
    f_values, p_values = f_classif(X, y)
    f_features = [(X.columns[i], f_values[i]) for i in range(len(f_values))]
    f_features.sort(key=lambda x: x[1], reverse=True)

    # 5. 计算蛋白质特征和化合物特征的平均区分能力
    protein_t_stats = [feature_stats[f] for f in protein_features if f in feature_stats]
    compound_t_stats = [feature_stats[f] for f in compound_features if f in feature_stats]

    protein_mi = [mi_scores[i] for i, f in enumerate(X.columns) if f in protein_features]
    compound_mi = [mi_scores[i] for i, f in enumerate(X.columns) if f in compound_features]

    protein_rf = [importances[i] for i, f in enumerate(X.columns) if f in protein_features]
    compound_rf = [importances[i] for i, f in enumerate(X.columns) if f in compound_features]

    protein_f = [f_values[i] for i, f in enumerate(X.columns) if f in protein_features]
    compound_f = [f_values[i] for i, f in enumerate(X.columns) if f in compound_features]

    # 6. 创建特征区分能力评分数据框
    discrimination_scores = []
    for feature in X.columns:
        idx = list(X.columns).index(feature)
        feature_type = "蛋白质特征" if feature in protein_features else "化合物特征"

        discrimination_scores.append({
            "特征名": feature,
            "特征类型": feature_type,
            "t统计量": feature_stats[feature],
            "p值": feature_pvalues[feature],
            "互信息": mi_scores[idx],
            "随机森林重要性": importances[idx],
            "F值": f_values[idx],
            "平均区分分数": (
                                # 归一化并平均不同指标
                                    abs(feature_stats[feature]) / max(abs(v) for v in feature_stats.values()) +
                                    mi_scores[idx] / max(mi_scores) +
                                    importances[idx] / max(importances) +
                                    f_values[idx] / max(f_values)
                            ) / 4
        })

    # 转换为DataFrame并排序
    df_scores = pd.DataFrame(discrimination_scores)
    df_scores.sort_values("平均区分分数", ascending=False, inplace=True)

    # 保存到CSV
    df_scores.to_csv(csv_file, index=False, encoding='utf-8-sig')

    # 7. 写入报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("特征区分能力分析报告\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"数据集信息:\n")
        f.write(f"- 样本总数: {len(X)}\n")
        f.write(f"- 正样本数: {len(X_pos)}\n")
        f.write(f"- 负样本数: {len(X_neg)}\n")
        f.write(f"- 蛋白质特征数: {len(protein_features)}\n")
        f.write(f"- 化合物特征数: {len(compound_features)}\n")
        f.write(f"- 特征总数: {len(X.columns)}\n\n")

        # 报告特征类型的平均区分能力
        f.write("特征类型平均区分能力比较:\n")
        f.write("-" * 50 + "\n")
        f.write(f"蛋白质特征 t统计量平均值: {np.mean(protein_t_stats):.4f}\n")
        f.write(f"化合物特征 t统计量平均值: {np.mean(compound_t_stats):.4f}\n")
        f.write(f"蛋白质特征 互信息平均值: {np.mean(protein_mi):.4f}\n")
        f.write(f"化合物特征 互信息平均值: {np.mean(compound_mi):.4f}\n")
        f.write(f"蛋白质特征 随机森林重要性平均值: {np.mean(protein_rf):.4f}\n")
        f.write(f"化合物特征 随机森林重要性平均值: {np.mean(compound_rf):.4f}\n")
        f.write(f"蛋白质特征 F值平均值: {np.mean(protein_f):.4f}\n")
        f.write(f"化合物特征 F值平均值: {np.mean(compound_f):.4f}\n\n")

        # 每种评分标准的显著差异特征数
        sig_t = sum(1 for p in feature_pvalues.values() if p < 0.05)
        sig_f = sum(1 for p in p_values if p < 0.05)

        f.write(f"统计学显著特征:\n")
        f.write(f"- t检验显著特征 (p<0.05): {sig_t}/{len(feature_pvalues)} ({sig_t / len(feature_pvalues):.2%})\n")
        f.write(f"- ANOVA检验显著特征 (p<0.05): {sig_f}/{len(p_values)} ({sig_f / len(p_values):.2%})\n\n")

        # 前N个最具区分能力的特征
        f.write(f"前{top_features}个最具区分能力的特征 (基于平均区分分数):\n")
        f.write("-" * 50 + "\n")
        for i, row in df_scores.head(top_features).iterrows():
            f.write(f"{i + 1}. {row['特征名']} ({row['特征类型']})\n")
            f.write(f"   平均区分分数: {row['平均区分分数']:.4f}\n")
            f.write(f"   t统计量: {row['t统计量']:.4f} (p={row['p值']:.4e})\n")
            f.write(f"   互信息: {row['互信息']:.4f}\n")
            f.write(f"   随机森林重要性: {row['随机森林重要性']:.4f}\n")
            f.write(f"   F值: {row['F值']:.4f}\n\n")

        # 整体区分能力评估
        avg_p_value = np.mean(list(feature_pvalues.values()))
        avg_mi = np.mean(mi_scores)
        f.write("整体区分能力评估:\n")
        f.write("-" * 50 + "\n")
        f.write(f"- 平均 p 值: {avg_p_value:.4f}\n")
        f.write(f"- 平均互信息: {avg_mi:.4f}\n")

        # 区分能力判断
        if sig_t > len(feature_pvalues) * 0.3 and avg_mi > 0.05:
            f.write("\n总体结论: 特征集具有良好的区分能力\n")
        elif sig_t > len(feature_pvalues) * 0.1 and avg_mi > 0.01:
            f.write("\n总体结论: 特征集具有中等区分能力\n")
        else:
            f.write("\n总体结论: 特征集的区分能力较弱，建议进行特征工程或收集更多数据\n")

    print(f"特征区分能力分析报告已保存至: {report_file}")
    return df_scores


# ================== 可视化函数 ==================
def visualize_feature_discrimination(X, y, df_scores, protein_features, compound_features, output_dir, top_features=20,
                                     plot_dpi=300):
    """创建特征区分能力的可视化图表"""
    print("生成特征区分能力可视化...")

    # 1. 特征重要性条形图
    plt.figure(figsize=(12, 10))
    top_feats = df_scores.head(top_features)

    # 创建颜色映射
    colors = ['#3498db' if feat in protein_features else '#e74c3c' for feat in top_feats['特征名']]

    plt.barh(top_feats['特征名'], top_feats['平均区分分数'], color=colors)
    plt.gca().invert_yaxis()  # 从上到下显示
    plot_with_chinese_font(
        title=f'前{top_features}个最具区分能力的特征',
        xlabel='平均区分分数',
        ylabel='特征名'
    )

    # 添加图例
    import matplotlib.patches as mpatches
    protein_patch = mpatches.Patch(color='#3498db', label='蛋白质特征')
    compound_patch = mpatches.Patch(color='#e74c3c', label='化合物特征')
    plt.legend(handles=[protein_patch, compound_patch], prop=chinese_font)

    plt.savefig(os.path.join(output_dir, 'top_features_importance.png'), dpi=plot_dpi, bbox_inches='tight')
    plt.close()

    # 2. 特征类型区分能力比较图
    plt.figure(figsize=(10, 6))

    # 准备数据
    feature_types = ['蛋白质特征', '化合物特征']
    metrics = ['t统计量', '互信息', '随机森林重要性', 'F值']

    protein_scores = [
        np.mean([row['t统计量'] for _, row in df_scores[df_scores['特征类型'] == '蛋白质特征'].iterrows()]),
        np.mean([row['互信息'] for _, row in df_scores[df_scores['特征类型'] == '蛋白质特征'].iterrows()]),
        np.mean([row['随机森林重要性'] for _, row in df_scores[df_scores['特征类型'] == '蛋白质特征'].iterrows()]),
        np.mean([row['F值'] for _, row in df_scores[df_scores['特征类型'] == '蛋白质特征'].iterrows()])
    ]

    compound_scores = [
        np.mean([row['t统计量'] for _, row in df_scores[df_scores['特征类型'] == '化合物特征'].iterrows()]),
        np.mean([row['互信息'] for _, row in df_scores[df_scores['特征类型'] == '化合物特征'].iterrows()]),
        np.mean([row['随机森林重要性'] for _, row in df_scores[df_scores['特征类型'] == '化合物特征'].iterrows()]),
        np.mean([row['F值'] for _, row in df_scores[df_scores['特征类型'] == '化合物特征'].iterrows()])
    ]

    # 数据归一化
    for i in range(len(metrics)):
        max_val = max(protein_scores[i], compound_scores[i])
        if max_val > 0:
            protein_scores[i] = protein_scores[i] / max_val
            compound_scores[i] = compound_scores[i] / max_val

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width / 2, protein_scores, width, label='蛋白质特征', color='#3498db')
    rects2 = ax.bar(x + width / 2, compound_scores, width, label='化合物特征', color='#e74c3c')

    if chinese_font:
        ax.set_title('特征类型区分能力比较 (归一化)', fontproperties=chinese_font, fontsize=14)
        ax.set_xlabel('评估指标', fontproperties=chinese_font, fontsize=12)
        ax.set_ylabel('归一化得分', fontproperties=chinese_font, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontproperties=chinese_font)
        ax.legend(prop=chinese_font)
    else:
        ax.set_title('特征类型区分能力比较 (归一化)')
        ax.set_xlabel('评估指标')
        ax.set_ylabel('归一化得分')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_type_comparison.png'), dpi=plot_dpi, bbox_inches='tight')
    plt.close()

    # 3. 正负样本特征值分布对比 (选取前5个最具区分能力的特征)
    top5_features = df_scores.head(5)['特征名'].tolist()

    # 创建2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # 绘制前5个特征的分布
    for i, feature in enumerate(top5_features):
        ax = axes[i]
        sns.histplot(X[feature][y == 0], ax=ax, color='blue', alpha=0.5, label='负样本', kde=True)
        sns.histplot(X[feature][y == 1], ax=ax, color='red', alpha=0.5, label='正样本', kde=True)

        if chinese_font:
            ax.set_title(f'{feature} 分布', fontproperties=chinese_font)
            ax.set_xlabel('特征值', fontproperties=chinese_font)
            ax.set_ylabel('频数', fontproperties=chinese_font)
            ax.legend(prop=chinese_font)
        else:
            ax.set_title(f'{feature} 分布')
            ax.set_xlabel('特征值')
            ax.set_ylabel('频数')
            ax.legend()

    # 设置最后一个子图显示总体结论
    ax = axes[5]
    ax.axis('off')

    t_stats_sig = sum(1 for p in df_scores['p值'] if p < 0.05)
    percent_sig = t_stats_sig / len(df_scores) * 100

    conclusion_text = (
        f"特征区分能力总结:\n\n"
        f"• 共有 {t_stats_sig}/{len(df_scores)} ({percent_sig:.1f}%) 个特征具有统计学显著性\n\n"
        f"• 蛋白质特征平均区分分数: {df_scores[df_scores['特征类型'] == '蛋白质特征']['平均区分分数'].mean():.4f}\n\n"
        f"• 化合物特征平均区分分数: {df_scores[df_scores['特征类型'] == '化合物特征']['平均区分分数'].mean():.4f}\n\n"
    )

    if percent_sig > 30:
        conclusion_text += "总体结论: 特征集具有良好的区分能力"
    elif percent_sig > 10:
        conclusion_text += "总体结论: 特征集具有中等区分能力"
    else:
        conclusion_text += "总体结论: 特征集的区分能力较弱"

    if chinese_font:
        ax.text(0.1, 0.5, conclusion_text, fontproperties=chinese_font, fontsize=12,
                verticalalignment='center', horizontalalignment='left',
                transform=ax.transAxes)
    else:
        ax.text(0.1, 0.5, conclusion_text, fontsize=12,
                verticalalignment='center', horizontalalignment='left',
                transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_features_distribution.png'), dpi=plot_dpi, bbox_inches='tight')
    plt.close()

    # 4. 特征显著性热图
    plt.figure(figsize=(12, 8))

    # 准备热图数据
    significance_data = []
    for _, row in df_scores.head(top_features).iterrows():
        significance_data.append({
            '特征名': row['特征名'],
            '特征类型': row['特征类型'],
            't统计量': row['t统计量'],
            '互信息': row['互信息'],
            '随机森林重要性': row['随机森林重要性'],
            'F值': row['F值'],
            '平均区分分数': row['平均区分分数']
        })

    sig_df = pd.DataFrame(significance_data)
    # 设置索引
    sig_df.set_index(['特征名', '特征类型'], inplace=True)

    # 准备热图数据列
    heatmap_columns = ['t统计量', '互信息', '随机森林重要性', 'F值', '平均区分分数']
    heatmap_data = sig_df[heatmap_columns]

    # 对每列进行归一化
    for col in heatmap_data.columns:
        heatmap_data[col] = heatmap_data[col] / heatmap_data[col].max()

    # 绘制热图
    sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt='.2f', linewidths=.5)

    if chinese_font:
        plt.title(f'前{top_features}个特征的区分能力评估 (归一化)', fontproperties=chinese_font, fontsize=14)
        plt.ylabel('特征名与类型', fontproperties=chinese_font, fontsize=12)
        plt.xticks(fontproperties=chinese_font)
    else:
        plt.title(f'前{top_features}个特征的区分能力评估 (归一化)')
        plt.ylabel('特征名与类型')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_significance_heatmap.png'), dpi=plot_dpi, bbox_inches='tight')
    plt.close()

    # 5. 降维可视化 - PCA
    print("执行PCA降维可视化...")
    try:
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA降维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # 绘制PCA散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis',
                              alpha=0.8, edgecolors='w', s=100, marker='o')

        # 添加图例
        legend_labels = ['负样本', '正样本']
        legend = plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels,
                            prop=chinese_font if chinese_font else None)

        # 添加解释方差比例
        variance_ratio = pca.explained_variance_ratio_

        plot_with_chinese_font(
            title='PCA降维可视化',
            xlabel=f'主成分1 (解释方差: {variance_ratio[0]:.2%})',
            ylabel=f'主成分2 (解释方差: {variance_ratio[1]:.2%})'
        )

        plt.savefig(os.path.join(output_dir, 'pca_visualization.png'), dpi=plot_dpi, bbox_inches='tight')
        plt.close()

        # 计算PCA降维的分类性能
        from sklearn.model_selection import cross_val_score
        from sklearn.svm import SVC

        svm = SVC(probability=True)
        pca_scores = cross_val_score(svm, X_pca, y, cv=5)
        print(f"PCA降维后5折交叉验证准确率: {np.mean(pca_scores):.4f} ± {np.std(pca_scores):.4f}")

        # 将PCA性能写入文件
        with open(os.path.join(output_dir, "dimensionality_reduction_results.txt"), 'w', encoding='utf-8') as f:
            f.write("降维方法性能比较\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"PCA降维 (2维) 解释方差: {sum(variance_ratio):.2%}\n")
            f.write(f"PCA降维后5折交叉验证准确率: {np.mean(pca_scores):.4f} ± {np.std(pca_scores):.4f}\n")
    except Exception as e:
        print(f"PCA降维可视化出错: {e}")

    # 6. 降维可视化 - t-SNE
    print("执行t-SNE降维可视化...")
    try:
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
        X_tsne = tsne.fit_transform(X_scaled)

        # 绘制t-SNE散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis',
                              alpha=0.8, edgecolors='w', s=100, marker='o')

        # 添加图例
        legend_labels = ['负样本', '正样本']
        legend = plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels,
                            prop=chinese_font if chinese_font else None)

        plot_with_chinese_font(
            title='t-SNE降维可视化',
            xlabel='t-SNE维度1',
            ylabel='t-SNE维度2'
        )

        plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'), dpi=plot_dpi, bbox_inches='tight')
        plt.close()

        # 计算t-SNE降维的分类性能
        svm = SVC(probability=True)
        tsne_scores = cross_val_score(svm, X_tsne, y, cv=5)
        print(f"t-SNE降维后5折交叉验证准确率: {np.mean(tsne_scores):.4f} ± {np.std(tsne_scores):.4f}")

        # 将t-SNE性能写入文件
        with open(os.path.join(output_dir, "dimensionality_reduction_results.txt"), 'a', encoding='utf-8') as f:
            f.write(f"\nt-SNE降维后5折交叉验证准确率: {np.mean(tsne_scores):.4f} ± {np.std(tsne_scores):.4f}\n")

            # 添加整体结论
            f.write("\n降维可视化分析结论:\n")

            pca_correct_classification = sum(1 for i in range(len(X_pca)) if
                                             (X_pca[i, 0] > 0 and y.iloc[i] == 1) or
                                             (X_pca[i, 0] < 0 and y.iloc[i] == 0))
            pca_correct_rate = pca_correct_classification / len(X_pca)

            tsne_clusters = sum(1 for i in range(len(X_tsne)) for j in range(i + 1, len(X_tsne)) if
                                np.linalg.norm(X_tsne[i] - X_tsne[j]) < np.std(X_tsne) and y.iloc[i] == y.iloc[j])
            tsne_cluster_rate = tsne_clusters / (len(X_tsne) * (len(X_tsne) - 1) / 2)

            if pca_correct_rate > 0.7 or tsne_cluster_rate > 0.7:
                f.write("数据具有良好的可分离性，特征集能够有效区分正负样本。\n")
            elif pca_correct_rate > 0.6 or tsne_cluster_rate > 0.6:
                f.write("数据具有中等可分离性，特征集对正负样本有一定的区分能力。\n")
            else:
                f.write("数据可分离性较弱，特征集区分正负样本的能力有限。\n")

            if np.mean(pca_scores) > 0.7 or np.mean(tsne_scores) > 0.7:
                f.write("降维后的分类性能良好，说明特征集中存在具有强区分能力的主要因素。\n")
            elif np.mean(pca_scores) > 0.6 or np.mean(tsne_scores) > 0.6:
                f.write("降维后的分类性能尚可，特征集中存在一定区分能力的因素。\n")
            else:
                f.write("降维后的分类性能不佳，可能需要更多或更好的特征来提高区分能力。\n")
    except Exception as e:
        print(f"t-SNE降维可视化出错: {e}")

    # 7. ROC曲线和PRC曲线
    print("计算ROC曲线和PR曲线...")
    try:
        from sklearn.model_selection import cross_val_predict

        # 使用随机森林进行预测
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        y_scores = cross_val_predict(rf, X, y, cv=5, method='predict_proba')[:, 1]

        # ROC曲线
        fpr, tpr, _ = roc_curve(y, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plot_with_chinese_font(
            title='ROC曲线',
            xlabel='假阳性率',
            ylabel='真阳性率'
        )

        if chinese_font:
            plt.legend(prop=chinese_font)
        else:
            plt.legend()

        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=plot_dpi, bbox_inches='tight')
        plt.close()

        # PR曲线
        precision, recall, _ = precision_recall_curve(y, y_scores)
        avg_precision = average_precision_score(y, y_scores)

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR曲线 (AP = {avg_precision:.4f})')

        # 计算基线
        baseline = np.sum(y) / len(y)
        plt.axhline(y=baseline, color='red', linestyle='--', label=f'基线 (阳性率 = {baseline:.4f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plot_with_chinese_font(
            title='精确率-召回率曲线',
            xlabel='召回率',
            ylabel='精确率'
        )

        if chinese_font:
            plt.legend(prop=chinese_font)
        else:
            plt.legend()

        plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=plot_dpi, bbox_inches='tight')
        plt.close()

        # 将曲线性能指标写入文件
        with open(os.path.join(output_dir, "discriminative_power_metrics.txt"), 'w', encoding='utf-8') as f:
            f.write("特征区分能力度量指标\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"ROC曲线下面积 (AUC): {roc_auc:.4f}\n")
            f.write(f"平均精确率 (AP): {avg_precision:.4f}\n")
            f.write(f"基线准确率 (随机猜测): {0.5:.4f}\n")
            f.write(f"基线精确率 (总是预测多数类): {max(baseline, 1 - baseline):.4f}\n\n")

            # 添加AUC和AP的解释
            if roc_auc > 0.9:
                f.write("AUC > 0.9: 特征集具有极佳的区分能力\n")
            elif roc_auc > 0.8:
                f.write("AUC > 0.8: 特征集具有良好的区分能力\n")
            elif roc_auc > 0.7:
                f.write("AUC > 0.7: 特征集具有中等区分能力\n")
            elif roc_auc > 0.6:
                f.write("AUC > 0.6: 特征集具有一定区分能力\n")
            else:
                f.write("AUC ≤ 0.6: 特征集区分能力较弱\n")

            if avg_precision > baseline * 2:
                f.write(
                    f"AP ({avg_precision:.4f}) 显著高于基线 ({baseline:.4f})，表明特征集具有良好的精确率-召回率平衡\n")
            elif avg_precision > baseline * 1.5:
                f.write(f"AP ({avg_precision:.4f}) 高于基线 ({baseline:.4f})，表明特征集具有中等的精确率-召回率平衡\n")
            else:
                f.write(
                    f"AP ({avg_precision:.4f}) 接近基线 ({baseline:.4f})，表明特征集在精确率-召回率平衡方面表现有限\n")
    except Exception as e:
        print(f"计算ROC和PR曲线出错: {e}")

    print("特征区分能力可视化完成！所有图形已保存到输出目录。")


# ================== 主函数 ==================
def main():
    """主函数，执行特征区分能力测试的完整流程"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试输入数据特征的区分能力')
    parser.add_argument('--input', type=str, default=INPUT_FILE, help='输入CSV文件路径')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='输出目录路径')
    parser.add_argument('--top', type=int, default=TOP_FEATURES_DEFAULT, help='显示前N个最具区分能力的特征')
    parser.add_argument('--dpi', type=int, default=PLOT_DPI_DEFAULT, help='图形DPI')
    args = parser.parse_args()

    print(f"特征区分能力测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"显示前 {args.top} 个最具区分能力的特征")
    print(f"图形DPI: {args.dpi}")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 加载数据
    df, X, y, protein_id_col, compound_id_col, protein_features, compound_features, label_col = load_data(args.input)

    # 测试特征区分能力，使用参数
    df_scores = test_feature_discrimination(X, y, protein_features, compound_features, args.output,
                                            top_features=args.top)

    # 可视化特征区分能力，使用参数
    visualize_feature_discrimination(X, y, df_scores, protein_features, compound_features, args.output,
                                     top_features=args.top, plot_dpi=args.dpi)

    print(f"\n特征区分能力测试完成！所有结果已保存到目录: {args.output}")
    print(f"总执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()