import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
import joblib
import os

try:
    from xgboost import XGBClassifier

    xgb_installed = True
except ImportError:
    xgb_installed = False

try:
    from lightgbm import LGBMClassifier

    lgbm_installed = True
except ImportError:
    lgbm_installed = False

import argparse
from datetime import datetime


def load_and_prepare_data(input_file):
    df = pd.read_csv(input_file)
    df = df.fillna(0)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    protein_id_col = 'gene_id'
    compound_id_col = 'compound_id'
    all_columns = df.columns.tolist()
    protein_id_idx = all_columns.index(protein_id_col)
    compound_id_idx = all_columns.index(compound_id_col)
    protein_features = all_columns[protein_id_idx + 1:compound_id_idx]
    compound_features = all_columns[compound_id_idx + 1:-1]
    label_col = all_columns[-1]
    X = df[protein_features + compound_features]
    y = df[label_col]
    return X, y


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, feature_names, output_dir):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_per_fold = []
    aucs = []
    print(f"\n{model_name} 训练集5折交叉验证各折指标：")
    fold_details = []

    for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        # 计算AUC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_val)
        else:
            y_score = y_pred
        auc = roc_auc_score(y_val, y_score)
        aucs.append(auc)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        metrics_per_fold.append([acc, prec, rec, f1])
        fold_details.append({
            "折": f"折{i + 1}",
            "准确率": f"{acc:.4f}",
            "精确率": f"{prec:.4f}",
            "召回率": f"{rec:.4f}",
            "F1": f"{f1:.4f}",
            "AUC": f"{auc:.4f}"
        })
        print(f"折{i + 1}: 准确率={acc:.4f} 精确率={prec:.4f} 召回率={rec:.4f} F1={f1:.4f} AUC={auc:.4f}")

    # 创建训练集表格数据
    fold_df = pd.DataFrame(fold_details)

    # 添加均值和方差行
    metrics_per_fold = np.array(metrics_per_fold)
    mean_row = {
        "折": "均值",
        "准确率": f"{metrics_per_fold[:, 0].mean():.4f}",
        "精确率": f"{metrics_per_fold[:, 1].mean():.4f}",
        "召回率": f"{metrics_per_fold[:, 2].mean():.4f}",
        "F1": f"{metrics_per_fold[:, 3].mean():.4f}",
        "AUC": f"{np.mean(aucs):.4f}"
    }
    std_row = {
        "折": "标准差",
        "准确率": f"{metrics_per_fold[:, 0].std():.4f}",
        "精确率": f"{metrics_per_fold[:, 1].std():.4f}",
        "召回率": f"{metrics_per_fold[:, 2].std():.4f}",
        "F1": f"{metrics_per_fold[:, 3].std():.4f}",
        "AUC": f"{np.std(aucs):.4f}"
    }

    fold_df = pd.concat([fold_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    # 保存为三线表格式的LaTeX表格
    latex_table = fold_df.to_latex(index=False, escape=False, column_format='c|ccccc')
    latex_table = latex_table.replace('\\toprule', '\\hline\\hline')
    latex_table = latex_table.replace('\\midrule', '\\hline')
    latex_table = latex_table.replace('\\bottomrule', '\\hline\\hline')

    # 在均值行前添加中线
    lines = latex_table.split('\n')
    mean_line_idx = len(fold_details) + 2  # 计算均值行的位置
    lines.insert(mean_line_idx, '\\hline')
    latex_table = '\n'.join(lines)

    # 保存LaTeX表格
    with open(os.path.join(output_dir, f'{model_name}_train_results_table.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_table)

    # 保存CSV格式表格
    fold_df.to_csv(os.path.join(output_dir, f'{model_name}_train_results_table.csv'), index=False, encoding='utf-8-sig')

    print(f"{model_name} 训练集5折交叉验证均值±方差:")
    print(f"准确率: {metrics_per_fold[:, 0].mean():.4f} ± {metrics_per_fold[:, 0].std():.4f}")
    print(f"精确率: {metrics_per_fold[:, 1].mean():.4f} ± {metrics_per_fold[:, 1].std():.4f}")
    print(f"召回率: {metrics_per_fold[:, 2].mean():.4f} ± {metrics_per_fold[:, 2].std():.4f}")
    print(f"F1分数: {metrics_per_fold[:, 3].mean():.4f} ± {metrics_per_fold[:, 3].std():.4f}")
    print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # 在全部训练集上训练
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test)
    rec = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)

    if hasattr(model, "predict_proba"):
        y_score_test = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score_test = model.decision_function(X_test)
    else:
        y_score_test = y_pred_test
    auc_test = roc_auc_score(y_test, y_score_test)
    print(f"{model_name} 测试集:")
    print(f"准确率: {acc:.4f}  精确率: {prec:.4f}  召回率: {rec:.4f}  F1分数: {f1:.4f}  AUC: {auc_test:.4f}")
    print("混淆矩阵:")
    print(conf_matrix)

    # 输出前十个最相关特征
    print(f"\n{model_name} 前十个最相关特征（按权重绝对值排序）：")
    importances = None
    top10_idx = None

    # 对不同模型采用不同的特征重要性获取方法
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
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importances = perm_importance.importances_mean
        sorted_idx = np.argsort(np.abs(importances))[::-1]
        top10_idx = sorted_idx[:10]
        feature_importance_type = "排列重要性"

    if top10_idx is not None:
        feature_importance_data = []
        for i, idx in enumerate(top10_idx):
            feature_importance_data.append({
                "排名": i + 1,
                "特征名": feature_names[idx],
                "重要性": abs(importances[idx]),
                "原始值": importances[idx]
            })
            print(f"{i + 1}. {feature_names[idx]}: {importances[idx]:.6f}")

        # 将特征重要性保存为CSV
        pd.DataFrame(feature_importance_data).to_csv(
            os.path.join(output_dir, f'{model_name}_top10_features.csv'),
            index=False, encoding='utf-8-sig'
        )

    return {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc_test,
        "top_features": top10_idx,
        "feature_importances": importances
    }


def load_q13133_ehdpp_sample(feature_names, protein_file, compound_file, train_file):
    df_protein = pd.read_csv(protein_file)
    df_compound = pd.read_csv(compound_file)
    all_columns = pd.read_csv(train_file, nrows=1).columns.tolist()
    protein_id_col = 'gene_id'
    compound_id_col = 'compound_id'
    protein_id_idx = all_columns.index(protein_id_col)
    compound_id_idx = all_columns.index(compound_id_col)
    protein_features = all_columns[protein_id_idx + 1:compound_id_idx]
    compound_features = all_columns[compound_id_idx + 1:-1]
    protein_row = df_protein.iloc[0]
    compound_row = df_compound.iloc[0]
    features = []
    for f in protein_features:
        features.append(protein_row[f] if f in df_protein.columns else 0)
    for f in compound_features:
        features.append(compound_row[f] if f in df_compound.columns else 0)
    sample_df = pd.DataFrame([features], columns=protein_features + compound_features)
    for col in feature_names:
        if col not in sample_df.columns:
            sample_df[col] = 0
    sample_df = sample_df[feature_names]
    sample_df = sample_df.fillna(0)
    return sample_df


def get_best_params(X, y, model_type, param_grid):
    """使用网格搜索找到最佳参数"""
    from sklearn.model_selection import GridSearchCV

    if model_type == 'SVM':
        base_model = SVC(probability=True, random_state=42)
    elif model_type == '随机森林':
        base_model = RandomForestClassifier(random_state=42)
    elif model_type == '梯度提升':
        base_model = GradientBoostingClassifier(random_state=42)
    elif model_type == 'AdaBoost':
        base_model = AdaBoostClassifier(random_state=42)
    elif model_type == '逻辑回归':
        base_model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'K近邻':
        base_model = KNeighborsClassifier()
    elif model_type == '朴素贝叶斯':
        base_model = GaussianNB()
    elif model_type == 'XGBoost' and xgb_installed:
        base_model = XGBClassifier(random_state=42)
    elif model_type == 'LightGBM' and lgbm_installed:
        base_model = LGBMClassifier(random_state=42)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    grid = GridSearchCV(base_model, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    print(f"{model_type}最优参数: {grid.best_params_}")
    return grid.best_estimator_, grid.best_params_


def select_optimal_features(X, y, output_dir=None):
    """使用多种方法选择最优特征"""
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt

    results = {}
    feature_names = X.columns.tolist()

    # 1. 方差筛选 - 过滤低方差特征
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    X_var = selector.fit_transform(X)
    var_support = selector.get_support()
    var_features = [feature_names[i] for i in range(len(feature_names)) if var_support[i]]
    print(f"方差筛选后保留特征数: {len(var_features)}")
    results['方差筛选'] = var_features

    # 2. 统计检验 - F检验
    selector_f = SelectKBest(f_classif, k=min(100, X.shape[1]))
    X_f = selector_f.fit_transform(X, y)
    f_support = selector_f.get_support()
    f_features = [feature_names[i] for i in range(len(feature_names)) if f_support[i]]
    f_scores = selector_f.scores_
    print(f"F检验选择后保留特征数: {len(f_features)}")
    results['F检验'] = f_features

    # 3. 互信息
    selector_mi = SelectKBest(mutual_info_classif, k=min(100, X.shape[1]))
    X_mi = selector_mi.fit_transform(X, y)
    mi_support = selector_mi.get_support()
    mi_features = [feature_names[i] for i in range(len(feature_names)) if mi_support[i]]
    mi_scores = selector_mi.scores_
    print(f"互信息选择后保留特征数: {len(mi_features)}")
    results['互信息'] = mi_features

    # 4. 递归特征消除(RFE)
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=estimator, n_features_to_select=min(100, X.shape[1]))
    X_rfe = rfe.fit_transform(X, y)
    rfe_support = rfe.get_support()
    rfe_features = [feature_names[i] for i in range(len(feature_names)) if rfe_support[i]]
    print(f"RFE选择后保留特征数: {len(rfe_features)}")
    results['RFE'] = rfe_features

    # 5. 基于模型的特征重要性
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    model_features = [feature_names[i] for i in indices[:100]]
    print(f"模型特征重要性选择后保留特征数: {len(model_features)}")
    results['模型重要性'] = model_features

    # 6. 综合特征选择 - 取并集和交集
    # 至少被两种方法选择的特征
    all_selected_features = set()
    feature_vote = {f: 0 for f in feature_names}

    for method, features in results.items():
        for f in features:
            feature_vote[f] += 1
            all_selected_features.add(f)

    # 被至少2种方法选中的特征
    consensus_features = [f for f, votes in feature_vote.items() if votes >= 2]
    print(f"综合特征选择后保留特征数: {len(consensus_features)}")

    # 保存特征选择结果
    if output_dir:
        import os
        # 保存每种方法选择的特征
        for method, features in results.items():
            pd.DataFrame({'Feature': features}).to_csv(
                os.path.join(output_dir, f'{method}_selected_features.csv'),
                index=False, encoding='utf-8-sig'
            )

        # 保存综合选择的特征
        pd.DataFrame({'Feature': consensus_features, 'Vote': [feature_vote[f] for f in consensus_features]}).to_csv(
            os.path.join(output_dir, 'consensus_features.csv'),
            index=False, encoding='utf-8-sig'
        )

        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(importances)[-20:]  # 展示前20个重要特征
        plt.barh(range(20), importances[sorted_idx])
        plt.yticks(range(20), [feature_names[i] for i in sorted_idx])
        plt.xlabel('特征重要性')
        plt.title('随机森林特征重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)

    return consensus_features


def enhance_features(X, protein_cols, compound_cols):
    """增强特征通过构造新特征和转换"""
    X_enhanced = X.copy()

    # 1. 对数变换 - 用于右偏特征
    for col in X.columns:
        # 跳过非数值列
        if X[col].dtype not in [np.float64, np.int64]:
            continue

        # 找出存在正值的列进行对数变换
        if (X[col] > 0).any():
            min_positive = X[col][X[col] > 0].min()
            X_enhanced[f"{col}_log"] = np.log1p(X[col] - min_positive + 1e-6)

    # 2. 平方和平方根变换 - 增强非线性关系
    for col in X.columns:
        if X[col].dtype not in [np.float64, np.int64]:
            continue

        X_enhanced[f"{col}_squared"] = X[col] ** 2
        X_enhanced[f"{col}_sqrt"] = np.sqrt(np.abs(X[col]) + 1e-10)

    # 3. 分箱处理 - 针对连续特征
    for col in X.columns:
        if X[col].dtype not in [np.float64, np.int64]:
            continue

        X_enhanced[f"{col}_bin"] = pd.qcut(X[col], q=5, labels=False, duplicates='drop')

    # 4. 多项式特征 - 基于重要特征
    # 假设前5个蛋白质特征和前5个化合物特征最重要
    top_protein = protein_cols[:min(5, len(protein_cols))]
    top_compound = compound_cols[:min(5, len(compound_cols))]

    # 蛋白质特征内部交互
    for i, col1 in enumerate(top_protein):
        for col2 in top_protein[i + 1:]:
            X_enhanced[f"{col1}_{col2}_interact"] = X[col1] * X[col2]

    # 化合物特征内部交互
    for i, col1 in enumerate(top_compound):
        for col2 in top_compound[i + 1:]:
            X_enhanced[f"{col1}_{col2}_interact"] = X[col1] * X[col2]

    # 5. 蛋白质-化合物交叉特征
    # 为每对重要的蛋白质和化合物特征创建交互项
    for p_col in top_protein:
        for c_col in top_compound:
            X_enhanced[f"{p_col}_{c_col}_cross"] = X[p_col] * X[c_col]

    # 6. 特征比率 - 可能捕捉一些比例关系
    for i, col1 in enumerate(top_protein):
        for col2 in top_compound:
            # 避免除零
            denominator = X[col2].replace(0, np.nan)
            ratio = X[col1] / denominator
            X_enhanced[f"{col1}_to_{col2}_ratio"] = ratio.fillna(0)

    # 7. 统计特征 - 计算蛋白质和化合物特征的统计量
    X_enhanced['protein_mean'] = X[protein_cols].mean(axis=1)
    X_enhanced['protein_std'] = X[protein_cols].std(axis=1)
    X_enhanced['compound_mean'] = X[compound_cols].mean(axis=1)
    X_enhanced['compound_std'] = X[compound_cols].std(axis=1)
    X_enhanced['protein_to_compound_ratio'] = X_enhanced['protein_mean'] / X_enhanced['compound_mean'].replace(0, 1)

    return X_enhanced


def scale_features(X_train, X_test):
    """应用多种缩放方法并选择最合适的"""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
    import numpy as np
    from scipy import stats
    from sklearn.impute import SimpleImputer

    # 首先处理 NaN 值
    print("处理缺失值...")
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # 存储不同缩放器的结果
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
        'power': PowerTransformer(method='yeo-johnson')
    }

    scaled_data = {}

    for name, scaler in scalers.items():
        try:
            # 对训练集拟合并转换
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            # 对测试集转换
            X_test_scaled = scaler.transform(X_test_imputed)

            # 计算统计量，安全处理可能的异常值
            train_mean = np.nanmean([np.mean(X_train_scaled[:, i]) for i in range(X_train_scaled.shape[1])])
            train_std = np.nanmean([np.std(X_train_scaled[:, i]) for i in range(X_train_scaled.shape[1])])

            # 计算偏度时添加安全处理
            train_skew_values = []
            for i in range(X_train_scaled.shape[1]):
                col_data = X_train_scaled[:, i]
                # 检查列是否有足够的变异性
                if np.std(col_data) > 1e-10:
                    try:
                        skew_value = stats.skew(col_data)
                        if not np.isnan(skew_value) and not np.isinf(skew_value):
                            train_skew_values.append(abs(skew_value))
                    except:
                        pass

            train_skew = np.mean(train_skew_values) if train_skew_values else 0

            # 计算质量分数
            quality_score = abs(train_mean) + abs(train_std - 1) + train_skew

            print(
                f"{name} 缩放器: 平均差异={train_mean:.3f}, 标准差差异={abs(train_std - 1):.3f}, 偏度={train_skew:.3f}")

            scaled_data[name] = {
                'scaler': scaler,
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
                'quality_score': quality_score,
                'imputer': imputer  # 保存imputer以便后续使用
            }
        except Exception as e:
            print(f"{name} 缩放器出错: {str(e)}")
            # 如果发生错误，使用一个很大的质量分数以避免选择这个缩放器
            scaled_data[name] = {
                'scaler': scaler,
                'X_train_scaled': X_train_imputed,  # 使用填充后的数据
                'X_test_scaled': X_test_imputed,  # 使用填充后的数据
                'quality_score': float('inf'),
                'imputer': imputer
            }

    # 选择质量分数最低的缩放器(分布差异最小)
    best_scaler_name = min(scaled_data.keys(), key=lambda x: scaled_data[x]['quality_score'])
    print(f"最佳缩放方法: {best_scaler_name}")

    # 检查结果中是否还有NaN值
    best_result = scaled_data[best_scaler_name]
    if np.isnan(best_result['X_train_scaled']).any() or np.isnan(best_result['X_test_scaled']).any():
        print("警告：缩放后的数据仍包含NaN值，将再次填充")
        best_result['X_train_scaled'] = np.nan_to_num(best_result['X_train_scaled'], nan=0.0)
        best_result['X_test_scaled'] = np.nan_to_num(best_result['X_test_scaled'], nan=0.0)

    return (
        best_result['scaler'],
        best_result['X_train_scaled'],
        best_result['X_test_scaled'],
        best_result['imputer']  # 同时返回imputer
    )

def add_domain_knowledge_features(X, protein_df, compound_df, protein_cols, compound_cols):
    """根据蛋白质-药物相互作用领域知识添加特征"""
    X_enhanced = X.copy()

    # 1. 基于物理化学性质的特征
    # 找出表示疏水性的特征
    hydrophobic_cols = [col for col in protein_cols if 'hydrophobic' in col.lower()
                        or 'alogp' in col.lower() or 'logp' in col.lower()]
    # 找出表示氢键的特征
    hbond_cols = [col for col in protein_cols if 'hbond' in col.lower()
                  or 'donor' in col.lower() or 'acceptor' in col.lower()]

    # 化合物相关特征
    logp_cols = [col for col in compound_cols if 'logp' in col.lower()]
    hba_cols = [col for col in compound_cols if 'hba' in col.lower() or 'acceptor' in col.lower()]
    hbd_cols = [col for col in compound_cols if 'hbd' in col.lower() or 'donor' in col.lower()]

    # 2. 创建物理化学互作用特征
    if hydrophobic_cols and logp_cols:
        # 疏水相互作用得分
        X_enhanced['hydrophobic_interaction'] = X[hydrophobic_cols].mean(axis=1) * X[logp_cols].mean(axis=1)

    if hbond_cols and (hba_cols or hbd_cols):
        # 氢键相互作用潜力
        hbond_protein = X[hbond_cols].mean(axis=1) if hbond_cols else 0
        hba = X[hba_cols].mean(axis=1) if hba_cols else 0
        hbd = X[hbd_cols].mean(axis=1) if hbd_cols else 0
        X_enhanced['hbond_potential'] = hbond_protein * (hba + hbd)

    # 3. 基于蛋白质结构域的特征
    # 假设有些蛋白质特征表示某些结构域的特性
    domain_cols = [col for col in protein_cols if 'domain' in col.lower()
                   or 'motif' in col.lower() or 'site' in col.lower()]

    if domain_cols:
        X_enhanced['domain_features'] = X[domain_cols].mean(axis=1)

        # 与化合物特性的交互
        for c_col in compound_cols[:5]:  # 使用前5个化合物特征
            X_enhanced[f'domain_{c_col}_interaction'] = X_enhanced['domain_features'] * X[c_col]

    # 4. 基于药效团的特征
    pharmacophore_cols = [col for col in compound_cols if 'ring' in col.lower()
                          or 'aromatic' in col.lower() or 'pharmacophore' in col.lower()]

    if pharmacophore_cols:
        X_enhanced['pharmacophore_features'] = X[pharmacophore_cols].mean(axis=1)

        # 与蛋白质特性的交互
        for p_col in protein_cols[:5]:  # 使用前5个蛋白质特征
            X_enhanced[f'pharmacophore_{p_col}_interaction'] = X_enhanced['pharmacophore_features'] * X[p_col]

    return X_enhanced


def optimize_features(X, y, protein_file, compound_file, train_file, output_dir=None):
    """特征工程优化流程"""
    print("\n开始特征工程优化...")

    # 确定蛋白质和化合物特征
    all_columns = pd.read_csv(train_file, nrows=1).columns.tolist()
    protein_id_col = 'gene_id'
    compound_id_col = 'compound_id'
    protein_id_idx = all_columns.index(protein_id_col)
    compound_id_idx = all_columns.index(compound_id_col)
    protein_cols = all_columns[protein_id_idx + 1:compound_id_idx]
    compound_cols = all_columns[compound_id_idx + 1:-1]

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 记录原始特征数
    original_feature_count = X.shape[1]

    # 步骤1：特征增强
    print("增强特征...")
    X_enhanced = enhance_features(X, protein_cols, compound_cols)

    # 应用增强的特征到训练集和测试集
    X_train_enhanced = X_enhanced.loc[X_train.index]
    X_test_enhanced = X_enhanced.loc[X_test.index]

    # 步骤2：特征选择
    print("选择最优特征...")
    X_train_enhanced = X_train_enhanced.select_dtypes(include=['number'])
    X_test_enhanced = X_test_enhanced.select_dtypes(include=['number'])

    # 确保没有无穷值
    X_train_enhanced.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_enhanced.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 步骤3：优化特征缩放
    print("优化特征缩放...")
    best_scaler, X_train_scaled, X_test_scaled, imputer = scale_features(X_train_enhanced, X_test_enhanced)

    # 构建特征管道
    feature_pipeline = {
        'protein_cols': protein_cols,
        'compound_cols': compound_cols,
        'enhanced_columns': X_train_enhanced.columns.tolist(),
        'scaler': best_scaler,
        'imputer': imputer
    }

    print(
        f"特征工程完成. 原始特征数: {original_feature_count}, 优化后特征数: {len(feature_pipeline['enhanced_columns'])}")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_pipeline


def process_q13133_ehdpp_sample(feature_pipeline, protein_file, compound_file):
    """使用特征工程流程处理样本"""
    # 加载蛋白质和化合物数据
    df_protein = pd.read_csv(protein_file)
    df_compound = pd.read_csv(compound_file)

    # 获取Q13133和EHDPP的特征行
    protein_row = df_protein.iloc[0]
    compound_row = df_compound.iloc[0]

    # 构建原始特征向量
    features = {}
    for col in feature_pipeline['protein_cols']:
        features[col] = protein_row[col] if col in df_protein.columns else 0

    for col in feature_pipeline['compound_cols']:
        features[col] = compound_row[col] if col in df_compound.columns else 0

    # 创建样本DataFrame
    sample_df = pd.DataFrame([features])

    # 1. 只保留选定的特征
    sample_selected = pd.DataFrame()
    for col in feature_pipeline['selected_features']:
        if col in sample_df.columns:
            sample_selected[col] = sample_df[col]
        else:
            sample_selected[col] = 0

    # 2. 添加领域知识特征
    sample_domain = add_domain_knowledge_features(
        sample_selected, df_protein, df_compound,
        [col for col in feature_pipeline['protein_cols'] if col in feature_pipeline['selected_features']],
        [col for col in feature_pipeline['compound_cols'] if col in feature_pipeline['selected_features']]
    )

    # 3. 特征增强
    sample_enhanced = enhance_features(
        sample_domain,
        [col for col in feature_pipeline['protein_cols'] if col in feature_pipeline['selected_features']],
        [col for col in feature_pipeline['compound_cols'] if col in feature_pipeline['selected_features']]
    )

    # 4. 确保所有特征存在
    for col in feature_pipeline['enhanced_columns']:
        if col not in sample_enhanced.columns:
            sample_enhanced[col] = 0

    # 只保留需要的列，并按正确顺序排列
    sample_enhanced = sample_enhanced[feature_pipeline['enhanced_columns']]

    # 5. 特征缩放
    sample_scaled = feature_pipeline['scaler'].transform(sample_enhanced)

    # 填充所有缺失特征
    for col in feature_names:
        if col not in sample_df.columns:
            sample_df[col] = 0

    sample_df = sample_df[feature_names]
    sample_df = sample_df.fillna(0)

    # 确保没有无穷大值
    sample_df.replace([np.inf, -np.inf], 0, inplace=True)

    return sample_scaled


def main():
    parser = argparse.ArgumentParser(description='多模型对比及特征工程优化')
    # 基本参数
    parser.add_argument('--input_file', required=True, help='输入CSV文件路径')
    parser.add_argument('--protein_file', required=True, help='蛋白质特征CSV文件路径')
    parser.add_argument('--compound_file', required=True, help='化合物特征CSV文件路径')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--output_dir', default='results', help='输出目录')
    parser.add_argument('--save_models', action='store_true', help='是否保存模型')
    parser.add_argument('--compare_only', nargs='+', help='只比较指定的模型')

    # 特征工程参数
    parser.add_argument('--feature_engineering', action='store_true', help='是否应用特征工程优化')
    parser.add_argument('--feature_selection_threshold', type=int, default=2,
                        help='特征选择的最低投票阈值(至少多少种方法选择)')
    parser.add_argument('--max_interaction_features', type=int, default=5,
                        help='构建交叉特征时使用的最大特征数量')

    # SVM参数
    parser.add_argument('--svm_C', type=float, nargs='+', default=[0.1, 1, 10], help='SVM的C参数')
    parser.add_argument('--svm_gamma', type=float, nargs='+', default=[0.01, 0.1, 1], help='SVM的gamma参数')
    parser.add_argument('--svm_kernel', nargs='+', default=['rbf'], help='SVM核函数类型')

    # 随机森林参数
    parser.add_argument('--rf_n_estimators', type=int, nargs='+', default=[50, 100, 200], help='随机森林树的数量')
    parser.add_argument('--rf_max_depth', type=int, nargs='+', default=[None, 10, 20], help='随机森林最大深度')
    parser.add_argument('--rf_min_samples_leaf', type=int, nargs='+', default=[1, 2, 5],
                        help='随机森林叶节点最小样本数')

    # 梯度提升参数
    parser.add_argument('--gb_n_estimators', type=int, nargs='+', default=[50, 100, 200], help='梯度提升树的数量')
    parser.add_argument('--gb_learning_rate', type=float, nargs='+', default=[0.01, 0.1, 0.5], help='梯度提升学习率')
    parser.add_argument('--gb_max_depth', type=int, nargs='+', default=[3, 5, 7], help='梯度提升树最大深度')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, f'run_{timestamp}')
    os.makedirs(output_dir)

    # 保存运行参数
    with open(os.path.join(output_dir, 'run_parameters.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # 加载数据
    print("\n加载数据...")
    X, y = load_and_prepare_data(args.input_file)
    print(f"数据集大小: {X.shape}")
    print(f"特征数量: {X.shape[1]}")
    print(f"类别分布: {y.value_counts().to_dict()}")

    # 确定蛋白质和化合物特征
    all_columns = pd.read_csv(args.input_file, nrows=1).columns.tolist()
    protein_id_col = 'gene_id'
    compound_id_col = 'compound_id'
    protein_id_idx = all_columns.index(protein_id_col)
    compound_id_idx = all_columns.index(compound_id_col)
    protein_cols = all_columns[protein_id_idx + 1:compound_id_idx]
    compound_cols = all_columns[compound_id_idx + 1:-1]

    if args.feature_engineering:
        # 应用特征工程优化流程
        print("\n开始特征工程优化...")
        X_train_scaled, X_test_scaled, y_train, y_test, feature_pipeline = optimize_features(
            X, y, args.protein_file, args.compound_file, args.input_file, output_dir
        )
        # 使用优化后的特征列名
        feature_names = feature_pipeline['enhanced_columns']
    else:
        # 传统的数据划分和标准化
        print("\n执行传统数据处理...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 将转换后的数据转换回DataFrame以保留特征名
        feature_names = X.columns.tolist()
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

        # 创建简单的特征管道(用于传统方法)
        feature_pipeline = {
            'selected_features': feature_names,
            'protein_cols': protein_cols,
            'compound_cols': compound_cols,
            'scaler': scaler,
            'enhanced_columns': feature_names
        }

        # 保存标准化器
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # 配置参数网格
    param_grids = {
        'SVM': {
            'C': args.svm_C,
            'gamma': args.svm_gamma,
            'kernel': args.svm_kernel
        },
        '随机森林': {
            'n_estimators': args.rf_n_estimators,
            'max_depth': args.rf_max_depth,
            'min_samples_leaf': args.rf_min_samples_leaf
        },
        '梯度提升': {
            'n_estimators': args.gb_n_estimators,
            'learning_rate': args.gb_learning_rate,
            'max_depth': args.gb_max_depth
        },
        'AdaBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.5, 1.0, 1.5]
        },
        '逻辑回归': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['liblinear', 'lbfgs', 'saga']
        },
        'K近邻': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        },
        '朴素贝叶斯': {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        }
    }

    if xgb_installed:
        param_grids['XGBoost'] = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 6, 9]
        }

    if lgbm_installed:
        param_grids['LightGBM'] = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [-1, 5, 10]
        }

    # 初始化模型字典和最佳参数字典
    model_dict = {}
    best_params_dict = {}

    # 根据参数添加模型
    for model_name in param_grids:
        if args.compare_only is None or model_name in args.compare_only:
            print(f"\n{'=' * 50}")
            print(f"优化模型参数: {model_name}")
            print(f"{'=' * 50}")
            best_model, best_params = get_best_params(
                X_train_scaled, y_train,
                model_name, param_grids[model_name]
            )
            model_dict[model_name] = best_model
            best_params_dict[model_name] = best_params

    # 保存最佳参数
    with open(os.path.join(output_dir, 'best_parameters.txt'), 'w') as f:
        for model_name, params in best_params_dict.items():
            f.write(f"{model_name}:\n")
            for param, value in params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")

    # 评估和比较模型
    results = {}
    for name, model in model_dict.items():
        print(f"\n{'=' * 50}")
        print(f"评估模型: {name}")
        print(f"{'=' * 50}")
        result = evaluate_model(
            model, X_train_scaled, y_train, X_test_scaled, y_test,
            name, feature_names, output_dir
        )
        results[name] = result

        # 保存模型
        if args.save_models:
            joblib.dump(model, os.path.join(output_dir, f'{name}_model.pkl'))
            print(f"{name}模型已保存")

    # 保存特征工程管道
    joblib.dump(feature_pipeline, os.path.join(output_dir, 'feature_pipeline.pkl'))

    # 预测q13133和ehdpp的相互作用概率
    print("\nq13133与ehdpp的相互作用概率预测：")

    # 根据使用的特征工程方法处理样本
    if args.feature_engineering:
        # 使用特征工程流程处理样本
        sample_scaled = process_q13133_ehdpp_sample(
            feature_pipeline, args.protein_file, args.compound_file
        )
    else:
        # 使用传统方法处理样本
        sample_df = load_q13133_ehdpp_sample(feature_names, args.protein_file, args.compound_file, args.input_file)
        sample_scaled = feature_pipeline['scaler'].transform(sample_df)

    # 模型预测
    pred_results = []
    ensemble_probs = []

    for name, model in model_dict.items():
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(sample_scaled)[0, 1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(sample_scaled)[0]
            prob = 1 / (1 + np.exp(-decision))
        else:
            prob = model.predict(sample_scaled)[0]

        print(f"{name}: 概率={prob:.4f}")
        pred_results.append({"模型": name, "交互概率": prob})
        ensemble_probs.append(prob)

    # 保存预测结果
    pd.DataFrame(pred_results).to_csv(
        os.path.join(output_dir, 'q13133_ehdpp_prediction.csv'),
        index=False, encoding='utf-8-sig'
    )

    # 执行模型融合预测
    print("\n执行模型融合预测...")
    avg_prob = sum(ensemble_probs) / len(ensemble_probs)
    print(f"模型融合预测概率: {avg_prob:.4f}")

    # 保存集成结果
    with open(os.path.join(output_dir, 'ensemble_prediction.txt'), 'w') as f:
        f.write(f"模型融合预测概率: {avg_prob:.4f}\n")

    # 生成测试集模型对比汇总表
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            "模型": name,
            "准确率": f"{result['accuracy']:.4f}",
            "精确率": f"{result['precision']:.4f}",
            "召回率": f"{result['recall']:.4f}",
            "F1分数": f"{result['f1']:.4f}",
            "AUC": f"{result['auc']:.4f}"
        })

    summary_df = pd.DataFrame(summary_data)

    # 保存为CSV
    summary_df.to_csv(os.path.join(output_dir, 'test_model_comparison.csv'),
                      index=False, encoding='utf-8-sig')

    # 保存为三线表格式的LaTeX表格
    latex_table = summary_df.to_latex(index=False, escape=False, column_format='c|ccccc')
    latex_table = latex_table.replace('\\toprule', '\\hline\\hline')
    latex_table = latex_table.replace('\\midrule', '\\hline')
    latex_table = latex_table.replace('\\bottomrule', '\\hline\\hline')

    with open(os.path.join(output_dir, 'test_model_comparison.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_table)

    # 如果应用了特征工程，保存特征重要性排序
    if args.feature_engineering and '随机森林' in model_dict:
        rf_model = model_dict['随机森林']
        feature_importances = rf_model.feature_importances_
        sorted_idx = np.argsort(feature_importances)[::-1]

        with open(os.path.join(output_dir, 'feature_importance.txt'), 'w') as f:
            f.write("特征重要性排序（随机森林）:\n")
            for i in sorted_idx[:50]:  # 前50个重要特征
                f.write(f"{feature_names[i]}: {feature_importances[i]:.6f}\n")

    print(f"\n所有结果已保存至目录: {output_dir}")
    return results, model_dict, feature_pipeline


# 如果是直接运行此脚本，则调用主函数
# 在文件末尾的 if __name__ == "__main__": 部分进行修改

if __name__ == "__main__":
    # 导入缺失的模块
    from scipy import stats


    # 创建一个类来模拟命令行参数
    class Args:
        def __init__(self):
            self.input_file = "GRPC的PSSM和ACC特征.csv"
            self.protein_file = "Q13133的PseACC和PsePSSM特征.csv"
            self.compound_file = "ehdpp.csv"
            self.test_size = 0.2
            self.output_dir = "results"
            self.save_models = True
            self.compare_only = None
            self.feature_engineering = True
            self.feature_selection_threshold = 2
            self.max_interaction_features = 10
            self.svm_C = [0.1, 1, 10]
            self.svm_gamma = [0.01, 0.1, 1]
            self.svm_kernel = ['rbf']
            self.rf_n_estimators = [50, 100, 200]
            self.rf_max_depth = [5, 10, 15]
            self.rf_min_samples_leaf = [1, 2, 5]
            self.gb_n_estimators = [50, 100, 200]
            self.gb_learning_rate = [0.01, 0.1, 0.5]
            self.gb_max_depth = [3, 5, 7]


    # 创建参数对象
    args = Args()


    # 修改main函数以接受命令行参数对象
    def modified_main(args):
        # 创建输出目录
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(output_dir, f'run_{timestamp}')
        os.makedirs(output_dir)

        # 保存运行参数
        with open(os.path.join(output_dir, 'run_parameters.txt'), 'w') as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")

        # 加载数据
        print("\n加载数据...")
        X, y = load_and_prepare_data(args.input_file)
        print(f"数据集大小: {X.shape}")
        print(f"特征数量: {X.shape[1]}")
        print(f"类别分布: {y.value_counts().to_dict()}")

        # 确定蛋白质和化合物特征
        all_columns = pd.read_csv(args.input_file, nrows=1).columns.tolist()
        protein_id_col = 'gene_id'
        compound_id_col = 'compound_id'
        protein_id_idx = all_columns.index(protein_id_col)
        compound_id_idx = all_columns.index(compound_id_col)
        protein_cols = all_columns[protein_id_idx + 1:compound_id_idx]
        compound_cols = all_columns[compound_id_idx + 1:-1]

        if args.feature_engineering:
            # 应用特征工程优化流程
            print("\n开始特征工程优化...")
            X_train_scaled, X_test_scaled, y_train, y_test, feature_pipeline = optimize_features(
                X, y, args.protein_file, args.compound_file, args.input_file, output_dir
            )
            # 使用优化后的特征列名
            feature_names = feature_pipeline['enhanced_columns']
        else:
            # 传统的数据划分和标准化
            print("\n执行传统数据处理...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42,
                                                                stratify=y)

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 将转换后的数据转换回DataFrame以保留特征名
            feature_names = X.columns.tolist()
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

            # 创建简单的特征管道(用于传统方法)
            feature_pipeline = {
                'selected_features': feature_names,
                'protein_cols': protein_cols,
                'compound_cols': compound_cols,
                'scaler': scaler,
                'enhanced_columns': feature_names
            }

            # 保存标准化器
            joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

        # 配置参数网格
        param_grids = {
            'SVM': {
                'C': args.svm_C,
                'gamma': args.svm_gamma,
                'kernel': args.svm_kernel
            },
            '随机森林': {
                'n_estimators': args.rf_n_estimators,
                'max_depth': args.rf_max_depth,
                'min_samples_leaf': args.rf_min_samples_leaf
            },
            '梯度提升': {
                'n_estimators': args.gb_n_estimators,
                'learning_rate': args.gb_learning_rate,
                'max_depth': args.gb_max_depth
            },
            'AdaBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0]
            },
            '逻辑回归': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l2']
            },
            'K近邻': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            },
            '朴素贝叶斯': {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            }
        }

        if xgb_installed:
            param_grids['XGBoost'] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7]
            }

        if lgbm_installed:
            param_grids['LightGBM'] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 63, 127]
            }

        # 初始化模型字典和最佳参数字典
        model_dict = {}
        best_params_dict = {}

        # 根据参数添加模型
        for model_name in param_grids:
            if args.compare_only is None or model_name in args.compare_only:
                print(f"\n优化 {model_name} 参数...")
                model, best_params = get_best_params(X_train_scaled, y_train, model_name, param_grids[model_name])
                model_dict[model_name] = model
                best_params_dict[model_name] = best_params

        # 保存最佳参数
        with open(os.path.join(output_dir, 'best_parameters.txt'), 'w') as f:
            for model_name, params in best_params_dict.items():
                f.write(f"{model_name}: {params}\n")

        # 评估和比较模型
        results = {}
        for name, model in model_dict.items():
            print(f"\n{'=' * 50}")
            print(f"评估模型: {name}")
            print(f"{'=' * 50}")
            result = evaluate_model(
                model, X_train_scaled, y_train, X_test_scaled, y_test, name, feature_names, output_dir
            )
            results[name] = result

            # 保存模型
            if args.save_models:
                joblib.dump(model, os.path.join(output_dir, f'{name}_model.pkl'))

        # 保存特征工程管道
        joblib.dump(feature_pipeline, os.path.join(output_dir, 'feature_pipeline.pkl'))

        # 预测q13133和ehdpp的相互作用概率
        sample_df = load_q13133_ehdpp_sample(feature_names, args.protein_file, args.compound_file, args.input_file)

        if hasattr(feature_pipeline, 'get') and feature_pipeline.get('scaler'):
            sample_scaled = feature_pipeline['scaler'].transform(sample_df)
        else:
            # 如果没有特定的scaler，使用默认值
            scaler = StandardScaler()
            sample_scaled = scaler.fit_transform(sample_df)

        print("\nq13133与ehdpp的相互作用概率预测：")
        pred_results = []
        ensemble_probs = []

        for name, model in model_dict.items():
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(sample_scaled)[0, 1]
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(sample_scaled)[0]
                prob = 1 / (1 + np.exp(-decision))
            else:
                prob = model.predict(sample_scaled)[0]
            print(f"{name}: 概率={prob:.4f}")
            pred_results.append({"模型": name, "交互概率": prob})
            ensemble_probs.append(prob)

        # 保存预测结果
        pd.DataFrame(pred_results).to_csv(
            os.path.join(output_dir, 'q13133_ehdpp_prediction.csv'),
            index=False, encoding='utf-8-sig'
        )

        # 执行模型融合预测
        print("\n执行模型融合预测...")
        avg_prob = sum(ensemble_probs) / len(ensemble_probs)
        print(f"模型融合预测概率: {avg_prob:.4f}")

        # 保存集成结果
        with open(os.path.join(output_dir, 'ensemble_prediction.txt'), 'w') as f:
            f.write(f"模型融合预测概率: {avg_prob:.6f}\n")

        # 生成测试集模型对比汇总表
        summary_data = []
        for name, result in results.items():
            summary_data.append({
                "模型": name,
                "准确率": f"{result['accuracy']:.4f}",
                "精确率": f"{result['precision']:.4f}",
                "召回率": f"{result['recall']:.4f}",
                "F1分数": f"{result['f1']:.4f}",
                "AUC": f"{result['auc']:.4f}"
            })

        summary_df = pd.DataFrame(summary_data)

        # 保存为CSV
        summary_df.to_csv(os.path.join(output_dir, 'test_model_comparison.csv'),
                          index=False, encoding='utf-8-sig')

        # 保存为三线表格式的LaTeX表格
        latex_table = summary_df.to_latex(index=False, escape=False, column_format='c|ccccc')
        latex_table = latex_table.replace('\\toprule', '\\hline\\hline')
        latex_table = latex_table.replace('\\midrule', '\\hline')
        latex_table = latex_table.replace('\\bottomrule', '\\hline\\hline')

        with open(os.path.join(output_dir, 'test_model_comparison.tex'), 'w', encoding='utf-8') as f:
            f.write(latex_table)

        # 如果应用了特征工程，保存特征重要性排序
        if args.feature_engineering and '随机森林' in model_dict:
            rf_model = model_dict['随机森林']
            feature_importances = rf_model.feature_importances_
            sorted_idx = np.argsort(feature_importances)[::-1]

            with open(os.path.join(output_dir, 'feature_importance.txt'), 'w') as f:
                for i in sorted_idx:
                    if i < len(feature_names):
                        f.write(f"{feature_names[i]}: {feature_importances[i]:.6f}\n")

        print(f"\n所有结果已保存至目录: {output_dir}")
        return results, model_dict, feature_pipeline


    # 调用修改后的main函数
    results, models, pipeline = modified_main(args)