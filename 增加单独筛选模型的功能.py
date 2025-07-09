# =============================================================================
# 增强模型详情版预测结果分析器 GUI - 包含单模型筛选信息
# =============================================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime
import threading
from collections import defaultdict, Counter
import warnings
import platform
import time

# 忽略警告
warnings.filterwarnings('ignore')


# =============================================================================
# 字体配置
# =============================================================================

def configure_chinese_fonts():
    """配置中文字体"""
    try:
        import matplotlib.font_manager as fm

        system = platform.system()
        chinese_fonts = []

        if system == "Windows":
            chinese_fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi']
        elif system == "Darwin":
            chinese_fonts = ['PingFang SC', 'Heiti SC', 'STSong']
        else:
            chinese_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']

        available_fonts = [f.name for f in fm.fontManager.ttflist]

        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                return font

        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return "DejaVu Sans"

    except Exception as e:
        print(f"字体配置出错: {e}")
        return "DejaVu Sans"


configure_chinese_fonts()


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
# 可滚动框架类
# =============================================================================

class ScrollableFrame(ttk.Frame):
    """可滚动的框架"""

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        # 创建Canvas和Scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        # 配置滚动
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # 创建Canvas窗口
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # 配置Canvas滚动
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # 布局
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # 绑定鼠标滚轮事件
        self.bind_mousewheel()

        # 绑定Canvas大小变化事件
        self.canvas.bind('<Configure>', self._on_canvas_configure)

    def _on_canvas_configure(self, event):
        """Canvas大小变化时调整scrollable_frame宽度"""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_frame, width=canvas_width)

    def bind_mousewheel(self):
        """绑定鼠标滚轮事件"""

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_to_mousewheel(event):
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_from_mousewheel(event):
            self.canvas.unbind_all("<MouseWheel>")

        self.canvas.bind('<Enter>', _bind_to_mousewheel)
        self.canvas.bind('<Leave>', _unbind_from_mousewheel)


# =============================================================================
# 快速文件扫描器
# =============================================================================

class FastFileScanner:
    """快速文件扫描器 - 优化性能"""

    def __init__(self):
        self.cache = {}

    def quick_scan_directory(self, directory, max_depth=3):
        """快速扫描目录"""
        print(f"🚀 快速扫描目录: {directory}")
        start_time = time.time()

        prediction_files = {}
        csv_files = []

        for root, dirs, files in os.walk(directory):
            depth = root[len(directory):].count(os.sep)
            if depth >= max_depth:
                dirs[:] = []
                continue

            for file in files:
                if file.endswith('.csv') and ('prediction' in file.lower() or 'result' in file.lower()):
                    full_path = os.path.join(root, file)
                    csv_files.append(full_path)

        print(f"   发现 {len(csv_files)} 个潜在预测文件")

        valid_files = []
        for file_path in csv_files:
            if self._quick_validate_file(file_path):
                valid_files.append(file_path)

        print(f"   验证通过 {len(valid_files)} 个文件")

        for file_path in valid_files:
            self._quick_categorize_file(file_path, prediction_files)

        scan_time = time.time() - start_time
        print(f"   扫描完成，耗时: {scan_time:.2f}秒")

        return prediction_files

    def _quick_validate_file(self, file_path):
        """快速验证文件"""
        try:
            if os.path.getsize(file_path) < 1024:
                return False

            if file_path in self.cache:
                return self.cache[file_path]

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.readline().lower()

                has_protein = 'protein' in header
                has_compound = 'compound' in header
                has_prediction = 'prediction' in header or 'pred' in header

                result = has_protein and has_compound and has_prediction
                self.cache[file_path] = result
                return result

        except Exception:
            return False

    def _quick_categorize_file(self, file_path, prediction_files):
        """快速分类文件"""
        try:
            rel_path = os.path.relpath(file_path)
            path_parts = rel_path.split(os.sep)

            model_type = 'unknown'
            model_name = os.path.basename(file_path).replace('.csv', '')

            for part in path_parts:
                if any(keyword in part.lower() for keyword in ['标准', '随机', 'random', 'standard']):
                    model_type = part
                    break
                elif len(part) > 3 and part not in ['prediction_results_batch', 'reuse_chunks']:
                    model_type = part

            model_name = model_name.replace('_prediction', '').replace('_result', '')

            if model_type not in prediction_files:
                prediction_files[model_type] = {}

            prediction_files[model_type][model_name] = file_path

        except Exception as e:
            print(f"分类文件失败 {file_path}: {e}")


# =============================================================================
# 优化的数据加载器
# =============================================================================

class OptimizedDataLoader:
    """优化的数据加载器"""

    def __init__(self):
        self.chunk_size = 10000

    def load_file_efficiently(self, file_path, progress_callback=None):
        """高效加载文件"""
        try:
            file_size = os.path.getsize(file_path)

            if progress_callback:
                progress_callback(f"加载文件: {os.path.basename(file_path)} ({file_size / 1024 / 1024:.1f}MB)")

            if file_size > 100 * 1024 * 1024:
                return self._load_large_file(file_path, progress_callback)
            else:
                return self._load_normal_file(file_path)

        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return None

    def _load_normal_file(self, file_path):
        """加载普通大小的文件"""
        try:
            df = pd.read_csv(file_path, low_memory=False)
            return self._standardize_dataframe_fast(df)
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None

    def _load_large_file(self, file_path, progress_callback=None):
        """分块加载大文件"""
        try:
            chunks = []
            total_chunks = 0

            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size, low_memory=False):
                chunks.append(self._standardize_dataframe_fast(chunk))
                total_chunks += 1

                if progress_callback and total_chunks % 10 == 0:
                    progress_callback(f"已处理 {total_chunks} 个数据块")

            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                return df
            else:
                return None

        except Exception as e:
            print(f"分块读取文件失败 {file_path}: {e}")
            return None

    def _standardize_dataframe_fast(self, df):
        """快速标准化DataFrame"""
        if df is None or df.empty:
            return None

        try:
            column_map = {}
            columns_lower = {col.lower(): col for col in df.columns}

            # 必需列映射
            for target, patterns in [
                ('protein_id', ['protein_id', 'protein', 'prot_id']),
                ('compound_id', ['compound_id', 'compound', 'comp_id']),
                ('prediction', ['prediction', 'pred', 'label'])
            ]:
                for pattern in patterns:
                    if pattern in columns_lower:
                        column_map[columns_lower[pattern]] = target
                        break

            # 可选列映射
            for target, patterns in [
                ('probability_0', ['probability_0', 'prob_0', 'p0']),
                ('probability_1', ['probability_1', 'prob_1', 'p1']),
                ('confidence', ['confidence', 'conf'])
            ]:
                for pattern in patterns:
                    if pattern in columns_lower:
                        column_map[columns_lower[pattern]] = target
                        break

            df = df.rename(columns=column_map)

            required_cols = ['protein_id', 'compound_id', 'prediction']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"缺少必需列: {missing_cols}")
                return None

            if 'probability_0' not in df.columns:
                if 'probability_1' in df.columns:
                    df['probability_0'] = 1 - df['probability_1']
                else:
                    df['probability_0'] = 0.5
                    df['probability_1'] = 0.5

            if 'probability_1' not in df.columns:
                df['probability_1'] = 1 - df['probability_0']

            if 'confidence' not in df.columns:
                df['confidence'] = df[['probability_0', 'probability_1']].max(axis=1)

            df['prediction'] = df['prediction'].astype(int)

            keep_cols = ['protein_id', 'compound_id', 'prediction', 'probability_0', 'probability_1', 'confidence']
            df = df[keep_cols]

            return df

        except Exception as e:
            print(f"标准化失败: {e}")
            return None


# =============================================================================
# 增强阈值分析器 - 包含单模型筛选信息
# =============================================================================

class EnhancedThresholdAnalyzer:
    """增强阈值分析器 - 支持自定义正例预测阈值和单模型筛选"""

    def __init__(self):
        self.scanner = FastFileScanner()
        self.loader = OptimizedDataLoader()
        self.raw_predictions = {}
        self.compound_stats = {}
        self.available_models = []
        self.result_dir = None
        self.detected_format = None

        # 新增：单模型筛选结果
        self.individual_model_results = {}

        # 分析参数
        self.current_min_consensus = 2
        self.current_prob_threshold = 0.6
        self.current_conf_threshold = 0.7
        # 自定义正例预测阈值
        self.current_positive_prediction_threshold = 0.5

    def quick_scan_directory(self, directory):
        """快速扫描目录"""
        return self.scanner.quick_scan_directory(directory)

    def load_prediction_results(self, result_dir, progress_callback=None):
        """优化的数据加载"""
        print(f"📊 开始加载预测结果...")
        start_time = time.time()

        self.result_dir = result_dir

        try:
            if progress_callback:
                progress_callback("正在扫描文件...")

            prediction_files = self.scanner.quick_scan_directory(result_dir)

            if not prediction_files:
                return False

            total_files = sum(len(models) for models in prediction_files.values())
            print(f"   需要加载 {total_files} 个文件")

            self.raw_predictions = {}
            self.available_models = []
            loaded_count = 0

            for model_type, models in prediction_files.items():
                if progress_callback:
                    progress_callback(f"加载模型类型: {model_type}")

                self.raw_predictions[model_type] = {}

                for model_name, file_path in models.items():
                    try:
                        if progress_callback:
                            progress_callback(f"加载: {model_name} ({loaded_count + 1}/{total_files})")

                        df = self.loader.load_file_efficiently(file_path, progress_callback)

                        if df is not None and len(df) > 0:
                            self.raw_predictions[model_type][model_name] = df
                            self.available_models.append(f"{model_type}_{model_name}")
                            loaded_count += 1
                            print(f"   ✓ {model_name}: {len(df):,} 行")
                        else:
                            print(f"   ✗ {model_name}: 加载失败或数据为空")

                    except Exception as e:
                        print(f"   ✗ {model_name}: {e}")
                        continue

            if progress_callback:
                progress_callback("正在分析化合物统计...")

            self._analyze_compound_statistics_fast()

            if progress_callback:
                progress_callback("正在分析单模型筛选结果...")

            # 新增：分析单模型筛选结果
            self._analyze_individual_model_results()

            load_time = time.time() - start_time
            print(f"📊 数据加载完成，耗时: {load_time:.2f}秒")
            print(f"   成功加载: {loaded_count}/{total_files} 个文件")
            print(f"   化合物对: {len(self.compound_stats):,}")
            print(f"   单模型筛选结果: {len(self.individual_model_results)} 个模型")

            return len(self.compound_stats) > 0

        except Exception as e:
            print(f"加载失败: {e}")
            return False

    def _analyze_compound_statistics_fast(self):
        """快速分析化合物统计 - 使用自定义正例预测阈值"""
        print("🔄 快速分析化合物统计...")
        start_time = time.time()

        self.compound_stats = {}
        compound_predictions = defaultdict(list)

        for model_type, models in self.raw_predictions.items():
            for model_name, df in models.items():
                for _, row in df.iterrows():
                    key = f"{row['protein_id']}_{row['compound_id']}"

                    # 使用自定义阈值重新计算预测结果
                    custom_prediction = 1 if row['probability_1'] >= self.current_positive_prediction_threshold else 0

                    compound_predictions[key].append({
                        'model_type': model_type,
                        'model_name': model_name,
                        'protein_id': row['protein_id'],
                        'compound_id': row['compound_id'],
                        'original_prediction': row['prediction'],  # 保存原始预测
                        'custom_prediction': custom_prediction,  # 基于自定义阈值的预测
                        'probability_0': row['probability_0'],
                        'probability_1': row['probability_1'],
                        'confidence': row['confidence']
                    })

        for compound_key, predictions in compound_predictions.items():
            protein_id, compound_id = compound_key.split('_', 1)

            total_models = len(predictions)
            # 使用自定义阈值的预测结果计算
            positive_predictions = sum(1 for p in predictions if p['custom_prediction'] == 1)

            prob_1_values = [p['probability_1'] for p in predictions]
            conf_values = [p['confidence'] for p in predictions]

            self.compound_stats[compound_key] = {
                'protein_id': protein_id,
                'compound_id': compound_id,
                'total_models': total_models,
                'positive_predictions': positive_predictions,
                'negative_predictions': total_models - positive_predictions,
                'positive_ratio': positive_predictions / total_models,
                'avg_probability_1': np.mean(prob_1_values),
                'avg_confidence': np.mean(conf_values),
                'predictions': predictions
            }

        analysis_time = time.time() - start_time
        print(f"   统计分析完成，耗时: {analysis_time:.2f}秒")
        print(f"   使用正例预测阈值: {self.current_positive_prediction_threshold:.3f}")

    def _analyze_individual_model_results(self):
        """新增：分析每个模型的单独筛选结果"""
        print("🔄 分析单模型筛选结果...")
        start_time = time.time()

        self.individual_model_results = {}

        for model_type, models in self.raw_predictions.items():
            for model_name, df in models.items():
                model_key = f"{model_type}_{model_name}"

                # 筛选大于阈值的化合物
                filtered_df = df[df['probability_1'] >= self.current_positive_prediction_threshold].copy()

                if len(filtered_df) > 0:
                    # 添加自定义预测列
                    filtered_df['custom_prediction'] = 1

                    # 按置信度排序
                    filtered_df = filtered_df.sort_values(['confidence', 'probability_1'], ascending=[False, False])

                    # 计算统计信息
                    total_compounds = len(df)
                    filtered_compounds = len(filtered_df)
                    avg_prob_1 = filtered_df['probability_1'].mean()
                    avg_confidence = filtered_df['confidence'].mean()
                    max_prob_1 = filtered_df['probability_1'].max()
                    min_prob_1 = filtered_df['probability_1'].min()

                    self.individual_model_results[model_key] = {
                        'model_type': model_type,
                        'model_name': model_name,
                        'total_compounds': total_compounds,
                        'filtered_compounds': filtered_compounds,
                        'filtered_ratio': filtered_compounds / total_compounds,
                        'avg_probability_1': avg_prob_1,
                        'avg_confidence': avg_confidence,
                        'max_probability_1': max_prob_1,
                        'min_probability_1': min_prob_1,
                        'filtered_data': filtered_df.to_dict('records'),
                        'threshold_used': self.current_positive_prediction_threshold
                    }
                else:
                    self.individual_model_results[model_key] = {
                        'model_type': model_type,
                        'model_name': model_name,
                        'total_compounds': len(df),
                        'filtered_compounds': 0,
                        'filtered_ratio': 0.0,
                        'avg_probability_1': 0.0,
                        'avg_confidence': 0.0,
                        'max_probability_1': 0.0,
                        'min_probability_1': 0.0,
                        'filtered_data': [],
                        'threshold_used': self.current_positive_prediction_threshold
                    }

        analysis_time = time.time() - start_time
        print(f"   单模型分析完成，耗时: {analysis_time:.2f}秒")
        print(f"   已分析 {len(self.individual_model_results)} 个模型")

    def reanalyze_with_new_threshold(self, new_threshold):
        """使用新的正例预测阈值重新分析"""
        if len(self.compound_stats) == 0:
            return False

        print(f"🔄 重新分析，新的正例预测阈值: {new_threshold:.3f}")

        self.current_positive_prediction_threshold = new_threshold
        self._analyze_compound_statistics_fast()
        self._analyze_individual_model_results()  # 重新分析单模型结果

        return True

    def get_summary_info(self):
        """获取摘要信息"""
        return {
            'total_compounds': len(self.compound_stats),
            'total_models': len(self.available_models),
            'directory': self.result_dir,
            'format': self.detected_format or 'auto_detected',
            'positive_prediction_threshold': self.current_positive_prediction_threshold,
            'individual_models_analyzed': len(self.individual_model_results)
        }

    def get_individual_model_summary(self):
        """新增：获取单模型筛选汇总信息"""
        if not self.individual_model_results:
            return {}

        summary = {}
        for model_key, result in self.individual_model_results.items():
            summary[model_key] = {
                'model_type': result['model_type'],
                'model_name': result['model_name'],
                'total_compounds': result['total_compounds'],
                'filtered_compounds': result['filtered_compounds'],
                'filtered_ratio': result['filtered_ratio'],
                'avg_probability_1': result['avg_probability_1'],
                'avg_confidence': result['avg_confidence']
            }

        return summary

    def get_top_compounds_by_model(self, model_key, top_n=20):
        """新增：获取指定模型的Top N化合物"""
        if model_key not in self.individual_model_results:
            return []

        filtered_data = self.individual_model_results[model_key]['filtered_data']
        return filtered_data[:top_n]

    # 分析方法 - 使用自定义阈值
    def _find_all_positive(self):
        """找到所有模型都预测为正例的化合物（基于自定义阈值）"""
        return [stats for stats in self.compound_stats.values()
                if (stats['total_models'] >= self.current_min_consensus and
                    stats['positive_predictions'] == stats['total_models'])]

    def _find_majority_positive(self):
        """找到大多数模型预测为正例的化合物（基于自定义阈值）"""
        return [stats for stats in self.compound_stats.values()
                if (stats['total_models'] >= self.current_min_consensus and
                    stats['positive_ratio'] > 0.5 and
                    stats['positive_predictions'] >= self.current_min_consensus)]

    def _find_high_confidence(self):
        """找到高置信度的化合物"""
        return [stats for stats in self.compound_stats.values()
                if (stats['total_models'] >= self.current_min_consensus and
                    stats['avg_confidence'] >= self.current_conf_threshold and
                    stats['positive_predictions'] >= self.current_min_consensus)]

    def _find_high_probability(self):
        """找到高概率的化合物"""
        return [stats for stats in self.compound_stats.values()
                if (stats['total_models'] >= self.current_min_consensus and
                    stats['avg_probability_1'] >= self.current_prob_threshold and
                    stats['positive_predictions'] >= self.current_min_consensus)]

    def _find_custom_consensus(self):
        """自定义共识分析（基于自定义阈值）"""
        return [stats for stats in self.compound_stats.values()
                if (stats['total_models'] >= self.current_min_consensus and
                    stats['positive_predictions'] >= self.current_min_consensus and
                    stats['avg_confidence'] >= self.current_conf_threshold and
                    stats['avg_probability_1'] >= self.current_prob_threshold)]


# =============================================================================
# 增强模型详情GUI - 包含单模型筛选功能
# =============================================================================

class EnhancedModelDetailsAnalyzerGUI:
    """增强模型详情GUI - 包含单模型筛选功能"""

    def __init__(self, root):
        self.root = root
        self.analyzer = EnhancedThresholdAnalyzer()
        self.current_figure = None

        # 状态变量
        self.status_var = tk.StringVar(value="准备就绪")
        self.progress_var = tk.DoubleVar()
        self.data_info_var = tk.StringVar(value="未加载数据")
        self.result_dir_var = tk.StringVar()

        # 参数变量
        self.min_consensus_var = tk.IntVar(value=2)
        self.prob_threshold_var = tk.DoubleVar(value=0.6)
        self.conf_threshold_var = tk.DoubleVar(value=0.7)

        # 正例预测阈值变量
        self.positive_prediction_threshold_var = tk.DoubleVar(value=0.5)
        self.positive_prediction_entry_var = tk.StringVar(value="0.500")

        # 直接输入的变量
        self.prob_entry_var = tk.StringVar(value="0.60")
        self.conf_entry_var = tk.StringVar(value="0.70")

        # 新增：模型选择变量
        self.selected_model_var = tk.StringVar()

        # 配置窗口
        self.setup_main_window()
        self.setup_styles()
        self.create_widgets()

    def setup_main_window(self):
        """配置主窗口"""
        self.root.title("🎯 增强模型详情预测结果分析器 v3.5 - 包含单模型筛选")
        self.root.geometry("1600x1000")
        self.root.minsize(1400, 900)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def setup_styles(self):
        """配置样式"""
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Warning.TLabel', foreground='orange')
        style.configure('Highlight.TLabel', foreground='blue', font=('Arial', 10, 'bold'))
        style.configure('Model.TLabel', foreground='purple', font=('Arial', 9, 'bold'))

    def create_widgets(self):
        """创建界面组件"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        self.create_header(main_frame)
        self.create_main_content(main_frame)
        self.create_status_bar(main_frame)

    def create_header(self, parent):
        """创建标题栏"""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.grid_columnconfigure(1, weight=1)

        title_label = ttk.Label(header_frame, text="🎯 增强模型详情预测结果分析器 v3.5", style='Title.TLabel')
        title_label.grid(row=0, column=0, sticky=tk.W)

        info_label = ttk.Label(header_frame,
                               text=f"包含单模型筛选 | 用户: woyaokaoyanhaha | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        info_label.grid(row=0, column=1, sticky=tk.E)

        separator = ttk.Separator(header_frame, orient='horizontal')
        separator.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

    def create_main_content(self, parent):
        """创建主要内容"""
        self.create_scrollable_control_panel(parent)
        self.create_display_area(parent)

    def create_scrollable_control_panel(self, parent):
        """创建可滚动的控制面板"""
        # 创建控制面板外框
        control_outer_frame = ttk.LabelFrame(parent, text="📊 控制面板 (可滚动)", padding="5")
        control_outer_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_outer_frame.grid_rowconfigure(0, weight=1)
        control_outer_frame.grid_columnconfigure(0, weight=1)

        # 设置固定宽度
        control_outer_frame.configure(width=420)

        # 创建可滚动框架
        self.scrollable_control = ScrollableFrame(control_outer_frame)
        self.scrollable_control.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 在可滚动框架内创建控制内容
        self.create_control_content(self.scrollable_control.scrollable_frame)

    def create_control_content(self, parent):
        """在可滚动框架内创建控制内容"""
        # 文件加载区域
        self.create_file_section(parent)

        # 正例预测阈值设置区域（重点）
        self.create_positive_threshold_section(parent)

        # 其他参数设置区域
        self.create_other_parameters_section(parent)

        # 新增：单模型选择区域
        self.create_model_selection_section(parent)

        # 分析功能区域
        self.create_analysis_section(parent)

        # 导出功能区域
        self.create_export_section(parent)

    def create_file_section(self, parent):
        """创建文件加载区域"""
        file_frame = ttk.LabelFrame(parent, text="📁 快速数据加载")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.grid_columnconfigure(1, weight=1)

        # 目录选择
        ttk.Label(file_frame, text="预测结果目录:").grid(row=0, column=0, sticky=tk.W, padx=(5, 5))

        result_dir_entry = ttk.Entry(file_frame, textvariable=self.result_dir_var, width=25)
        result_dir_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))

        browse_btn = ttk.Button(file_frame, text="浏览", command=self.browse_result_dir)
        browse_btn.grid(row=0, column=2, padx=(0, 5))

        # 快速操作按钮
        quick_scan_btn = ttk.Button(file_frame, text="🔍 快速扫描", command=self.quick_scan)
        quick_scan_btn.grid(row=1, column=0, pady=(5, 0), sticky=(tk.W, tk.E))

        load_btn = ttk.Button(file_frame, text="⚡ 快速加载", command=self.fast_load_data)
        load_btn.grid(row=1, column=1, columnspan=2, pady=(5, 0), sticky=(tk.W, tk.E))

        # 状态显示
        self.load_status_label = ttk.Label(file_frame, text="", style='Success.TLabel', font=('Arial', 8))
        self.load_status_label.grid(row=2, column=0, columnspan=3, pady=(5, 0))

        # 数据信息
        info_label = ttk.Label(file_frame, textvariable=self.data_info_var, font=('Arial', 8), wraplength=350)
        info_label.grid(row=3, column=0, columnspan=3, pady=(5, 0))

    def create_positive_threshold_section(self, parent):
        """创建正例预测阈值设置区域"""
        threshold_frame = ttk.LabelFrame(parent, text="🎯 正例预测阈值 (核心参数)", padding="8")
        threshold_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        threshold_frame.grid_columnconfigure(1, weight=1)

        # 说明标签
        explanation_label = ttk.Label(threshold_frame,
                                      text="设置 probability_1 ≥ 阈值时认为预测为正例",
                                      font=('Arial', 9), style='Highlight.TLabel')
        explanation_label.grid(row=0, column=0, columnspan=3, pady=(0, 5), sticky=tk.W)

        # 正例预测阈值设置
        ttk.Label(threshold_frame, text="正例预测阈值:", font=('Arial', 9, 'bold')).grid(row=1, column=0, sticky=tk.W,
                                                                                         padx=(5, 5))

        # 滑块
        pos_pred_scale = ttk.Scale(threshold_frame, from_=0.1, to=0.9, variable=self.positive_prediction_threshold_var,
                                   orient=tk.HORIZONTAL, length=120)
        pos_pred_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5))

        # 直接输入框
        pos_pred_entry = ttk.Entry(threshold_frame, textvariable=self.positive_prediction_entry_var, width=8)
        pos_pred_entry.grid(row=1, column=2, padx=(5, 0))

        # 同步按钮
        pos_pred_sync_btn = ttk.Button(threshold_frame, text="↔", width=3,
                                       command=self.sync_positive_prediction_from_entry)
        pos_pred_sync_btn.grid(row=2, column=2, pady=(2, 0))

        # 快速设置按钮框架
        quick_frame = ttk.Frame(threshold_frame)
        quick_frame.grid(row=3, column=0, columnspan=3, pady=(8, 0), sticky=(tk.W, tk.E))

        ttk.Label(quick_frame, text="快速设置:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 5))

        threshold_buttons = [
            ("严格(0.7)", lambda: self.set_quick_positive_threshold(0.7)),
            ("中等(0.5)", lambda: self.set_quick_positive_threshold(0.5)),
            ("宽松(0.3)", lambda: self.set_quick_positive_threshold(0.3))
        ]

        for text, command in threshold_buttons:
            btn = ttk.Button(quick_frame, text=text, command=command, width=8)
            btn.pack(side=tk.LEFT, padx=1)

        # 重新分析按钮
        reanalyze_btn = ttk.Button(threshold_frame, text="🔄 重新分析", command=self.reanalyze_with_threshold)
        reanalyze_btn.grid(row=4, column=0, columnspan=3, pady=(8, 0), sticky=(tk.W, tk.E))

        # 绑定滑块更新事件
        def update_positive_prediction_entry(*args):
            self.positive_prediction_entry_var.set(f"{self.positive_prediction_threshold_var.get():.3f}")

        self.positive_prediction_threshold_var.trace('w', update_positive_prediction_entry)

        # 绑定输入框回车事件
        pos_pred_entry.bind('<Return>', lambda e: self.sync_positive_prediction_from_entry())

    def create_other_parameters_section(self, parent):
        """创建其他参数设置区域"""
        param_frame = ttk.LabelFrame(parent, text="⚙️ 其他筛选参数", padding="8")
        param_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        param_frame.grid_columnconfigure(1, weight=1)

        # 最小共识模型数
        ttk.Label(param_frame, text="最小共识模型数:", font=('Arial', 9)).grid(row=0, column=0, sticky=tk.W,
                                                                               padx=(5, 5))
        consensus_spin = ttk.Spinbox(param_frame, from_=1, to=10, textvariable=self.min_consensus_var, width=8)
        consensus_spin.grid(row=0, column=1, sticky=tk.W, padx=(0, 5))

        # 概率阈值
        ttk.Label(param_frame, text="概率阈值:", font=('Arial', 9)).grid(row=1, column=0, sticky=tk.W, padx=(5, 5))

        prob_scale = ttk.Scale(param_frame, from_=0.0, to=1.0, variable=self.prob_threshold_var,
                               orient=tk.HORIZONTAL, length=100)
        prob_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5))

        prob_entry = ttk.Entry(param_frame, textvariable=self.prob_entry_var, width=8)
        prob_entry.grid(row=1, column=2, padx=(5, 0))

        prob_sync_btn = ttk.Button(param_frame, text="↔", width=3,
                                   command=self.sync_prob_from_entry)
        prob_sync_btn.grid(row=2, column=2, pady=(2, 0))

        # 置信度阈值
        ttk.Label(param_frame, text="置信度阈值:", font=('Arial', 9)).grid(row=3, column=0, sticky=tk.W, padx=(5, 5))

        conf_scale = ttk.Scale(param_frame, from_=0.0, to=1.0, variable=self.conf_threshold_var,
                               orient=tk.HORIZONTAL, length=100)
        conf_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(0, 5))

        conf_entry = ttk.Entry(param_frame, textvariable=self.conf_entry_var, width=8)
        conf_entry.grid(row=3, column=2, padx=(5, 0))

        conf_sync_btn = ttk.Button(param_frame, text="↔", width=3,
                                   command=self.sync_conf_from_entry)
        conf_sync_btn.grid(row=4, column=2, pady=(2, 0))

        # 绑定滑块更新事件
        def update_prob_entry(*args):
            self.prob_entry_var.set(f"{self.prob_threshold_var.get():.3f}")

        self.prob_threshold_var.trace('w', update_prob_entry)

        def update_conf_entry(*args):
            self.conf_entry_var.set(f"{self.conf_threshold_var.get():.3f}")

        self.conf_threshold_var.trace('w', update_conf_entry)

        # 绑定输入框回车事件
        prob_entry.bind('<Return>', lambda e: self.sync_prob_from_entry())
        conf_entry.bind('<Return>', lambda e: self.sync_conf_from_entry())

        # 应用所有参数按钮
        apply_btn = ttk.Button(param_frame, text="✅ 应用所有参数", command=self.apply_parameters)
        apply_btn.grid(row=5, column=0, columnspan=3, pady=(10, 0), sticky=(tk.W, tk.E))

    def create_model_selection_section(self, parent):
        """新增：创建单模型选择区域"""
        model_frame = ttk.LabelFrame(parent, text="🤖 单模型筛选分析", padding="8")
        model_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.grid_columnconfigure(1, weight=1)

        # 说明标签
        explanation_label = ttk.Label(model_frame,
                                      text="查看单个模型筛选出的大于阈值的化合物",
                                      font=('Arial', 9), style='Model.TLabel')
        explanation_label.grid(row=0, column=0, columnspan=3, pady=(0, 5), sticky=tk.W)

        # 模型选择下拉框
        ttk.Label(model_frame, text="选择模型:", font=('Arial', 9)).grid(row=1, column=0, sticky=tk.W, padx=(5, 5))

        self.model_combobox = ttk.Combobox(model_frame, textvariable=self.selected_model_var,
                                           width=25, state="readonly")
        self.model_combobox.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(0, 5))

        # 单模型分析按钮
        model_analysis_buttons = [
            ("📊 单模型统计", self.show_individual_model_stats),
            ("📋 模型筛选汇总", self.show_model_filtering_summary),
            ("🎯 查看筛选结果", self.show_model_filtered_compounds)
        ]

        for i, (text, command) in enumerate(model_analysis_buttons):
            btn = ttk.Button(model_frame, text=text, command=command)
            btn.grid(row=i + 2, column=0, columnspan=3, pady=2, sticky=(tk.W, tk.E))

    def create_analysis_section(self, parent):
        """创建分析功能区域"""
        analysis_frame = ttk.LabelFrame(parent, text="🔍 综合分析功能", padding="8")
        analysis_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        analysis_frame.grid_columnconfigure(0, weight=1)

        buttons = [
            ("📊 基础统计", self.show_basic_stats),
            ("🎯 共识分析", self.show_consensus_analysis),
            ("📈 阈值敏感性", self.show_threshold_sensitivity),
            ("🎨 分布图", self.show_distribution_plots)
        ]

        for i, (text, command) in enumerate(buttons):
            btn = ttk.Button(analysis_frame, text=text, command=command)
            btn.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2)

    def create_export_section(self, parent):
        """创建导出功能区域"""
        export_frame = ttk.LabelFrame(parent, text="💾 导出功能", padding="8")
        export_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        export_frame.grid_columnconfigure(0, weight=1)

        export_buttons = [
            ("📋 生成分析报告", self.generate_enhanced_report),
            ("📁 简单结果导出", self.export_simple_results),
            ("🔍 详细结果导出", self.export_detailed_results),
            ("🤖 单模型筛选导出", self.export_individual_model_results),  # 新增
            ("🖼️ 保存当前图表", self.save_current_plot)
        ]

        for i, (text, command) in enumerate(export_buttons):
            btn = ttk.Button(export_frame, text=text, command=command)
            btn.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2)

    def create_display_area(self, parent):
        """创建显示区域"""
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(display_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 图表标签页
        plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(plot_frame, text="📊 图表")

        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加工具栏
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)

        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # 新增：单模型数据表格标签页
        model_table_frame = ttk.Frame(self.notebook)
        self.notebook.add(model_table_frame, text="🤖 单模型数据")

        # 创建单模型树形视图
        self.model_tree = ttk.Treeview(model_table_frame, show='headings')

        # 单模型滚动条
        model_v_scrollbar = ttk.Scrollbar(model_table_frame, orient=tk.VERTICAL, command=self.model_tree.yview)
        model_h_scrollbar = ttk.Scrollbar(model_table_frame, orient=tk.HORIZONTAL, command=self.model_tree.xview)

        self.model_tree.configure(yscrollcommand=model_v_scrollbar.set, xscrollcommand=model_h_scrollbar.set)

        # 布局
        self.model_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        model_v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        model_h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        model_table_frame.grid_rowconfigure(0, weight=1)
        model_table_frame.grid_columnconfigure(0, weight=1)

        # 信息标签页
        info_frame = ttk.Frame(self.notebook)
        self.notebook.add(info_frame, text="ℹ️ 详细信息")

        self.info_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, font=('Consolas', 10))
        self.info_text.pack(fill=tk.BOTH, expand=True)

        self.show_welcome_info()

    def create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(status_frame, text="状态:").grid(row=0, column=0, padx=(0, 5))
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.grid(row=0, column=1, sticky=tk.W)

        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, mode='determinate', length=200)
        self.progress_bar.grid(row=0, column=2, padx=(10, 0))

    # ===============================
    # 参数同步函数
    # ===============================

    def sync_positive_prediction_from_entry(self):
        """从输入框同步正例预测阈值到滑块"""
        try:
            value = float(self.positive_prediction_entry_var.get())
            if 0.1 <= value <= 0.9:
                self.positive_prediction_threshold_var.set(value)
            else:
                messagebox.showwarning("输入错误", "正例预测阈值必须在0.1-0.9之间")
                self.positive_prediction_entry_var.set(f"{self.positive_prediction_threshold_var.get():.3f}")
        except ValueError:
            messagebox.showwarning("输入错误", "请输入有效的数值")
            self.positive_prediction_entry_var.set(f"{self.positive_prediction_threshold_var.get():.3f}")

    def sync_prob_from_entry(self):
        """从输入框同步概率阈值到滑块"""
        try:
            value = float(self.prob_entry_var.get())
            if 0.0 <= value <= 1.0:
                self.prob_threshold_var.set(value)
            else:
                messagebox.showwarning("输入错误", "概率阈值必须在0.0-1.0之间")
                self.prob_entry_var.set(f"{self.prob_threshold_var.get():.3f}")
        except ValueError:
            messagebox.showwarning("输入错误", "请输入有效的数值")
            self.prob_entry_var.set(f"{self.prob_threshold_var.get():.3f}")

    def sync_conf_from_entry(self):
        """从输入框同步置信度阈值到滑块"""
        try:
            value = float(self.conf_entry_var.get())
            if 0.0 <= value <= 1.0:
                self.conf_threshold_var.set(value)
            else:
                messagebox.showwarning("输入错误", "置信度阈值必须在0.0-1.0之间")
                self.conf_entry_var.set(f"{self.conf_threshold_var.get():.3f}")
        except ValueError:
            messagebox.showwarning("输入错误", "请输入有效的数值")
            self.conf_entry_var.set(f"{self.conf_threshold_var.get():.3f}")

    def set_quick_positive_threshold(self, threshold):
        """快速设置正例预测阈值"""
        self.positive_prediction_threshold_var.set(threshold)
        self.positive_prediction_entry_var.set(f"{threshold:.3f}")
        messagebox.showinfo("阈值设置", f"已设置正例预测阈值={threshold:.3f}\n\n点击'🔄 重新分析'按钮应用新阈值")

    def reanalyze_with_threshold(self):
        """使用新的正例预测阈值重新分析"""
        if len(self.analyzer.compound_stats) == 0:
            messagebox.showwarning("警告", "请先加载数据")
            return

        new_threshold = self.positive_prediction_threshold_var.get()

        self.status_var.set(f"重新分析中，正例预测阈值: {new_threshold:.3f}...")

        try:
            success = self.analyzer.reanalyze_with_new_threshold(new_threshold)

            if success:
                # 更新显示信息
                summary = self.analyzer.get_summary_info()
                info_text = f"✅ 重新分析完成 | 化合物: {summary['total_compounds']:,} | 模型: {summary['total_models']} | 正例阈值: {summary['positive_prediction_threshold']:.3f}"
                self.data_info_var.set(info_text)

                # 更新模型选择下拉框
                self.update_model_combobox()

                # 刷新图表
                self.show_basic_stats()

                # 更新信息显示
                self.update_info_display()

                self.status_var.set("重新分析完成")
                messagebox.showinfo("成功",
                                    f"已使用新的正例预测阈值 {new_threshold:.3f} 重新分析数据\n\n单模型筛选结果已更新")
            else:
                messagebox.showerror("错误", "重新分析失败")
                self.status_var.set("重新分析失败")

        except Exception as e:
            messagebox.showerror("错误", f"重新分析过程中出错: {e}")
            self.status_var.set("重新分析失败")

    def update_model_combobox(self):
        """更新模型选择下拉框"""
        if hasattr(self, 'model_combobox') and self.analyzer.individual_model_results:
            model_list = list(self.analyzer.individual_model_results.keys())
            self.model_combobox['values'] = model_list
            if model_list and not self.selected_model_var.get():
                self.selected_model_var.set(model_list[0])

    # ===============================
    # 事件处理函数
    # ===============================

    def browse_result_dir(self):
        """浏览目录"""
        directory = filedialog.askdirectory(title="选择预测结果目录", initialdir=os.getcwd())
        if directory:
            self.result_dir_var.set(directory)

    def quick_scan(self):
        """快速扫描"""
        result_dir = self.result_dir_var.get().strip()

        if not result_dir:
            messagebox.showwarning("警告", "请先选择目录")
            return

        if not os.path.exists(result_dir):
            messagebox.showerror("错误", f"目录不存在: {result_dir}")
            return

        self.status_var.set("正在快速扫描...")

        try:
            prediction_files = self.analyzer.quick_scan_directory(result_dir)

            if prediction_files:
                total_files = sum(len(models) for models in prediction_files.values())
                info_msg = f"快速扫描完成！\n"
                info_msg += f"发现 {len(prediction_files)} 个模型类型\n"
                info_msg += f"总计 {total_files} 个预测文件\n\n"

                for model_type, models in prediction_files.items():
                    info_msg += f"• {model_type}: {len(models)} 个模型\n"

                self.load_status_label.config(text=f"✅ 发现 {total_files} 个文件")
                messagebox.showinfo("快速扫描结果", info_msg)
            else:
                self.load_status_label.config(text="❌ 未发现预测文件")
                messagebox.showwarning("扫描结果", "未发现有效的预测文件")

            self.status_var.set("扫描完成")

        except Exception as e:
            messagebox.showerror("错误", f"扫描失败: {e}")
            self.status_var.set("扫描失败")

    def fast_load_data(self):
        """快速加载数据"""
        result_dir = self.result_dir_var.get().strip()

        if not result_dir:
            messagebox.showwarning("警告", "请先选择目录")
            return

        if not os.path.exists(result_dir):
            messagebox.showerror("错误", f"目录不存在: {result_dir}")
            return

        # 在后台线程中加载
        def load_thread():
            try:
                def progress_callback(message):
                    self.root.after(0, lambda: self.status_var.set(message))

                success = self.analyzer.load_prediction_results(result_dir, progress_callback)
                self.root.after(0, self.on_data_loaded, success)

            except Exception as e:
                self.root.after(0, self.on_data_load_error, str(e))

        self.status_var.set("正在快速加载数据...")
        self.progress_var.set(0)

        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()

    def on_data_loaded(self, success):
        """数据加载完成"""
        if success:
            summary = self.analyzer.get_summary_info()

            info_text = f"✅ 快速加载成功 | 化合物: {summary['total_compounds']:,} | 模型: {summary['total_models']} | 正例阈值: {summary['positive_prediction_threshold']:.3f}"
            self.data_info_var.set(info_text)
            self.load_status_label.config(text="✅ 数据加载成功")
            self.status_var.set("加载完成")

            # 更新模型选择下拉框
            self.update_model_combobox()

            self.update_info_display()
            self.show_basic_stats()

        else:
            self.data_info_var.set("❌ 数据加载失败")
            self.load_status_label.config(text="❌ 加载失败")
            self.status_var.set("加载失败")
            messagebox.showerror("错误", "数据加载失败，请检查目录和文件格式")

        self.progress_var.set(100)

    def on_data_load_error(self, error_msg):
        """加载错误处理"""
        self.data_info_var.set("❌ 数据加载失败")
        self.load_status_label.config(text="❌ 加载错误")
        self.status_var.set("加载失败")
        messagebox.showerror("错误", f"数据加载失败: {error_msg}")
        self.progress_var.set(0)

    def apply_parameters(self):
        """应用所有参数"""
        if len(self.analyzer.compound_stats) == 0:
            messagebox.showwarning("警告", "请先加载数据")
            return

        # 应用所有参数
        self.analyzer.current_min_consensus = self.min_consensus_var.get()
        self.analyzer.current_prob_threshold = self.prob_threshold_var.get()
        self.analyzer.current_conf_threshold = self.conf_threshold_var.get()

        # 应用正例预测阈值并重新分析
        new_threshold = self.positive_prediction_threshold_var.get()
        self.analyzer.reanalyze_with_new_threshold(new_threshold)

        # 更新模型选择下拉框
        self.update_model_combobox()

        # 更新显示
        summary = self.analyzer.get_summary_info()
        info_text = f"✅ 参数已应用 | 化合物: {summary['total_compounds']:,} | 模型: {summary['total_models']} | 正例阈值: {summary['positive_prediction_threshold']:.3f}"
        self.data_info_var.set(info_text)

        self.status_var.set("所有参数已更新")
        messagebox.showinfo("成功", f"所有筛选参数已应用:\n"
                                    f"正例预测阈值: {new_threshold:.3f}\n"
                                    f"最小共识模型数: {self.analyzer.current_min_consensus}\n"
                                    f"概率阈值: {self.analyzer.current_prob_threshold:.3f}\n"
                                    f"置信度阈值: {self.analyzer.current_conf_threshold:.3f}\n\n"
                                    f"单模型筛选结果已更新")

    def check_analyzer(self):
        """检查是否已加载数据"""
        if len(self.analyzer.compound_stats) == 0:
            messagebox.showwarning("警告", "请先加载预测数据")
            return False
        return True

    # ===============================
    # 分析功能
    # ===============================

    def show_basic_stats(self):
        """显示基础统计"""
        if not self.check_analyzer():
            return

        self.status_var.set("生成基础统计...")

        self.figure.clear()

        # 创建2x2子图
        fig = self.figure

        # 获取数据
        all_probs = [stats['avg_probability_1'] for stats in self.analyzer.compound_stats.values()]
        all_confs = [stats['avg_confidence'] for stats in self.analyzer.compound_stats.values()]
        all_ratios = [stats['positive_ratio'] for stats in self.analyzer.compound_stats.values()]

        # 子图1: 概率分布直方图
        ax1 = fig.add_subplot(221)
        ax1.hist(all_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(self.analyzer.current_prob_threshold, color='red', linestyle='--',
                    label=f'概率阈值 ({self.analyzer.current_prob_threshold:.3f})')
        ax1.axvline(self.analyzer.current_positive_prediction_threshold, color='blue', linestyle='-',
                    label=f'正例预测阈值 ({self.analyzer.current_positive_prediction_threshold:.3f})', linewidth=2)
        ax1.set_title('概率分布', fontsize=12)
        ax1.set_xlabel('平均正例概率')
        ax1.set_ylabel('化合物数量')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 置信度分布直方图
        ax2 = fig.add_subplot(222)
        ax2.hist(all_confs, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(self.analyzer.current_conf_threshold, color='red', linestyle='--',
                    label=f'置信度阈值 ({self.analyzer.current_conf_threshold:.3f})')
        ax2.set_title('置信度分布', fontsize=12)
        ax2.set_xlabel('平均置信度')
        ax2.set_ylabel('化合物数量')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子图3: 正例比例分布
        ax3 = fig.add_subplot(223)
        ax3.hist(all_ratios, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.set_title(f'正例比例分布 (阈值: {self.analyzer.current_positive_prediction_threshold:.3f})', fontsize=12)
        ax3.set_xlabel('正例预测比例')
        ax3.set_ylabel('化合物数量')
        ax3.grid(True, alpha=0.3)

        # 子图4: 概率vs置信度散点图
        ax4 = fig.add_subplot(224)
        scatter = ax4.scatter(all_probs, all_confs, c=all_ratios, cmap='RdYlBu_r',
                              alpha=0.6, s=30)
        ax4.axhline(self.analyzer.current_conf_threshold, color='red', linestyle='--', alpha=0.7)
        ax4.axvline(self.analyzer.current_prob_threshold, color='red', linestyle='--', alpha=0.7)
        ax4.axvline(self.analyzer.current_positive_prediction_threshold, color='blue', linestyle='-', alpha=0.8,
                    linewidth=2)
        ax4.set_title('概率 vs 置信度', fontsize=12)
        ax4.set_xlabel('平均正例概率')
        ax4.set_ylabel('平均置信度')
        ax4.grid(True, alpha=0.3)

        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax4)
        cbar.set_label('正例比例')

        fig.tight_layout()
        self.canvas.draw()

        self.status_var.set("基础统计完成")

    def show_consensus_analysis(self):
        """显示共识分析"""
        if not self.check_analyzer():
            return

        self.status_var.set("进行共识分析...")

        # 执行共识分析
        all_positive = self.analyzer._find_all_positive()
        majority_positive = self.analyzer._find_majority_positive()
        high_confidence = self.analyzer._find_high_confidence()
        high_probability = self.analyzer._find_high_probability()
        custom_consensus = self.analyzer._find_custom_consensus()

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        categories = ['所有模型\n一致', '大多数\n同意', '高置信度', '高概率', '综合\n筛选']
        counts = [len(all_positive), len(majority_positive), len(high_confidence),
                  len(high_probability), len(custom_consensus)]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

        bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')

        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')

        ax.set_title(f'共识分析结果 (正例预测阈值: {self.analyzer.current_positive_prediction_threshold:.3f})',
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('化合物数量')
        ax.grid(True, alpha=0.3, axis='y')

        self.figure.tight_layout()
        self.canvas.draw()

        # 更新详细信息
        self.update_consensus_info(all_positive, majority_positive, high_confidence,
                                   high_probability, custom_consensus)

        self.status_var.set("共识分析完成")

    def show_threshold_sensitivity(self):
        """显示阈值敏感性分析"""
        if not self.check_analyzer():
            return

        self.status_var.set("分析阈值敏感性...")

        self.figure.clear()

        # 创建子图
        fig = self.figure
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # 概率阈值敏感性
        prob_thresholds = np.arange(0.5, 1.0, 0.05)
        prob_counts = []

        for threshold in prob_thresholds:
            count = sum(1 for stats in self.analyzer.compound_stats.values()
                        if stats['avg_probability_1'] >= threshold and
                        stats['positive_predictions'] >= self.analyzer.current_min_consensus)
            prob_counts.append(count)

        ax1.plot(prob_thresholds, prob_counts, 'b-o', linewidth=2, markersize=6)
        ax1.axvline(self.analyzer.current_prob_threshold, color='red', linestyle='--',
                    label=f'当前阈值 ({self.analyzer.current_prob_threshold:.2f})')
        ax1.axvline(self.analyzer.current_positive_prediction_threshold, color='blue', linestyle='-',
                    label=f'正例预测阈值 ({self.analyzer.current_positive_prediction_threshold:.2f})', linewidth=2)
        ax1.set_xlabel('概率阈值')
        ax1.set_ylabel('符合条件的化合物数量')
        ax1.set_title('概率阈值敏感性', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 置信度阈值敏感性
        conf_thresholds = np.arange(0.5, 1.0, 0.05)
        conf_counts = []

        for threshold in conf_thresholds:
            count = sum(1 for stats in self.analyzer.compound_stats.values()
                        if stats['avg_confidence'] >= threshold and
                        stats['positive_predictions'] >= self.analyzer.current_min_consensus)
            conf_counts.append(count)

        ax2.plot(conf_thresholds, conf_counts, 'g-o', linewidth=2, markersize=6)
        ax2.axvline(self.analyzer.current_conf_threshold, color='red', linestyle='--',
                    label=f'当前阈值 ({self.analyzer.current_conf_threshold:.2f})')
        ax2.set_xlabel('置信度阈值')
        ax2.set_ylabel('符合条件的化合物数量')
        ax2.set_title('置信度阈值敏感性', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        self.canvas.draw()

        self.status_var.set("阈值敏感性分析完成")

    def show_distribution_plots(self):
        """显示分布图"""
        if not self.check_analyzer():
            return

        self.status_var.set("生成分布图...")

        self.figure.clear()

        all_probs = [stats['avg_probability_1'] for stats in self.analyzer.compound_stats.values()]
        all_confs = [stats['avg_confidence'] for stats in self.analyzer.compound_stats.values()]
        all_ratios = [stats['positive_ratio'] for stats in self.analyzer.compound_stats.values()]

        # 创建子图
        fig = self.figure

        # 子图1: 概率分布（密度图）
        ax1 = fig.add_subplot(221)
        ax1.hist(all_probs, bins=30, density=True, alpha=0.7, color='skyblue')
        ax1.axvline(self.analyzer.current_prob_threshold, color='red', linestyle='--',
                    label=f'概率阈值 ({self.analyzer.current_prob_threshold:.2f})')
        ax1.axvline(self.analyzer.current_positive_prediction_threshold, color='blue', linestyle='-',
                    label=f'正例预测阈值 ({self.analyzer.current_positive_prediction_threshold:.2f})', linewidth=2)
        ax1.set_title('概率密度分布')
        ax1.set_xlabel('平均正例概率')
        ax1.set_ylabel('密度')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 置信度分布（密度图）
        ax2 = fig.add_subplot(222)
        ax2.hist(all_confs, bins=30, density=True, alpha=0.7, color='lightgreen')
        ax2.axvline(self.analyzer.current_conf_threshold, color='red', linestyle='--',
                    label=f'置信度阈值 ({self.analyzer.current_conf_threshold:.2f})')
        ax2.set_title('置信度密度分布')
        ax2.set_xlabel('平均置信度')
        ax2.set_ylabel('密度')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子图3: 正例比例饼图
        ax3 = fig.add_subplot(223)
        ratio_bins = [0, 0.25, 0.5, 0.75, 1.0]
        ratio_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
        ratio_counts = []

        for i in range(len(ratio_bins) - 1):
            count = sum(1 for r in all_ratios if ratio_bins[i] <= r < ratio_bins[i + 1])
            ratio_counts.append(count)

        # 添加100%的化合物
        ratio_counts[-1] += sum(1 for r in all_ratios if r == 1.0)

        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700']
        wedges, texts, autotexts = ax3.pie(ratio_counts, labels=ratio_labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90)
        ax3.set_title(f'正例比例分布\n(阈值: {self.analyzer.current_positive_prediction_threshold:.3f})')

        # 子图4: 箱线图
        ax4 = fig.add_subplot(224)
        data_to_plot = [all_probs, all_confs, all_ratios]
        box_plot = ax4.boxplot(data_to_plot, labels=['概率', '置信度', '正例比例'])

        # 添加阈值线
        ax4.axhline(self.analyzer.current_positive_prediction_threshold, color='blue', linestyle='-',
                    alpha=0.7, linewidth=2,
                    label=f'正例预测阈值 ({self.analyzer.current_positive_prediction_threshold:.3f})')

        ax4.set_title('数据分布箱线图')
        ax4.set_ylabel('数值')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        fig.tight_layout()
        self.canvas.draw()

        self.status_var.set("分布图生成完成")

    # ===============================
    # 新增：单模型分析功能
    # ===============================

    def show_individual_model_stats(self):
        """显示单个模型统计"""
        if not self.check_analyzer():
            return

        selected_model = self.selected_model_var.get()
        if not selected_model:
            messagebox.showwarning("警告", "请先选择一个模型")
            return

        if selected_model not in self.analyzer.individual_model_results:
            messagebox.showerror("错误", f"未找到模型 {selected_model} 的数据")
            return

        self.status_var.set(f"生成模型 {selected_model} 的统计...")

        model_result = self.analyzer.individual_model_results[selected_model]

        self.figure.clear()
        fig = self.figure

        # 创建2x2子图显示单模型统计
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        # 子图1: 筛选结果饼图
        labels = ['筛选出', '未筛选']
        sizes = [model_result['filtered_compounds'],
                 model_result['total_compounds'] - model_result['filtered_compounds']]
        colors = ['#FF6B6B', '#C0C0C0']

        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'{model_result["model_name"]}\n筛选结果分布')

        # 子图2: 筛选出化合物的概率分布
        if model_result['filtered_data']:
            probs = [item['probability_1'] for item in model_result['filtered_data']]
            ax2.hist(probs, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(self.analyzer.current_positive_prediction_threshold, color='red', linestyle='--',
                        label=f'阈值 ({self.analyzer.current_positive_prediction_threshold:.3f})')
            ax2.set_title('筛选化合物概率分布')
            ax2.set_xlabel('Probability_1')
            ax2.set_ylabel('化合物数量')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '无筛选数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('筛选化合物概率分布')

        # 子图3: 筛选出化合物的置信度分布
        if model_result['filtered_data']:
            confs = [item['confidence'] for item in model_result['filtered_data']]
            ax3.hist(confs, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_title('筛选化合物置信度分布')
            ax3.set_xlabel('Confidence')
            ax3.set_ylabel('化合物数量')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '无筛选数据', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('筛选化合物置信度分布')

        # 子图4: 统计信息文本
        ax4.axis('off')
        stats_text = f"""模型统计信息:
模型类型: {model_result['model_type']}
模型名称: {model_result['model_name']}

总化合物数: {model_result['total_compounds']:,}
筛选出数量: {model_result['filtered_compounds']:,}
筛选比例: {model_result['filtered_ratio']:.3f}

筛选化合物统计:
平均概率: {model_result['avg_probability_1']:.4f}
平均置信度: {model_result['avg_confidence']:.4f}
最大概率: {model_result['max_probability_1']:.4f}
最小概率: {model_result['min_probability_1']:.4f}

使用阈值: {model_result['threshold_used']:.3f}"""

        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace')

        fig.tight_layout()
        self.canvas.draw()

        self.status_var.set(f"模型 {selected_model} 统计完成")

    def show_model_filtering_summary(self):
        """显示所有模型筛选汇总"""
        if not self.check_analyzer():
            return

        if not self.analyzer.individual_model_results:
            messagebox.showwarning("警告", "无单模型筛选数据")
            return

        self.status_var.set("生成模型筛选汇总...")

        self.figure.clear()
        fig = self.figure

        # 准备数据
        model_names = []
        filtered_counts = []
        filtered_ratios = []
        avg_probs = []
        avg_confs = []

        for model_key, result in self.analyzer.individual_model_results.items():
            model_names.append(result['model_name'][:15])  # 截断长名称
            filtered_counts.append(result['filtered_compounds'])
            filtered_ratios.append(result['filtered_ratio'])
            avg_probs.append(result['avg_probability_1'])
            avg_confs.append(result['avg_confidence'])

        # 创建2x2子图
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        # 子图1: 各模型筛选化合物数量
        bars1 = ax1.bar(range(len(model_names)), filtered_counts, color='skyblue', alpha=0.8)
        ax1.set_title(f'各模型筛选化合物数量\n(阈值: {self.analyzer.current_positive_prediction_threshold:.3f})')
        ax1.set_ylabel('筛选化合物数量')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, count in zip(bars1, filtered_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + max(filtered_counts) * 0.01,
                     f'{count}', ha='center', va='bottom', fontsize=8)

        # 子图2: 各模型筛选比例
        bars2 = ax2.bar(range(len(model_names)), filtered_ratios, color='lightgreen', alpha=0.8)
        ax2.set_title('各模型筛选比例')
        ax2.set_ylabel('筛选比例')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, ratio in zip(bars2, filtered_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + max(filtered_ratios) * 0.01,
                     f'{ratio:.3f}', ha='center', va='bottom', fontsize=8)

        # 子图3: 各模型平均概率
        bars3 = ax3.bar(range(len(model_names)), avg_probs, color='lightcoral', alpha=0.8)
        ax3.axhline(self.analyzer.current_positive_prediction_threshold, color='red', linestyle='--',
                    label=f'阈值 ({self.analyzer.current_positive_prediction_threshold:.3f})')
        ax3.set_title('各模型筛选化合物平均概率')
        ax3.set_ylabel('平均 Probability_1')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 子图4: 各模型平均置信度
        bars4 = ax4.bar(range(len(model_names)), avg_confs, color='gold', alpha=0.8)
        ax4.set_title('各模型筛选化合物平均置信度')
        ax4.set_ylabel('平均 Confidence')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        self.canvas.draw()

        # 更新信息显示
        self.update_model_summary_info()

        self.status_var.set("模型筛选汇总完成")

    def show_model_filtered_compounds(self):
        """在表格中显示选定模型的筛选化合物"""
        if not self.check_analyzer():
            return

        selected_model = self.selected_model_var.get()
        if not selected_model:
            messagebox.showwarning("警告", "请先选择一个模型")
            return

        if selected_model not in self.analyzer.individual_model_results:
            messagebox.showerror("错误", f"未找到模型 {selected_model} 的数据")
            return

        self.status_var.set(f"显示模型 {selected_model} 的筛选化合物...")

        model_result = self.analyzer.individual_model_results[selected_model]
        filtered_data = model_result['filtered_data']

        # 清除现有数据
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)

        # 设置列
        columns = ('蛋白质ID', '化合物ID', '预测', '概率0', '概率1', '置信度')
        self.model_tree['columns'] = columns

        for col in columns:
            self.model_tree.heading(col, text=col)
            self.model_tree.column(col, width=100)

        # 添加数据
        for item in filtered_data[:100]:  # 限制显示前100个
            self.model_tree.insert('', 'end', values=(
                item['protein_id'],
                item['compound_id'],
                '相互作用' if item['custom_prediction'] == 1 else '无相互作用',
                f"{item['probability_0']:.4f}",
                f"{item['probability_1']:.4f}",
                f"{item['confidence']:.4f}"
            ))

        # 切换到单模型数据标签页
        self.notebook.select(1)  # 单模型数据是第二个标签页

        self.status_var.set(f"已显示模型 {selected_model} 的筛选结果")

    # ===============================
    # 信息更新函数
    # ===============================

    def show_welcome_info(self):
        """显示欢迎信息"""
        welcome_text = """
🎯 增强模型详情预测结果分析器 v3.5 - 包含单模型筛选
════════════════════════════════════════

🆕 新增功能:
✨ 单模型筛选分析：查看每个模型筛选出的大于阈值的化合物
✨ 模型筛选汇总：对比所有模型的筛选效果
✨ 模型筛选详情：显示具体的筛选化合物列表
✨ 单模型导出：导出每个模型的筛选结果

🎉 核心功能:
✨ 自定义正例预测阈值：可设置probability_1 ≥ 阈值时认为预测为正例
✨ 实时重新分析：修改阈值后立即重新计算所有统计结果
✨ 智能参数控制：滑块+直接输入+快速设置
✨ 完整导出功能：简单/详细/单模型多种导出选项

⚡ 性能优化特性:
• 快速文件扫描：限制搜索深度，避免深度递归
• 智能文件验证：只检查文件大小和基本格式
• 分块数据加载：大文件自动分块处理
• 内存优化处理：及时释放不需要的数据

🖱️ 界面操作说明:
• 控制面板支持鼠标滚轮滚动
• 新增单模型选择区域，可选择查看具体模型
• 单模型数据标签页显示筛选化合物详情
• 所有功能按钮都可正常访问

📋 使用步骤:
1. 点击"浏览"选择预测结果目录
2. 点击"🔍 快速扫描"预览文件结构
3. 点击"⚡ 快速加载"导入数据
4. 设置正例预测阈值（核心功能）：
   • 使用滑块拖拽调整（0.1-0.9）
   • 直接在输入框中输入精确数值
   • 使用快速设置按钮（严格0.7/中等0.5/宽松0.3）
   • 点击"🔄 重新分析"应用新阈值
5. 调整其他筛选参数
6. 点击"✅ 应用所有参数"确认设置
7. 选择综合分析功能查看整体结果
8. 使用单模型分析功能：
   • 在下拉框中选择要分析的模型
   • 点击"📊 单模型统计"查看该模型的详细统计
   • 点击"📋 模型筛选汇总"对比所有模型
   • 点击"🎯 查看筛选结果"查看具体化合物列表
9. 使用导出功能保存结果

🤖 单模型筛选功能说明:
• 每个模型的筛选：基于当前正例预测阈值筛选probability_1 ≥ 阈值的化合物
• 筛选统计：显示筛选数量、比例、平均概率、平均置信度等
• 筛选结果：按置信度和概率排序的具体化合物列表
• 模型对比：横向对比所有模型的筛选效果

🎯 正例预测阈值功能:
• 传统方式：固定使用0.5作为正例预测阈值
• 增强方式：可自定义设置0.1-0.9之间的任意阈值
• 实时调整：修改阈值后点击"🔄 重新分析"立即生效
• 单模型应用：新阈值同时应用于综合分析和单模型分析

💾 完整导出功能:
• 📋 自动生成详细分析报告（包含单模型信息）
• 📁 简单结果导出（摘要数据）
• 🔍 详细结果导出（含每个模型预测）
• 🤖 单模型筛选导出（每个模型的筛选结果）
• 🖼️ 高质量图表保存

💡 应用场景:
• 模型性能评估：比较不同模型的筛选效果
• 化合物优先级：根据单模型结果确定化合物优先级
• 阈值优化：通过单模型分析优化正例预测阈值
• 结果验证：交叉验证不同模型的预测一致性

开发者: woyaokaoyanhaha
版本: 3.5.0 (增强模型详情版)
更新日期: 2025-06-18 13:59:41
"""

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, welcome_text)

    def update_info_display(self):
        """更新信息显示"""
        if len(self.analyzer.compound_stats) == 0:
            return

        summary = self.analyzer.get_summary_info()

        info_text = f"""
📊 数据概况 (增强模型详情版v3.5)
═══════════════════════════════════════════
数据目录: {summary['directory']}
检测格式: {summary['format']}
化合物总数: {summary['total_compounds']:,}
模型总数: {summary['total_models']}
单模型分析数: {summary['individual_models_analyzed']}

🎯 核心阈值参数 (可在滚动控制面板中调整)
═══════════════════════════════════════════
正例预测阈值: {summary['positive_prediction_threshold']:.3f} ⭐
最小共识模型数: {self.analyzer.current_min_consensus}
概率阈值: {self.analyzer.current_prob_threshold:.3f}
置信度阈值: {self.analyzer.current_conf_threshold:.3f}

🆕 单模型筛选功能 (v3.5新增)
═══════════════════════════════════════════
已分析模型数量: {len(self.analyzer.individual_model_results)}
当前选择模型: {self.selected_model_var.get() or '未选择'}
单模型数据标签页: ✅ 可用
模型筛选导出: ✅ 支持

🤖 加载的模型列表
═══════════════════════════════════════════
"""

        for model_type, models in self.analyzer.raw_predictions.items():
            info_text += f"\n{model_type}:\n"
            for model_name in models.keys():
                row_count = len(models[model_name])
                model_key = f"{model_type}_{model_name}"
                if model_key in self.analyzer.individual_model_results:
                    filtered_count = self.analyzer.individual_model_results[model_key]['filtered_compounds']
                    info_text += f"  • {model_name} ({row_count:,} 条预测, 筛选: {filtered_count:,})\n"
                else:
                    info_text += f"  • {model_name} ({row_count:,} 条预测)\n"

        info_text += f"""

🎯 正例预测阈值说明 (v3.5增强模型详情版)
═══════════════════════════════════════════
• 当前设置: probability_1 ≥ {summary['positive_prediction_threshold']:.3f} → 正例
• 阈值范围: 0.1 - 0.9 (在控制面板中可调节)
• 重分析功能: 修改阈值后可实时重新计算统计
• 单模型应用: 新阈值同时应用于综合分析和单模型分析

🤖 单模型筛选说明 (v3.5新功能)
═══════════════════════════════════════════
• 筛选逻辑: 每个模型独立筛选 probability_1 ≥ 阈值的化合物
• 筛选统计: 筛选数量、比例、平均概率、平均置信度
• 结果排序: 按置信度和概率降序排列
• 数据展示: 在"单模型数据"标签页中显示具体化合物

💡 使用提示 (v3.5)
═══════════════════════════════════════════
• 在单模型选择区域选择要分析的模型
• 使用"📊 单模型统计"查看详细的模型表现
• 使用"📋 模型筛选汇总"对比所有模型的筛选效果
• 使用"🎯 查看筛选结果"查看具体的筛选化合物
• 使用"🤖 单模型筛选导出"导出每个模型的筛选结果
• 支持单层批处理.py和批量预测的结果格式
"""

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info_text)

    def update_consensus_info(self, all_positive, majority_positive, high_confidence,
                              high_probability, custom_consensus):
        """更新共识分析信息"""
        info_text = f"""
🎯 共识分析结果 (增强模型详情版)
═══════════════════════════════════════════
正例预测阈值: {self.analyzer.current_positive_prediction_threshold:.3f} ⭐
分析模型数: {len(self.analyzer.available_models)}

📊 筛选结果统计 (基于自定义阈值)
───────────────────────────────────────────
所有模型都预测为正例: {len(all_positive):,} 个化合物
大多数模型预测为正例: {len(majority_positive):,} 个化合物
高置信度预测: {len(high_confidence):,} 个化合物
高概率预测: {len(high_probability):,} 个化合物
综合筛选结果: {len(custom_consensus):,} 个化合物

🥇 最高优先级化合物 (所有模型一致, 阈值: {self.analyzer.current_positive_prediction_threshold:.3f})
───────────────────────────────────────────
"""

        if all_positive:
            for i, compound in enumerate(all_positive[:10], 1):
                info_text += f"{i:2d}. {compound['compound_id']} (蛋白质: {compound['protein_id']}) "
                info_text += f"概率: {compound['avg_probability_1']:.3f}, 置信度: {compound['avg_confidence']:.3f}\n"
        else:
            info_text += "暂无符合条件的化合物\n"

        info_text += f"""
🥈 高置信度化合物
───────────────────────────────────────────
"""

        if high_confidence:
            for i, compound in enumerate(high_confidence[:10], 1):
                info_text += f"{i:2d}. {compound['compound_id']} (蛋白质: {compound['protein_id']}) "
                info_text += f"概率: {compound['avg_probability_1']:.3f}, 置信度: {compound['avg_confidence']:.3f}\n"
        else:
            info_text += "暂无符合条件的化合物\n"

        info_text += f"""
🤖 单模型筛选摘要 (v3.5新增)
───────────────────────────────────────────
"""

        # 显示前5个模型的筛选摘要
        model_summary = self.analyzer.get_individual_model_summary()
        for i, (model_key, result) in enumerate(list(model_summary.items())[:5], 1):
            info_text += f"{i}. {result['model_name']}: 筛选 {result['filtered_compounds']} 个 "
            info_text += f"(比例: {result['filtered_ratio']:.3f})\n"

        if len(model_summary) > 5:
            info_text += f"... 还有 {len(model_summary) - 5} 个模型\n"

        info_text += f"""

💡 分析建议 (增强模型详情版)
───────────────────────────────────────────
"""

        if len(all_positive) > 0:
            info_text += f"• 优先验证所有模型都预测为正例的化合物 (阈值: {self.analyzer.current_positive_prediction_threshold:.3f})\n"
        if len(high_confidence) > len(all_positive):
            info_text += "• 考虑高置信度化合物作为二线选择\n"
        if len(majority_positive) > 50:
            info_text += "• 大多数模型预测为正例的化合物数量较多，建议进一步筛选\n"

        info_text += "• 建议结合生物学知识和化合物特性进行最终筛选\n"
        info_text += "• 考虑分批进行实验验证，从最高置信度开始\n"
        info_text += f"• 当前使用自定义正例预测阈值: {self.analyzer.current_positive_prediction_threshold:.3f}\n"
        info_text += "• 使用单模型分析功能对比不同模型的筛选效果\n"
        info_text += "• 查看单模型筛选结果，识别表现突出的模型\n"
        info_text += "• 使用导出功能获取详细的筛选结果和单模型数据\n"

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info_text)

    def update_model_summary_info(self):
        """更新模型筛选汇总信息"""
        if not self.analyzer.individual_model_results:
            return

        info_text = f"""
🤖 模型筛选汇总信息 (v3.5)
═══════════════════════════════════════════
正例预测阈值: {self.analyzer.current_positive_prediction_threshold:.3f}
分析模型总数: {len(self.analyzer.individual_model_results)}

📊 筛选效果排名 (按筛选数量)
───────────────────────────────────────────
"""

        # 按筛选数量排序
        sorted_models = sorted(
            self.analyzer.individual_model_results.items(),
            key=lambda x: x[1]['filtered_compounds'],
            reverse=True
        )

        for i, (model_key, result) in enumerate(sorted_models[:10], 1):
            info_text += f"{i:2d}. {result['model_name']}\n"
            info_text += f"    筛选: {result['filtered_compounds']:,} / {result['total_compounds']:,} "
            info_text += f"({result['filtered_ratio']:.3f})\n"
            info_text += f"    平均概率: {result['avg_probability_1']:.4f}, "
            info_text += f"平均置信度: {result['avg_confidence']:.4f}\n"

        info_text += f"""

📈 筛选效果分析
───────────────────────────────────────────
"""

        # 计算汇总统计
        all_filtered = [result['filtered_compounds'] for result in self.analyzer.individual_model_results.values()]
        all_ratios = [result['filtered_ratio'] for result in self.analyzer.individual_model_results.values()]
        all_avg_probs = [result['avg_probability_1'] for result in self.analyzer.individual_model_results.values() if
                         result['filtered_compounds'] > 0]
        all_avg_confs = [result['avg_confidence'] for result in self.analyzer.individual_model_results.values() if
                         result['filtered_compounds'] > 0]

        if all_filtered:
            info_text += f"总筛选化合物数: {sum(all_filtered):,}\n"
            info_text += f"平均筛选数量: {np.mean(all_filtered):.1f}\n"
            info_text += f"平均筛选比例: {np.mean(all_ratios):.4f}\n"

            if all_avg_probs:
                info_text += f"筛选化合物平均概率: {np.mean(all_avg_probs):.4f}\n"
            if all_avg_confs:
                info_text += f"筛选化合物平均置信度: {np.mean(all_avg_confs):.4f}\n"

        info_text += f"""

🎯 模型表现分类
───────────────────────────────────────────
"""

        # 模型表现分类
        high_performers = [result for result in self.analyzer.individual_model_results.values()
                           if result['filtered_ratio'] > np.mean(all_ratios) + np.std(all_ratios)]
        low_performers = [result for result in self.analyzer.individual_model_results.values()
                          if result['filtered_ratio'] < np.mean(all_ratios) - np.std(all_ratios)]

        info_text += f"高筛选率模型 ({len(high_performers)} 个):\n"
        for result in high_performers[:5]:
            info_text += f"  • {result['model_name']} (比例: {result['filtered_ratio']:.3f})\n"

        info_text += f"\n低筛选率模型 ({len(low_performers)} 个):\n"
        for result in low_performers[:5]:
            info_text += f"  • {result['model_name']} (比例: {result['filtered_ratio']:.3f})\n"

        info_text += f"""

💡 使用建议
───────────────────────────────────────────
• 重点关注高筛选率且高置信度的模型
• 对比不同模型类型的筛选表现
• 考虑多个高表现模型的交集化合物
• 使用单模型导出功能获取详细数据
• 根据模型表现调整筛选策略
"""

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info_text)

    # ===============================
    # 导出功能
    # ===============================

    def export_simple_results(self):
        """简单结果导出"""
        if not self.check_analyzer():
            return

        directory = filedialog.askdirectory(title="选择导出目录 - 简单结果")
        if not directory:
            return

        self.status_var.set("正在导出简单筛选结果...")

        try:
            results = {
                'all_positive': self.analyzer._find_all_positive(),
                'majority_positive': self.analyzer._find_majority_positive(),
                'high_confidence': self.analyzer._find_high_confidence(),
                'high_probability': self.analyzer._find_high_probability(),
                'custom_consensus': self.analyzer._find_custom_consensus()
            }

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = os.path.join(directory, f"enhanced_model_results_{timestamp}")
            os.makedirs(export_dir, exist_ok=True)

            exported_files = []

            for result_type, compounds in results.items():
                if compounds:
                    df_data = []
                    for compound in compounds:
                        row = {
                            'protein_id': compound['protein_id'],
                            'compound_id': compound['compound_id'],
                            'total_models': compound['total_models'],
                            'positive_predictions': compound['positive_predictions'],
                            'negative_predictions': compound['negative_predictions'],
                            'positive_ratio': f"{compound['positive_ratio']:.4f}",
                            'avg_probability_0': f"{1 - compound['avg_probability_1']:.4f}",
                            'avg_probability_1': f"{compound['avg_probability_1']:.4f}",
                            'avg_confidence': f"{compound['avg_confidence']:.4f}",
                            'positive_prediction_threshold': f"{self.analyzer.current_positive_prediction_threshold:.3f}"
                        }
                        df_data.append(row)

                    df = pd.DataFrame(df_data)
                    output_file = os.path.join(export_dir, f"{result_type}_simple.csv")
                    df.to_csv(output_file, index=False, encoding='utf-8-sig')
                    exported_files.append(output_file)

            self._save_enhanced_export_parameters(export_dir, "simple")

            files_list = '\n'.join([os.path.basename(f) for f in exported_files])
            messagebox.showinfo("导出成功",
                                f"增强模型详情筛选结果已导出到:\n{export_dir}\n\n"
                                f"正例预测阈值: {self.analyzer.current_positive_prediction_threshold:.3f}\n\n"
                                f"导出文件:\n{files_list}")

            self.status_var.set("简单结果导出完成")

        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")
            self.status_var.set("导出失败")

    def export_detailed_results(self):
        """详细结果导出"""
        if not self.check_analyzer():
            return

        directory = filedialog.askdirectory(title="选择导出目录 - 详细结果")
        if not directory:
            return

        self.status_var.set("正在导出详细筛选结果...")

        try:
            results = {
                'all_positive': self.analyzer._find_all_positive(),
                'majority_positive': self.analyzer._find_majority_positive(),
                'high_confidence': self.analyzer._find_high_confidence(),
                'high_probability': self.analyzer._find_high_probability(),
                'custom_consensus': self.analyzer._find_custom_consensus()
            }

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = os.path.join(directory, f"enhanced_model_detailed_{timestamp}")
            os.makedirs(export_dir, exist_ok=True)

            exported_files = []

            for result_type, compounds in results.items():
                if compounds:
                    summary_data = []
                    detailed_data = []

                    for compound in compounds:
                        summary_row = {
                            'protein_id': compound['protein_id'],
                            'compound_id': compound['compound_id'],
                            'total_models': compound['total_models'],
                            'positive_predictions': compound['positive_predictions'],
                            'negative_predictions': compound['negative_predictions'],
                            'positive_ratio': f"{compound['positive_ratio']:.4f}",
                            'avg_probability_0': f"{1 - compound['avg_probability_1']:.4f}",
                            'avg_probability_1': f"{compound['avg_probability_1']:.4f}",
                            'avg_confidence': f"{compound['avg_confidence']:.4f}",
                            'positive_prediction_threshold': f"{self.analyzer.current_positive_prediction_threshold:.3f}"
                        }
                        summary_data.append(summary_row)

                        for pred in compound['predictions']:
                            detailed_row = {
                                'protein_id': compound['protein_id'],
                                'compound_id': compound['compound_id'],
                                'model_type': pred['model_type'],
                                'model_name': pred['model_name'],
                                'original_prediction': pred['original_prediction'],
                                'custom_prediction': pred['custom_prediction'],
                                'prediction_label': '相互作用' if pred['custom_prediction'] == 1 else '无相互作用',
                                'probability_0': f"{pred['probability_0']:.4f}",
                                'probability_1': f"{pred['probability_1']:.4f}",
                                'confidence': f"{pred['confidence']:.4f}",
                                'positive_prediction_threshold': f"{self.analyzer.current_positive_prediction_threshold:.3f}",
                                'avg_probability_1': f"{compound['avg_probability_1']:.4f}",
                                'avg_confidence': f"{compound['avg_confidence']:.4f}",
                                'positive_ratio': f"{compound['positive_ratio']:.4f}"
                            }
                            detailed_data.append(detailed_row)

                    summary_df = pd.DataFrame(summary_data)
                    summary_file = os.path.join(export_dir, f"{result_type}_summary.csv")
                    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
                    exported_files.append(summary_file)

                    detailed_df = pd.DataFrame(detailed_data)
                    detailed_file = os.path.join(export_dir, f"{result_type}_detailed.csv")
                    detailed_df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
                    exported_files.append(detailed_file)

            self._save_enhanced_export_parameters(export_dir, "detailed")

            files_list = '\n'.join([os.path.basename(f) for f in exported_files])
            messagebox.showinfo("导出成功",
                                f"增强模型详情详细结果已导出到:\n{export_dir}\n\n"
                                f"正例预测阈值: {self.analyzer.current_positive_prediction_threshold:.3f}\n\n"
                                f"导出文件:\n{files_list}\n\n"
                                f"每个筛选类型包含两个文件:\n"
                                f"• *_summary.csv: 基础摘要数据\n"
                                f"• *_detailed.csv: 每个模型的详细预测\n"
                                f"包含原始预测和基于自定义阈值的预测")

            self.status_var.set("详细结果导出完成")

        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")
            self.status_var.set("导出失败")

    def export_individual_model_results(self):
        """新增：导出单模型筛选结果"""
        if not self.check_analyzer():
            return

        if not self.analyzer.individual_model_results:
            messagebox.showwarning("警告", "无单模型筛选数据可导出")
            return

        directory = filedialog.askdirectory(title="选择导出目录 - 单模型筛选结果")
        if not directory:
            return

        self.status_var.set("正在导出单模型筛选结果...")

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = os.path.join(directory, f"individual_model_results_{timestamp}")
            os.makedirs(export_dir, exist_ok=True)

            exported_files = []

            # 导出模型筛选汇总
            summary_data = []
            for model_key, result in self.analyzer.individual_model_results.items():
                summary_row = {
                    'model_type': result['model_type'],
                    'model_name': result['model_name'],
                    'total_compounds': result['total_compounds'],
                    'filtered_compounds': result['filtered_compounds'],
                    'filtered_ratio': f"{result['filtered_ratio']:.4f}",
                    'avg_probability_1': f"{result['avg_probability_1']:.4f}",
                    'avg_confidence': f"{result['avg_confidence']:.4f}",
                    'max_probability_1': f"{result['max_probability_1']:.4f}",
                    'min_probability_1': f"{result['min_probability_1']:.4f}",
                    'threshold_used': f"{result['threshold_used']:.3f}"
                }
                summary_data.append(summary_row)

            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(export_dir, "model_filtering_summary.csv")
            summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
            exported_files.append(summary_file)

            # 为每个模型导出详细的筛选化合物
            model_details_dir = os.path.join(export_dir, "individual_models")
            os.makedirs(model_details_dir, exist_ok=True)

            for model_key, result in self.analyzer.individual_model_results.items():
                if result['filtered_data']:
                    model_df_data = []
                    for item in result['filtered_data']:
                        row = {
                            'protein_id': item['protein_id'],
                            'compound_id': item['compound_id'],
                            'original_prediction': item['prediction'],
                            'custom_prediction': item['custom_prediction'],
                            'prediction_label': '相互作用' if item['custom_prediction'] == 1 else '无相互作用',
                            'probability_0': f"{item['probability_0']:.4f}",
                            'probability_1': f"{item['probability_1']:.4f}",
                            'confidence': f"{item['confidence']:.4f}",
                            'threshold_used': f"{result['threshold_used']:.3f}",
                            'model_type': result['model_type'],
                            'model_name': result['model_name']
                        }
                        model_df_data.append(row)

                    model_df = pd.DataFrame(model_df_data)
                    # 安全的文件名处理
                    safe_model_name = "".join(
                        c for c in result['model_name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_model_name = safe_model_name.replace(' ', '_')
                    model_file = os.path.join(model_details_dir, f"{safe_model_name}_filtered_compounds.csv")
                    model_df.to_csv(model_file, index=False, encoding='utf-8-sig')
                    exported_files.append(model_file)

            # 导出Top化合物汇总（所有模型的Top10化合物）
            top_compounds_data = []
            for model_key, result in self.analyzer.individual_model_results.items():
                top_compounds = result['filtered_data'][:10]  # Top 10
                for i, item in enumerate(top_compounds, 1):
                    row = {
                        'model_type': result['model_type'],
                        'model_name': result['model_name'],
                        'rank': i,
                        'protein_id': item['protein_id'],
                        'compound_id': item['compound_id'],
                        'probability_1': f"{item['probability_1']:.4f}",
                        'confidence': f"{item['confidence']:.4f}",
                        'threshold_used': f"{result['threshold_used']:.3f}"
                    }
                    top_compounds_data.append(row)

            if top_compounds_data:
                top_df = pd.DataFrame(top_compounds_data)
                top_file = os.path.join(export_dir, "all_models_top10_compounds.csv")
                top_df.to_csv(top_file, index=False, encoding='utf-8-sig')
                exported_files.append(top_file)

            self._save_enhanced_export_parameters(export_dir, "individual_model")

            # 创建README文件
            readme_file = os.path.join(export_dir, "README.txt")
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write("单模型筛选结果导出说明\n")
                f.write("=" * 40 + "\n")
                f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据来源: {self.analyzer.result_dir}\n")
                f.write(f"正例预测阈值: {self.analyzer.current_positive_prediction_threshold:.3f}\n")
                f.write(f"用户: woyaokaoyanhaha\n\n")

                f.write("文件结构:\n")
                f.write("- model_filtering_summary.csv: 所有模型的筛选汇总统计\n")
                f.write("- all_models_top10_compounds.csv: 所有模型的Top10化合物汇总\n")
                f.write("- individual_models/: 每个模型的详细筛选化合物文件\n")
                f.write("- export_parameters.json: 导出时的参数设置\n\n")

                f.write("数据说明:\n")
                f.write("- filtered_compounds: 基于正例预测阈值筛选出的化合物数量\n")
                f.write("- filtered_ratio: 筛选化合物占总化合物的比例\n")
                f.write("- custom_prediction: 基于自定义阈值的预测结果 (1=正例, 0=负例)\n")
                f.write("- 所有化合物按置信度和概率降序排列\n")

            files_count = len(exported_files)
            model_count = len([f for f in exported_files if 'individual_models' in f])

            messagebox.showinfo("导出成功",
                                f"单模型筛选结果已导出到:\n{export_dir}\n\n"
                                f"导出统计:\n"
                                f"• 分析模型数量: {len(self.analyzer.individual_model_results)}\n"
                                f"• 导出文件数量: {files_count}\n"
                                f"• 单模型详细文件: {model_count}\n"
                                f"• 正例预测阈值: {self.analyzer.current_positive_prediction_threshold:.3f}\n\n"
                                f"包含:\n"
                                f"• 模型筛选汇总统计\n"
                                f"• 每个模型的详细筛选化合物\n"
                                f"• 所有模型Top10化合物汇总\n"
                                f"• 详细的使用说明")

            self.status_var.set("单模型结果导出完成")

        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")
            self.status_var.set("导出失败")

    def generate_enhanced_report(self):
        """生成增强模型详情分析报告"""
        if not self.check_analyzer():
            return

        file_path = filedialog.asksaveasfilename(
            title="保存增强模型详情分析报告",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if not file_path:
            return

        self.status_var.set("正在生成增强模型详情报告...")

        try:
            all_positive = self.analyzer._find_all_positive()
            majority_positive = self.analyzer._find_majority_positive()
            high_confidence = self.analyzer._find_high_confidence()
            high_probability = self.analyzer._find_high_probability()
            custom_consensus = self.analyzer._find_custom_consensus()

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("增强模型详情预测结果分析报告 v3.5\n")
                f.write("=" * 80 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"分析用户: woyaokaoyanhaha\n")
                f.write(f"数据来源: {self.analyzer.result_dir}\n")
                f.write(f"检测格式: {self.analyzer.get_summary_info()['format']}\n")
                f.write(f"分析工具: 增强模型详情预测结果分析器 GUI v3.5\n\n")

                f.write("新增功能 (v3.5)\n")
                f.write("-" * 40 + "\n")
                f.write("🆕 单模型筛选分析：查看每个模型筛选出的大于阈值的化合物\n")
                f.write("🆕 模型筛选汇总：对比所有模型的筛选效果\n")
                f.write("🆕 模型筛选详情：显示具体的筛选化合物列表\n")
                f.write("🆕 单模型导出：导出每个模型的筛选结果\n\n")

                f.write("核心参数设置\n")
                f.write("-" * 40 + "\n")
                f.write(f"🎯 正例预测阈值: {self.analyzer.current_positive_prediction_threshold:.3f} (核心参数)\n")
                f.write(
                    f"   说明: probability_1 ≥ {self.analyzer.current_positive_prediction_threshold:.3f} 时认为预测为正例\n")
                f.write(f"最小共识模型数: {self.analyzer.current_min_consensus}\n")
                f.write(f"概率阈值: {self.analyzer.current_prob_threshold:.3f}\n")
                f.write(f"置信度阈值: {self.analyzer.current_conf_threshold:.3f}\n\n")

                f.write("数据概况\n")
                f.write("-" * 40 + "\n")
                f.write(f"化合物总数: {len(self.analyzer.compound_stats):,}\n")
                f.write(f"模型总数: {len(self.analyzer.available_models)}\n")
                f.write(f"单模型分析数: {len(self.analyzer.individual_model_results)}\n")
                f.write(f"数据标准化: ✅ 成功\n")
                f.write(f"自定义阈值分析: ✅ 启用\n")
                f.write(f"单模型筛选功能: ✅ 启用\n\n")

                f.write("基于自定义阈值的筛选结果统计\n")
                f.write("-" * 40 + "\n")
                f.write(f"所有模型都预测为正例: {len(all_positive):,} 个化合物\n")
                f.write(f"大多数模型预测为正例: {len(majority_positive):,} 个化合物\n")
                f.write(f"高置信度预测: {len(high_confidence):,} 个化合物\n")
                f.write(f"高概率预测: {len(high_probability):,} 个化合物\n")
                f.write(f"综合筛选结果: {len(custom_consensus):,} 个化合物\n\n")

                # 单模型筛选结果分析
                if self.analyzer.individual_model_results:
                    f.write("单模型筛选结果分析 (v3.5新增)\n")
                    f.write("-" * 40 + "\n")

                    # 按筛选数量排序
                    sorted_models = sorted(
                        self.analyzer.individual_model_results.items(),
                        key=lambda x: x[1]['filtered_compounds'],
                        reverse=True
                    )

                    f.write(f"分析模型数量: {len(self.analyzer.individual_model_results)}\n")
                    f.write(f"使用阈值: {self.analyzer.current_positive_prediction_threshold:.3f}\n\n")

                    f.write("模型筛选排名 (按筛选数量):\n")
                    for i, (model_key, result) in enumerate(sorted_models[:10], 1):
                        f.write(f"  {i:2d}. {result['model_name']}\n")
                        f.write(f"      筛选: {result['filtered_compounds']:,} / {result['total_compounds']:,} ")
                        f.write(f"({result['filtered_ratio']:.3f})\n")
                        f.write(f"      平均概率: {result['avg_probability_1']:.4f}, ")
                        f.write(f"平均置信度: {result['avg_confidence']:.4f}\n")

                    # 筛选效果统计
                    all_filtered = [result['filtered_compounds'] for result in
                                    self.analyzer.individual_model_results.values()]
                    all_ratios = [result['filtered_ratio'] for result in
                                  self.analyzer.individual_model_results.values()]

                    if all_filtered:
                        f.write(f"\n筛选效果统计:\n")
                        f.write(f"  总筛选化合物数: {sum(all_filtered):,}\n")
                        f.write(f"  平均筛选数量: {np.mean(all_filtered):.1f}\n")
                        f.write(f"  平均筛选比例: {np.mean(all_ratios):.4f}\n")
                        f.write(f"  筛选比例标准差: {np.std(all_ratios):.4f}\n")
                        f.write(f"  最高筛选比例: {max(all_ratios):.4f}\n")
                        f.write(f"  最低筛选比例: {min(all_ratios):.4f}\n\n")

                if all_positive:
                    f.write(f"重点化合物推荐 (阈值: {self.analyzer.current_positive_prediction_threshold:.3f})\n")
                    f.write("-" * 40 + "\n")
                    f.write("🥇 最高优先级化合物 (所有模型都预测为正例):\n")
                    for i, compound in enumerate(all_positive[:20], 1):
                        f.write(f"  {i:2d}. {compound['compound_id']} (蛋白质: {compound['protein_id']}) - "
                                f"概率: {compound['avg_probability_1']:.3f}, 置信度: {compound['avg_confidence']:.3f}\n")
                    f.write("\n")

                # 统计分析
                all_probs = [stats['avg_probability_1'] for stats in self.analyzer.compound_stats.values()]
                all_confs = [stats['avg_confidence'] for stats in self.analyzer.compound_stats.values()]

                f.write("统计分析\n")
                f.write("-" * 40 + "\n")
                f.write(f"概率分布 (相对于正例预测阈值 {self.analyzer.current_positive_prediction_threshold:.3f}):\n")
                f.write(f"  平均值: {np.mean(all_probs):.4f}\n")
                f.write(f"  中位数: {np.median(all_probs):.4f}\n")
                f.write(f"  标准差: {np.std(all_probs):.4f}\n")
                f.write(f"  最小值: {np.min(all_probs):.4f}\n")
                f.write(f"  最大值: {np.max(all_probs):.4f}\n")
                f.write(
                    f"  ≥ 正例预测阈值的化合物: {sum(1 for p in all_probs if p >= self.analyzer.current_positive_prediction_threshold)} 个\n\n")

                f.write(f"置信度分布:\n")
                f.write(f"  平均值: {np.mean(all_confs):.4f}\n")
                f.write(f"  中位数: {np.median(all_confs):.4f}\n")
                f.write(f"  标准差: {np.std(all_confs):.4f}\n")
                f.write(f"  最小值: {np.min(all_confs):.4f}\n")
                f.write(f"  最大值: {np.max(all_confs):.4f}\n\n")

                # 模型详细信息
                f.write("模型详细信息\n")
                f.write("-" * 40 + "\n")
                for model_type, models in self.analyzer.raw_predictions.items():
                    f.write(f"{model_type}:\n")
                    for model_name, df in models.items():
                        total_count = len(df)
                        model_key = f"{model_type}_{model_name}"
                        if model_key in self.analyzer.individual_model_results:
                            filtered_count = self.analyzer.individual_model_results[model_key]['filtered_compounds']
                            filtered_ratio = self.analyzer.individual_model_results[model_key]['filtered_ratio']
                            f.write(
                                f"  • {model_name}: {total_count:,} 预测, 筛选: {filtered_count:,} ({filtered_ratio:.3f})\n")
                        else:
                            f.write(f"  • {model_name}: {total_count:,} 预测\n")
                    f.write("\n")

                f.write("增强模型详情版分析建议 (v3.5)\n")
                f.write("-" * 40 + "\n")
                f.write(f"1. 当前使用的正例预测阈值: {self.analyzer.current_positive_prediction_threshold:.3f}\n")
                f.write("2. 优先验证所有模型都预测为正例的化合物，成功率最高\n")
                f.write("3. 关注高筛选率且高置信度的单个模型结果\n")
                f.write("4. 对比不同模型类型的筛选表现，识别最佳模型\n")
                f.write("5. 考虑多个高表现模型的交集化合物\n")
                f.write("6. 使用单模型导出功能获取详细的筛选数据\n")
                f.write("7. 根据单模型表现调整筛选策略和阈值设置\n")
                f.write("8. 结合生物学知识验证单模型筛选结果\n")

                f.write("\n单模型筛选功能说明 (v3.5新增)\n")
                f.write("-" * 40 + "\n")
                f.write("• 筛选逻辑: 每个模型独立筛选 probability_1 ≥ 阈值的化合物\n")
                f.write("• 筛选统计: 提供筛选数量、比例、平均概率、平均置信度\n")
                f.write("• 结果排序: 按置信度和概率降序排列\n")
                f.write("• 模型对比: 横向对比所有模型的筛选效果\n")
                f.write("• 数据导出: 支持单模型筛选结果的专门导出\n")
                f.write("• 实时更新: 阈值修改后单模型结果同步更新\n")

                f.write("\n使用场景建议\n")
                f.write("-" * 40 + "\n")
                f.write("• 模型性能评估: 比较不同模型的筛选效果和一致性\n")
                f.write("• 化合物优先级: 根据单模型和综合结果确定化合物优先级\n")
                f.write("• 阈值优化: 通过单模型分析优化正例预测阈值设置\n")
                f.write("• 结果验证: 交叉验证不同模型的预测一致性和可靠性\n")
                f.write("• 模型选择: 识别表现最佳的模型类型和配置\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("增强模型详情分析报告结束\n")
                f.write("=" * 80 + "\n")

            messagebox.showinfo("成功", f"增强模型详情分析报告已保存到:\n{file_path}")
            self.status_var.set("报告生成完成")

        except Exception as e:
            messagebox.showerror("错误", f"生成报告失败: {e}")
            self.status_var.set("报告生成失败")

    def save_current_plot(self):
        """保存当前图表"""
        if not self.figure:
            messagebox.showwarning("警告", "没有可保存的图表")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存图表",
            defaultextension=".png",
            filetypes=[
                ("PNG图片", "*.png"),
                ("JPEG图片", "*.jpg"),
                ("PDF文件", "*.pdf"),
                ("SVG矢量图", "*.svg"),
                ("所有文件", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("成功", f"图表已保存到:\n{file_path}")
            self.status_var.set("图表保存成功")

        except Exception as e:
            messagebox.showerror("错误", f"保存图表失败: {e}")
            self.status_var.set("保存失败")

    def _save_enhanced_export_parameters(self, export_dir, export_type):
        """保存增强版导出参数"""
        params = {
            'export_type': export_type,
            'positive_prediction_threshold': self.analyzer.current_positive_prediction_threshold,
            'min_consensus_models': self.analyzer.current_min_consensus,
            'probability_threshold': self.analyzer.current_prob_threshold,
            'confidence_threshold': self.analyzer.current_conf_threshold,
            'export_time': datetime.now().isoformat(),
            'total_compounds_analyzed': len(self.analyzer.compound_stats),
            'total_models': len(self.analyzer.available_models),
            'individual_models_analyzed': len(self.analyzer.individual_model_results),
            'data_source': self.analyzer.result_dir,
            'user': 'woyaokaoyanhaha',
            'version': '3.5.0_enhanced_model_details',
            'new_features': {
                'individual_model_filtering': True,
                'model_comparison': True,
                'model_specific_export': True,
                'threshold_based_filtering': True
            }
        }

        params_file = os.path.join(export_dir, "export_parameters.json")
        if safe_json_dump(params, params_file, indent=2, ensure_ascii=False):
            print(f"增强模型详情版导出参数已保存: {params_file}")


def main():
    """主函数"""
    # 创建主窗口
    root = tk.Tk()

    # 创建应用
    app = EnhancedModelDetailsAnalyzerGUI(root)

    # 设置窗口关闭事件
    def on_closing():
        if messagebox.askokcancel("退出", "确定要退出增强模型详情预测结果分析器吗？"):
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # 运行应用
    try:
        print("🎯 启动增强模型详情预测结果分析器 GUI v3.5...")
        print("🆕 新增功能: 单模型筛选分析")
        print("✨ 核心功能: 支持自定义正例预测阈值")
        print("⚡ 性能优化: 快速加载 + 智能重分析")
        print("🖱️ 界面优化: 可滚动控制面板 + 鼠标滚轮支持")
        print("🤖 单模型功能: 筛选分析 + 模型对比 + 专门导出")
        print("📊 完整导出: 简单/详细/单模型多种导出选项")
        print("✅ 中文字体已配置")
        print("✅ 界面组件已加载")
        print("✅ 可滚动功能已启用")
        print("✅ 增强阈值功能已启用")
        print("✅ 单模型筛选功能已启用")
        print("✅ 所有功能都可正常访问")
        print("✅ 准备就绪！")

        root.mainloop()

    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()