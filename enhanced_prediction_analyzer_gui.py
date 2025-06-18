# =============================================================================
# 增强版预测结果分析器 GUI - 支持直接输入和详细导出
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
from datetime import datetime
import threading
from collections import defaultdict, Counter
import warnings
import platform

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
        
        # 尝试不同的中文字体
        chinese_fonts = []
        
        if system == "Windows":
            chinese_fonts = [
                'Microsoft YaHei',
                'SimHei', 
                'KaiTi',
                'SimSun',
                'FangSong'
            ]
        elif system == "Darwin":  # macOS
            chinese_fonts = [
                'PingFang SC',
                'Heiti SC',
                'STSong',
                'Arial Unicode MS'
            ]
        else:  # Linux
            chinese_fonts = [
                'WenQuanYi Micro Hei',
                'WenQuanYi Zen Hei',
                'Noto Sans CJK SC',
                'Source Han Sans CN'
            ]
        
        # 查找可用的中文字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 已配置中文字体: {font}")
                return font
        
        # 如果没有找到中文字体，使用默认设置
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        print("⚠️  使用默认字体设置")
        return "DejaVu Sans"
    
    except Exception as e:
        print(f"字体配置出错: {e}")
        return "DejaVu Sans"

# 配置字体
configure_chinese_fonts()

# =============================================================================
# 核心分析器类
# =============================================================================

class PredictionAnalyzer:
    """内置的预测结果分析器"""
    
    def __init__(self):
        self.raw_predictions = {}
        self.model_summary = {}
        self.available_models = []
        self.compound_stats = {}
        self.result_dir = None
        
        # 当前分析参数
        self.current_min_consensus = 2
        self.current_prob_threshold = 0.6
        self.current_conf_threshold = 0.7
    
    def find_latest_result_dir(self, base_dir="prediction_results_batch"):
        """查找最新的预测结果目录"""
        if not os.path.exists(base_dir):
            return None
        
        # 查找所有batch_prediction_开头的目录
        result_dirs = [d for d in os.listdir(base_dir) 
                      if d.startswith('batch_prediction_') and 
                      os.path.isdir(os.path.join(base_dir, d))]
        
        if not result_dirs:
            return None
        
        # 按时间戳排序，获取最新的
        result_dirs.sort(reverse=True)
        latest_dir = os.path.join(base_dir, result_dirs[0])
        return latest_dir
    
    def load_prediction_results(self, result_dir=None):
        """加载预测结果"""
        if not result_dir:
            result_dir = self.find_latest_result_dir()
            if not result_dir:
                return False
        
        if not os.path.exists(result_dir):
            return False
        
        self.result_dir = result_dir
        
        # 加载模型摘要
        summary_file = os.path.join(result_dir, "prediction_summary.json")
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    self.model_summary = json.load(f)
            except:
                pass
        
        # 加载原始预测结果
        self._load_individual_predictions()
        
        # 分析化合物统计
        self._analyze_compound_statistics()
        
        return len(self.compound_stats) > 0
    
    def _load_individual_predictions(self):
        """加载各个模型的预测结果"""
        self.raw_predictions = {}
        self.available_models = []
        
        # 如果有模型摘要，使用摘要信息
        if self.model_summary:
            for model_type, models in self.model_summary.items():
                self.raw_predictions[model_type] = {}
                
                model_type_dir = os.path.join(self.result_dir, model_type.replace('/', '_'))
                if not os.path.exists(model_type_dir):
                    continue
                
                for model_name in models.keys():
                    prediction_file = os.path.join(model_type_dir, f"{model_name}_prediction.csv")
                    
                    if os.path.exists(prediction_file):
                        try:
                            df = pd.read_csv(prediction_file)
                            self.raw_predictions[model_type][model_name] = df
                            self.available_models.append(f"{model_type}_{model_name}")
                        except Exception as e:
                            print(f"加载失败 {model_type}/{model_name}: {e}")
        else:
            # 如果没有摘要，尝试自动发现
            self._auto_discover_predictions()
    
    def _auto_discover_predictions(self):
        """自动发现预测结果文件"""
        for item in os.listdir(self.result_dir):
            item_path = os.path.join(self.result_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                model_type = item
                self.raw_predictions[model_type] = {}
                
                for file in os.listdir(item_path):
                    if file.endswith('_prediction.csv'):
                        model_name = file.replace('_prediction.csv', '')
                        file_path = os.path.join(item_path, file)
                        
                        try:
                            df = pd.read_csv(file_path)
                            self.raw_predictions[model_type][model_name] = df
                            self.available_models.append(f"{model_type}_{model_name}")
                        except Exception as e:
                            print(f"加载失败 {model_type}/{model_name}: {e}")
    
    def _analyze_compound_statistics(self):
        """分析化合物统计信息"""
        self.compound_stats = {}
        
        # 按化合物聚合所有预测
        compound_predictions = defaultdict(list)
        
        for model_type, models in self.raw_predictions.items():
            for model_name, df in models.items():
                for _, row in df.iterrows():
                    key = f"{row['protein_id']}_{row['compound_id']}"
                    compound_predictions[key].append({
                        'model_type': model_type,
                        'model_name': model_name,
                        'protein_id': row['protein_id'],
                        'compound_id': row['compound_id'],
                        'prediction': row['prediction'],
                        'probability_0': row.get('probability_0', 0.5),
                        'probability_1': row.get('probability_1', 0.5),
                        'confidence': row.get('confidence', 0.5)
                    })
        
        # 计算每个化合物的统计信息
        for compound_key, predictions in compound_predictions.items():
            protein_id, compound_id = compound_key.split('_', 1)
            
            total_models = len(predictions)
            positive_predictions = sum(1 for p in predictions if p['prediction'] == 1)
            negative_predictions = total_models - positive_predictions
            
            avg_prob_1 = np.mean([p['probability_1'] for p in predictions])
            avg_confidence = np.mean([p['confidence'] for p in predictions])
            
            self.compound_stats[compound_key] = {
                'protein_id': protein_id,
                'compound_id': compound_id,
                'total_models': total_models,
                'positive_predictions': positive_predictions,
                'negative_predictions': negative_predictions,
                'positive_ratio': positive_predictions / total_models,
                'avg_probability_1': avg_prob_1,
                'avg_confidence': avg_confidence,
                'predictions': predictions
            }
    
    def _find_all_positive(self):
        """找到所有模型都预测为正例的化合物"""
        all_positive = []
        
        for compound_key, stats in self.compound_stats.items():
            if (stats['total_models'] >= self.current_min_consensus and 
                stats['positive_predictions'] == stats['total_models']):
                all_positive.append(stats)
        
        return sorted(all_positive, key=lambda x: x['avg_confidence'], reverse=True)
    
    def _find_majority_positive(self):
        """找到大多数模型预测为正例的化合物"""
        majority_positive = []
        
        for compound_key, stats in self.compound_stats.items():
            if (stats['total_models'] >= self.current_min_consensus and 
                stats['positive_ratio'] > 0.5 and
                stats['positive_predictions'] >= self.current_min_consensus):
                majority_positive.append(stats)
        
        return sorted(majority_positive, key=lambda x: x['avg_confidence'], reverse=True)
    
    def _find_high_confidence(self):
        """找到高置信度的化合物"""
        high_confidence = []
        
        for compound_key, stats in self.compound_stats.items():
            if (stats['total_models'] >= self.current_min_consensus and 
                stats['avg_confidence'] >= self.current_conf_threshold and
                stats['positive_predictions'] >= self.current_min_consensus):
                high_confidence.append(stats)
        
        return sorted(high_confidence, key=lambda x: x['avg_confidence'], reverse=True)
    
    def _find_high_probability(self):
        """找到高概率的化合物"""
        high_probability = []
        
        for compound_key, stats in self.compound_stats.items():
            if (stats['total_models'] >= self.current_min_consensus and 
                stats['avg_probability_1'] >= self.current_prob_threshold and
                stats['positive_predictions'] >= self.current_min_consensus):
                high_probability.append(stats)
        
        return sorted(high_probability, key=lambda x: x['avg_probability_1'], reverse=True)
    
    def _find_custom_consensus(self):
        """自定义共识分析"""
        custom_consensus = []
        
        for compound_key, stats in self.compound_stats.items():
            if (stats['total_models'] >= self.current_min_consensus and 
                stats['positive_predictions'] >= self.current_min_consensus and
                stats['avg_confidence'] >= self.current_conf_threshold and
                stats['avg_probability_1'] >= self.current_prob_threshold):
                custom_consensus.append(stats)
        
        return sorted(custom_consensus, key=lambda x: (x['avg_confidence'] + x['avg_probability_1'])/2, reverse=True)

# =============================================================================
# 增强版GUI主类
# =============================================================================

class EnhancedPredictionAnalyzerGUI:
    """增强版预测结果分析器GUI"""
    
    def __init__(self, root):
        self.root = root
        self.analyzer = PredictionAnalyzer()
        self.current_figure = None
        
        # 初始化状态变量（必须在create_widgets之前）
        self.status_var = tk.StringVar(value="准备就绪")
        self.progress_var = tk.DoubleVar()
        self.data_info_var = tk.StringVar(value="未加载数据")
        self.result_dir_var = tk.StringVar()
        
        # 初始化参数变量
        self.min_consensus_var = tk.IntVar(value=2)
        self.prob_threshold_var = tk.DoubleVar(value=0.6)
        self.conf_threshold_var = tk.DoubleVar(value=0.7)
        
        # 新增：直接输入的变量
        self.prob_entry_var = tk.StringVar(value="0.60")
        self.conf_entry_var = tk.StringVar(value="0.70")
        
        # 配置主窗口
        self.setup_main_window()
        
        # 配置样式
        self.setup_styles()
        
        # 创建界面
        self.create_widgets()
    
    def setup_main_window(self):
        """配置主窗口"""
        self.root.title("🔬 预测结果分析器 v2.1 - 增强版")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # 配置网格权重
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def setup_styles(self):
        """配置样式"""
        style = ttk.Style()
        
        # 配置现代化主题
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
        
        # 自定义样式
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Warning.TLabel', foreground='orange')
    
    def create_widgets(self):
        """创建主界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # 顶部标题栏
        self.create_header(main_frame)
        
        # 创建左侧控制面板和右侧显示区域
        self.create_main_content(main_frame)
        
        # 底部状态栏
        self.create_status_bar(main_frame)
    
    def create_header(self, parent):
        """创建顶部标题栏"""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.grid_columnconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(header_frame, text="🔬 预测结果分析器 v2.1", style='Title.TLabel')
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # 用户信息
        user_info = f"用户: woyaokaoyanhaha | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        user_label = ttk.Label(header_frame, text=user_info)
        user_label.grid(row=0, column=1, sticky=tk.E)
        
        # 分隔线
        separator = ttk.Separator(header_frame, orient='horizontal')
        separator.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def create_main_content(self, parent):
        """创建主要内容区域"""
        # 左侧控制面板
        self.create_control_panel(parent)
        
        # 右侧显示区域
        self.create_display_area(parent)
    
    def create_control_panel(self, parent):
        """创建左侧控制面板"""
        # 控制面板框架
        control_frame = ttk.LabelFrame(parent, text="📊 控制面板", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_frame.grid_columnconfigure(0, weight=1)
        
        # 文件加载区域
        self.create_file_section(control_frame)
        
        # 增强版参数设置区域
        self.create_enhanced_parameter_section(control_frame)
        
        # 分析功能区域
        self.create_analysis_section(control_frame)
        
        # 增强版导出功能区域
        self.create_enhanced_export_section(control_frame)
    
    def create_file_section(self, parent):
        """创建文件加载区域"""
        file_frame = ttk.LabelFrame(parent, text="📁 数据加载")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.grid_columnconfigure(1, weight=1)
        
        # 结果目录选择
        ttk.Label(file_frame, text="预测结果目录:").grid(row=0, column=0, sticky=tk.W, padx=(5, 5))
        
        result_dir_entry = ttk.Entry(file_frame, textvariable=self.result_dir_var, width=30)
        result_dir_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        browse_btn = ttk.Button(file_frame, text="浏览", command=self.browse_result_dir)
        browse_btn.grid(row=0, column=2, padx=(0, 5))
        
        # 快速加载按钮
        quick_load_btn = ttk.Button(file_frame, text="🔍 自动查找", command=self.auto_load_latest)
        quick_load_btn.grid(row=1, column=0, pady=(5, 0), sticky=(tk.W, tk.E))
        
        # 加载按钮
        load_btn = ttk.Button(file_frame, text="🔄 加载数据", command=self.load_data)
        load_btn.grid(row=1, column=1, columnspan=2, pady=(5, 0), sticky=(tk.W, tk.E))
        
        # 数据信息显示
        info_label = ttk.Label(file_frame, textvariable=self.data_info_var, style='Success.TLabel')
        info_label.grid(row=2, column=0, columnspan=3, pady=(5, 0))
    
    def create_enhanced_parameter_section(self, parent):
        """创建增强版参数设置区域（支持滑块和直接输入）"""
        param_frame = ttk.LabelFrame(parent, text="⚙️ 筛选参数（滑块+直接输入）")
        param_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        param_frame.grid_columnconfigure(2, weight=1)
        
        # 最小共识模型数
        ttk.Label(param_frame, text="最小共识模型数:").grid(row=0, column=0, sticky=tk.W, padx=(5, 5))
        consensus_spin = ttk.Spinbox(param_frame, from_=1, to=10, textvariable=self.min_consensus_var, width=10)
        consensus_spin.grid(row=0, column=1, sticky=tk.W, padx=(0, 5))
        
        # 概率阈值 - 滑块+输入框组合
        ttk.Label(param_frame, text="概率阈值:").grid(row=1, column=0, sticky=tk.W, padx=(5, 5))
        
        # 概率滑块
        prob_scale = ttk.Scale(param_frame, from_=0.0, to=1.0, variable=self.prob_threshold_var, 
                              orient=tk.HORIZONTAL, length=120)
        prob_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # 概率直接输入框
        prob_entry = ttk.Entry(param_frame, textvariable=self.prob_entry_var, width=8)
        prob_entry.grid(row=1, column=2, padx=(5, 5))
        
        # 概率同步按钮
        prob_sync_btn = ttk.Button(param_frame, text="↔", width=3, 
                                  command=self.sync_prob_from_entry)
        prob_sync_btn.grid(row=1, column=3, padx=(0, 5))
        
        # 置信度阈值 - 滑块+输入框组合
        ttk.Label(param_frame, text="置信度阈值:").grid(row=2, column=0, sticky=tk.W, padx=(5, 5))
        
        # 置信度滑块
        conf_scale = ttk.Scale(param_frame, from_=0.0, to=1.0, variable=self.conf_threshold_var, 
                              orient=tk.HORIZONTAL, length=120)
        conf_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # 置信度直接输入框
        conf_entry = ttk.Entry(param_frame, textvariable=self.conf_entry_var, width=8)
        conf_entry.grid(row=2, column=2, padx=(5, 5))
        
        # 置信度同步按钮
        conf_sync_btn = ttk.Button(param_frame, text="↔", width=3, 
                                  command=self.sync_conf_from_entry)
        conf_sync_btn.grid(row=2, column=3, padx=(0, 5))
        
        # 绑定滑块更新事件（滑块→输入框）
        def update_prob_entry(*args):
            self.prob_entry_var.set(f"{self.prob_threshold_var.get():.3f}")
        self.prob_threshold_var.trace('w', update_prob_entry)
        
        def update_conf_entry(*args):
            self.conf_entry_var.set(f"{self.conf_threshold_var.get():.3f}")
        self.conf_threshold_var.trace('w', update_conf_entry)
        
        # 绑定输入框回车事件（输入框→滑块）
        prob_entry.bind('<Return>', lambda e: self.sync_prob_from_entry())
        conf_entry.bind('<Return>', lambda e: self.sync_conf_from_entry())
        
        # 快速设置按钮
        quick_frame = ttk.Frame(param_frame)
        quick_frame.grid(row=3, column=0, columnspan=4, pady=(10, 0), sticky=(tk.W, tk.E))
        
        ttk.Label(quick_frame, text="快速设置:").pack(side=tk.LEFT, padx=(0, 5))
        
        quick_buttons = [
            ("严格 (0.8/0.9)", lambda: self.set_quick_params(0.8, 0.9)),
            ("中等 (0.6/0.7)", lambda: self.set_quick_params(0.6, 0.7)),
            ("宽松 (0.5/0.6)", lambda: self.set_quick_params(0.5, 0.6))
        ]
        
        for text, command in quick_buttons:
            btn = ttk.Button(quick_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=2)
        
        # 应用参数按钮
        apply_btn = ttk.Button(param_frame, text="✅ 应用参数", command=self.apply_parameters)
        apply_btn.grid(row=4, column=0, columnspan=4, pady=(10, 5), sticky=(tk.W, tk.E))
    
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
    
    def set_quick_params(self, prob, conf):
        """快速设置参数"""
        self.prob_threshold_var.set(prob)
        self.conf_threshold_var.set(conf)
        self.prob_entry_var.set(f"{prob:.3f}")
        self.conf_entry_var.set(f"{conf:.3f}")
        messagebox.showinfo("参数设置", f"已设置概率阈值={prob}, 置信度阈值={conf}")
    
    def create_analysis_section(self, parent):
        """创建分析功能区域"""
        analysis_frame = ttk.LabelFrame(parent, text="🔍 分析功能")
        analysis_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        analysis_frame.grid_columnconfigure(0, weight=1)
        
        # 分析按钮
        buttons = [
            ("📊 基础统计", self.show_basic_stats),
            ("🎯 共识分析", self.show_consensus_analysis),
            ("📈 阈值敏感性", self.show_threshold_sensitivity),
            ("🔥 模型一致性", self.show_model_consistency),
            ("🎨 分布可视化", self.show_distribution_plots),
            ("🎯 筛选漏斗", self.show_funnel_analysis)
        ]
        
        for i, (text, command) in enumerate(buttons):
            btn = ttk.Button(analysis_frame, text=text, command=command)
            btn.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2)
    
    def create_enhanced_export_section(self, parent):
        """创建增强版导出功能区域"""
        export_frame = ttk.LabelFrame(parent, text="💾 导出功能（增强版）")
        export_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        export_frame.grid_columnconfigure(0, weight=1)
        
        # 导出按钮
        export_buttons = [
            ("📋 生成分析报告", self.generate_report),
            ("📁 导出筛选结果（基础）", self.export_filtered_results_basic),
            ("🔍 导出筛选结果（详细）", self.export_filtered_results_detailed),
            ("📊 导出所有模型数据", self.export_all_model_data),
            ("🖼️ 保存当前图表", self.save_current_plot)
        ]
        
        for i, (text, command) in enumerate(export_buttons):
            btn = ttk.Button(export_frame, text=text, command=command)
            btn.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2)
    
    def create_display_area(self, parent):
        """创建右侧显示区域"""
        # 显示区域框架
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)
        
        # 创建笔记本控件（标签页）
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 图表显示标签页
        self.create_plot_tab()
        
        # 数据表格标签页
        self.create_table_tab()
        
        # 详细信息标签页
        self.create_info_tab()
    
    def create_plot_tab(self):
        """创建图表显示标签页"""
        plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(plot_frame, text="📊 图表")
        
        # 创建matplotlib图表
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # 初始化空图表
        self.show_welcome_plot()
    
    def create_table_tab(self):
        """创建数据表格标签页"""
        table_frame = ttk.Frame(self.notebook)
        self.notebook.add(table_frame, text="📋 数据表格")
        
        # 创建树形视图
        self.tree = ttk.Treeview(table_frame, show='headings')
        
        # 滚动条
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # 布局
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
    
    def create_info_tab(self):
        """创建详细信息标签页"""
        info_frame = ttk.Frame(self.notebook)
        self.notebook.add(info_frame, text="ℹ️ 详细信息")
        
        # 创建滚动文本框
        self.info_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, 
                                                  font=('Consolas', 10))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # 初始化欢迎信息
        self.show_welcome_info()
    
    def create_status_bar(self, parent):
        """创建底部状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.grid_columnconfigure(1, weight=1)
        
        # 状态标签
        ttk.Label(status_frame, text="状态:").grid(row=0, column=0, padx=(0, 5))
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.grid(row=0, column=1, sticky=tk.W)
        
        # 进度条
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                           mode='determinate', length=200)
        self.progress_bar.grid(row=0, column=2, padx=(10, 0))
    
    def show_welcome_plot(self):
        """显示欢迎图表"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # 创建一个简单的欢迎图表
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * np.exp(-x/5)
        
        ax.plot(x, y, 'b-', linewidth=2, alpha=0.7)
        ax.set_title('欢迎使用预测结果分析器 v2.1', fontsize=16, fontweight='bold')
        ax.set_xlabel('请先加载预测数据', fontsize=12)
        ax.set_ylabel('然后选择分析功能', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def show_welcome_info(self):
        """显示欢迎信息"""
        welcome_text = """
🔬 预测结果分析器 GUI v2.1 - 增强版
════════════════════════════════════════

🎉 新功能亮点:
✨ 参数调整：支持滑块调整 + 直接数值输入
✨ 导出增强：基础统计 + 详细模型数据导出
✨ 快速设置：预设参数组合，一键应用

📋 使用步骤:
1. 点击"浏览"选择预测结果目录，或点击"🔍 自动查找"
2. 点击"🔄 加载数据"导入分析数据
3. 调整筛选参数：
   • 使用滑块拖拽调整
   • 直接在输入框中输入精确数值
   • 点击"↔"按钮同步输入框到滑块
   • 使用快速设置按钮应用预设组合
4. 点击"✅ 应用参数"确认设置
5. 选择分析功能查看结果
6. 使用增强版导出功能：
   • 📁 基础导出：平均统计数据
   • 🔍 详细导出：包含每个模型的预测数据
   • 📊 全量导出：所有模型的完整数据

🎯 主要功能:
• 📊 基础统计 - 查看数据概况
• 🎯 共识分析 - 不同筛选策略的结果
• 📈 阈值敏感性 - 参数对结果的影响
• 🔥 模型一致性 - 模型间预测一致性
• 🎨 分布可视化 - 概率和置信度分布
• 🎯 筛选漏斗 - 逐层筛选可视化

💾 增强版导出功能:
• 📋 自动生成分析报告
• 📁 基础筛选结果（摘要数据）
• 🔍 详细筛选结果（含每个模型预测）
• 📊 全量模型数据（完整原始数据）
• 🖼️ 高质量图表保存

💡 使用提示:
• 参数输入支持回车键快速确认
• 快速设置可以一键应用常用参数组合
• 详细导出包含每个模型的单独预测结果
• 所有图表支持缩放、平移和保存

开发者: woyaokaoyanhaha
版本: 2.1.0 (增强版)
时间: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, welcome_text)
    
    # ===============================
    # 事件处理函数
    # ===============================
    
    def browse_result_dir(self):
        """浏览选择结果目录"""
        directory = filedialog.askdirectory(
            title="选择预测结果目录",
            initialdir=os.getcwd()
        )
        if directory:
            self.result_dir_var.set(directory)
    
    def auto_load_latest(self):
        """自动查找并加载最新的预测结果"""
        self.status_var.set("正在查找最新预测结果...")
        
        latest_dir = self.analyzer.find_latest_result_dir()
        if latest_dir:
            self.result_dir_var.set(latest_dir)
            self.status_var.set(f"找到最新结果: {os.path.basename(latest_dir)}")
            messagebox.showinfo("找到结果", f"找到最新预测结果:\n{latest_dir}")
        else:
            self.status_var.set("未找到预测结果")
            messagebox.showwarning("未找到", "未找到预测结果目录\n请手动选择或检查 prediction_results_batch 目录")
    
    def load_data(self):
        """加载数据"""
        result_dir = self.result_dir_var.get().strip()
        
        if not result_dir:
            messagebox.showwarning("警告", "请先选择预测结果目录或点击'🔍 自动查找'")
            return
        
        if not os.path.exists(result_dir):
            messagebox.showerror("错误", f"目录不存在: {result_dir}")
            return
        
        # 显示加载进度
        self.status_var.set("正在加载数据...")
        self.progress_var.set(0)
        self.root.update()
        
        try:
            # 在后台线程中加载数据
            def load_thread():
                try:
                    success = self.analyzer.load_prediction_results(result_dir)
                    
                    # 更新UI（需要在主线程中执行）
                    self.root.after(0, self.on_data_loaded, success)
                    
                except Exception as e:
                    self.root.after(0, self.on_data_load_error, str(e))
            
            thread = threading.Thread(target=load_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            messagebox.showerror("错误", f"加载数据失败: {e}")
            self.status_var.set("加载失败")
    
    def on_data_loaded(self, success):
        """数据加载完成回调"""
        if success:
            compound_count = len(self.analyzer.compound_stats)
            model_count = len(self.analyzer.available_models)
            
            info_text = f"✅ 数据加载成功 | 化合物: {compound_count} | 模型: {model_count}"
            self.data_info_var.set(info_text)
            self.status_var.set("数据加载成功")
            
            # 更新详细信息
            self.update_info_display()
            
            # 显示基础统计图表
            self.show_basic_stats()
            
        else:
            self.data_info_var.set("❌ 数据加载失败")
            self.status_var.set("加载失败")
            messagebox.showerror("错误", "数据加载失败，请检查目录结构")
        
        self.progress_var.set(100)
    
    def on_data_load_error(self, error_msg):
        """数据加载错误回调"""
        self.data_info_var.set("❌ 数据加载失败")
        self.status_var.set("加载失败")
        messagebox.showerror("错误", f"数据加载失败: {error_msg}")
        self.progress_var.set(0)
    
    def apply_parameters(self):
        """应用参数设置"""
        if len(self.analyzer.compound_stats) == 0:
            messagebox.showwarning("警告", "请先加载数据")
            return
        
        # 更新分析器参数
        self.analyzer.current_min_consensus = self.min_consensus_var.get()
        self.analyzer.current_prob_threshold = self.prob_threshold_var.get()
        self.analyzer.current_conf_threshold = self.conf_threshold_var.get()
        
        self.status_var.set("参数已更新")
        messagebox.showinfo("成功", f"筛选参数已应用:\n"
                                   f"最小共识模型数: {self.analyzer.current_min_consensus}\n"
                                   f"概率阈值: {self.analyzer.current_prob_threshold:.3f}\n"
                                   f"置信度阈值: {self.analyzer.current_conf_threshold:.3f}")
    
    def check_analyzer(self):
        """检查分析器是否已加载"""
        if len(self.analyzer.compound_stats) == 0:
            messagebox.showwarning("警告", "请先加载预测数据")
            return False
        return True
    
    # ===============================
    # 增强版导出功能
    # ===============================
    
    def export_filtered_results_basic(self):
        """导出基础筛选结果（平均数据）"""
        if not self.check_analyzer():
            return
        
        # 选择保存目录
        directory = filedialog.askdirectory(title="选择导出目录 - 基础结果")
        if not directory:
            return
        
        self.status_var.set("正在导出基础筛选结果...")
        
        try:
            # 执行各种分析
            results = {
                'all_positive': self.analyzer._find_all_positive(),
                'majority_positive': self.analyzer._find_majority_positive(),
                'high_confidence': self.analyzer._find_high_confidence(),
                'high_probability': self.analyzer._find_high_probability(),
                'custom_consensus': self.analyzer._find_custom_consensus()
            }
            
            # 创建时间戳目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = os.path.join(directory, f"basic_results_{timestamp}")
            os.makedirs(export_dir, exist_ok=True)
            
            exported_files = []
            
            # 保存各类筛选结果（基础版 - 只有平均数据）
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
                            'avg_confidence': f"{compound['avg_confidence']:.4f}"
                        }
                        df_data.append(row)
                    
                    df = pd.DataFrame(df_data)
                    output_file = os.path.join(export_dir, f"{result_type}_basic.csv")
                    df.to_csv(output_file, index=False, encoding='utf-8-sig')
                    exported_files.append(output_file)
            
            self._save_export_parameters(export_dir, "basic")
            
            # 显示成功消息
            files_list = '\n'.join([os.path.basename(f) for f in exported_files])
            messagebox.showinfo("导出成功", 
                              f"基础筛选结果已导出到:\n{export_dir}\n\n导出文件:\n{files_list}")
            
            self.status_var.set("基础结果导出完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")
            self.status_var.set("导出失败")
    
    def export_filtered_results_detailed(self):
        """导出详细筛选结果（包含每个模型的预测数据）"""
        if not self.check_analyzer():
            return
        
        # 选择保存目录
        directory = filedialog.askdirectory(title="选择导出目录 - 详细结果")
        if not directory:
            return
        
        self.status_var.set("正在导出详细筛选结果...")
        
        try:
            # 执行各种分析
            results = {
                'all_positive': self.analyzer._find_all_positive(),
                'majority_positive': self.analyzer._find_majority_positive(),
                'high_confidence': self.analyzer._find_high_confidence(),
                'high_probability': self.analyzer._find_high_probability(),
                'custom_consensus': self.analyzer._find_custom_consensus()
            }
            
            # 创建时间戳目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = os.path.join(directory, f"detailed_results_{timestamp}")
            os.makedirs(export_dir, exist_ok=True)
            
            exported_files = []
            
            # 保存各类筛选结果（详细版 - 包含每个模型的数据）
            for result_type, compounds in results.items():
                if compounds:
                    # 创建基础摘要文件
                    summary_data = []
                    detailed_data = []
                    
                    for compound in compounds:
                        # 基础摘要行
                        summary_row = {
                            'protein_id': compound['protein_id'],
                            'compound_id': compound['compound_id'],
                            'total_models': compound['total_models'],
                            'positive_predictions': compound['positive_predictions'],
                            'negative_predictions': compound['negative_predictions'],
                            'positive_ratio': f"{compound['positive_ratio']:.4f}",
                            'avg_probability_0': f"{1 - compound['avg_probability_1']:.4f}",
                            'avg_probability_1': f"{compound['avg_probability_1']:.4f}",
                            'avg_confidence': f"{compound['avg_confidence']:.4f}"
                        }
                        summary_data.append(summary_row)
                        
                        # 详细的每个模型数据
                        for pred in compound['predictions']:
                            detailed_row = {
                                'protein_id': compound['protein_id'],
                                'compound_id': compound['compound_id'],
                                'model_type': pred['model_type'],
                                'model_name': pred['model_name'],
                                'prediction': pred['prediction'],
                                'prediction_label': '相互作用' if pred['prediction'] == 1 else '无相互作用',
                                'probability_0': f"{pred['probability_0']:.4f}",
                                'probability_1': f"{pred['probability_1']:.4f}",
                                'confidence': f"{pred['confidence']:.4f}",
                                # 添加平均信息作为参考
                                'avg_probability_1': f"{compound['avg_probability_1']:.4f}",
                                'avg_confidence': f"{compound['avg_confidence']:.4f}",
                                'positive_ratio': f"{compound['positive_ratio']:.4f}"
                            }
                            detailed_data.append(detailed_row)
                    
                    # 保存摘要文件
                    summary_df = pd.DataFrame(summary_data)
                    summary_file = os.path.join(export_dir, f"{result_type}_summary.csv")
                    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
                    exported_files.append(summary_file)
                    
                    # 保存详细文件
                    detailed_df = pd.DataFrame(detailed_data)
                    detailed_file = os.path.join(export_dir, f"{result_type}_detailed.csv")
                    detailed_df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
                    exported_files.append(detailed_file)
            
            self._save_export_parameters(export_dir, "detailed")
            
            # 显示成功消息
            files_list = '\n'.join([os.path.basename(f) for f in exported_files])
            messagebox.showinfo("导出成功", 
                              f"详细筛选结果已导出到:\n{export_dir}\n\n导出文件:\n{files_list}\n\n"
                              f"每个筛选类型包含两个文件:\n"
                              f"• *_summary.csv: 基础摘要数据\n"
                              f"• *_detailed.csv: 每个模型的详细预测")
            
            self.status_var.set("详细结果导出完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")
            self.status_var.set("导出失败")
    
    def export_all_model_data(self):
        """导出所有模型的完整原始数据"""
        if not self.check_analyzer():
            return
        
        # 选择保存目录
        directory = filedialog.askdirectory(title="选择导出目录 - 完整模型数据")
        if not directory:
            return
        
        self.status_var.set("正在导出所有模型数据...")
        
        try:
            # 创建时间戳目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = os.path.join(directory, f"all_model_data_{timestamp}")
            os.makedirs(export_dir, exist_ok=True)
            
            exported_files = []
            
            # 导出每个模型类型的数据
            for model_type, models in self.analyzer.raw_predictions.items():
                model_type_dir = os.path.join(export_dir, model_type.replace('/', '_'))
                os.makedirs(model_type_dir, exist_ok=True)
                
                for model_name, df in models.items():
                    output_file = os.path.join(model_type_dir, f"{model_name}_complete.csv")
                    df.to_csv(output_file, index=False, encoding='utf-8-sig')
                    exported_files.append(output_file)
            
            # 创建合并的全量数据文件
            all_predictions = []
            for model_type, models in self.analyzer.raw_predictions.items():
                for model_name, df in models.items():
                    df_copy = df.copy()
                    df_copy['model_type'] = model_type
                    df_copy['model_name'] = model_name
                    all_predictions.append(df_copy)
            
            if all_predictions:
                combined_df = pd.concat(all_predictions, ignore_index=True)
                combined_file = os.path.join(export_dir, "all_predictions_combined.csv")
                combined_df.to_csv(combined_file, index=False, encoding='utf-8-sig')
                exported_files.append(combined_file)
            
            self._save_export_parameters(export_dir, "complete")
            
            # 创建数据说明文件
            readme_file = os.path.join(export_dir, "README.txt")
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write("完整模型数据导出说明\n")
                f.write("=" * 40 + "\n")
                f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据来源: {self.analyzer.result_dir}\n\n")
                
                f.write("文件结构:\n")
                f.write("- 各模型类型目录: 包含该类型下所有模型的完整预测数据\n")
                f.write("- all_predictions_combined.csv: 所有模型预测数据的合并文件\n")
                f.write("- export_parameters.json: 导出时的参数设置\n\n")
                
                f.write("数据列说明:\n")
                f.write("- protein_id: 蛋白质ID\n")
                f.write("- compound_id: 化合物ID\n")
                f.write("- prediction: 预测结果 (0=无相互作用, 1=相互作用)\n")
                f.write("- probability_0: 预测为无相互作用的概率\n")
                f.write("- probability_1: 预测为相互作用的概率\n")
                f.write("- confidence: 预测置信度\n")
                f.write("- model_type: 模型类型\n")
                f.write("- model_name: 模型名称\n")
            
            # 显示成功消息
            model_count = sum(len(models) for models in self.analyzer.raw_predictions.values())
            total_predictions = sum(len(df) for models in self.analyzer.raw_predictions.values() 
                                  for df in models.values())
            
            messagebox.showinfo("导出成功", 
                              f"完整模型数据已导出到:\n{export_dir}\n\n"
                              f"导出统计:\n"
                              f"• 模型数量: {model_count}\n"
                              f"• 预测记录总数: {total_predictions:,}\n"
                              f"• 文件数量: {len(exported_files)}\n\n"
                              f"包含:\n"
                              f"• 各模型类型的单独数据文件\n"
                              f"• 合并的完整数据文件\n"
                              f"• 数据说明文件")
            
            self.status_var.set("完整数据导出完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")
            self.status_var.set("导出失败")
    
    def _save_export_parameters(self, export_dir, export_type):
        """保存导出参数"""
        params = {
            'export_type': export_type,
            'min_consensus_models': self.analyzer.current_min_consensus,
            'probability_threshold': self.analyzer.current_prob_threshold,
            'confidence_threshold': self.analyzer.current_conf_threshold,
            'export_time': datetime.now().isoformat(),
            'total_compounds_analyzed': len(self.analyzer.compound_stats),
            'total_models': len(self.analyzer.available_models),
            'data_source': self.analyzer.result_dir,
            'version': '2.1.0'
        }
        
        params_file = os.path.join(export_dir, "export_parameters.json")
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
    
    # ===============================
    # 其他功能函数（继承原有功能）
    # ===============================
    
    def show_basic_stats(self):
        """显示基础统计"""
        if not self.check_analyzer():
            return
        
        self.status_var.set("生成基础统计图表...")
        
        # 清除之前的图表
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
                    label=f'当前阈值 ({self.analyzer.current_prob_threshold:.3f})')
        ax1.set_title('概率分布', fontsize=12)
        ax1.set_xlabel('平均正例概率')
        ax1.set_ylabel('化合物数量')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 置信度分布直方图
        ax2 = fig.add_subplot(222)
        ax2.hist(all_confs, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(self.analyzer.current_conf_threshold, color='red', linestyle='--',
                    label=f'当前阈值 ({self.analyzer.current_conf_threshold:.3f})')
        ax2.set_title('置信度分布', fontsize=12)
        ax2.set_xlabel('平均置信度')
        ax2.set_ylabel('化合物数量')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子图3: 正例比例分布
        ax3 = fig.add_subplot(223)
        ax3.hist(all_ratios, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.set_title('正例比例分布', fontsize=12)
        ax3.set_xlabel('正例预测比例')
        ax3.set_ylabel('化合物数量')
        ax3.grid(True, alpha=0.3)

        # 子图4: 概率vs置信度散点图
        ax4 = fig.add_subplot(224)
        scatter = ax4.scatter(all_probs, all_confs, c=all_ratios, cmap='RdYlBu_r',
                              alpha=0.6, s=30)
        ax4.axhline(self.analyzer.current_conf_threshold, color='red', linestyle='--', alpha=0.7)
        ax4.axvline(self.analyzer.current_prob_threshold, color='red', linestyle='--', alpha=0.7)
        ax4.set_title('概率 vs 置信度', fontsize=12)
        ax4.set_xlabel('平均正例概率')
        ax4.set_ylabel('平均置信度')
        ax4.grid(True, alpha=0.3)

        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax4)
        cbar.set_label('正例比例')

        fig.tight_layout()
        self.canvas.draw()

        # 更新详细信息
        self.update_basic_stats_info()

        self.status_var.set("基础统计图表已生成")

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

        # 清除之前的图表
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 准备数据
        categories = ['所有模型\n一致', '大多数\n同意', '高置信度', '高概率', '综合\n筛选']
        counts = [len(all_positive), len(majority_positive), len(high_confidence),
                  len(high_probability), len(custom_consensus)]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

        # 创建柱状图
        bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')

        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')

        ax.set_title('不同筛选策略的化合物数量', fontsize=14, fontweight='bold')
        ax.set_ylabel('化合物数量')
        ax.grid(True, alpha=0.3, axis='y')

        self.figure.tight_layout()
        self.canvas.draw()

        # 更新表格数据
        self.update_consensus_table(all_positive, majority_positive, high_confidence,
                                    high_probability, custom_consensus)

        # 更新详细信息
        self.update_consensus_info(all_positive, majority_positive, high_confidence,
                                   high_probability, custom_consensus)

        self.status_var.set("共识分析完成")

    def show_threshold_sensitivity(self):
        """显示阈值敏感性分析"""
        if not self.check_analyzer():
            return

        self.status_var.set("分析阈值敏感性...")

        # 清除之前的图表
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

    def show_model_consistency(self):
        """显示模型正例预测一致性热图"""
        if not self.check_analyzer():
            return

        self.status_var.set("分析模型正例预测一致性...")

        # 清除之前的图表
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 获取模型名称
        model_names = []
        for model_type, models in self.analyzer.raw_predictions.items():
            for model_name in models.keys():
                type_short = model_type.split('_')[-1][:3] if '_' in model_type else model_type[:3]
                model_short = model_name[:3]
                short_name = f"{type_short}_{model_short}"
                model_names.append(short_name)

        n_models = len(model_names)

        if n_models < 2:
            ax.text(0.5, 0.5, '需要至少2个模型才能分析一致性',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            self.canvas.draw()
            return

        # 计算正例预测一致性
        consistency_matrix = self._calculate_model_consistency(model_names)

        # 创建热图
        im = ax.imshow(consistency_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

        # 设置标签
        ax.set_xticks(range(n_models))
        ax.set_yticks(range(n_models))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_yticklabels(model_names)

        # 添加数值标注
        for i in range(n_models):
            for j in range(n_models):
                color = 'white' if consistency_matrix[i, j] < 0.5 else 'black'
                text = ax.text(j, i, f'{consistency_matrix[i, j]:.2f}',
                               ha="center", va="center", color=color, fontsize=8)

        ax.set_title('模型正例预测一致性热图 (Jaccard相似度)', fontsize=14, fontweight='bold')

        # 添加颜色条
        cbar = self.figure.colorbar(im, ax=ax)
        cbar.set_label('Jaccard相似度 (0=无重叠, 1=完全重叠)')

        self.figure.tight_layout()
        self.canvas.draw()

        self.status_var.set("正例预测一致性分析完成")

    def _calculate_model_consistency(self, model_names):
        """计算模型正例预测一致性（使用Jaccard相似度）"""
        n_models = len(model_names)
        consistency_matrix = np.zeros((n_models, n_models))

        print("开始计算模型正例预测一致性...")

        # 重新生成正确的映射关系
        full_model_mapping = {}
        for model_type, models in self.analyzer.raw_predictions.items():
            for model_name in models.keys():
                type_short = model_type.split('_')[-1][:3] if '_' in model_type else model_type[:3]
                model_short = model_name[:3]
                short_name = f"{type_short}_{model_short}"
                full_model_mapping[short_name] = (model_type, model_name)

        # 获取所有模型的正例预测
        model_positive_predictions = {}

        for short_name in model_names:
            if short_name in full_model_mapping:
                model_type, model_name = full_model_mapping[short_name]

                if model_type in self.analyzer.raw_predictions and model_name in self.analyzer.raw_predictions[
                    model_type]:
                    df = self.analyzer.raw_predictions[model_type][model_name]

                    # 只保存正例预测
                    positive_predictions = set()
                    total_predictions = 0

                    for _, row in df.iterrows():
                        key = f"{row['protein_id']}_{row['compound_id']}"
                        prediction = int(row['prediction'])
                        total_predictions += 1

                        # 如果是正例，加入正例集合
                        if prediction == 1:
                            positive_predictions.add(key)

                    model_positive_predictions[short_name] = positive_predictions

                    positive_rate = len(positive_predictions) / total_predictions * 100
                    print(
                        f"✓ 模型 {short_name}: 总预测={total_predictions}, 正例={len(positive_predictions)} ({positive_rate:.1f}%)")

        print(f"成功加载 {len(model_positive_predictions)} 个模型的正例预测数据")

        # 计算正例预测的Jaccard相似度
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    consistency_matrix[i][j] = 1.0
                elif model1 in model_positive_predictions and model2 in model_positive_predictions:
                    pos1 = model_positive_predictions[model1]
                    pos2 = model_positive_predictions[model2]

                    # 计算Jaccard相似度：交集 / 并集
                    intersection = pos1 & pos2
                    union = pos1 | pos2

                    if len(union) > 0:
                        jaccard_similarity = len(intersection) / len(union)
                        consistency_matrix[i][j] = jaccard_similarity

                        # 计算其他有用指标
                        if len(pos1) > 0:
                            recall_1_to_2 = len(intersection) / len(pos1)
                        else:
                            recall_1_to_2 = 0

                        if len(pos2) > 0:
                            recall_2_to_1 = len(intersection) / len(pos2)
                        else:
                            recall_2_to_1 = 0

                        print(f"  {model1} vs {model2}:")
                        print(f"    模型1正例: {len(pos1)}, 模型2正例: {len(pos2)}")
                        print(f"    共同正例: {len(intersection)}, 总正例: {len(union)}")
                        print(f"    Jaccard相似度: {jaccard_similarity:.3f}")
                        print(f"    模型1→模型2覆盖率: {recall_1_to_2:.3f}")
                        print(f"    模型2→模型1覆盖率: {recall_2_to_1:.3f}")
                    else:
                        consistency_matrix[i][j] = 0.0
                        print(f"  {model1} vs {model2}: 都无正例预测")
                else:
                    consistency_matrix[i][j] = 0.0
                    print(f"  {model1} vs {model2}: 数据缺失")

        # 输出统计摘要
        off_diagonal = [consistency_matrix[i][j] for i in range(n_models) for j in range(n_models) if i != j]

        if off_diagonal and any(x > 0 for x in off_diagonal):
            print(f"\n📊 正例预测Jaccard相似度统计:")
            print(f"  平均相似度: {np.mean(off_diagonal):.3f}")
            print(f"  最小相似度: {np.min(off_diagonal):.3f}")
            print(f"  最大相似度: {np.max(off_diagonal):.3f}")

            # 找出最相似和最不相似的模型对
            max_idx = np.unravel_index(np.argmax(consistency_matrix - np.eye(n_models)), consistency_matrix.shape)
            min_idx = np.unravel_index(np.argmin(consistency_matrix + np.eye(n_models) * 2), consistency_matrix.shape)

            print(f"\n🎯 正例预测分析:")
            print(
                f"  最相似模型对: {model_names[max_idx[0]]} vs {model_names[max_idx[1]]} (相似度: {consistency_matrix[max_idx]:.3f})")
            print(
                f"  最不相似模型对: {model_names[min_idx[0]]} vs {model_names[min_idx[1]]} (相似度: {consistency_matrix[min_idx]:.3f})")

            # 模型互补性分析
            print(f"\n💡 模型互补性建议:")
            for i, model1 in enumerate(model_names):
                other_similarities = [consistency_matrix[i][j] for j in range(n_models) if i != j]
                avg_similarity = np.mean(other_similarities)
                if avg_similarity < 0.5:
                    print(f"  {model1}: 与其他模型差异较大，建议保留用于集成")
                elif avg_similarity > 0.8:
                    print(f"  {model1}: 与其他模型高度相似，可考虑替换")
                else:
                    print(f"  {model1}: 与其他模型中等相似，适合集成使用")
        else:
            print(f"\n❌ 无法计算正例预测相似度")

        return consistency_matrix

    def show_distribution_plots(self):
        """显示分布可视化"""
        if not self.check_analyzer():
            return

        self.status_var.set("生成分布图...")

        # 清除之前的图表
        self.figure.clear()

        # 获取数据
        all_probs = [stats['avg_probability_1'] for stats in self.analyzer.compound_stats.values()]
        all_confs = [stats['avg_confidence'] for stats in self.analyzer.compound_stats.values()]
        all_ratios = [stats['positive_ratio'] for stats in self.analyzer.compound_stats.values()]

        # 创建子图
        fig = self.figure

        # 子图1: 概率分布（密度图）
        ax1 = fig.add_subplot(221)
        ax1.hist(all_probs, bins=30, density=True, alpha=0.7, color='skyblue')
        ax1.axvline(self.analyzer.current_prob_threshold, color='red', linestyle='--',
                    label=f'阈值 ({self.analyzer.current_prob_threshold:.2f})')
        ax1.set_title('概率密度分布')
        ax1.set_xlabel('平均正例概率')
        ax1.set_ylabel('密度')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 置信度分布（密度图）
        ax2 = fig.add_subplot(222)
        ax2.hist(all_confs, bins=30, density=True, alpha=0.7, color='lightgreen')
        ax2.axvline(self.analyzer.current_conf_threshold, color='red', linestyle='--',
                    label=f'阈值 ({self.analyzer.current_conf_threshold:.2f})')
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
        ax3.set_title('正例比例分布')

        # 子图4: 箱线图
        ax4 = fig.add_subplot(224)
        data_to_plot = [all_probs, all_confs, all_ratios]
        box_plot = ax4.boxplot(data_to_plot, labels=['概率', '置信度', '正例比例'])
        ax4.set_title('数据分布箱线图')
        ax4.set_ylabel('数值')
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        self.canvas.draw()

        self.status_var.set("分布图生成完成")

    def show_funnel_analysis(self):
        """显示筛选漏斗分析"""
        if not self.check_analyzer():
            return

        self.status_var.set("生成筛选漏斗...")

        # 计算不同筛选条件下的化合物数量
        stages = [
            ('总化合物', len(self.analyzer.compound_stats)),
            ('至少1个正例', sum(1 for s in self.analyzer.compound_stats.values() if s['positive_predictions'] > 0)),
            ('至少2个正例', sum(1 for s in self.analyzer.compound_stats.values() if s['positive_predictions'] >= 2)),
            ('大多数正例', len(self.analyzer._find_majority_positive())),
            ('高概率', len(self.analyzer._find_high_probability())),
            ('高置信度', len(self.analyzer._find_high_confidence())),
            ('所有模型一致', len(self.analyzer._find_all_positive())),
            ('综合筛选', len(self.analyzer._find_custom_consensus()))
        ]

        # 清除之前的图表
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        stage_names = [s[0] for s in stages]
        counts = [s[1] for s in stages]

        # 创建水平条形图
        y_pos = np.arange(len(stages))
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(stages)))

        bars = ax.barh(y_pos, counts, color=colors)

        # 添加数值标签
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{count:,}', ha='left', va='center', fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(stage_names)
        ax.set_xlabel('化合物数量')
        ax.set_title('化合物筛选漏斗图', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # 反转y轴使其呈现漏斗效果
        ax.invert_yaxis()

        self.figure.tight_layout()
        self.canvas.draw()

        self.status_var.set("筛选漏斗分析完成")

    # ===============================
    # 信息更新函数
    # ===============================

    def update_info_display(self):
        """更新详细信息显示"""
        if len(self.analyzer.compound_stats) == 0:
            return

        info_text = f"""
📊 数据概况
═══════════════════════════════════════════
化合物总数: {len(self.analyzer.compound_stats):,}
模型总数: {len(self.analyzer.available_models)}
数据目录: {self.analyzer.result_dir}

⚙️ 当前筛选参数
═══════════════════════════════════════════
最小共识模型数: {self.analyzer.current_min_consensus}
概率阈值: {self.analyzer.current_prob_threshold:.3f}
置信度阈值: {self.analyzer.current_conf_threshold:.3f}

🤖 可用模型列表
═══════════════════════════════════════════
"""

        for model_type, models in self.analyzer.raw_predictions.items():
            info_text += f"\n{model_type}:\n"
            for model_name in models.keys():
                info_text += f"  • {model_name}\n"

        info_text += f"""

💡 使用提示 (v2.1增强版)
═══════════════════════════════════════════
• 参数调整: 支持滑块拖拽 + 直接输入数值
• 输入框支持回车键快速确认
• 快速设置按钮可一键应用预设参数
• 导出功能包含详细的模型预测数据
• 使用"详细导出"获取每个模型的单独预测
"""

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info_text)

    def update_basic_stats_info(self):
        """更新基础统计信息"""
        if len(self.analyzer.compound_stats) == 0:
            return

        all_probs = [stats['avg_probability_1'] for stats in self.analyzer.compound_stats.values()]
        all_confs = [stats['avg_confidence'] for stats in self.analyzer.compound_stats.values()]
        all_ratios = [stats['positive_ratio'] for stats in self.analyzer.compound_stats.values()]

        # 计算在当前阈值下符合条件的化合物数量
        prob_filtered = sum(1 for p in all_probs if p >= self.analyzer.current_prob_threshold)
        conf_filtered = sum(1 for c in all_confs if c >= self.analyzer.current_conf_threshold)
        both_filtered = sum(1 for stats in self.analyzer.compound_stats.values()
                            if stats['avg_probability_1'] >= self.analyzer.current_prob_threshold and
                            stats['avg_confidence'] >= self.analyzer.current_conf_threshold)

        stats_text = f"""
📊 基础统计分析结果
═══════════════════════════════════════════

📈 概率统计
───────────────────────────────────────────
平均值: {np.mean(all_probs):.4f}
中位数: {np.median(all_probs):.4f}
标准差: {np.std(all_probs):.4f}
最小值: {np.min(all_probs):.4f}
最大值: {np.max(all_probs):.4f}
≥ 当前阈值({self.analyzer.current_prob_threshold:.3f}): {prob_filtered} 个

🎯 置信度统计
───────────────────────────────────────────
平均值: {np.mean(all_confs):.4f}
中位数: {np.median(all_confs):.4f}
标准差: {np.std(all_confs):.4f}
最小值: {np.min(all_confs):.4f}
最大值: {np.max(all_confs):.4f}
≥ 当前阈值({self.analyzer.current_conf_threshold:.3f}): {conf_filtered} 个

📊 正例比例统计
───────────────────────────────────────────
平均值: {np.mean(all_ratios):.4f}
中位数: {np.median(all_ratios):.4f}
标准差: {np.std(all_ratios):.4f}

🎯 当前阈值筛选
───────────────────────────────────────────
同时满足概率和置信度阈值: {both_filtered} 个

📋 分布情况
───────────────────────────────────────────
"""

        # 正例比例分组统计
        ratio_groups = {
            '无正例 (0%)': sum(1 for r in all_ratios if r == 0),
            '少数正例 (0-25%)': sum(1 for r in all_ratios if 0 < r <= 0.25),
            '部分正例 (25-50%)': sum(1 for r in all_ratios if 0.25 < r <= 0.5),
            '多数正例 (50-75%)': sum(1 for r in all_ratios if 0.5 < r <= 0.75),
            '大多数正例 (75-100%)': sum(1 for r in all_ratios if 0.75 < r < 1),
            '全部正例 (100%)': sum(1 for r in all_ratios if r == 1)
        }

        for group, count in ratio_groups.items():
            percentage = (count / len(all_ratios)) * 100 if len(all_ratios) > 0 else 0
            stats_text += f"{group}: {count:,} 个 ({percentage:.1f}%)\n"

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, stats_text)

    def update_consensus_table(self, all_positive, majority_positive, high_confidence,
                               high_probability, custom_consensus):
        """更新共识分析表格"""
        # 清除现有数据
        for item in self.tree.get_children():
            self.tree.delete(item)

        # 设置列
        columns = ('类型', '化合物ID', '蛋白质ID', '正例/总数', '概率', '置信度')
        self.tree['columns'] = columns

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        # 添加数据
        def add_compounds(compounds, category):
            for compound in compounds[:20]:  # 限制显示前20个
                self.tree.insert('', 'end', values=(
                    category,
                    compound['compound_id'],
                    compound['protein_id'],
                    f"{compound['positive_predictions']}/{compound['total_models']}",
                    f"{compound['avg_probability_1']:.3f}",
                    f"{compound['avg_confidence']:.3f}"
                ))

        add_compounds(all_positive, "所有模型一致")
        add_compounds(majority_positive, "大多数同意")
        add_compounds(high_confidence, "高置信度")
        add_compounds(high_probability, "高概率")
        add_compounds(custom_consensus, "综合筛选")

    def update_consensus_info(self, all_positive, majority_positive, high_confidence,
                              high_probability, custom_consensus):
        """更新共识分析信息"""
        info_text = f"""
🎯 共识分析结果
═══════════════════════════════════════════

📊 筛选结果统计
───────────────────────────────────────────
所有模型都预测为正例: {len(all_positive):,} 个化合物
大多数模型预测为正例: {len(majority_positive):,} 个化合物
高置信度预测: {len(high_confidence):,} 个化合物
高概率预测: {len(high_probability):,} 个化合物
综合筛选结果: {len(custom_consensus):,} 个化合物

🥇 最高优先级化合物 (所有模型一致)
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
💡 分析建议
───────────────────────────────────────────
"""

        if len(all_positive) > 0:
            info_text += "• 优先验证所有模型都预测为正例的化合物，成功率最高\n"
        if len(high_confidence) > len(all_positive):
            info_text += "• 考虑高置信度化合物作为二线选择\n"
        if len(majority_positive) > 50:
            info_text += "• 大多数模型预测为正例的化合物数量较多，建议进一步筛选\n"

        info_text += "• 建议结合生物学知识和化合物特性进行最终筛选\n"
        info_text += "• 考虑分批进行实验验证，从最高置信度开始\n"
        info_text += "• 使用'详细导出'功能获取每个模型的具体预测数据\n"

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info_text)

    # ===============================
    # 其他导出功能
    # ===============================

    def generate_report(self):
        """生成分析报告"""
        if not self.check_analyzer():
            return

        # 选择保存位置
        file_path = filedialog.asksaveasfilename(
            title="保存分析报告",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if not file_path:
            return

        self.status_var.set("正在生成报告...")

        try:
            # 执行各种分析
            all_positive = self.analyzer._find_all_positive()
            majority_positive = self.analyzer._find_majority_positive()
            high_confidence = self.analyzer._find_high_confidence()
            high_probability = self.analyzer._find_high_probability()
            custom_consensus = self.analyzer._find_custom_consensus()

            # 生成报告内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("预测结果分析报告 (增强版)\n")
                f.write("=" * 80 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"分析用户: {os.getenv('USERNAME', 'woyaokaoyanhaha')}\n")
                f.write(f"数据来源: {self.analyzer.result_dir}\n")
                f.write(f"分析工具: 预测结果分析器 GUI v2.1 (增强版)\n\n")

                f.write("数据概况\n")
                f.write("-" * 40 + "\n")
                f.write(f"化合物总数: {len(self.analyzer.compound_stats):,}\n")
                f.write(f"模型总数: {len(self.analyzer.available_models)}\n")
                f.write(f"分析参数:\n")
                f.write(f"  最小共识模型数: {self.analyzer.current_min_consensus}\n")
                f.write(f"  概率阈值: {self.analyzer.current_prob_threshold:.3f}\n")
                f.write(f"  置信度阈值: {self.analyzer.current_conf_threshold:.3f}\n\n")

                f.write("筛选结果统计\n")
                f.write("-" * 40 + "\n")
                f.write(f"所有模型都预测为正例: {len(all_positive):,} 个化合物\n")
                f.write(f"大多数模型预测为正例: {len(majority_positive):,} 个化合物\n")
                f.write(f"高置信度预测: {len(high_confidence):,} 个化合物\n")
                f.write(f"高概率预测: {len(high_probability):,} 个化合物\n")
                f.write(f"综合筛选结果: {len(custom_consensus):,} 个化合物\n\n")

                # 重点化合物推荐
                f.write("重点化合物推荐\n")
                f.write("-" * 40 + "\n")

                if all_positive:
                    f.write("🥇 最高优先级化合物 (所有模型都预测为正例):\n")
                    for i, compound in enumerate(all_positive[:20], 1):
                        f.write(f"  {i:2d}. {compound['compound_id']} (蛋白质: {compound['protein_id']}) - "
                                f"概率: {compound['avg_probability_1']:.3f}, 置信度: {compound['avg_confidence']:.3f}\n")
                    f.write("\n")

                if high_confidence:
                    f.write("🥈 高置信度化合物:\n")
                    for i, compound in enumerate(high_confidence[:20], 1):
                        f.write(f"  {i:2d}. {compound['compound_id']} (蛋白质: {compound['protein_id']}) - "
                                f"概率: {compound['avg_probability_1']:.3f}, 置信度: {compound['avg_confidence']:.3f}\n")
                    f.write("\n")

                # 统计分析
                all_probs = [stats['avg_probability_1'] for stats in self.analyzer.compound_stats.values()]
                all_confs = [stats['avg_confidence'] for stats in self.analyzer.compound_stats.values()]

                f.write("统计分析\n")
                f.write("-" * 40 + "\n")
                f.write(f"概率分布:\n")
                f.write(f"  平均值: {np.mean(all_probs):.4f}\n")
                f.write(f"  中位数: {np.median(all_probs):.4f}\n")
                f.write(f"  标准差: {np.std(all_probs):.4f}\n")
                f.write(f"  最小值: {np.min(all_probs):.4f}\n")
                f.write(f"  最大值: {np.max(all_probs):.4f}\n\n")

                f.write(f"置信度分布:\n")
                f.write(f"  平均值: {np.mean(all_confs):.4f}\n")
                f.write(f"  中位数: {np.median(all_confs):.4f}\n")
                f.write(f"  标准差: {np.std(all_confs):.4f}\n")
                f.write(f"  最小值: {np.min(all_confs):.4f}\n")
                f.write(f"  最大值: {np.max(all_confs):.4f}\n\n")

                # 增强版分析建议
                f.write("分析建议 (增强版)\n")
                f.write("-" * 40 + "\n")
                if len(all_positive) > 0:
                    f.write("1. 优先验证所有模型都预测为正例的化合物，成功率最高\n")
                if len(high_confidence) > len(all_positive):
                    f.write("2. 考虑高置信度化合物作为二线选择\n")
                if len(majority_positive) > 50:
                    f.write("3. 大多数模型预测为正例的化合物数量较多，建议进一步筛选\n")

                f.write("4. 建议结合生物学知识和化合物特性进行最终筛选\n")
                f.write("5. 考虑分批进行实验验证，从最高置信度开始\n")
                f.write("6. 使用详细导出功能获取每个模型的具体预测数据\n")
                f.write("7. 可以调整参数阈值进行敏感性分析\n")
                f.write("8. 定期更新模型和重新评估预测结果\n")

                f.write("\n导出功能说明\n")
                f.write("-" * 40 + "\n")
                f.write("• 基础导出: 包含平均统计数据，适合快速筛选\n")
                f.write("• 详细导出: 包含每个模型的预测数据，适合深入分析\n")
                f.write("• 完整导出: 包含所有原始模型数据，适合数据挖掘\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("报告结束\n")
                f.write("=" * 80 + "\n")

            messagebox.showinfo("成功", f"增强版分析报告已保存到:\n{file_path}")
            self.status_var.set("报告生成完成")

        except Exception as e:
            messagebox.showerror("错误", f"生成报告失败: {e}")
            self.status_var.set("报告生成失败")

    def save_current_plot(self):
        """保存当前图表"""
        if not self.figure:
            messagebox.showwarning("警告", "没有可保存的图表")
            return

        # 选择保存位置
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


def main():
    """主函数"""
    # 创建主窗口
    root = tk.Tk()

    # 创建应用
    app = EnhancedPredictionAnalyzerGUI(root)

    # 设置窗口关闭事件
    def on_closing():
        if messagebox.askokcancel("退出", "确定要退出预测结果分析器吗？"):
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # 运行应用
    try:
        print("🚀 启动预测结果分析器 GUI v2.1 (增强版)...")
        print("✨ 新功能: 滑块+直接输入 + 详细导出")
        print("✅ 中文字体已配置")
        print("✅ 界面组件已加载")
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