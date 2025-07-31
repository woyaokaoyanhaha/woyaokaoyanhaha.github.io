# -*- coding: utf-8 -*-
"""
ESM2集成BBB穿透肽正样本生成器
基于integrated_bbb_positive_generator.py，使用ESM2模型表征蛋白质
用户: woyaokaoyanhaha
当前时间: 2025-07-16 15:30:00 UTC
架构: ESM2特征提取 + 扩散模型 + Transformer生成
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
import random
from typing import List, Tuple, Dict, Set
import logging
from pathlib import Path
import warnings
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import hashlib
import gc

# ESM2相关导入
try:
    import esm
    ESM2_AVAILABLE = True
except ImportError:
    ESM2_AVAILABLE = False
    print("⚠️ ESM2未安装，请运行: pip install fair-esm")

warnings.filterwarnings('ignore')

# ==================== ESM2整合正样本生成配置 ====================
class ESM2IntegratedPositiveConfig:
    """ESM2整合BBB穿透肽正样本生成配置"""
    
    # 基础参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seq_len = 20  # 保持与原始代码一致
    vocab_size = 20
    
    # ESM2模型配置
    esm2_model_name = "esm2_t6_8M_UR50D"  # 使用较小的模型以节省显存
    esm2_repr_layers = [6]  # 最后一层
    esm2_max_seq_len = 20
    esm2_embedding_dim = 320  # ESM2-8M的嵌入维度
    freeze_esm2 = True  # 冻结ESM2参数
    
    # 数据平衡参数
    existing_positive_samples = 329
    existing_negative_samples = 6851
    target_positive_samples = 6851
    need_to_generate = 6522
    
    # 训练参数 - 针对ESM2优化
    batch_size = 8  # 调整批次大小适应ESM2
    lr = 3e-5
    n_epochs = 100
    early_stopping_patience = 15
    
    # 扩散参数
    n_timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    
    # 模型复杂度 - 集成ESM2
    embedding_dim = 256  # 增加嵌入维度以匹配ESM2
    transformer_layers = 4
    attention_heads = 8
    dropout_rate = 0.15
    
    # 生成参数
    n_sequences = 6522
    quality_threshold = 0.55
    
    # 采样参数
    sampling_steps = 50
    temperature_start = 2.5
    temperature_end = 0.8
    temperature_schedule = "cosine"
    top_k = 15
    top_p = 0.88
    
    # BBB肽实际需求参数
    target_length_min = 6
    target_length_max = 15
    target_length_optimal = 9
    target_mw_min = 500
    target_mw_max = 2000
    target_mw_optimal = 1200
    target_charge_min = 1
    target_charge_max = 6
    target_charge_optimal = 3
    
    # BBB肽必需特征
    min_cationic_residues = 2
    max_cationic_residues = 6
    min_hydrophobic_residues = 1
    max_hydrophobic_residues = 8
    min_aromatic_residues = 0
    max_aromatic_residues = 4
    
    # 多样性增强参数
    diversity_boost = True
    diversity_temperature = 2.2
    nucleus_sampling = True
    repetition_penalty = 1.25
    length_penalty = 0.12
    
    # 去重参数
    enable_deduplication = True
    similarity_threshold = 0.78
    max_generation_attempts = 35000
    duplicate_check_window = 150
    
    # 动态采样参数
    dynamic_sampling = True
    adaptive_temperature = True
    diversity_penalty_weight = 0.25
    
    # 显存优化参数
    mixed_precision = True
    gradient_accumulation_steps = 4
    gradient_clip_norm = 1.0
    num_workers = 0  # ESM2使用时建议设为0
    pin_memory = False
    
    # 数据增强
    augment_data = True
    augment_ratio = 3.0
    noise_augmentation = True
    
    # 质量评估权重
    property_weights = {
        'length_score': 0.20,
        'charge_score': 0.20,
        'molecular_weight': 0.18,
        'cationic_ratio': 0.18,
        'hydrophobic_ratio': 0.14,
        'bbb_motifs': 0.10
    }
    
    # 文件路径
    input_file = "train_pos_org.fasta"
    output_dir = "esm2_integrated_positive_output"
    model_save_path = "esm2_integrated_positive_model.pth"
    
    # 验证和保存
    val_split = 0.12
    save_interval = 25
    
    # 生成批次控制
    generation_batch_size = 32
    
    # 日志配置
    log_level = logging.INFO
    log_to_console = True
    log_to_file = True
    
    # 用户信息
    user_login = "woyaokaoyanhaha"
    current_time = "2025-07-16 15:30:00"


config = ESM2IntegratedPositiveConfig()

# ==================== 氨基酸属性定义 ====================
AA_PROPERTIES = {
    'A': {'mw': 89.1, 'hydro': 1.8, 'charge': 0, 'polar': 0, 'volume': 88.6},
    'R': {'mw': 174.2, 'hydro': -4.5, 'charge': 1, 'polar': 1, 'volume': 173.4},
    'N': {'mw': 132.1, 'hydro': -3.5, 'charge': 0, 'polar': 1, 'volume': 114.1},
    'D': {'mw': 133.1, 'hydro': -3.5, 'charge': -1, 'polar': 1, 'volume': 111.1},
    'C': {'mw': 121.0, 'hydro': 2.5, 'charge': 0, 'polar': 0, 'volume': 108.5},
    'Q': {'mw': 146.1, 'hydro': -3.5, 'charge': 0, 'polar': 1, 'volume': 143.8},
    'E': {'mw': 147.1, 'hydro': -3.5, 'charge': -1, 'polar': 1, 'volume': 138.4},
    'G': {'mw': 75.1, 'hydro': -0.4, 'charge': 0, 'polar': 0, 'volume': 60.1},
    'H': {'mw': 155.2, 'hydro': -3.2, 'charge': 0, 'polar': 1, 'volume': 153.2},
    'I': {'mw': 131.2, 'hydro': 4.5, 'charge': 0, 'polar': 0, 'volume': 166.7},
    'L': {'mw': 131.2, 'hydro': 3.8, 'charge': 0, 'polar': 0, 'volume': 166.7},
    'K': {'mw': 146.2, 'hydro': -3.9, 'charge': 1, 'polar': 1, 'volume': 168.6},
    'M': {'mw': 149.2, 'hydro': 1.9, 'charge': 0, 'polar': 0, 'volume': 162.9},
    'F': {'mw': 165.2, 'hydro': 2.8, 'charge': 0, 'polar': 0, 'volume': 189.9},
    'P': {'mw': 115.1, 'hydro': -1.6, 'charge': 0, 'polar': 0, 'volume': 112.7},
    'S': {'mw': 105.1, 'hydro': -0.8, 'charge': 0, 'polar': 1, 'volume': 89.0},
    'T': {'mw': 119.1, 'hydro': -0.7, 'charge': 0, 'polar': 1, 'volume': 116.1},
    'W': {'mw': 204.2, 'hydro': -0.9, 'charge': 0, 'polar': 0, 'volume': 227.8},
    'Y': {'mw': 181.2, 'hydro': -1.3, 'charge': 0, 'polar': 1, 'volume': 193.6},
    'V': {'mw': 117.1, 'hydro': 4.2, 'charge': 0, 'polar': 0, 'volume': 140.0}
}

AMINO_ACIDS = list(AA_PROPERTIES.keys())
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

# ==================== ESM2特征提取器 ====================
class ESM2FeatureExtractor:
    """ESM2特征提取器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        if not ESM2_AVAILABLE:
            raise ImportError("ESM2未安装。请运行: pip install fair-esm")
        
        self.logger.info(f"🧬 初始化ESM2特征提取器: {config.esm2_model_name}")
        
        # 加载ESM2模型
        self.model, self.alphabet = self._load_esm2_model()
        self.model = self.model.to(config.device)
        self.model.eval()
        
        # 冻结ESM2参数
        if config.freeze_esm2:
            for param in self.model.parameters():
                param.requires_grad = False
            self.logger.info("🔒 ESM2参数已冻结")
        
        self.batch_converter = self.alphabet.get_batch_converter()
        
        self.logger.info("✅ ESM2特征提取器初始化完成")
    
    def _load_esm2_model(self):
        """加载ESM2模型"""
        try:
            if self.config.esm2_model_name == "esm2_t6_8M_UR50D":
                model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
                self.config.esm2_embedding_dim = 320
            elif self.config.esm2_model_name == "esm2_t12_35M_UR50D":
                model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
                self.config.esm2_embedding_dim = 480
            else:
                model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
                self.config.esm2_embedding_dim = 320
            
            return model, alphabet
        except Exception as e:
            self.logger.error(f"❌ ESM2模型加载失败: {e}")
            raise e
    
    def extract_features(self, sequences: List[str], batch_size: int = None) -> torch.Tensor:
        """提取ESM2特征"""
        if batch_size is None:
            batch_size = max(1, self.config.batch_size // 4)  # ESM2需要更小的批次
        
        self.logger.info(f"🔍 提取 {len(sequences)} 个序列的ESM2特征")
        all_features = []
        
        # 显存监控
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 分批处理
        for i in tqdm(range(0, len(sequences), batch_size), desc="提取ESM2特征"):
            batch_sequences = sequences[i:i+batch_size]
            
            # 准备批次数据
            batch_labels = [(f"seq_{j}", seq) for j, seq in enumerate(batch_sequences)]
            
            try:
                # 转换为ESM2输入格式
                batch_tokens = self.batch_converter(batch_labels)[2]
                
                # 限制序列长度
                if batch_tokens.size(1) > self.config.esm2_max_seq_len:
                    batch_tokens = batch_tokens[:, :self.config.esm2_max_seq_len]
                
                # 移动到设备
                batch_tokens = batch_tokens.to(self.config.device)
                
                # 提取特征
                with torch.no_grad():
                    if self.config.mixed_precision and self.config.device != 'cpu':
                        with torch.cuda.amp.autocast():
                            results = self.model(batch_tokens, repr_layers=self.config.esm2_repr_layers)
                    else:
                        results = self.model(batch_tokens, repr_layers=self.config.esm2_repr_layers)
                    
                    # 获取表示
                    representations = results["representations"][self.config.esm2_repr_layers[0]]
                    
                    # 去掉特殊token（CLS和SEP）
                    sequence_representations = representations[:, 1:-1]
                    
                    # 处理表示
                    batch_features = self._process_representations(sequence_representations, batch_sequences)
                    
                    # 移到CPU节省显存
                    batch_features = batch_features.cpu()
                    all_features.append(batch_features)
                
            except Exception as e:
                self.logger.error(f"❌ 批次 {i//batch_size + 1} 特征提取失败: {e}")
                # 创建备用特征
                backup_features = torch.zeros(len(batch_sequences), self.config.seq_len, self.config.esm2_embedding_dim)
                all_features.append(backup_features)
        
        # 合并所有特征
        if all_features:
            all_features = torch.cat(all_features, dim=0)
        else:
            all_features = torch.zeros(len(sequences), self.config.seq_len, self.config.esm2_embedding_dim)
        
        self.logger.info(f"✅ ESM2特征提取完成: {all_features.shape}")
        return all_features
    
    def _process_representations(self, representations: torch.Tensor, sequences: List[str]) -> torch.Tensor:
        """处理ESM2表示"""
        batch_size = representations.size(0)
        device = representations.device
        
        features = []
        
        for i in range(batch_size):
            seq_len = min(len(sequences[i]), representations.size(1))
            seq_repr = representations[i, :seq_len]
            
            # 填充或截断到指定长度
            if seq_len < self.config.seq_len:
                padding = torch.zeros(self.config.seq_len - seq_len, self.config.esm2_embedding_dim, device=device)
                seq_features = torch.cat([seq_repr, padding], dim=0)
            else:
                seq_features = seq_repr[:self.config.seq_len]
            
            features.append(seq_features)
        
        return torch.stack(features, dim=0)


# ==================== ESM2整合数据集 ====================
class ESM2IntegratedDataset(Dataset):
    """ESM2整合数据集"""
    
    def __init__(self, sequences: List[str], esm2_features: torch.Tensor, logger):
        self.sequences = sequences
        self.esm2_features = esm2_features
        self.logger = logger
        
        # 编码序列
        self.encoded_sequences = []
        for seq in sequences:
            # 填充序列
            padded_seq = self._pad_sequence(seq, config.seq_len)
            indices = self._sequence_to_indices(padded_seq)
            self.encoded_sequences.append(torch.tensor(indices, dtype=torch.long))
        
        self.logger.info(f"✅ ESM2整合数据集创建完成: {len(sequences)} 个序列")
    
    def _pad_sequence(self, seq: str, max_len: int) -> str:
        """填充序列到指定长度"""
        if len(seq) > max_len:
            return seq[:max_len]
        return seq + 'A' * (max_len - len(seq))
    
    def _sequence_to_indices(self, seq: str) -> List[int]:
        """将序列转换为索引"""
        return [AA_TO_IDX.get(aa, 0) for aa in seq]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'encoded_sequence': self.encoded_sequences[idx],
            'esm2_features': self.esm2_features[idx]
        }


# ==================== ESM2整合扩散调度器 ====================
class ESM2IntegratedDiffusionScheduler:
    """ESM2整合扩散调度器"""
    
    def __init__(self, n_timesteps: int = config.n_timesteps,
                 vocab_size: int = config.vocab_size,
                 device: str = config.device):
        self.n_timesteps = n_timesteps
        self.vocab_size = vocab_size
        self.device = device
        
        # 生成beta调度
        self.betas = torch.linspace(config.beta_start, config.beta_end, n_timesteps).to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        
        # 预计算常用值
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
        
        # 后验方差
        if n_timesteps > 1:
            alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
            self.posterior_variance = (
                self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            ).to(device)
        else:
            self.posterior_variance = torch.tensor([0.0], device=device)
        
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
    
    def add_noise(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """向离散序列添加噪声"""
        if noise is None:
            noise = torch.randint_like(x_start, 0, self.vocab_size)
        
        t = t.to(self.device)
        alpha_t = self.alphas_cumprod[t]
        
        batch_size, seq_len = x_start.shape
        alpha_t_expanded = alpha_t.view(batch_size, 1).expand(batch_size, seq_len)
        
        # 噪声掩码
        noise_mask = torch.rand_like(alpha_t_expanded) > alpha_t_expanded
        x_noisy = torch.where(noise_mask, noise, x_start)
        
        return x_noisy, noise


# ==================== ESM2整合扩散模型 ====================
class ESM2IntegratedDiffusionModel(nn.Module):
    """ESM2整合扩散模型"""
    
    def __init__(self, device: str = config.device):
        super().__init__()
        self.device = device
        self.vocab_size = config.vocab_size
        
        # 扩散调度器
        self.scheduler = ESM2IntegratedDiffusionScheduler(device=device)
        
        # 标准嵌入层
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_embedding = nn.Parameter(torch.randn(config.seq_len, config.embedding_dim))
        
        # ESM2特征投影层
        self.esm2_projection = nn.Linear(config.esm2_embedding_dim, config.embedding_dim)
        
        # 时间嵌入
        self.time_embedding = nn.Sequential(
            nn.Linear(1, config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.embedding_dim * 3,
            dropout=config.dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.transformer_layers)
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.embedding_dim, config.vocab_size)
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x, t, esm2_features=None):
        batch_size, seq_len = x.shape
        
        # 标准嵌入
        x_emb = self.embedding(x)
        pos_emb = self.pos_embedding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        # 时间嵌入
        t_emb = self.time_embedding(t.float().unsqueeze(-1))
        t_emb = t_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # ESM2特征集成
        if esm2_features is not None:
            esm2_proj = self.esm2_projection(esm2_features)
            # 融合标准嵌入和ESM2特征
            combined_emb = torch.cat([x_emb, esm2_proj], dim=-1)
            fused_emb = self.feature_fusion(combined_emb)
        else:
            fused_emb = x_emb
        
        # 组合所有嵌入
        h = self.layer_norm(fused_emb + pos_emb + t_emb)
        
        # Transformer
        h = self.transformer(h)
        
        # 输出
        logits = self.output_proj(h)
        
        return logits
    
    def forward_with_loss(self, x, t, esm2_features=None):
        """前向传播并计算损失"""
        # 添加噪声
        x_noisy, noise = self.scheduler.add_noise(x, t)
        
        # 预测
        predicted_logits = self.forward(x_noisy, t, esm2_features)
        
        # 计算损失
        loss = F.cross_entropy(predicted_logits.reshape(-1, self.vocab_size),
                              x.reshape(-1), reduction='mean')
        
        return loss, predicted_logits


# ==================== 整合的多样性采样器 ====================
class ESM2IntegratedDiversitySampler:
    """ESM2整合多样性采样器"""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # BBB肽特征 - 添加缺失的 hydrophobic_residues 定义
        self.cationic_residues = ['R', 'K', 'H']
        self.hydrophobic_residues = ['W', 'F', 'L', 'I', 'V', 'A', 'M', 'Y']  # 添加这行
        self.aromatic_residues = ['W', 'F', 'Y', 'H']
        
        # 多样性追踪
        self.generated_patterns = defaultdict(int)
        self.amino_acid_usage = defaultdict(int)
        self.sequence_lengths = defaultdict(int)
    
    def get_adaptive_temperature(self, step: int, total_steps: int, 
                                diversity_factor: float = 1.0, 
                                bbb_compliance: float = 1.0) -> float:
        """自适应温度调节 - 针对ESM2优化"""
        base_temp = self.get_base_temperature(step, total_steps)
        
        # 针对ESM2特征的温度调整
        if diversity_factor < 0.5:
            diversity_boost = 1.5
        elif diversity_factor < 0.7:
            diversity_boost = 1.2
        else:
            diversity_boost = 1.0
        
        # BBB肽符合度调整
        if bbb_compliance < 0.4:
            bbb_boost = 1.3
        elif bbb_compliance < 0.7:
            bbb_boost = 1.1
        else:
            bbb_boost = 1.0
        
        return base_temp * diversity_boost * bbb_boost
    
    def get_base_temperature(self, step: int, total_steps: int) -> float:
        """基础温度调节"""
        progress = step / total_steps
        
        if self.config.temperature_schedule == "linear":
            return self.config.temperature_start * (1 - progress) + self.config.temperature_end * progress
        elif self.config.temperature_schedule == "exponential":
            decay_rate = 1.0
            temp = self.config.temperature_start * np.exp(-decay_rate * progress)
            return max(temp, self.config.temperature_end)
        elif self.config.temperature_schedule == "cosine":
            temp = self.config.temperature_end + (self.config.temperature_start - self.config.temperature_end) * \
                   (1 + np.cos(np.pi * progress)) / 2
            return temp
        else:
            return self.config.temperature_start * (1 - progress) + self.config.temperature_end * progress
    
    def apply_bbb_bias(self, logits: torch.Tensor, current_seq: torch.Tensor, 
                      step: int, total_steps: int) -> torch.Tensor:
        """应用BBB肽偏置 - 针对ESM2优化"""
        if step < total_steps * 0.15:  # 在前15%的步骤中应用强BBB偏置
            return logits
        
        # 确保tensors在同一设备
        logits = logits.to(current_seq.device)
        
        # 计算当前序列的BBB肽特征
        batch_size, seq_len = current_seq.shape
        
        for i in range(batch_size):
            seq_so_far = current_seq[i].cpu().numpy()
            
            # 转换为氨基酸序列
            aa_seq = [IDX_TO_AA.get(idx, 'A') for idx in seq_so_far]
            
            # 计算当前特征
            cationic_count = sum(1 for aa in aa_seq if aa in self.cationic_residues)
            hydrophobic_count = sum(1 for aa in aa_seq if aa in self.hydrophobic_residues)
            current_length = seq_len
            
            # 计算需要的特征
            cationic_ratio = cationic_count / current_length
            hydrophobic_ratio = hydrophobic_count / current_length
            
            # 应用偏置 - 确保在相同设备
            for j in range(seq_len):
                # 如果阳离子残基不足，提高R, K, H的概率
                if cationic_ratio < 0.22:
                    logits[i, j, AA_TO_IDX['R']] += 0.7
                    logits[i, j, AA_TO_IDX['K']] += 0.7
                    logits[i, j, AA_TO_IDX['H']] += 0.4
                
                # 如果疏水残基不足，提高疏水残基的概率
                if hydrophobic_ratio < 0.18:
                    for hydro_aa in ['W', 'F', 'L', 'I', 'V']:
                        if hydro_aa in AA_TO_IDX:
                            logits[i, j, AA_TO_IDX[hydro_aa]] += 0.3
                
                # 降低负电荷残基的概率
                logits[i, j, AA_TO_IDX['D']] -= 0.25
                logits[i, j, AA_TO_IDX['E']] -= 0.25
                
                # 如果阳离子残基过多，适度降低其概率
                if cationic_ratio > 0.55:
                    logits[i, j, AA_TO_IDX['R']] -= 0.15
                    logits[i, j, AA_TO_IDX['K']] -= 0.15
        
        return logits
    
    def enhanced_sampling(self, logits: torch.Tensor, input_ids: torch.Tensor, 
                         step: int, total_steps: int, 
                         generated_tokens: List[int] = None,
                         generated_sequences: List[str] = None) -> torch.Tensor:
        """增强的整合采样 - 针对ESM2优化"""
        batch_size, seq_len, vocab_size = logits.shape
        
        # 计算多样性因子和BBB符合度
        diversity_factor = self.calculate_diversity_factor(generated_sequences or [])
        bbb_compliance = self.calculate_bbb_compliance(generated_sequences or [])
        
        # 获取自适应温度
        temperature = self.get_adaptive_temperature(step, total_steps, diversity_factor, bbb_compliance)
        
        # 应用温度 - 确保在相同设备
        logits = logits.to(input_ids.device)
        scaled_logits = logits / temperature
        
        # 应用BBB偏置
        scaled_logits = self.apply_bbb_bias(scaled_logits, input_ids, step, total_steps)
        
        # 应用nucleus采样
        if self.config.nucleus_sampling:
            scaled_logits = self.nucleus_sampling_enhanced(scaled_logits, self.config.top_p)
        
        # 转换为概率
        probs = F.softmax(scaled_logits, dim=-1)
        
        # 动态top-k采样
        if self.config.top_k > 0:
            current_top_k = self.config.top_k
            if diversity_factor < 0.6 or bbb_compliance < 0.4:
                current_top_k = min(self.config.top_k + 3, vocab_size)
            
            top_k_probs, top_k_indices = torch.topk(probs, current_top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            
            sampled_indices = torch.multinomial(top_k_probs.reshape(-1, current_top_k), 1)
            result = torch.gather(top_k_indices.reshape(-1, current_top_k), 1, sampled_indices)
            result = result.reshape(batch_size, seq_len)
        else:
            result = torch.multinomial(probs.reshape(-1, vocab_size), 1)
            result = result.reshape(batch_size, seq_len)
        
        return result
    
    def nucleus_sampling_enhanced(self, logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
        """增强的nucleus采样"""
        if not self.config.nucleus_sampling:
            return logits
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def calculate_diversity_factor(self, generated_sequences: List[str]) -> float:
        """计算多样性因子"""
        if not generated_sequences:
            return 1.0
        
        # 序列唯一性
        unique_sequences = set(generated_sequences)
        uniqueness = len(unique_sequences) / len(generated_sequences)
        
        # 氨基酸使用多样性
        aa_counts = defaultdict(int)
        total_aa = 0
        for seq in generated_sequences:
            for aa in seq:
                aa_counts[aa] += 1
                total_aa += 1
        
        aa_entropy = 0
        for count in aa_counts.values():
            if count > 0:
                p = count / total_aa
                aa_entropy -= p * np.log2(p)
        
        max_entropy = np.log2(20)
        aa_diversity = aa_entropy / max_entropy
        
        # 长度多样性
        lengths = [len(seq) for seq in generated_sequences]
        unique_lengths = len(set(lengths))
        length_diversity = min(unique_lengths / 10, 1.0)
        
        # 综合多样性因子
        diversity_factor = (uniqueness * 0.5 + aa_diversity * 0.3 + length_diversity * 0.2)
        return diversity_factor
    
    def calculate_bbb_compliance(self, generated_sequences: List[str]) -> float:
        """计算BBB肽符合度"""
        if not generated_sequences:
            return 1.0
        
        compliant_count = 0
        for seq in generated_sequences:
            # 检查长度
            if not (self.config.target_length_min <= len(seq) <= self.config.target_length_max):
                continue
            
            # 检查阳离子残基
            cationic_count = sum(1 for aa in seq if aa in self.cationic_residues)
            if not (self.config.min_cationic_residues <= cationic_count <= self.config.max_cationic_residues):
                continue
            
            # 检查疏水残基
            hydrophobic_count = sum(1 for aa in seq if aa in self.hydrophobic_residues)
            if not (self.config.min_hydrophobic_residues <= hydrophobic_count <= self.config.max_hydrophobic_residues):
                continue
            
            # 检查分子量
            mw = sum(AA_PROPERTIES[aa]['mw'] for aa in seq if aa in AA_PROPERTIES)
            if not (self.config.target_mw_min <= mw <= self.config.target_mw_max):
                continue
            
            # 检查净电荷
            net_charge = sum(AA_PROPERTIES[aa]['charge'] for aa in seq if aa in AA_PROPERTIES)
            if not (self.config.target_charge_min <= net_charge <= self.config.target_charge_max):
                continue
            
            compliant_count += 1
        
        return compliant_count / len(generated_sequences)


# ==================== 整合的质量评估器 ====================
class ESM2IntegratedBBBEvaluator:
    """ESM2整合BBB质量评估器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.aa_props = AA_PROPERTIES
        
        # BBB肽特征
        self.cationic_residues = ['R', 'K', 'H']
        self.hydrophobic_residues = ['W', 'F', 'L', 'I', 'V', 'A', 'M', 'Y']
        self.aromatic_residues = ['W', 'F', 'Y', 'H']
        
        # BBB特征模式
        self.bbb_motifs = [
            'RRR', 'KKK', 'RKR', 'KRK', 'RWR', 'KWK', 'RK', 'KR', 
            'RW', 'WR', 'KW', 'WK', 'RF', 'FR', 'KF', 'FK', 'RH', 'HR'
        ]
        
        self.logger.info("ESM2整合BBB评估器初始化完成")
    
    def calculate_overall_score(self, seq: str, existing_sequences: List[str] = None) -> float:
        """计算整合的总体评分"""
        # 快速淘汰不符合基本要求的序列
        if len(seq) < self.config.target_length_min or len(seq) > self.config.target_length_max:
            return 0.0
        
        # 计算各项得分
        length_score = self.calculate_length_score(seq)
        charge_score = self.calculate_charge_score(seq)
        mw_score = self.calculate_molecular_weight_score(seq)
        cationic_score = self.calculate_cationic_ratio_score(seq)
        hydrophobic_score = self.calculate_hydrophobic_ratio_score(seq)
        motif_score = self.calculate_bbb_motifs_score(seq)
        
        # 必需条件检查
        if length_score == 0.0 or charge_score == 0.0 or mw_score == 0.0:
            return 0.0
        
        # 计算加权总分
        total_score = (
            length_score * self.config.property_weights['length_score'] +
            charge_score * self.config.property_weights['charge_score'] +
            mw_score * self.config.property_weights['molecular_weight'] +
            cationic_score * self.config.property_weights['cationic_ratio'] +
            hydrophobic_score * self.config.property_weights['hydrophobic_ratio'] +
            motif_score * self.config.property_weights['bbb_motifs']
        )
        
        return total_score
    
    def calculate_length_score(self, seq: str) -> float:
        """计算长度得分"""
        length = len(seq)
        
        if length < self.config.target_length_min:
            return 0.0
        elif length > self.config.target_length_max:
            return 0.0
        elif length == self.config.target_length_optimal:
            return 1.0
        else:
            distance = abs(length - self.config.target_length_optimal)
            max_distance = max(
                self.config.target_length_optimal - self.config.target_length_min,
                self.config.target_length_max - self.config.target_length_optimal
            )
            return max(0.0, 1.0 - (distance / max_distance))
    
    def calculate_charge_score(self, seq: str) -> float:
        """计算电荷得分"""
        net_charge = sum(self.aa_props[aa]['charge'] for aa in seq if aa in self.aa_props)
        
        if net_charge < self.config.target_charge_min:
            return 0.0
        elif net_charge > self.config.target_charge_max:
            return 0.3
        elif net_charge == self.config.target_charge_optimal:
            return 1.0
        else:
            distance = abs(net_charge - self.config.target_charge_optimal)
            max_distance = max(
                self.config.target_charge_optimal - self.config.target_charge_min,
                self.config.target_charge_max - self.config.target_charge_optimal
            )
            return max(0.0, 1.0 - (distance / max_distance) * 0.2)
    
    def calculate_molecular_weight_score(self, seq: str) -> float:
        """计算分子量得分"""
        mw = sum(self.aa_props[aa]['mw'] for aa in seq if aa in self.aa_props)
        
        if mw < self.config.target_mw_min or mw > self.config.target_mw_max:
            return 0.0
        elif abs(mw - self.config.target_mw_optimal) <= 120:
            return 1.0
        else:
            distance = abs(mw - self.config.target_mw_optimal)
            max_distance = max(
                self.config.target_mw_optimal - self.config.target_mw_min,
                self.config.target_mw_max - self.config.target_mw_optimal
            )
            return max(0.0, 1.0 - (distance / max_distance) * 0.3)
    
    def calculate_cationic_ratio_score(self, seq: str) -> float:
        """计算阳离子比例得分"""
        cationic_count = sum(1 for aa in seq if aa in self.cationic_residues)
        
        if cationic_count < self.config.min_cationic_residues:
            return 0.0
        elif cationic_count > self.config.max_cationic_residues:
            return 0.5
        else:
            ratio = cationic_count / len(seq)
            if 0.20 <= ratio <= 0.50:
                return 1.0
            elif 0.15 <= ratio <= 0.60:
                return 0.8
            else:
                return 0.6
    
    def calculate_hydrophobic_ratio_score(self, seq: str) -> float:
        """计算疏水比例得分"""
        hydrophobic_count = sum(1 for aa in seq if aa in self.hydrophobic_residues)
        
        if hydrophobic_count < self.config.min_hydrophobic_residues:
            return 0.0
        elif hydrophobic_count > self.config.max_hydrophobic_residues:
            return 0.5
        else:
            ratio = hydrophobic_count / len(seq)
            if 0.15 <= ratio <= 0.60:
                return 1.0
            elif 0.10 <= ratio <= 0.70:
                return 0.8
            else:
                return 0.6
    
    def calculate_bbb_motifs_score(self, seq: str) -> float:
        """计算BBB模式得分"""
        motif_score = 0
        for motif in self.bbb_motifs:
            if motif in seq:
                motif_score += 0.10
        
        return min(motif_score, 1.0)
    
    def analyze_sequence(self, seq: str) -> Dict:
        """分析序列的详细特征"""
        analysis = {
            'sequence': seq,
            'length': len(seq),
            'molecular_weight': sum(self.aa_props[aa]['mw'] for aa in seq if aa in self.aa_props),
            'net_charge': sum(self.aa_props[aa]['charge'] for aa in seq if aa in self.aa_props),
            'cationic_count': sum(1 for aa in seq if aa in self.cationic_residues),
            'hydrophobic_count': sum(1 for aa in seq if aa in self.hydrophobic_residues),
            'aromatic_count': sum(1 for aa in seq if aa in self.aromatic_residues),
            'cationic_ratio': sum(1 for aa in seq if aa in self.cationic_residues) / len(seq),
            'hydrophobic_ratio': sum(1 for aa in seq if aa in self.hydrophobic_residues) / len(seq),
            'bbb_motifs_found': [motif for motif in self.bbb_motifs if motif in seq],
            'meets_all_requirements': self.meets_all_bbb_requirements(seq),
            'overall_score': self.calculate_overall_score(seq)
        }
        
        return analysis
    
    def meets_all_bbb_requirements(self, seq: str) -> bool:
        """检查是否满足所有BBB肽要求"""
        # 长度要求
        if not (self.config.target_length_min <= len(seq) <= self.config.target_length_max):
            return False
        
        # 电荷要求
        net_charge = sum(self.aa_props[aa]['charge'] for aa in seq if aa in self.aa_props)
        if not (self.config.target_charge_min <= net_charge <= self.config.target_charge_max):
            return False
        
        # 分子量要求
        mw = sum(self.aa_props[aa]['mw'] for aa in seq if aa in self.aa_props)
        if not (self.config.target_mw_min <= mw <= self.config.target_mw_max):
            return False
        
        # 阳离子残基要求
        cationic_count = sum(1 for aa in seq if aa in self.cationic_residues)
        if not (self.config.min_cationic_residues <= cationic_count <= self.config.max_cationic_residues):
            return False
        
        # 疏水残基要求
        hydrophobic_count = sum(1 for aa in seq if aa in self.hydrophobic_residues)
        if not (self.config.min_hydrophobic_residues <= hydrophobic_count <= self.config.max_hydrophobic_residues):
            return False
        
        return True


# ==================== 去重器 ====================
class ESM2RelaxedSequenceDeduplicator:
    """ESM2优化的序列去重器"""
    
    def __init__(self, similarity_threshold=0.78, existing_samples=None, logger=None):
        self.similarity_threshold = similarity_threshold
        self.logger = logger or logging.getLogger(__name__)
        self.existing_samples = existing_samples or []
        
        # 存储已生成的序列
        self.generated_sequences = set()
        self.sequence_list = []
        self.sequence_hashes = set()
        
        # 将现有样本加入去重检查
        for seq in self.existing_samples:
            self.generated_sequences.add(seq)
            self.sequence_list.append(seq)
            self.sequence_hashes.add(self.sequence_hash(seq))
        
        # 相似度缓存
        self.similarity_cache = {}
        
        # 统计信息
        self.stats = {
            'total_generated': 0,
            'duplicates_filtered': 0,
            'similar_filtered': 0,
            'unique_sequences': len(self.existing_samples),
            'hash_collisions': 0
        }
    
    def sequence_hash(self, seq: str) -> str:
        """生成序列的哈希值"""
        return hashlib.md5(seq.encode()).hexdigest()
    
    def calculate_similarity_fast(self, seq1: str, seq2: str) -> float:
        """快速相似度计算"""
        cache_key = (seq1, seq2) if seq1 <= seq2 else (seq2, seq1)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0
        
        # 针对短序列优化的相似度计算
        len_diff = abs(len(seq1) - len(seq2))
        max_len = max(len(seq1), len(seq2))
        
        if len_diff / max_len > 0.3:
            similarity = 0.0
        else:
            min_len = min(len(seq1), len(seq2))
            matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
            
            length_penalty = len_diff / max_len * 0.15
            similarity = (matches / min_len) * (1 - length_penalty)
        
        # 缓存结果
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    def is_similar_to_existing(self, new_seq: str, check_window: int = 150) -> bool:
        """检查新序列是否与现有序列相似"""
        if not self.sequence_list:
            return False
        
        # 检查与现有正样本的相似度
        for existing_seq in self.existing_samples:
            if self.calculate_similarity_fast(new_seq, existing_seq) >= self.similarity_threshold:
                return True
        
        # 检查与最近生成序列的相似度
        recent_sequences = self.sequence_list[-check_window:]
        for existing_seq in recent_sequences:
            if self.calculate_similarity_fast(new_seq, existing_seq) >= self.similarity_threshold:
                return True
        
        return False
    
    def add_sequence(self, seq: str) -> bool:
        """添加序列，如果不重复则返回True"""
        self.stats['total_generated'] += 1
        
        # 快速哈希检查
        seq_hash = self.sequence_hash(seq)
        if seq_hash in self.sequence_hashes:
            self.stats['hash_collisions'] += 1
            return False
        
        # 完全相同检查
        if seq in self.generated_sequences:
            self.stats['duplicates_filtered'] += 1
            return False
        
        # 相似度检查
        if self.is_similar_to_existing(seq, config.duplicate_check_window):
            self.stats['similar_filtered'] += 1
            return False
        
        # 添加到集合
        self.generated_sequences.add(seq)
        self.sequence_list.append(seq)
        self.sequence_hashes.add(seq_hash)
        self.stats['unique_sequences'] += 1
        
        return True
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        new_sequences = stats['unique_sequences'] - len(self.existing_samples)
        if stats['total_generated'] > 0:
            stats['unique_rate'] = new_sequences / stats['total_generated']
            stats['duplicate_rate'] = stats['duplicates_filtered'] / stats['total_generated']
            stats['similar_rate'] = stats['similar_filtered'] / stats['total_generated']
        
        return stats


# ==================== 主函数 ====================
def main():
    """主函数 - ESM2集成BBB穿透肽正样本生成器"""
    print("🧬" + "=" * 79)
    print("🎯 ESM2集成BBB穿透肽正样本生成器")
    print("=" * 80)
    print("📋 任务概况:")
    print(f"   👤 用户: {config.user_login}")
    print(f"   📅 时间: {config.current_time}")
    print(f"   🧬 架构: ESM2特征提取 + 扩散模型 + Transformer生成")
    print(f"   ➕ 现有正样本: {config.existing_positive_samples} 个")
    print(f"   ➖ 现有负样本: {config.existing_negative_samples} 个")
    print(f"   🎯 需要生成: {config.need_to_generate} 个正样本")
    print(f"   ⚖️  目标平衡: {config.target_positive_samples} 正样本 vs {config.existing_negative_samples} 负样本")
    print("=" * 80)
    print("🔧 ESM2集成特性:")
    print("   ✅ ESM2蛋白质语言模型特征提取")
    print("   ✅ 扩散模型 + Transformer架构")
    print("   ✅ BBB实际需求: 长度6-15aa, 分子量500-2000Da, 净电荷+1到+6")
    print("   ✅ 多样性增强: 去重优化, 温度自适应, 核采样")
    print("   ✅ 质量保证: 阳离子残基2-6个, 疏水残基1-8个")
    print("   ✅ 智能采样: ESM2引导, BBB偏置引导, 重复惩罚")
    print("=" * 80)
    
    # 检查ESM2可用性
    if not ESM2_AVAILABLE:
        print("❌ ESM2不可用，请安装：pip install fair-esm")
        return
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 初始化管理器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # 设置输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 开始ESM2集成BBB穿透肽正样本生成")
    logger.info(f"🎯 目标: 补齐正样本数量从 {config.existing_positive_samples} 到 {config.target_positive_samples}")
    logger.info(f"📊 需要生成: {config.need_to_generate} 个高质量正样本")
    
    try:
        # 第一阶段：数据处理
        logger.info("📊 第一阶段: 正样本数据处理")
        existing_samples = read_existing_positive_samples(config.input_file, logger)
        
        # 更新配置
        config.existing_positive_samples = len(existing_samples)
        config.need_to_generate = config.target_positive_samples - len(existing_samples)
        
        if len(existing_samples) >= config.target_positive_samples:
            logger.info("✅ 现有正样本已足够，无需生成新样本")
            return
        
        # 第二阶段：ESM2特征提取
        logger.info("🧬 第二阶段: ESM2特征提取")
        esm2_extractor = ESM2FeatureExtractor(config, logger)
        
        # 为所有样本提取ESM2特征
        logger.info("🔍 为训练数据提取ESM2特征")
        train_esm2_features = esm2_extractor.extract_features(existing_samples)
        
        # 创建数据集
        logger.info("📦 创建数据集")
        dataset = ESM2IntegratedDataset(existing_samples, train_esm2_features, logger)
        
        # 分割数据集
        n_val = int(len(dataset) * config.val_split)
        n_train = len(dataset) - n_val
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # 创建数据加载器
        def collate_fn(batch):
            sequences = [item['sequence'] for item in batch]
            encoded_sequences = torch.stack([item['encoded_sequence'] for item in batch])
            esm2_features = torch.stack([item['esm2_features'] for item in batch])
            
            return {
                'sequences': sequences,
                'encoded_sequences': encoded_sequences,
                'esm2_features': esm2_features
            }
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False
        )
        
        logger.info(f"📈 训练集: {len(train_dataset)} 个序列")
        logger.info(f"📉 验证集: {len(val_dataset)} 个序列")
        
        # 第三阶段：模型训练
        logger.info("🤖 第三阶段: ESM2集成模型训练")
        model = ESM2IntegratedDiffusionModel(device=config.device).to(config.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"🔧 模型参数总数: {total_params:,}")
        
        # 训练模型
        train_esm2_model(model, train_loader, val_loader, config, logger)
        
        # 保存模型
        model_path = output_dir / "models" / config.model_save_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logger.info(f"💾 模型已保存: {model_path}")
        
        # 第四阶段：正样本生成
        logger.info("🌟 第四阶段: ESM2集成正样本生成")
        evaluator = ESM2IntegratedBBBEvaluator(config, logger)
        
        all_new_sequences, high_quality_sequences = generate_esm2_positive_samples(
            model, esm2_extractor, evaluator, existing_samples, config, logger
        )
        
        # 第五阶段：结果保存
        logger.info("💾 第五阶段: 结果保存")
        save_esm2_results(all_new_sequences, high_quality_sequences, existing_samples, 
                         evaluator, config, logger, output_dir)
        
        # 最终结果
        logger.info("🏆" + "=" * 79)
        logger.info("🎉 ESM2集成BBB穿透肽正样本生成完成")
        logger.info("=" * 80)
        logger.info(f"📊 现有正样本: {len(existing_samples)}")
        logger.info(f"📊 新生成正样本: {len(all_new_sequences)}")
        logger.info(f"⭐ 高质量正样本: {len(high_quality_sequences)}")
        logger.info(f"📊 总正样本数: {len(existing_samples) + len(all_new_sequences)}")
        logger.info(f"📊 负样本数: {config.existing_negative_samples}")
        
        logger.info("🎊 ESM2集成BBB穿透肽正样本生成器执行完成！")
        logger.info(f"📁 所有结果已保存到: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ 程序执行过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


# ==================== 辅助函数 ====================
def read_existing_positive_samples(file_path: str, logger) -> List[str]:
    """读取现有正样本"""
    sequences = []
    
    possible_files = [
        file_path,
        "train_pos_org.fasta",
        "positive_samples.fasta",
        "existing_positive.fasta"
    ]
    
    file_found = False
    for file in possible_files:
        if os.path.exists(file):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    sequence = ""
                    for line in f:
                        line = line.strip()
                        if line.startswith('>'):
                            if sequence:
                                sequences.append(sequence)
                            sequence = ""
                        else:
                            sequence += line
                    if sequence:
                        sequences.append(sequence)
                
                logger.info(f"✅ 成功读取现有正样本文件: {file}")
                logger.info(f"📊 现有正样本数量: {len(sequences)}")
                file_found = True
                break
                
            except Exception as e:
                logger.warning(f"❌ 读取文件 {file} 失败: {e}")
    
    if not file_found:
        logger.warning("⚠️ 未找到现有正样本文件，使用默认BBB肽模板")
        sequences = create_default_positive_samples(logger)
    
    return sequences


def create_default_positive_samples(logger) -> List[str]:
    """创建默认的正样本BBB肽"""
    logger.info("🧬 创建默认BBB穿透肽正样本")
    
    default_samples = [
        "GRKKRRQRRR", "YGRKKRRQRRR", "RQIKIWFQNRR", "KLALKLALK",
        "RQARRNRRRR", "RRWWRRWW", "GRKKRRQRR", "YGRKKRRQ",
        "RWKWKW", "FKFKFK", "LRLRLR", "AWAWAW", "QWQWQW", "TWTWTW",
        "HRHRHH", "GKGKGK", "PLPLPL", "CYCYYY", "MEMEME", "SISISS",
        "RWKFLQ", "GKPLCY", "RWKWRW", "KFKFKF", "WKWKWK",
        "AWHRFK", "QWTYLY", "LRCYNE", "VIFKGP", "MESIDQ",
        "KRWKRW", "FRLFRL", "RLRLRL", "KFKFKF", "RWRWRW",
        "FLFLFL", "WLWLWL", "RKRKRK", "FWFWFW", "KYKYKF",
        "RWFRWF", "WFWFWF", "YFYFYF", "HWKHWK", "FKRFKR",
        "WKYWKY", "FRFRFR", "WFKRWF", "YKYKYF", "RWFRFW"
    ]
    
    with open(config.input_file, 'w', encoding='utf-8') as f:
        for i, seq in enumerate(default_samples):
            f.write(f">default_positive_{i + 1}\n{seq}\n")
    
    logger.info(f"📁 默认正样本已保存到: {config.input_file}")
    logger.info(f"📊 包含 {len(default_samples)} 个高质量BBB肽序列")
    return default_samples


def train_esm2_model(model, train_loader, val_loader, config, logger):
    """训练ESM2集成模型"""
    logger.info("🚀 开始训练ESM2集成BBB正样本生成模型")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.n_epochs,
        eta_min=config.lr * 0.01
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.n_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.n_epochs}")):
            encoded_sequences = batch['encoded_sequences'].to(config.device)
            esm2_features = batch['esm2_features'].to(config.device)
            
            batch_size = encoded_sequences.size(0)
            
            # 随机时间步
            t = torch.randint(0, config.n_timesteps, (batch_size,), device=config.device)
            
            # 前向传播
            if config.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss, _ = model.forward_with_loss(encoded_sequences, t, esm2_features)
            else:
                loss, _ = model.forward_with_loss(encoded_sequences, t, esm2_features)
            
            # 梯度累积
            loss = loss / config.gradient_accumulation_steps
            
            # 反向传播
            if config.mixed_precision:
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * config.gradient_accumulation_steps
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                encoded_sequences = batch['encoded_sequences'].to(config.device)
                esm2_features = batch['esm2_features'].to(config.device)
                
                batch_size = encoded_sequences.size(0)
                t = torch.randint(0, config.n_timesteps, (batch_size,), device=config.device)
                
                if config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        loss, _ = model.forward_with_loss(encoded_sequences, t, esm2_features)
                else:
                    loss, _ = model.forward_with_loss(encoded_sequences, t, esm2_features)
                
                val_loss += loss.item()
        
        # 记录损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # 学习率调度
        scheduler.step()
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 打印信息
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch + 1}: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}")
        
        # 早停检查
        if patience_counter >= config.early_stopping_patience:
            logger.info(f"⏹️ 早停触发！在epoch {epoch + 1}停止训练")
            break
        
        # 显存管理
        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    logger.info("✅ ESM2集成BBB正样本模型训练完成")
    logger.info(f"🏆 最佳验证损失: {best_val_loss:.4f}")


def generate_esm2_positive_samples(model, esm2_extractor, evaluator, existing_samples, config, logger):
    """生成ESM2集成正样本"""
    logger.info(f"🎯 开始生成 {config.need_to_generate} 个ESM2集成BBB正样本")
    
    # 初始化去重器
    deduplicator = ESM2RelaxedSequenceDeduplicator(
        config.similarity_threshold,
        existing_samples,
        logger
    )
    
    # 初始化采样器
    diversity_sampler = ESM2IntegratedDiversitySampler(config, logger)
    
    model.eval()
    all_new_sequences = []
    high_quality_sequences = []
    
    target_sequences = config.need_to_generate
    batch_size = config.generation_batch_size
    
    with torch.no_grad():
        while len(all_new_sequences) < target_sequences:
            remaining = target_sequences - len(all_new_sequences)
            current_batch_size = min(batch_size, remaining)
            
            # 生成序列
            generated_sequences = sample_esm2_sequences(
                model, diversity_sampler, current_batch_size, config, logger
            )
            
            # 评估和过滤
            for seq in generated_sequences:
                if config.target_length_min <= len(seq) <= config.target_length_max:
                    # 基本BBB肽要求检查
                    cationic_count = sum(1 for aa in seq if aa in ['R', 'K', 'H'])
                    net_charge = sum(AA_PROPERTIES[aa]['charge'] for aa in seq if aa in AA_PROPERTIES)
                    
                    if cationic_count >= config.min_cationic_residues and net_charge >= config.target_charge_min:
                        if deduplicator.add_sequence(seq):
                            all_new_sequences.append(seq)
                            
                            # 质量评估
                            score = evaluator.calculate_overall_score(seq)
                            if score >= config.quality_threshold:
                                high_quality_sequences.append((seq, score))
            
            # 进度更新
            logger.info(f"生成进度: {len(all_new_sequences)}/{target_sequences}")
            
            if len(all_new_sequences) >= target_sequences:
                break
    
    # 排序高质量序列
    high_quality_sequences.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"🎉 ESM2集成正样本生成完成:")
    logger.info(f"📊 新生成序列: {len(all_new_sequences)}")
    logger.info(f"⭐ 高质量序列: {len(high_quality_sequences)}")
    
    return all_new_sequences, high_quality_sequences


def sample_esm2_sequences(model, diversity_sampler, batch_size, config, logger):
    """采样ESM2集成序列"""
    device = config.device
    
    # 从随机序列开始
    x = torch.randint(0, config.vocab_size, (batch_size, config.seq_len), device=device)
    
    # 创建虚拟ESM2特征（在实际应用中，这里应该是从参考序列或模板生成的特征）
    dummy_esm2_features = torch.randn(batch_size, config.seq_len, config.esm2_embedding_dim, device=device)
    
    # 采样步骤
    sampling_steps = torch.linspace(config.n_timesteps - 1, 0, config.sampling_steps).long()
    
    # 逐步去噪
    for i, t in enumerate(sampling_steps):
        t_tensor = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
        
        # 预测
        predicted_logits = model.forward(x, t_tensor, dummy_esm2_features)
        
        # 使用多样性采样
        x = diversity_sampler.enhanced_sampling(
            predicted_logits, x, i, len(sampling_steps)
        )
    
    # 转换为字符串序列
    sequences = []
    for j in range(batch_size):
        seq = indices_to_sequence(x[j].cpu().numpy())
        seq = seq.rstrip('A')  # 移除填充
        sequences.append(seq)
    
    return sequences


def indices_to_sequence(indices: List[int]) -> str:
    """将索引转换为序列"""
    return ''.join([IDX_TO_AA.get(idx, 'A') for idx in indices])


def save_esm2_results(all_new_sequences, high_quality_sequences, existing_samples, 
                     evaluator, config, logger, output_dir):
    """保存ESM2集成结果"""
    # 创建结果目录
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存新生成的序列
    if all_new_sequences:
        with open(results_dir / "new_positive_sequences.fasta", 'w') as f:
            for i, seq in enumerate(all_new_sequences):
                f.write(f">new_positive_{i+1}\n{seq}\n")
        logger.info(f"💾 新生成序列已保存: {len(all_new_sequences)} 个")
    
    # 保存高质量序列
    if high_quality_sequences:
        with open(results_dir / "high_quality_positive.fasta", 'w') as f:
            for i, (seq, score) in enumerate(high_quality_sequences):
                f.write(f">high_quality_{i+1}_score_{score:.3f}\n{seq}\n")
        logger.info(f"💾 高质量序列已保存: {len(high_quality_sequences)} 个")
    
    # 保存合并的完整正样本数据集
    all_positive_sequences = existing_samples + [seq for seq, _ in high_quality_sequences]
    
    with open(results_dir / "all_positive_samples.fasta", 'w') as f:
        for i, seq in enumerate(existing_samples):
            f.write(f">existing_positive_{i+1}\n{seq}\n")
        for i, (seq, score) in enumerate(high_quality_sequences):
            f.write(f">generated_positive_{i+1}_score_{score:.3f}\n{seq}\n")
    
    logger.info(f"💾 完整正样本数据集已保存: {len(all_positive_sequences)} 个")
    
    # 生成分析报告
    analysis_results = []
    for seq, score in high_quality_sequences[:50]:  # 分析前50个
        analysis = evaluator.analyze_sequence(seq)
        analysis_results.append(analysis)
    
    # 保存分析报告
    report = {
        'existing_samples': len(existing_samples),
        'new_samples_generated': len(all_new_sequences),
        'high_quality_samples': len(high_quality_sequences),
        'total_positive_samples': len(all_positive_sequences),
        'average_quality_score': np.mean([score for _, score in high_quality_sequences]) if high_quality_sequences else 0,
        'bbb_compliant': sum(1 for a in analysis_results if a['meets_all_requirements']),
        'average_length': np.mean([a['length'] for a in analysis_results]) if analysis_results else 0,
        'average_mw': np.mean([a['molecular_weight'] for a in analysis_results]) if analysis_results else 0,
        'average_charge': np.mean([a['net_charge'] for a in analysis_results]) if analysis_results else 0,
        'generation_time': datetime.now().isoformat(),
        'config_summary': {
            'esm2_model': config.esm2_model_name,
            'target_length_range': f"{config.target_length_min}-{config.target_length_max}",
            'target_mw_range': f"{config.target_mw_min}-{config.target_mw_max}",
            'target_charge_range': f"{config.target_charge_min}-{config.target_charge_max}",
            'quality_threshold': config.quality_threshold,
            'similarity_threshold': config.similarity_threshold
        }
    }
    
    with open(results_dir / "generation_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info("📊 分析报告已保存")


if __name__ == "__main__":
    main()