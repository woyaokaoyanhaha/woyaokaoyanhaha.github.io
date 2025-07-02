#!/usr/bin/env python3
"""
蛋白质-化合物分离分批特征提取脚本 - 修改版
作者: woyaokaoyanhaha
版本: 18.1 (集成Chemprop预训练模型)
日期: 2025-07-02
"""

import csv
import os
import json
import time
import warnings
import gc
from pathlib import Path
from collections import defaultdict
import traceback
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
import torch
from torch.cuda.amp import autocast
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# 安装Chemprop
try:
    from chemprop.models import MoleculeModel
    from chemprop.data import MoleculeDataset
    from chemprop.features import get_features_generator
except ImportError:
    print("❌ Chemprop 未安装，请运行：pip install chemprop")
    sys.exit(1)

# ============================================================================
# 用户配置参数区域 - 在此修改所有参数
# ============================================================================

# 输入文件路径 (必须设置)
INPUT_CSV_FILE = "test2.csv"  # 修改为您的输入文件路径

# 输出目录设置 (可选，留空则自动生成)
CUSTOM_OUTPUT_DIR = "Q13133和EHDPP"  # 例如: "./my_output" 或留空 ""

# 恢复运行设置 (可选)
RESUME_FROM_DIR = ""  # 例如: "./previous_run_dir" 或留空 ""

# 运行模式设置
TEST_MODE = False  # True表示仅测试，False表示正常运行
DEBUG_MODE = True  # True表示显示详细调试信息

# 分批处理配置
COMPOUND_BATCH_SIZE = 1000000  # 每批处理的化合物数量
MEMORY_LIMIT_MB = 6144  # 内存限制(MB)，超过时强制垃圾回收

# 特征提取配置
EXTRACT_PROTEIN_FEATURES = True  # True表示提取蛋白质特征
EXTRACT_COMPOUND_FEATURES = True  # True表示提取化合物特征
SAVE_ORIGINAL_DATA = False  # False表示不保存sequence和smiles
GNN_FEATURE_DIM = 128  # Chemprop默认输出300维，需调整至128-256

# ============================================================================
# 特征维度配置
# ============================================================================

# 蛋白质特征维度
AAC_DIM = 20  # 氨基酸组成特征维度
DPC_DIM = 400  # 二肽组成特征维度
PROTEIN_ENHANCED_DIM = 10  # 增强特征维度
PROTEIN_TOTAL_DIM = AAC_DIM + DPC_DIM + PROTEIN_ENHANCED_DIM  # 430

# 化合物特征维度
GNN_DIM = GNN_FEATURE_DIM  # Chemprop GNN特征维度
ECFP4_BITS = 2048  # ECFP4指纹位数
MACCS_BITS = 167  # MACCS指纹位数
DTI_SUBSTRUCTURES_COUNT = 26  # DTI重要子结构数
PHARMACOPHORE_COUNT = 6  # 药效团特征数
RDKIT_DESCRIPTOR_COUNT = 50  # RDKit描述符数量
COMPOUND_TOTAL_DIM = (GNN_DIM + ECFP4_BITS + MACCS_BITS +
                     DTI_SUBSTRUCTURES_COUNT * 2 + PHARMACOPHORE_COUNT +
                     RDKIT_DESCRIPTOR_COUNT)  # 2521-2649

# 总特征维度
TOTAL_FEATURE_DIM = PROTEIN_TOTAL_DIM + COMPOUND_TOTAL_DIM

# DTI重要子结构（SMARTS格式）
DTI_IMPORTANT_SUBSTRUCTURES = {
    'benzene_ring': 'c1ccccc1',
    'pyridine': 'c1ccncc1',
    'pyrimidine': 'c1cncnc1',
    'imidazole': 'c1cnc[nH]1',
    'indole': 'c1ccc2c(c1)cc[nH]2',
    'quinoline': 'c1ccc2c(c1)cccn2',
    'hydroxyl': '[OH]',
    'primary_amine': '[NH2]',
    'secondary_amine': '[NH1]',
    'carboxyl': 'C(=O)[OH]',
    'amide': '[NX3][CX3](=[OX1])[#6]',
    'carbonyl': '[CX3]=[OX1]',
    'sulfonamide': '[SX4](=[OX1])(=[OX1])([NX3])[#6]',
    'urea': '[NX3][CX3](=[OX1])[NX3]',
    'ester': '[#6][CX3](=O)[OX2H0][#6]',
    'ether': '[OD2]([#6])[#6]',
    'morpholine': 'C1COCCN1',
    'piperidine': 'C1CCNCC1',
    'piperazine': 'C1CNCCN1',
    'pyrrolidine': 'C1CCNC1',
    'thiophene': 'c1ccsc1',
    'furan': 'c1ccoc1',
    'aromatic_hydroxyl': 'c[OH]',
    'aromatic_amine': 'c[NH2]',
    'beta_lactam': '[C@H]1[C@@H](N1[*])[*]',
    'guanidine': '[NX3][CX3](=[NX3+])[NX3]'
}

# RDKit描述符选择
RDKIT_DESCRIPTORS = [
    'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
    'NumRotatableBonds', 'NumAromaticRings', 'NumAliphaticRings',
    'NumSaturatedRings', 'NumHeteroatoms', 'FractionCSP3',
    'MaxPartialCharge', 'MinPartialCharge', 'NumValenceElectrons',
    'NumRadicalElectrons', 'LabuteASA', 'BalabanJ', 'BertzCT',
    'Chi0', 'Chi1', 'Chi2n', 'Chi3n', 'Chi4n', 'HallKierAlpha',
    'Kappa1', 'Kappa2', 'Kappa3', 'PEOE_VSA1', 'PEOE_VSA2',
    'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6',
    'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
    'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4',
    'SlogP_VSA5', 'EState_VSA1', 'EState_VSA2', 'EState_VSA3',
    'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4',
    'VSA_EState5'
]

# ============================================================================

# 检查和导入依赖库
def check_dependencies():
    """检查并导入必要的依赖库"""
    print("正在检查依赖库...")
    dependencies_met = True

    for module, name, install_name in [
        (np, 'numpy', 'numpy'),
        (pd, 'pandas', 'pandas'),
        (Chem, 'rdkit', 'rdkit'),
        (torch, 'pytorch', 'torch'),
        (MoleculeModel, 'chemprop', 'chemprop')
    ]:
        try:
            print(f"✅ {name} {module.__version__ if hasattr(module, '__version__') else ''}")
        except:
            print(f"❌ {name} 未安装")
            dependencies_met = False

    if PSUTIL_AVAILABLE:
        print(f"✅ psutil {psutil.__version__}")
    else:
        print("⚠️ psutil 未安装，将无法监控内存使用")

    if not dependencies_met:
        print("\n请安装缺失的依赖库:")
        print("pip install numpy pandas rdkit torch chemprop")
        sys.exit(1)

    print("🎉 所有依赖库检查完成")
    return True

# 检查依赖库
if not check_dependencies():
    sys.exit(1)

# 列名映射配置
COLUMN_MAPPING = {
    'protein_accession': ['Protein_Accession', 'ProteinAccession', 'Accession', 'Protein_ID', 'ProteinID'],
    'sequence': ['Sequence', 'Protein_Sequence', 'ProteinSequence', 'Seq'],
    'compound_cid': ['Compound_CID', 'CompoundCID', 'CID', 'Compound_ID', 'CompoundID'],
    'smile': ['Smile', 'SMILES', 'Canonical_SMILES', 'CanonicalSMILES']
}

def get_memory_usage():
    """获取当前内存使用情况"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # MB
    return 0

def force_garbage_collection():
    """强制垃圾回收"""
    gc.collect()
    if PSUTIL_AVAILABLE:
        return get_memory_usage()
    return 0

def detect_column_names(csv_file):
    """自动检测CSV文件的列名"""
    print(f"正在检测CSV文件的列名: {csv_file}")
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
    except Exception as e:
        print(f"❌ 无法读取CSV文件: {e}")
        return None, None

    detected_columns = {}
    for field_type, possible_names in COLUMN_MAPPING.items():
        detected_columns[field_type] = None
        for possible_name in possible_names:
            if possible_name in header:
                detected_columns[field_type] = possible_name
                break

    print("检测到的列名映射:")
    for field_type, column_name in detected_columns.items():
        status = "✅" if column_name else "❌"
        print(f"  {status} {field_type}: {column_name or '未找到'}")
    return detected_columns, header

def analyze_data_distribution(csv_file, detected_columns):
    """分析数据分布情况"""
    print("\n🔍 正在分析数据分布...")
    protein_data = {}
    compound_data = {}
    total_rows = 0

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                accession = row[detected_columns['protein_accession']].strip()
                sequence = row[detected_columns['sequence']].strip().upper()
                compound_cid = row[detected_columns['compound_cid']].strip()
                smile = row[detected_columns['smile']].strip()

                if accession in protein_data:
                    if protein_data[accession] != sequence:
                        print(f"⚠️ 警告: 蛋白质 {accession} 对应多个不同序列!")
                else:
                    protein_data[accession] = sequence

                if compound_cid not in compound_data:
                    compound_data[compound_cid] = smile

    except Exception as e:
        print(f"❌ 数据分析失败: {e}")
        return None, None, 0

    print(f"📊 数据分布统计:")
    print(f"  总记录数: {total_rows}")
    print(f"  唯一蛋白质数: {len(protein_data)}")
    print(f"  唯一化合物数: {len(compound_data)}")
    return protein_data, compound_data, total_rows

def get_output_dir_name(input_csv_path, custom_output=None):
    """生成输出目录名"""
    if custom_output:
        return custom_output
    filename = os.path.basename(input_csv_path)
    basename = os.path.splitext(filename)[0]
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    safe_basename = "".join(c for c in basename if c.isalnum() or c in ('-', '_')).rstrip()
    if len(safe_basename) < 3:
        safe_basename = "protein_compound_features"
    return f"./{safe_basename}_separated_batch_{timestamp}"

class ProgressManager:
    """进度管理器"""
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.progress_file = os.path.join(work_dir, "progress.json")
        self.progress = self.load_progress()

    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 进度文件读取失败: {e}")
        return {
            'start_time': time.time(),
            'total_records': 0,
            'unique_proteins': 0,
            'unique_compounds': 0,
            'protein_extraction_completed': False,
            'compound_batches_total': 0,
            'compound_batches_completed': 0,
            'compound_batch_files': [],
            'protein_file': None,
            'completed': False,
            'last_update': time.time(),
            'memory_peak_mb': 0,
            'processing_errors': []
        }

    def save_progress(self):
        self.progress['last_update'] = time.time()
        current_memory = get_memory_usage()
        if current_memory > self.progress['memory_peak_mb']:
            self.progress['memory_peak_mb'] = current_memory
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            print(f"⚠️ 进度保存失败: {e}")

    def update_totals(self, total_records, unique_proteins, unique_compounds, compound_batches_total):
        self Metz['total_records'] = total_records
        self.progress['unique_proteins'] = unique_proteins
        self.progress['unique_compounds'] = unique_compounds
        self.progress['compound_batches_total'] = compound_batches_total
        self.save_progress()

    def mark_protein_completed(self, protein_file):
        self.progress['protein_extraction_completed'] = True
        self.progress['protein_file'] = protein_file
        self.save_progress()

    def mark_compound_batch_completed(self, batch_number, batch_file):
        self.progress['compound_batches_completed'] = max(self.progress['compound_batches_completed'], batch_number)
        if batch_file not in self.progress['compound_batch_files']:
            self.progress['compound_batch_files'].append(batch_file)
        self.save_progress()

    def add_error(self, error_msg):
        error_record = {
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(error_msg)
        }
        self.progress['processing_errors'].append(error_record)

    def mark_completed(self):
        self.progress['completed'] = True
        self.progress['end_time'] = time.time()
        self.save_progress()

    def get_progress_info(self):
        return {
            'protein_completed': self.progress['protein_extraction_completed'],
            'compound_batches': f"{self.progress['compound_batches_completed']}/{self.progress['compound_batches_total']}",
            'compound_percent': (self.progress['compound_batches_completed'] / max(1, self.progress['compound_batches_total'])) * 100,
            'memory_mb': get_memory_usage(),
            'memory_peak_mb': self.progress['memory_peak_mb']
        }

class AACDPCProteinExtractor:
    """AAC+DPC蛋白质特征提取器"""
    def __init__(self):
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.aa_groups = {
            'hydrophobic': 'AILMFWYV',
            'polar': 'STYNQ',
            'charged': 'DEKR',
            'aromatic': 'FWY',
            'aliphatic': 'ILV',
        }
        self.aa_properties = {
            'A': [1.8, 89.1, 6.0], 'C': [2.5, 121.0, 5.1], 'D': [-3.5, 133.1, 3.0],
            'E': [-3.5, 147.1, 4.2], 'F': [2.8, 165.2, 5.5], 'G': [-0.4, 75.1, 6.0],
            'H': [-3.2, 155.2, 7.6], 'I': [4.5, 131.2, 6.0], 'K': [-3.9, 146.2, 9.7],
            'L': [3.8, 131.2, 6.0], 'M': [1.9, 149.2, 5.7], 'N': [-3.5, 132.1, 5.4],
            'P': [-1.6, 115.1, 6.3], 'Q': [-3.5, 146.1, 5.7], 'R': [-4.5, 174.2, 10.8],
            'S': [-0.8, 105.1, 5.7], 'T': [-0.7, 119.1, 5.6], 'V': [4.2, 117.1, 6.0],
            'W': [-0.9, 204.2, 5.9], 'Y': [-1.3, 181.2, 5.7]
        }

    def extract_aac_features(self, sequence):
        aac_features = []
        total_length = len(sequence)
        for aa in self.amino_acids:
            count = sequence.count(aa)
            frequency = count / total_length if total_length > 0 else 0.0
            aac_features.append(frequency)
        return aac_features

    def extract_dpc_features(self, sequence):
        dpc_features = []
        dipeptides = []
        for aa1 in self.amino_acids:
            for aa2 in self.amino_acids:
                dipeptides.append(aa1 + aa2)
        total_dipeptides = len(sequence) - 1
        if total_dipeptides <= 0:
            return [0.0] * DPC_DIM
        for dipeptide in dipeptides:
            count = 0
            for i in range(len(sequence) - 1):
                if sequence[i:i + 2] == dipeptide:
                    count += 1
            frequency = count / total_dipeptides
            dpc_features.append(frequency)
        return dpc_features

    def extract_enhanced_features(self, sequence):
        enhanced_features = []
        if len(sequence) == 0:
            return [0.0] * PROTEIN_ENHANCED_DIM
        for group_name, group_aa in list(self.aa_groups.items())[:5]:
            ratio = sum(sequence.count(aa) for aa in group_aa) / len(sequence)
            enhanced_features.append(ratio)
        length_feature = min(len(sequence) / 1000.0, 1.0)
        enhanced_features.append(length_feature)
        total_hydrophobicity = sum(self.aa_properties.get(aa, [0, 0, 0])[0]
                                 for aa in sequence if aa in self.aa_properties)
        avg_hydrophobicity = total_hydrophobicity / len(sequence) if len(sequence) > 0 else 0.0
        enhanced_features.append(avg_hydrophobicity)
        total_mw = sum(self.aa_properties.get(aa, [0, 0, 0])[1]
                      for aa in sequence if aa in self.aa_properties)
        avg_mw = total_mw / len(sequence) if len(sequence) > 0 else 0.0
        normalized_mw = (avg_mw - 75) / (200 - 75) if avg_mw > 0 else 0.0
        enhanced_features.append(normalized_mw)
        total_pi = sum(self.aa_properties.get(aa, [0, 0, 0])[2]
                      for aa in sequence if aa in self.aa_properties)
        avg_pi = total_pi / len(sequence) if len(sequence) > 0 else 0.0
        normalized_pi = (avg_pi - 3) / (11 - 3) if avg_pi > 0 else 0.0
        enhanced_features.append(normalized_pi)
        n_terminal_feature = 1.0 if len(sequence) > 0 and sequence[0] == 'M' else 0.0
        enhanced_features.append(n_terminal_feature)
        while len(enhanced_features) < PROTEIN_ENHANCED_DIM:
            enhanced_features.append(0.0)
        return enhanced_features[:PROTEIN_ENHANCED_DIM]

    def extract_all_features(self, sequence):
        cleaned_sequence = ''.join([aa for aa in sequence.upper() if aa in self.amino_acids])
        if len(cleaned_sequence) == 0:
            return [0.0] * PROTEIN_TOTAL_DIM
        aac_features = self.extract_aac_features(cleaned_sequence)
        dpc_features = self.extract_dpc_features(cleaned_sequence)
        enhanced_features = self.extract_enhanced_features(cleaned_sequence)
        all_features = aac_features + dpc_features + enhanced_features
        if len(all_features) != PROTEIN_TOTAL_DIM:
            if len(all_features) < PROTEIN_TOTAL_DIM:
                all_features.extend([0.0] * (PROTEIN_TOTAL_DIM - len(all_features)))
            else:
                all_features = all_features[:PROTEIN_TOTAL_DIM]
        return all_features

class DTIOptimizedFingerprintExtractor:
    """DTI优化的分子指纹提取器"""
    def __init__(self):
        self.substructure_patterns = DTI_IMPORTANT_SUBSTRUCTURES
        # 加载Chemprop预训练模型
        pretrained_path = 'chemprop_pretrain.pth'  # 替换为实际权重路径
        try:
            self.gnn_model = MoleculeModel.load_from_file(pretrained_path)
            self.gnn_model.to('cuda' if torch.cuda.is_available() else 'cpu')
            self.gnn_model.eval()
            print(f"✅ 加载Chemprop预训练模型: {pretrained_path}")
        except Exception as e:
            print(f"❌ 无法加载Chemprop模型: {e}")
            sys.exit(1)
        # 维度调整层（Chemprop默认输出300维，调整至GNN_DIM）
        self.dim_reducer = torch.nn.Linear(300, GNN_DIM)
        self.dim_reducer.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim_reducer.eval()

    def smiles_to_mol(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                Chem.SanitizeMol(mol)
            return mol
        except:
            return None

    def extract_gnn_features(self, smiles):
        """使用Chemprop提取GNN特征"""
        if not smiles or not isinstance(smiles, str):
            return [0.0] * GNN_DIM
        try:
            dataset = MoleculeDataset([{'smiles': smiles}])
            features_generator = get_features_generator('morgan')
            with torch.no_grad(), autocast():
                features = self.gnn_model(dataset, features_generator=features_generator)
                features = features.squeeze(0)  # 移除batch维度
                features = self.dim_reducer(features).cpu().numpy()
            return features.tolist()[:GNN_DIM]
        except Exception as e:
            print(f"⚠️ GNN特征提取失败: {e}")
            return [0.0] * GNN_DIM

    def extract_ecfp4_fingerprint(self, mol):
        if mol is None:
            return [0] * ECFP4_BITS
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=ECFP4_BITS,
                useChirality=True, useBondTypes=True
            )
            return list(map(int, fp.ToBitString()))
        except:
            return [0] * ECFP4_BITS

    def extract_maccs_fingerprint(self, mol):
        if mol is None:
            return [0] * MACCS_BITS
        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
            return list(map(int, fp.ToBitString()))
        except:
            return [0] * MACCS_BITS

    def extract_dti_substructures(self, mol):
        if mol is None:
            return [0] * (DTI_SUBSTRUCTURES_COUNT * 2)
        features = []
        try:
            for name, pattern in self.substructure_patterns.items():
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol is not None:
                    has_match = mol.HasSubstructMatch(pattern_mol)
                    features.append(int(has_match))
                    matches = len(mol.GetSubstructMatches(pattern_mol))
                    features.append(matches)
                else:
                    features.extend([0, 0])
        except:
            features = [0] * (DTI_SUBSTRUCTURES_COUNT * 2)
        while len(features) < DTI_SUBSTRUCTURES_COUNT * 2:
            features.append(0)
        return features[:DTI_SUBSTRUCTURES_COUNT * 2]

    def extract_pharmacophore_features(self, mol):
        if mol is None:
            return [0] * PHARMACOPHORE_COUNT
        try:
            hbd_count = Descriptors.NumHBD(mol)
            hba_count = Descriptors.NumHBA(mol)
            aromatic_count = Descriptors.NumAromaticRings(mol)
            hydrophobic_count = sum(1 for atom in mol.GetAtoms()
                                  if atom.GetSymbol() == 'C' and atom.GetIsAromatic())
            pos_ionizable = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NH3+,NH2+,NH+]')))
            neg_ionizable = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[O-,COO-]')))
            return [hbd_count, hba_count, aromatic_count, hydrophobic_count, pos_ionizable, neg_ionizable]
        except:
            return [0] * PHARMACOPHORE_COUNT

    def extract_rdkit_descriptors(self, mol):
        if mol is None:
            return [0.0] * RDKIT_DESCRIPTOR_COUNT
        try:
            features = []
            for desc_name in RDKIT_DESCRIPTORS:
                desc_func = getattr(Descriptors, desc_name)
                value = desc_func(mol)
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                value = min(max(value, -10.0), 10.0) / 10.0
                features.append(value)
            return features[:RDKIT_DESCRIPTOR_COUNT]
        except:
            return [0.0] * RDKIT_DESCRIPTOR_COUNT

    def extract_all_features(self, smiles):
        mol = self.smiles_to_mol(smiles)
        gnn_features = self.extract_gnn_features(smiles)  # 直接传递SMILES
        ecfp4_features = self.extract_ecfp4_fingerprint(mol)
        maccs_features = self.extract_maccs_fingerprint(mol)
        dti_sub_features = self.extract_dti_substructures(mol)
        pharm_features = self.extract_pharmacophore_features(mol)
        rdkit_features = self.extract_rdkit_descriptors(mol)
        all_features = (gnn_features + ecfp4_features + maccs_features +
                       dti_sub_features + pharm_features + rdkit_features)
        return all_features

class SeparatedBatchExtractor:
    """分离分批特征提取器主类"""
    def __init__(self, work_dir, input_filename, resume_mode=False):
        self.work_dir = work_dir
        self.input_filename = input_filename
        self.resume_mode = resume_mode
        self.output_dir = os.path.join(work_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.progress_manager = ProgressManager(work_dir)
        self.protein_extractor = AACDPCProteinExtractor()
        self.fingerprint_extractor = DTIOptimizedFingerprintExtractor()
        self.protein_data = {}
        self.compound_data = {}
        self.total_rows = 0
        self.column_mapping = {}
        print(f"📁 工作目录: {work_dir}")
        print(f"🔄 运行模式: {'恢复运行' if resume_mode else '新建运行'}")
        print(f"🧬 蛋白质特征维度: {PROTEIN_TOTAL_DIM}")
        print(f"💊 化合物特征维度: {COMPOUND_TOTAL_DIM}")

    def analyze_input_file(self, input_csv):
        print("\n" + "=" * 60)
        print("📂 输入文件分析阶段")
        print("=" * 60)
        detected_columns, header = detect_column_names(input_csv)
        if not detected_columns:
            return False
        self.column_mapping = detected_columns
        required_fields = ['protein_accession', 'sequence', 'compound_cid', 'smile']
        missing_fields = [field for field in required_fields if not detected_columns[field]]
        if missing_fields:
            print(f"\n❌ 错误: 未找到必需的列: {missing_fields}")
            return False
        protein_data, compound_data, total_rows = analyze_data_distribution(input_csv, detected_columns)
        if protein_data is None:
            return False
        self.protein_data = protein_data
        self.compound_data = compound_data
        self.total_rows = total_rows
        compound_batches_total = (len(compound_data) + COMPOUND_BATCH_SIZE - 1) // COMPOUND_BATCH_SIZE
        self.progress_manager.update_totals(
            total_records=total_rows,
            unique_proteins=len(protein_data),
            unique_compounds=len(compound_data),
            compound_batches_total=compound_batches_total
        )
        return True

    def extract_protein_features(self):
        if self.progress_manager.progress['protein_extraction_completed']:
            print("\n🧬 蛋白质特征已提取，跳过")
            return True
        print("\n" + "=" * 60)
        print("🧬 蛋白质特征提取阶段")
        print("=" * 60)
        base_name = os.path.splitext(os.path.basename(self.input_filename))[0]
        protein_file = os.path.join(self.output_dir, f'{base_name}_protein_features.csv')
        print(f"正在提取 {len(self.protein_data)} 个蛋白质特征...")
        try:
            protein_names = self._generate_protein_feature_names()
            with open(protein_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = ['Protein_Accession'] + protein_names
                writer.writerow(header)
                for i, (accession, sequence) in enumerate(self.protein_data.items(), 1):
                    print(f"🔄 提取蛋白质特征 {i}/{len(self.protein_data)}: {accession}")
                    try:
                        features = self.protein_extractor.extract_all_features(sequence)
                        row = [accession] + features
                        writer.writerow(row)
                    except Exception as e:
                        print(f"  ❌ 蛋白质特征提取失败: {e}")
                        self.progress_manager.add_error(f"Protein {accession}: {e}")
                        row = [accession] + [0.0] * PROTEIN_TOTAL_DIM
                        writer.writerow(row)
            self.progress_manager.mark_protein_completed(protein_file)
            print(f"✅ 蛋白质特征提取完成: {protein_file}")
            return True
        except Exception as e:
            print(f"❌ 蛋白质特征提取失败: {e}")
            self.progress_manager.add_error(f"Protein extraction: {e}")
            return False

    def extract_compound_features_in_batches(self):
        print("\n" + "=" * 60)
        print("💊 化合物特征分批提取阶段")
        print("=" * 60)
        base_name = os.path.splitext(os.path.basename(self.input_filename))[0]
        compound_list = list(self.compound_data.items())
        total_batches = (len(compound_list) + COMPOUND_BATCH_SIZE - 1) // COMPOUND_BATCH_SIZE
        print(f"开始分批提取 {len(compound_list)} 个化合物特征，共 {total_batches} 批")
        for batch_num in range(1, total_batches + 1):
            if batch_num <= self.progress_manager.progress['compound_batches_completed']:
                print(f"📦 批次 {batch_num}/{total_batches} 已处理，跳过")
                continue
            print(f"\n🔄 处理化合物批次 {batch_num}/{total_batches}")
            start_idx = (batch_num - 1) * COMPOUND_BATCH_SIZE
            end_idx = min(start_idx + COMPOUND_BATCH_SIZE, len(compound_list))
            batch_compounds = compound_list[start_idx:end_idx]
            print(f"📝 批次大小: {len(batch_compounds)} 个化合物")
            batch_file = os.path.join(self.output_dir, f'{base_name}_compounds_batch_{batch_num:04d}.csv')
            if self._extract_compound_batch(batch_compounds, batch_file, batch_num):
                self.progress_manager.mark_compound_batch_completed(batch_num, batch_file)
                if get_memory_usage() > MEMORY_LIMIT_MB:
                    print(f"  💾 内存使用超限，强制垃圾回收...")
                    force_garbage_collection()
            else:
                print(f"❌ 批次 {batch_num} 处理失败")
        print(f"\n✅ 化合物特征分批提取完成，共 {total_batches} 批")
        return True

    def _extract_compound_batch(self, batch_compounds, batch_file, batch_num):
        try:
            compound_names = self._generate_compound_feature_names()
            with open(batch_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Compound_CID'] + compound_names)
                # 批量提取GNN特征
                smiles_list = [smile for _, smile in batch_compounds if smile and isinstance(smile, str)]
                if smiles_list:
                    dataset = MoleculeDataset([{'smiles': smi} for smi in smiles_list])
                    features_generator = get_features_generator('morgan')
                    with torch.no_grad(), autocast():
                        gnn_features = self.fingerprint_extractor.gnn_model(dataset, features_generator=features_generator)
                        gnn_features = self.fingerprint_extractor.dim_reducer(gnn_features).cpu().numpy()
                else:
                    gnn_features = np.zeros((len(batch_compounds), GNN_DIM))
                gnn_idx = 0
                for i, (compound_cid, smile) in enumerate(batch_compounds, 1):
                    if i % 100 == 0:
                        memory_mb = get_memory_usage()
                        print(f"  进度: {i}/{len(batch_compounds)} (内存: {memory_mb:.1f}MB)")
                    try:
                        mol = self.fingerprint_extractor.smiles_to_mol(smile)
                        gnn_feat = gnn_features[gnn_idx].tolist() if gnn_idx < len(gnn_features) and smile in smiles_list else [0.0] * GNN_DIM
                        gnn_idx += 1
                        other_features = (self.fingerprint_extractor.extract_ecfp4_fingerprint(mol) +
                                         self.fingerprint_extractor.extract_maccs_fingerprint(mol) +
                                         self.fingerprint_extractor.extract_dti_substructures(mol) +
                                         self.fingerprint_extractor.extract_pharmacophore_features(mol) +
                                         self.fingerprint_extractor.extract_rdkit_descriptors(mol))
                        features = gnn_feat + other_features
                        writer.writerow([compound_cid] + features)
                    except Exception as e:
                        print(f"  ❌ 化合物 {compound_cid} 特征提取失败: {e}")
                        self.progress_manager.add_error(f"Compound {compound_cid}: {e}")
                        writer.writerow([compound_cid] + [0.0] * COMPOUND_TOTAL_DIM)
            print(f"  ✅ 批次 {batch_num} 完成: {os.path.basename(batch_file)}")
            return True
        except Exception as e:
            print(f"  ❌ 批次 {batch_num} 处理失败: {e}")
            self.progress_manager.add_error(f"Batch {batch_num}: {e}")
            return False

    def _generate_protein_feature_names(self):
        protein_names = []
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        for aa in amino_acids:
            protein_names.append(f'AAC_{aa}')
        for aa1 in amino_acids:
            for aa2 in amino_acids:
                protein_names.append(f'DPC_{aa1}{aa2}')
        enhanced_names = ['Enhanced_Hydrophobic', 'Enhanced_Polar', 'Enhanced_Charged',
                         'Enhanced_Aromatic', 'Enhanced_Aliphatic', 'Enhanced_Length',
                         'Enhanced_AvgHydrophobicity', 'Enhanced_AvgMW', 'Enhanced_AvgPI',
                         'Enhanced_NTerminal']
        protein_names.extend(enhanced_names)
        return protein_names

    def _generate_compound_feature_names(self):
        compound_names = []
        for i in range(GNN_DIM):
            compound_names.append(f'GNN_{i}')
        for i in range(ECFP4_BITS):
            compound_names.append(f'ECFP4_{i}')
        for i in range(MACCS_BITS):
            compound_names.append(f'MACCS_{i}')
        for name in DTI_IMPORTANT_SUBSTRUCTURES.keys():
            compound_names.append(f'DTI_Sub_{name}')
            compound_names.append(f'DTI_Count_{name}')
        pharm_features = ['HBD_count', 'HBA_count', 'Aromatic_count',
                         'Hydrophobic_count', 'PosIonizable', 'NegIonizable']
        for pharm in pharm_features:
            compound_names.append(f'Pharm_{pharm}')
        for desc in RDKIT_DESCRIPTORS:
            compound_names.append(f'RDKit_{desc}')
        return compound_names

    def save_processing_stats(self):
        base_name = os.path.splitext(os.path.basename(self.input_filename))[0]
        stats_file = os.path.join(self.output_dir, f'{base_name}_processing_stats.json')
        progress_info = self.progress_manager.get_progress_info()
        stats = {
            'input_file': self.input_filename,
            'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'user': 'woyaokaoyanhaha',
            'version': '18.1 (集成Chemprop预训练模型)',
            'work_directory': self.work_dir,
            'configuration': {
                'compound_batch_size': COMPOUND_BATCH_SIZE,
                'memory_limit_mb': MEMORY_LIMIT_MB,
                'extract_protein_features': EXTRACT_PROTEIN_FEATURES,
                'extract_compound_features': EXTRACT_COMPOUND_FEATURES,
                'save_original_data': SAVE_ORIGINAL_DATA,
                'gnn_feature_dim': GNN_DIM
            },
            'processing_statistics': {
                'total_records': self.total_rows,
                'unique_proteins': len(self.protein_data),
                'unique_compounds': len(self.compound_data),
                'compound_batches_total': self.progress_manager.progress['compound_batches_total'],
                'memory_peak_mb': progress_info['memory_peak_mb'],
                'processing_errors': len(self.progress_manager.progress['processing_errors'])
            },
            'output_files': {
                'protein_file': self.progress_manager.progress['protein_file'],
                'compound_batch_files': self.progress_manager.progress['compound_batch_files']
            },
            'feature_dimensions': {
                'protein_total': PROTEIN_TOTAL_DIM,
                'compound_total': COMPOUND_TOTAL_DIM,
                'total_features': TOTAL_FEATURE_DIM
            }
        }
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"📈 统计信息已保存: {os.path.basename(stats_file)}")
        except Exception as e:
            print(f"❌ 统计信息保存失败: {e}")

def main():
    print("\n" + "=" * 80)
    print("🧬 蛋白质-化合物分离分批特征提取脚本")
    print(f"👤 用户: woyaokaoyanhaha")
    print(f"📅 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 特征维度: 蛋白质{PROTEIN_TOTAL_DIM} + 化合物{COMPOUND_TOTAL_DIM} = 总计{TOTAL_FEATURE_DIM}")
    print(f"🔧 版本: 18.1 (集成Chemprop预训练模型)")
    print("=" * 80)

    try:
        if TEST_MODE:
            print("🧪 测试模式 - 检查文件和环境")
            if INPUT_CSV_FILE:
                detected_columns, header = detect_column_names(INPUT_CSV_FILE)
                if detected_columns:
                    print("✅ CSV文件格式检查通过")
                    if PSUTIL_AVAILABLE:
                        initial_memory = get_memory_usage()
                        print(f"✅ 当前内存使用: {initial_memory:.1f} MB")
                else:
                    print("❌ CSV文件格式检查失败")
                    return 1
                print("✅ 所有测试通过")
            else:
                print("❌ 缺少输入文件参数")
                return 1
            return 0

        if not INPUT_CSV_FILE:
            print("❌ 错误: 请在代码开头设置 INPUT_CSV_FILE 参数")
            return 1

        if not os.path.exists(INPUT_CSV_FILE):
            print(f"❌ 输入文件不存在: {INPUT_CSV_FILE}")
            return 1

        file_size_mb = os.path.getsize(INPUT_CSV_FILE) / 1024 / 1024
        print(f"📄 输入文件大小: {file_size_mb:.1f} MB")

        resume_mode = False
        if RESUME_FROM_DIR:
            if not os.path.exists(RESUME_FROM_DIR):
                print(f"❌ 恢复目录不存在: {RESUME_FROM_DIR}")
                return 1
            progress_file = os.path.join(RESUME_FROM_DIR, "progress.json")
            if not os.path.exists(progress_file):
                print(f"❌ 恢复目录中没有找到进度文件")
                return 1
            work_dir = RESUME_FROM_DIR
            resume_mode = True
        else:
            work_dir = get_output_dir_name(INPUT_CSV_FILE, CUSTOM_OUTPUT_DIR)
            if os.path.exists(work_dir):
                print(f"⚠️ 输出目录已存在: {work_dir}")
                print("正在删除并重建目录...")
                import shutil
                shutil.rmtree(work_dir)
                print(f"🗑️ 已删除目录: {work_dir}")

        extractor = SeparatedBatchExtractor(
            work_dir,
            INPUT_CSV_FILE,
            resume_mode
        )

        start_time = time.time()
        initial_memory = get_memory_usage()

        if not extractor.analyze_input_file(INPUT_CSV_FILE):
            print("❌ 输入文件分析失败")
            return 1

        print(f"\n💾 初始内存使用: {initial_memory:.1f} MB")

        if EXTRACT_PROTEIN_FEATURES:
            if not extractor.extract_protein_features():
                print("❌ 蛋白质特征提取失败")
                return 1

        if EXTRACT_COMPOUND_FEATURES:
            if not extractor.extract_compound_features_in_batches():
                print("❌ 化合物特征提取失败")
                return 1

        extractor.save_processing_stats()
        extractor.progress_manager.mark_completed()

        end_time = time.time()
        processing_time = end_time - start_time
        final_memory = get_memory_usage()

        print("\n" + "=" * 80)
        print("🎉 分离分批特征提取完成!")
        print(f"⏱️ 总处理时间: {processing_time:.2f} 秒")
        print(f"📁 结果保存在: {work_dir}")
        print(f"\n📊 处理统计:")
        print(f"  总记录数: {extractor.total_rows}")
        print(f"  唯一蛋白质数: {len(extractor.protein_data)} 个")
        print(f"  唯一化合物数: {len(extractor.compound_data)} 个")
        print(f"  化合物批次数: {extractor.progress_manager.progress['compound_batches_total']}")
        if extractor.total_rows > 0:
            print(f"  平均处理速度: {extractor.total_rows / processing_time:.1f} 记录/秒")
        print(f"\n💾 内存使用统计:")
        print(f"  初始内存: {initial_memory:.1f} MB")
        print(f"  最终内存: {final_memory:.1f} MB")
        progress_info = extractor.progress_manager.get_progress_info()
        print(f"  峰值内存: {progress_info['memory_peak_mb']:.1f} MB")
        print(f"\n🎯 特征维度详情:")
        print(f"  蛋白质特征: {PROTEIN_TOTAL_DIM} 维 (AAC+DPC+Enhanced)")
        print(f"  化合物特征: {COMPOUND_TOTAL_DIM} 维 (GNN+ECFP4+MACCS+DTI+Pharm+RDKit)")
        error_count = len(extractor.progress_manager.progress['processing_errors'])
        if error_count > 0:
            print(f"\n⚠️ 处理错误: {error_count} 个")
        else:
            print(f"\n✅ 处理过程无错误")
        print("=" * 80)
        return 0

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断处理")
        print("可通过设置 RESUME_FROM_DIR 参数恢复运行")
        if 'work_dir' in locals():
            print(f"恢复目录设置为: RESUME_FROM_DIR = '{work_dir}'")
        return 1
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    print("\n🔧 当前配置信息:")
    print(f"  输入文件: {INPUT_CSV_FILE}")
    print(f"  输出目录: {CUSTOM_OUTPUT_DIR or '自动生成'}")
    print(f"  恢复运行目录: {RESUME_FROM_DIR or '无'}")
    print(f"  测试模式: {TEST_MODE}")
    print(f"  调试模式: {DEBUG_MODE}")
    print(f"\n📦 分批配置:")
    print(f"  化合物批次大小: {COMPOUND_BATCH_SIZE}")
    print(f"  内存限制: {MEMORY_LIMIT_MB} MB")
    print(f"\n🎯 提取配置:")
    print(f"  提取蛋白质特征: {EXTRACT_PROTEIN_FEATURES}")
    print(f"  提取化合物特征: {EXTRACT_COMPOUND_FEATURES}")
    print(f"  保存原始数据: {SAVE_ORIGINAL_DATA}")
    print(f"  GNN特征维度: {GNN_DIM}")
    print(f"\n🎯 特征维度:")
    print(f"  蛋白质: {PROTEIN_TOTAL_DIM} 维")
    print(f"  化合物: {COMPOUND_TOTAL_DIM} 维")
    print(f"  总计: {TOTAL_FEATURE_DIM} 维")
    exit_code = main()
    sys.exit(exit_code)