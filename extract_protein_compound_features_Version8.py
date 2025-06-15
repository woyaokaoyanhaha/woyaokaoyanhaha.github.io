#!/usr/bin/env python3
"""
蛋白质-化合物特征提取脚本
作者: woyaokaoyanhaha
日期: 2025-06-15 01:33:18
分别对蛋白质和化合物去重计算特征，最后组合
使用全面的分子描述符特征
"""

# ================================
# 列名配置 (可根据输入文件修改)
# ================================

# 输入文件列名配置
COLUMN_NAMES = {
    'protein_accession': 'Protein_Accession',  # 蛋白质登录号列名
    'sequence': 'Sequence',                    # 蛋白质序列列名
    'compound_cid': 'Compound_CID',            # 化合物CID列名
    'smile': 'Smile',                          # 化合物SMILES列名
    'label': 'label'                           # 标签列名 (也支持 'Label')
}

# 可选的替代列名
ALTERNATIVE_COLUMN_NAMES = {
    'protein_accession': ['Protein_Accession', 'ProteinAccession', 'Accession', 'Protein_ID', 'ProteinID'],
    'sequence': ['Sequence', 'Protein_Sequence', 'ProteinSequence', 'Seq'],
    'compound_cid': ['Compound_CID', 'CompoundCID', 'CID', 'Compound_ID', 'CompoundID'],
    'smile': ['Smile', 'SMILES', 'Canonical_SMILES', 'CanonicalSMILES'],
    'label': ['label', 'Label', 'Class', 'Target', 'Y']
}

# ================================
# 特征提取参数配置 (可根据需要修改)
# ================================

# PSE-AAC特征配置
PSEAAC_LAMBDA = 10           # 伪氨基酸成分的lambda值
PSEAAC_WEIGHT = 0.1          # 权重因子w
PSEAAC_TOTAL_DIM = 51        # PSE-AAC总维度 (1+20+30)

# PSE-PSSM特征配置
PSEPSSM_TOTAL_DIM = 220      # PSE-PSSM总维度
PSEPSSM_LAMBDA = 10          # 伪PSSM成分的lambda值
PSSM_BASIC_FEATURES = 80     # 基础PSSM特征维度

# 化合物分子描述符配置 - 使用全面的描述符
MOLECULAR_DESCRIPTORS = [
    # 基本性质
    'MW', 'ExactMW', 'LogP', 'HBA', 'HBD', 'RotBonds', 'AromaticRings', 
    'RingCount', 'HeteroRings', 'SaturatedRings', 'AliphaticRings', 
    'HeavyAtomCount', 'AtomCount',
    
    # 拓扑性质
    'TPSA', 'LabuteASA', 'MolSurfaceArea', 'BertzCT', 'Chi0v', 'Chi1v', 
    'Kappa1', 'Kappa2', 'Kappa3',
    
    # 药物相似性
    'LipinskiViolations', 'QED', 'RO5_Violations', 'GhoseViolations', 'VeberViolations',
    
    # 原子组成
    'CarbonCount', 'NitrogenCount', 'OxygenCount', 'SulfurCount', 'PhosphorusCount',
    'FluorineCount', 'ChlorineCount', 'BromineCount', 'IodineCount', 'HalogenCount', 'FractionCSP3',
    
    # 电子性质
    'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'SMR_VSA1', 'SMR_VSA2', 
    'SlogP_VSA1', 'SlogP_VSA2',
    
    # 其他物理化学性质
    'MR', 'NHOH_Count', 'NO_Count', 'HallKierAlpha', 'ValenceElectrons',
    'MolLogP', 'MolMR',
    
    # 复杂度相关
    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
    
    # 功能团计数
    'SP3_N_Count', 'SP2_N_Count', 'Amide_Count', 'Ester_Count', 
    'Carboxylic_Count', 'Ether_Count', 'Alcohol_Count', 'Amine_Count'
]
COMPOUND_TOTAL_DIM = len(MOLECULAR_DESCRIPTORS)  # 化合物特征总维度

# PSI-BLAST参数
PSIBLAST_ITERATIONS = 3      # PSI-BLAST迭代次数
PSIBLAST_EVALUE = 0.001      # E-value阈值
PSIBLAST_THREADS = 4         # 线程数

# 文件处理参数
CACHE_ENABLED = True         # 是否启用缓存
VERBOSE_OUTPUT = True        # 是否显示详细输出

# ================================
# 主程序代码
# ================================

import csv
import os
import subprocess
import sys
import json
from pathlib import Path
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    import pandas as pd
    # 化合物处理库 - 使用全面的RDKit模块
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, MolSurf, QED, GraphDescriptors, Crippen, AllChem
    print("所有依赖库导入成功")
except ImportError as e:
    print("错误: 需要安装必要的Python包")
    print("运行: pip install biopython numpy pandas rdkit")
    print(f"缺少的库: {e}")
    sys.exit(1)

def get_output_dir_name(input_csv_path):
    """根据输入文件名生成输出目录名"""
    filename = os.path.basename(input_csv_path)
    basename = os.path.splitext(filename)[0]
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    safe_basename = "".join(c for c in basename if c.isalnum() or c in ('-', '_')).rstrip()
    
    if len(safe_basename) < 3:
        safe_basename = "protein_compound_features"
    
    return f"./{safe_basename}_features_{timestamp}"

def detect_column_names(csv_file):
    """自动检测CSV文件的列名"""
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    
    detected_columns = {}
    
    for field_type, default_name in COLUMN_NAMES.items():
        detected_columns[field_type] = None
        
        # 首先尝试默认名称
        if default_name in header:
            detected_columns[field_type] = default_name
            continue
        
        # 然后尝试替代名称
        for alt_name in ALTERNATIVE_COLUMN_NAMES[field_type]:
            if alt_name in header:
                detected_columns[field_type] = alt_name
                break
    
    return detected_columns, header

class ProteinCompoundFeatureExtractor:
    def __init__(self, swissprot_db, work_dir, input_filename):
        self.swissprot_db = swissprot_db
        self.work_dir = work_dir
        self.input_filename = input_filename
        self.temp_dir = os.path.join(work_dir, "temp")
        self.output_dir = os.path.join(work_dir, "output")
        self.cache_dir = os.path.join(work_dir, "cache")
        
        # 创建必要的目录
        for dir_path in [self.temp_dir, self.output_dir, self.cache_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 存储数据
        self.unique_proteins = {}        # accession -> protein_info
        self.unique_compounds = {}       # cid -> compound_info
        self.all_records = []            # 所有原始记录，保持顺序
        self.column_mapping = {}         # 列名映射
        
        # 特征名称定义
        self.pseaac_feature_names = self._get_pseaac_feature_names()
        self.psepssm_feature_names = self._get_psepssm_feature_names()
        self.compound_feature_names = self._get_compound_feature_names()
        
        print(f"输入文件: {input_filename}")
        print(f"工作目录: {work_dir}")
        print(f"去重模式: 分别对Protein_Accession和Compound_CID去重")
        print(f"PSE-AAC维度: {PSEAAC_TOTAL_DIM}")
        print(f"PSE-PSSM维度: {PSEPSSM_TOTAL_DIM}")
        print(f"化合物描述符维度: {COMPOUND_TOTAL_DIM}")
        print(f"总特征维度: {PSEAAC_TOTAL_DIM + PSEPSSM_TOTAL_DIM + COMPOUND_TOTAL_DIM}")
    
    def _get_pseaac_feature_names(self):
        """获取PSE-AAC特征的真实名称"""
        feature_names = []
        
        # 1. 序列长度特征
        feature_names.append('Length')
        
        # 2. 20种氨基酸组成特征 (按字母顺序)
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
        for aa in amino_acids:
            feature_names.append(f'AA_{aa}')
        
        # 3. 疏水性 (Hydrophobicity) 伪氨基酸特征
        for i in range(1, PSEAAC_LAMBDA + 1):
            feature_names.append(f'Hydrophobicity_lambda_{i}')
        
        # 4. 等电点 (Isoelectric Point) 伪氨基酸特征
        for i in range(1, PSEAAC_LAMBDA + 1):
            feature_names.append(f'Isoelectric_Point_lambda_{i}')
        
        # 5. 分子量 (Molecular Weight) 伪氨基酸特征
        for i in range(1, PSEAAC_LAMBDA + 1):
            feature_names.append(f'Molecular_Weight_lambda_{i}')
        
        return feature_names
    
    def _get_psepssm_feature_names(self):
        """获取PSE-PSSM特征的真实名称"""
        feature_names = []
        
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        
        # PSSM基础统计特征
        for aa in amino_acids:
            feature_names.append(f'PSSM_Mean_{aa}')
        
        for aa in amino_acids:
            feature_names.append(f'PSSM_Std_{aa}')
        
        for aa in amino_acids:
            feature_names.append(f'PSSM_Max_{aa}')
        
        for aa in amino_acids:
            feature_names.append(f'PSSM_Min_{aa}')
        
        # 伪PSSM特征
        remaining_dims = PSEPSSM_TOTAL_DIM - PSSM_BASIC_FEATURES
        pse_features_per_lag = remaining_dims // PSEPSSM_LAMBDA
        
        for lag in range(1, PSEPSSM_LAMBDA + 1):
            for i in range(pse_features_per_lag):
                if len(feature_names) < PSEPSSM_TOTAL_DIM:
                    aa_idx = i % 20
                    aa = amino_acids[aa_idx]
                    feature_names.append(f'PsePSSM_Lag{lag}_{aa}')
        
        # 填充剩余维度
        while len(feature_names) < PSEPSSM_TOTAL_DIM:
            feature_names.append(f'PsePSSM_Extra_{len(feature_names) - PSSM_BASIC_FEATURES + 1}')
        
        return feature_names[:PSEPSSM_TOTAL_DIM]
    
    def _get_compound_feature_names(self):
        """获取化合物分子描述符特征名称"""
        return [f'Compound_{desc}' for desc in MOLECULAR_DESCRIPTORS]
    
    def load_and_deduplicate(self, input_csv):
        """分别对蛋白质和化合物去重"""
        print("正在检测列名和加载数据...")
        
        # 自动检测列名
        detected_columns, header = detect_column_names(input_csv)
        self.column_mapping = detected_columns
        
        print(f"检测到的列名映射:")
        for field_type, column_name in detected_columns.items():
            if column_name:
                print(f"  {field_type}: {column_name}")
            else:
                print(f"  {field_type}: 未找到")
        
        # 检查必需列是否存在
        required_fields = ['protein_accession', 'sequence', 'compound_cid', 'smile']
        missing_fields = [field for field in required_fields if not detected_columns[field]]
        
        if missing_fields:
            print(f"\n错误: 未找到必需的列: {missing_fields}")
            print(f"可用的列: {header}")
            print(f"请检查列名是否正确，或修改脚本顶部的COLUMN_NAMES配置")
            sys.exit(1)
        
        print("\n正在加载数据并分别去重...")
        
        total_count = 0
        seen_proteins = set()
        seen_compounds = set()
        
        with open(input_csv, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                accession = row[detected_columns['protein_accession']].strip()
                sequence = row[detected_columns['sequence']].strip().upper()
                compound_cid = row[detected_columns['compound_cid']].strip()
                smile = row[detected_columns['smile']].strip()
                
                # 处理标签列（可能不存在）
                label = ''
                if detected_columns['label']:
                    label = row[detected_columns['label']].strip()
                
                total_count += 1
                
                # 记录所有原始数据（保持顺序）
                record = {
                    'accession': accession,
                    'sequence': sequence,
                    'compound_cid': compound_cid,
                    'smile': smile,
                    'label': label,
                    'row_number': total_count,
                    'protein_is_unique': accession not in seen_proteins,
                    'compound_is_unique': compound_cid not in seen_compounds
                }
                self.all_records.append(record)
                
                # 蛋白质去重
                if accession not in seen_proteins:
                    self.unique_proteins[accession] = {
                        'accession': accession,
                        'sequence': sequence,
                        'first_occurrence': total_count
                    }
                    seen_proteins.add(accession)
                
                # 化合物去重
                if compound_cid not in seen_compounds:
                    self.unique_compounds[compound_cid] = {
                        'compound_cid': compound_cid,
                        'smile': smile,
                        'first_occurrence': total_count
                    }
                    seen_compounds.add(compound_cid)
        
        print(f"总记录数: {total_count}")
        print(f"唯一蛋白质数: {len(self.unique_proteins)}")
        print(f"唯一化合物数: {len(self.unique_compounds)}")
        print(f"蛋白质重复率: {(total_count - len(self.unique_proteins)) / total_count * 100:.1f}%")
        print(f"化合物重复率: {(total_count - len(self.unique_compounds)) / total_count * 100:.1f}%")
        
        return len(self.unique_proteins), len(self.unique_compounds)
    
    def create_fasta_file(self, accession, protein_info):
        """为唯一蛋白质创建FASTA文件"""
        safe_acc = accession.replace('/', '_').replace('\\', '_').replace('|', '_')
        fasta_file = os.path.join(self.temp_dir, f"{safe_acc}.fasta")
        
        with open(fasta_file, 'w') as f:
            f.write(f">{accession}\n{protein_info['sequence']}\n")
        return fasta_file
    
    def run_psiblast(self, fasta_file, accession):
        """运行PSI-BLAST生成PSSM"""
        safe_acc = accession.replace('/', '_').replace('\\', '_').replace('|', '_')
        pssm_file = os.path.join(self.cache_dir, f"{safe_acc}.pssm")
        
        # 检查缓存
        if CACHE_ENABLED and os.path.exists(pssm_file):
            if VERBOSE_OUTPUT:
                print(f"使用缓存的PSSM: {accession}")
            return pssm_file
        
        cmd = [
            "psiblast",
            "-query", fasta_file,
            "-db", self.swissprot_db,
            "-num_iterations", str(PSIBLAST_ITERATIONS),
            "-evalue", str(PSIBLAST_EVALUE),
            "-out_ascii_pssm", pssm_file,
            "-num_threads", str(PSIBLAST_THREADS)
        ]
        
        try:
            if VERBOSE_OUTPUT:
                print(f"运行PSI-BLAST: {accession}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return pssm_file if os.path.exists(pssm_file) else None
        except subprocess.CalledProcessError as e:
            print(f"警告: {accession} PSI-BLAST运行失败: {e}")
            return None
    
    def extract_pseaac_features(self, sequence):
        """提取PSE-AAC特征"""
        # 氨基酸的真实理化性质 (疏水性, 等电点, 分子量)
        aa_properties = {
            'A': [1.8, 6.0, 89.1], 'R': [-4.5, 10.8, 174.2], 'N': [-3.5, 5.4, 132.1],
            'D': [-3.5, 3.0, 133.1], 'C': [2.5, 5.1, 121.0], 'Q': [-3.5, 5.7, 146.1],
            'E': [-3.5, 4.2, 147.1], 'G': [-0.4, 6.0, 75.1], 'H': [-3.2, 7.6, 155.2],
            'I': [4.5, 6.0, 131.2], 'L': [3.8, 6.0, 131.2], 'K': [-3.9, 9.7, 146.2],
            'M': [1.9, 5.7, 149.2], 'F': [2.8, 5.5, 165.2], 'P': [-1.6, 6.3, 115.1],
            'S': [-0.8, 5.7, 105.1], 'T': [-0.7, 5.6, 119.1], 'V': [4.2, 6.0, 117.1],
            'W': [-0.9, 5.9, 204.2], 'Y': [-1.3, 5.7, 181.2]
        }
        
        # 标准化理化性质
        all_props = np.array([aa_properties[aa] for aa in aa_properties])
        means = np.mean(all_props, axis=0)
        stds = np.std(all_props, axis=0)
        
        for aa in aa_properties:
            aa_properties[aa] = [(aa_properties[aa][i] - means[i]) / stds[i] for i in range(3)]
        
        features = []
        
        # 1. 序列长度
        features.append(len(sequence))
        
        # 2. 氨基酸组成 (20维) - 按字母顺序
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
        aa_count = {aa: 0 for aa in amino_acids}
        for aa in sequence:
            if aa in aa_count:
                aa_count[aa] += 1
        
        total = len(sequence)
        for aa in amino_acids:
            features.append(aa_count[aa] / total if total > 0 else 0)
        
        # 3. 计算三种理化性质的伪氨基酸组成
        for property_idx in range(3):  # 疏水性, 等电点, 分子量
            pseudo_comp = []
            lambda_val = min(PSEAAC_LAMBDA, len(sequence) - 1)
            
            for lag in range(1, lambda_val + 1):
                theta = 0
                count = 0
                for j in range(len(sequence) - lag):
                    aa1, aa2 = sequence[j], sequence[j + lag]
                    if aa1 in aa_properties and aa2 in aa_properties:
                        theta += (aa_properties[aa1][property_idx] - aa_properties[aa2][property_idx]) ** 2
                        count += 1
                
                theta = theta / count if count > 0 else 0
                pseudo_comp.append(theta)
            
            # 填充到指定维度
            while len(pseudo_comp) < PSEAAC_LAMBDA:
                pseudo_comp.append(0.0)
            
            features.extend(pseudo_comp)
        
        return features
    
    def extract_psepssm_features(self, pssm_file):
        """提取PSE-PSSM特征"""
        if not pssm_file or not os.path.exists(pssm_file):
            return [0.0] * PSEPSSM_TOTAL_DIM
        
        try:
            pssm_matrix = []
            with open(pssm_file, 'r') as f:
                lines = f.readlines()
            
            # 解析PSSM矩阵
            start_reading = False
            for line in lines:
                if line.strip() and len(line.split()) > 0 and line.split()[0].isdigit():
                    start_reading = True
                    parts = line.strip().split()
                    if len(parts) >= 22:
                        try:
                            scores = [float(x) for x in parts[2:22]]
                            pssm_matrix.append(scores)
                        except ValueError:
                            continue
                elif start_reading and (not line.strip() or line.strip().startswith('Lambda')):
                    break
            
            if not pssm_matrix:
                return [0.0] * PSEPSSM_TOTAL_DIM
            
            pssm_array = np.array(pssm_matrix)
            features = []
            
            # 1. PSSM基础统计特征 (80维: 20*4)
            pssm_mean = np.mean(pssm_array, axis=0)
            features.extend(pssm_mean.tolist())
            
            pssm_std = np.std(pssm_array, axis=0)
            features.extend(pssm_std.tolist())
            
            pssm_max = np.max(pssm_array, axis=0)
            features.extend(pssm_max.tolist())
            
            pssm_min = np.min(pssm_array, axis=0)
            features.extend(pssm_min.tolist())
            
            # 2. 伪PSSM特征 (140维)
            seq_len = len(pssm_array)
            lambda_val = min(PSEPSSM_LAMBDA, seq_len - 1)
            
            remaining_dims = PSEPSSM_TOTAL_DIM - len(features)
            features_per_lag = remaining_dims // lambda_val if lambda_val > 0 else 0
            
            for lag in range(1, lambda_val + 1):
                lag_features = []
                for aa_idx in range(20):
                    if len(lag_features) < features_per_lag:
                        theta = 0
                        count = 0
                        for pos in range(seq_len - lag):
                            theta += (pssm_array[pos][aa_idx] - pssm_array[pos + lag][aa_idx]) ** 2
                            count += 1
                        
                        theta = theta / count if count > 0 else 0
                        lag_features.append(theta)
                
                features.extend(lag_features)
            
            # 填充或截断到固定长度
            while len(features) < PSEPSSM_TOTAL_DIM:
                features.append(0.0)
            
            return features[:PSEPSSM_TOTAL_DIM]
            
        except Exception as e:
            print(f"PSSM特征提取错误: {e}")
            return [0.0] * PSEPSSM_TOTAL_DIM
    
    def extract_compound_features(self, smile):
        """提取化合物分子描述符特征 - 基于你提供的代码"""
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                print(f"警告: 无法解析SMILES: {smile}")
                return [0.0] * COMPOUND_TOTAL_DIM
            
            descriptors = {}
            
            # 基本性质
            descriptors['MW'] = Descriptors.MolWt(mol)
            descriptors['ExactMW'] = Descriptors.ExactMolWt(mol)
            descriptors['LogP'] = Descriptors.MolLogP(mol)
            descriptors['HBA'] = Descriptors.NumHAcceptors(mol)
            descriptors['HBD'] = Descriptors.NumHDonors(mol)
            descriptors['RotBonds'] = Descriptors.NumRotatableBonds(mol)
            descriptors['AromaticRings'] = Descriptors.NumAromaticRings(mol)
            descriptors['RingCount'] = Descriptors.RingCount(mol)
            descriptors['HeteroRings'] = Descriptors.NumHeterocycles(mol)
            descriptors['SaturatedRings'] = Descriptors.NumSaturatedRings(mol)
            descriptors['AliphaticRings'] = Descriptors.NumAliphaticRings(mol)
            descriptors['HeavyAtomCount'] = mol.GetNumHeavyAtoms()
            descriptors['AtomCount'] = mol.GetNumAtoms()

            # 拓扑性质
            descriptors['TPSA'] = MolSurf.TPSA(mol)
            descriptors['LabuteASA'] = MolSurf.LabuteASA(mol)

            # 尝试计算分子表面积，需要先生成3D构象
            try:
                mol_3d = Chem.AddHs(mol)  # 添加氢原子
                success = AllChem.EmbedMolecule(mol_3d, randomSeed=42)  # 生成3D构象
                if success == 0:  # 成功生成构象
                    # 优化结构
                    AllChem.UFFOptimizeMolecule(mol_3d)
                    descriptors['MolSurfaceArea'] = AllChem.ComputeMolVolume(mol_3d)
                else:
                    descriptors['MolSurfaceArea'] = 0.0
            except:
                descriptors['MolSurfaceArea'] = 0.0

            descriptors['BertzCT'] = GraphDescriptors.BertzCT(mol)
            descriptors['Chi0v'] = GraphDescriptors.Chi0v(mol)
            descriptors['Chi1v'] = GraphDescriptors.Chi1v(mol)
            descriptors['Kappa1'] = GraphDescriptors.Kappa1(mol)
            descriptors['Kappa2'] = GraphDescriptors.Kappa2(mol)
            descriptors['Kappa3'] = GraphDescriptors.Kappa3(mol)

            # 药物相似性和成药性相关指标
            descriptors['LipinskiViolations'] = Lipinski.NumRotatableBonds(mol)
            descriptors['QED'] = QED.qed(mol)
            descriptors['RO5_Violations'] = int(Lipinski.NumHDonors(mol) > 5 or Lipinski.NumHAcceptors(mol) > 10 or 
                                            Descriptors.MolWt(mol) > 500 or Descriptors.MolLogP(mol) > 5)
            descriptors['GhoseViolations'] = int(not (160 <= Descriptors.MolWt(mol) <= 480 and 
                                              -0.4 <= Descriptors.MolLogP(mol) <= 5.6 and 
                                              20 <= mol.GetNumAtoms() <= 70 and 
                                              0 <= Descriptors.NumRotatableBonds(mol) <= 10))
            descriptors['VeberViolations'] = int(Descriptors.NumRotatableBonds(mol) > 10 or MolSurf.TPSA(mol) > 140)

            # 原子组成
            descriptors['CarbonCount'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
            descriptors['NitrogenCount'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
            descriptors['OxygenCount'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
            descriptors['SulfurCount'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16)
            descriptors['PhosphorusCount'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 15)
            descriptors['FluorineCount'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
            descriptors['ChlorineCount'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 17)
            descriptors['BromineCount'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 35)
            descriptors['IodineCount'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 53)
            descriptors['HalogenCount'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [9, 17, 35, 53])
            descriptors['FractionCSP3'] = Descriptors.FractionCSP3(mol)

            # 电子性质
            descriptors['PEOE_VSA1'] = MolSurf.PEOE_VSA1(mol)
            descriptors['PEOE_VSA2'] = MolSurf.PEOE_VSA2(mol)
            descriptors['PEOE_VSA3'] = MolSurf.PEOE_VSA3(mol)
            descriptors['SMR_VSA1'] = MolSurf.SMR_VSA1(mol)
            descriptors['SMR_VSA2'] = MolSurf.SMR_VSA2(mol)
            descriptors['SlogP_VSA1'] = MolSurf.SlogP_VSA1(mol)
            descriptors['SlogP_VSA2'] = MolSurf.SlogP_VSA2(mol)

            # 其他物理化学性质
            descriptors['MR'] = Crippen.MolMR(mol)  # 摩尔折射率
            descriptors['NHOH_Count'] = Lipinski.NHOHCount(mol)
            descriptors['NO_Count'] = Lipinski.NOCount(mol)
            descriptors['HallKierAlpha'] = GraphDescriptors.HallKierAlpha(mol)
            descriptors['ValenceElectrons'] = Descriptors.NumValenceElectrons(mol)

            # 复杂度相关指标
            try:
                descriptors['BCUT2D_MWHI'] = Descriptors.BCUT2D_MWHI(mol)
                descriptors['BCUT2D_MWLOW'] = Descriptors.BCUT2D_MWLOW(mol)
                descriptors['BCUT2D_CHGHI'] = Descriptors.BCUT2D_CHGHI(mol)
                descriptors['BCUT2D_CHGLO'] = Descriptors.BCUT2D_CHGLO(mol)
            except:
                descriptors['BCUT2D_MWHI'] = 0.0
                descriptors['BCUT2D_MWLOW'] = 0.0
                descriptors['BCUT2D_CHGHI'] = 0.0
                descriptors['BCUT2D_CHGLO'] = 0.0

            # 适溶性和膜穿透性
            descriptors['MolLogP'] = Descriptors.MolLogP(mol)
            descriptors['MolMR'] = Descriptors.MolMR(mol)

            # 集团贡献
            descriptors['SP3_N_Count'] = len([atom for atom in mol.GetAtoms() if 
                                            atom.GetAtomicNum() == 7 and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3])
            descriptors['SP2_N_Count'] = len([atom for atom in mol.GetAtoms() if 
                                            atom.GetAtomicNum() == 7 and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2])
            descriptors['Amide_Count'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NX3][CX3](=[OX1])')))
            descriptors['Ester_Count'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][CX3](=[OX1])[OX2][#6]')))
            descriptors['Carboxylic_Count'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=[OX1])[OX2H]')))
            descriptors['Ether_Count'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OD2]([#6])[#6]')))
            descriptors['Alcohol_Count'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OX2H]')))
            descriptors['Amine_Count'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NX3;H2,H1,H0;!$(NC=O)]')))

            # 按照指定顺序提取特征值
            features = []
            for desc_name in MOLECULAR_DESCRIPTORS:
                if desc_name in descriptors:
                    value = descriptors[desc_name]
                    # 处理可能的NaN或无穷大值
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    features.append(float(value))
                else:
                    print(f"警告: 未找到描述符: {desc_name}")
                    features.append(0.0)
            
            return features
            
        except Exception as e:
            print(f"化合物特征提取错误: {e}")
            return [0.0] * COMPOUND_TOTAL_DIM
    
    def process_unique_proteins(self):
        """处理所有唯一蛋白质"""
        print(f"\n开始处理 {len(self.unique_proteins)} 个唯一蛋白质...")
        
        protein_features = {}
        processed = 0
        
        for accession, protein_info in self.unique_proteins.items():
            processed += 1
            
            if VERBOSE_OUTPUT:
                print(f"蛋白质处理进度: {processed}/{len(self.unique_proteins)} ({accession})")
            
            # 检查蛋白质特征缓存
            safe_acc = accession.replace('/', '_').replace('\\', '_').replace('|', '_')
            cache_file = os.path.join(self.cache_dir, f"protein_{safe_acc}_features.json")
            
            if CACHE_ENABLED and os.path.exists(cache_file):
                if VERBOSE_OUTPUT:
                    print(f"使用缓存的蛋白质特征: {accession}")
                with open(cache_file, 'r') as f:
                    cached_features = json.load(f)
                protein_features[accession] = cached_features
                continue
            
            # 创建FASTA文件
            fasta_file = self.create_fasta_file(accession, protein_info)
            
            # 运行PSI-BLAST
            pssm_file = self.run_psiblast(fasta_file, accession)
            
            # 提取特征
            if VERBOSE_OUTPUT:
                print(f"提取蛋白质特征: {accession}")
            
            sequence = protein_info['sequence']
            
            pseaac_features = self.extract_pseaac_features(sequence)
            psepssm_features = self.extract_psepssm_features(pssm_file)
            
            # 创建特征字典
            pseaac_dict = dict(zip(self.pseaac_feature_names, pseaac_features))
            psepssm_dict = dict(zip(self.psepssm_feature_names, psepssm_features))
            
            # 保存结果
            feature_data = {
                'accession': accession,
                'sequence_length': len(sequence),
                'pseaac_features': pseaac_dict,
                'psepssm_features': psepssm_dict,
                'pssm_available': pssm_file is not None and os.path.exists(pssm_file)
            }
            
            protein_features[accession] = feature_data
            
            # 缓存特征
            if CACHE_ENABLED:
                with open(cache_file, 'w') as f:
                    json.dump(feature_data, f)
            
            # 清理临时文件
            if os.path.exists(fasta_file):
                os.remove(fasta_file)
        
        return protein_features
    
    def process_unique_compounds(self):
        """处理所有唯一化合物"""
        print(f"\n开始处理 {len(self.unique_compounds)} 个唯一化合物...")
        
        compound_features = {}
        processed = 0
        
        for compound_cid, compound_info in self.unique_compounds.items():
            processed += 1
            
            if VERBOSE_OUTPUT:
                print(f"化合物处理进度: {processed}/{len(self.unique_compounds)} ({compound_cid})")
            
            # 检查化合物特征缓存
            safe_cid = str(compound_cid).replace('/', '_').replace('\\', '_').replace('|', '_')
            cache_file = os.path.join(self.cache_dir, f"compound_{safe_cid}_features.json")
            
            if CACHE_ENABLED and os.path.exists(cache_file):
                if VERBOSE_OUTPUT:
                    print(f"使用缓存的化合物特征: {compound_cid}")
                with open(cache_file, 'r') as f:
                    cached_features = json.load(f)
                compound_features[compound_cid] = cached_features
                continue
            
            # 提取特征
            if VERBOSE_OUTPUT:
                print(f"提取化合物特征: {compound_cid}")
            
            smile = compound_info['smile']
            features = self.extract_compound_features(smile)
            
            # 创建特征字典
            compound_dict = dict(zip(self.compound_feature_names, features))
            
            # 保存结果
            feature_data = {
                'compound_cid': compound_cid,
                'smile': smile,
                'compound_features': compound_dict
            }
            
            compound_features[compound_cid] = feature_data
            
            # 缓存特征
            if CACHE_ENABLED:
                with open(cache_file, 'w') as f:
                    json.dump(feature_data, f)
        
        return compound_features
    
    def combine_features_to_all_records(self, protein_features, compound_features):
        """将蛋白质和化合物特征组合到所有记录"""
        print("\n组合特征到所有记录...")
        
        all_results = []
        
        for record in self.all_records:
            accession = record['accession']
            compound_cid = record['compound_cid']
            
            result = {
                'Protein_Accession': accession,
                'Compound_CID': compound_cid,
                'Smile': record['smile'],
                'Row_Number': record['row_number'],
                'Protein_Is_Unique': record['protein_is_unique'],
                'Compound_Is_Unique': record['compound_is_unique']
            }
            
            # 添加蛋白质特征
            if accession in protein_features:
                prot_features = protein_features[accession]
                result['Sequence_Length'] = prot_features['sequence_length']
                result['PSSM_Available'] = prot_features['pssm_available']
                result.update(prot_features['pseaac_features'])
                result.update(prot_features['psepssm_features'])
            else:
                # 处理失败的情况
                result['Sequence_Length'] = len(record['sequence'])
                result['PSSM_Available'] = False
                for feature_name in self.pseaac_feature_names:
                    result[feature_name] = 0.0 if feature_name != 'Length' else len(record['sequence'])
                for feature_name in self.psepssm_feature_names:
                    result[feature_name] = 0.0
            
            # 添加化合物特征
            if compound_cid in compound_features:
                comp_features = compound_features[compound_cid]
                result.update(comp_features['compound_features'])
            else:
                # 处理失败的情况
                for feature_name in self.compound_feature_names:
                    result[feature_name] = 0.0
            
            # 添加标签
            result['Label'] = record['label']
            
            all_results.append(result)
        
        # 按原始行号排序，保持输入顺序
        all_results.sort(key=lambda x: x['Row_Number'])
        
        return all_results
    
    def save_features_to_files(self, results):
        """保存特征到文件"""
        print(f"\n保存特征到文件...")
        
        base_name = os.path.splitext(os.path.basename(self.input_filename))[0]
        
        # 1. 主要结果文件 - 按要求的格式: Protein_Accession，蛋白特征，Compound_CID，化合物特征，label
        combined_file = os.path.join(self.output_dir, f'{base_name}_combined_features.csv')
        with open(combined_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 构建表头：Protein_Accession + 蛋白特征 + Compound_CID + 化合物特征 + Label
            header = ['Protein_Accession']
            header.extend(self.pseaac_feature_names)
            header.extend(self.psepssm_feature_names)
            header.append('Compound_CID')
            header.extend(self.compound_feature_names)
            header.append('Label')
            writer.writerow(header)
            
            for result in results:
                row = [result['Protein_Accession']]
                
                # 添加蛋白质特征
                for feature_name in self.pseaac_feature_names:
                    row.append(result[feature_name])
                for feature_name in self.psepssm_feature_names:
                    row.append(result[feature_name])
                
                # 添加化合物CID
                row.append(result['Compound_CID'])
                
                # 添加化合物特征
                for feature_name in self.compound_feature_names:
                    row.append(result[feature_name])
                
                # 添加标签
                row.append(result['Label'])
                
                writer.writerow(row)
        
        # 2. 详细结果文件 (包含所有信息)
        detailed_file = os.path.join(self.output_dir, f'{base_name}_detailed_features.csv')
        with open(detailed_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            header = ['Protein_Accession', 'Compound_CID', 'Smile', 'Sequence_Length', 'PSSM_Available', 
                     'Row_Number', 'Protein_Is_Unique', 'Compound_Is_Unique']
            header.extend(self.pseaac_feature_names)
            header.extend(self.psepssm_feature_names)
            header.extend(self.compound_feature_names)
            header.append('Label')
            writer.writerow(header)
            
            for result in results:
                row = [
                    result['Protein_Accession'],
                    result['Compound_CID'],
                    result['Smile'],
                    result['Sequence_Length'],
                    result['PSSM_Available'],
                    result['Row_Number'],
                    result['Protein_Is_Unique'],
                    result['Compound_Is_Unique']
                ]
                
                for feature_name in self.pseaac_feature_names:
                    row.append(result[feature_name])
                for feature_name in self.psepssm_feature_names:
                    row.append(result[feature_name])
                for feature_name in self.compound_feature_names:
                    row.append(result[feature_name])
                
                row.append(result['Label'])
                
                writer.writerow(row)
        
        # 3. 特征描述文件
        desc_file = os.path.join(self.output_dir, f'{base_name}_feature_descriptions.json')
        feature_info = {
            'input_file': self.input_filename,
            'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'user': 'woyaokaoyanhaha',
            'output_format': 'Protein_Accession + Protein_Features + Compound_CID + Compound_Features + Label',
            'deduplication_strategy': 'Separate deduplication for proteins and compounds',
            'feature_extraction_params': {
                'pseaac_total_dim': PSEAAC_TOTAL_DIM,
                'pseaac_lambda': PSEAAC_LAMBDA,
                'psepssm_total_dim': PSEPSSM_TOTAL_DIM,
                'psepssm_lambda': PSEPSSM_LAMBDA,
                'compound_total_dim': COMPOUND_TOTAL_DIM,
                'total_features': PSEAAC_TOTAL_DIM + PSEPSSM_TOTAL_DIM + COMPOUND_TOTAL_DIM,
                'psiblast_iterations': PSIBLAST_ITERATIONS,
                'psiblast_evalue': PSIBLAST_EVALUE
            },
            'protein_features': {
                'pseaac_features': {
                    'description': 'Pseudo Amino Acid Composition features',
                    'dimensions': len(self.pseaac_feature_names),
                    'feature_names': self.pseaac_feature_names
                },
                'psepssm_features': {
                    'description': 'Pseudo Position-Specific Scoring Matrix features',
                    'dimensions': len(self.psepssm_feature_names),
                    'feature_names': self.psepssm_feature_names
                }
            },
            'compound_features': {
                'description': 'Comprehensive molecular descriptors for compounds',
                'dimensions': len(self.compound_feature_names),
                'feature_names': self.compound_feature_names,
                'descriptors': MOLECULAR_DESCRIPTORS,
                'feature_groups': {
                    'basic_properties': ['MW', 'ExactMW', 'LogP', 'HBA', 'HBD', 'RotBonds', 'AromaticRings', 'RingCount'],
                    'topological_properties': ['TPSA', 'LabuteASA', 'BertzCT', 'Chi0v', 'Chi1v'],
                    'drug_likeness': ['LipinskiViolations', 'QED', 'RO5_Violations', 'GhoseViolations', 'VeberViolations'],
                    'atom_composition': ['CarbonCount', 'NitrogenCount', 'OxygenCount', 'HalogenCount'],
                    'functional_groups': ['Amide_Count', 'Ester_Count', 'Carboxylic_Count', 'Ether_Count', 'Alcohol_Count']
                }
            }
        }
        
        with open(desc_file, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        # 4. 处理统计信息
        stats_file = os.path.join(self.output_dir, f'{base_name}_processing_stats.json')
        stats = {
            'input_file': self.input_filename,
            'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'user': 'woyaokaoyanhaha',
            'dedup_mode': 'Separate: Protein_Accession & Compound_CID',
            'total_records': len(results),
            'unique_proteins': len(self.unique_proteins),
            'unique_compounds': len(self.unique_compounds),
            'protein_deduplication_rate': (len(results) - len(self.unique_proteins)) / len(results) * 100,
            'compound_deduplication_rate': (len(results) - len(self.unique_compounds)) / len(results) * 100,
            'successful_pssm': sum(1 for r in results if r['PSSM_Available']),
            'feature_dimensions': {
                'protein_pseaac': PSEAAC_TOTAL_DIM,
                'protein_psepssm': PSEPSSM_TOTAL_DIM,
                'compound': COMPOUND_TOTAL_DIM,
                'total': PSEAAC_TOTAL_DIM + PSEPSSM_TOTAL_DIM + COMPOUND_TOTAL_DIM
            }
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"特征文件已保存:")
        print(f"  主要结果 (按要求格式): {combined_file}")
        print(f"  详细结果: {detailed_file}")
        print(f"  特征说明: {desc_file}")
        print(f"  统计信息: {stats_file}")
        
        print(f"\n主要结果文件格式:")
        print(f"  Protein_Accession + 蛋白质特征({PSEAAC_TOTAL_DIM + PSEPSSM_TOTAL_DIM}维) + Compound_CID + 化合物特征({COMPOUND_TOTAL_DIM}维) + Label")
        
        return stats

def main():
    if len(sys.argv) != 3:
        print("用法: python3 extract_protein_compound_features.py <input.csv> <swissprot_db_path>")
        print("示例: python3 extract_protein_compound_features.py data.csv ./swissprot_db/uniprot_sprot")
        print("\n输入CSV文件应包含以下列 (支持自动检测):")
        print("  - 蛋白质登录号: Protein_Accession, ProteinAccession, Accession, Protein_ID, ProteinID")
        print("  - 蛋白质序列: Sequence, Protein_Sequence, ProteinSequence, Seq")
        print("  - 化合物CID: Compound_CID, CompoundCID, CID, Compound_ID, CompoundID")
        print("  - 化合物SMILES: Smile, SMILES, Canonical_SMILES, CanonicalSMILES")
        print("  - 标签 (可选): label, Label, Class, Target, Y")
        print("\n去重策略: 分别对Protein_Accession和Compound_CID去重计算特征，最后组合")
        print("输出格式: Protein_Accession + 蛋白特征 + Compound_CID + 化合物特征 + Label")
        print(f"化合物特征: {COMPOUND_TOTAL_DIM}维全面分子描述符")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    swissprot_db = sys.argv[2]
    
    # 检查输入文件
    if not os.path.exists(input_csv):
        print(f"错误: 输入文件不存在: {input_csv}")
        sys.exit(1)
    
    # 检查数据库
    if not os.path.exists(swissprot_db + ".pin"):
        print(f"错误: SwissProt数据库不存在或未格式化: {swissprot_db}")
        sys.exit(1)
    
    # 根据输入文件名创建工作目录
    work_dir = get_output_dir_name(input_csv)
    
    print("=" * 80)
    print("蛋白质-化合物特征提取脚本")
    print(f"用户: woyaokaoyanhaha")
    print(f"时间: 2025-06-15 01:33:18")
    print(f"输入文件: {input_csv}")
    print(f"输出目录: {work_dir}")
    print(f"去重策略: 分别对蛋白质和化合物去重")
    print(f"输出格式: Protein_Accession + 蛋白特征 + Compound_CID + 化合物特征 + Label")
    print(f"蛋白质特征维度: {PSEAAC_TOTAL_DIM + PSEPSSM_TOTAL_DIM}")
    print(f"化合物特征维度: {COMPOUND_TOTAL_DIM} (全面分子描述符)")
    print(f"总特征维度: {PSEAAC_TOTAL_DIM + PSEPSSM_TOTAL_DIM + COMPOUND_TOTAL_DIM}")
    print("=" * 80)
    
    # 初始化提取器
    extractor = ProteinCompoundFeatureExtractor(swissprot_db, work_dir, input_csv)
    
    start_time = time.time()
    
    try:
        # 1. 加载和分别去重
        unique_protein_count, unique_compound_count = extractor.load_and_deduplicate(input_csv)
        
        # 2. 处理唯一蛋白质
        protein_features = extractor.process_unique_proteins()
        
        # 3. 处理唯一化合物
        compound_features = extractor.process_unique_compounds()
        
        # 4. 组合特征到所有记录
        all_results = extractor.combine_features_to_all_records(protein_features, compound_features)
        
        # 5. 保存结果
        stats = extractor.save_features_to_files(all_results)
        
        # 6. 输出统计信息
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n" + "=" * 80)
        print("处理完成!")
        print(f"总处理时间: {processing_time:.2f} 秒")
        print(f"平均每个唯一蛋白质: {processing_time/unique_protein_count:.2f} 秒")
        print(f"平均每个唯一化合物: {processing_time/unique_compound_count:.2f} 秒")
        print(f"分别对蛋白质和化合物去重计算特征")
        print(f"使用全面的分子描述符 ({COMPOUND_TOTAL_DIM}维)")
        print(f"结果保存在: {work_dir}")
        print(f"\n最终输出格式:")
        print(f"  Protein_Accession + 蛋白质特征({PSEAAC_TOTAL_DIM + PSEPSSM_TOTAL_DIM}维) + Compound_CID + 化合物特征({COMPOUND_TOTAL_DIM}维) + Label")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n用户中断处理")
        sys.exit(1)
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()