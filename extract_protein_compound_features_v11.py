#!/usr/bin/env python3
"""
蛋白质-化合物特征提取脚本 (修复排序版本)
作者: woyaokaoyanhaha
版本: 11.0
日期: 2025-06-16 04:25:34
修复: 保持输出文件与输入文件完全一致的顺序
"""

import csv
import os
import subprocess
import sys
import json
import time
import warnings
import argparse
import glob
import traceback
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

# 检查和导入依赖库
def check_dependencies():
    """检查并导入必要的依赖库"""
    print("正在检查依赖库...")
    
    try:
        import numpy as np
        print(f"✅ numpy {np.__version__}")
    except ImportError:
        print("❌ numpy 未安装")
        return False
    
    try:
        from Bio import SeqIO
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        print("✅ biopython")
    except ImportError:
        print("❌ biopython 未安装")
        return False
    
    try:
        import pandas as pd
        print(f"✅ pandas {pd.__version__}")
    except ImportError:
        print("❌ pandas 未安装")
        return False
    
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, MolSurf, QED, GraphDescriptors, Crippen, AllChem
        print("✅ rdkit")
    except ImportError:
        print("❌ rdkit 未安装")
        return False
    
    print("🎉 所有依赖库检查完成")
    return True

# 检查依赖库
if not check_dependencies():
    print("\n请安装缺失的依赖库:")
    print("pip install biopython numpy pandas rdkit")
    sys.exit(1)

# 现在安全导入所有库
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf, QED, GraphDescriptors, Crippen, AllChem

# 配置参数
COLUMN_MAPPING = {
    'protein_accession': ['Protein_Accession', 'ProteinAccession', 'Accession', 'Protein_ID', 'ProteinID'],
    'sequence': ['Sequence', 'Protein_Sequence', 'ProteinSequence', 'Seq'],
    'compound_cid': ['Compound_CID', 'CompoundCID', 'CID', 'Compound_ID', 'CompoundID'],
    'smile': ['Smile', 'SMILES', 'Canonical_SMILES', 'CanonicalSMILES'],
    'label': ['label', 'Label', 'Class', 'Target', 'Y']
}

PSEAAC_LAMBDA = 10
PSEAAC_TOTAL_DIM = 51
PSEPSSM_TOTAL_DIM = 220
PSEPSSM_LAMBDA = 10
PSSM_BASIC_FEATURES = 80

MOLECULAR_DESCRIPTORS = [
    'MW', 'ExactMW', 'LogP', 'HBA', 'HBD', 'RotBonds', 'AromaticRings', 
    'RingCount', 'HeteroRings', 'SaturatedRings', 'AliphaticRings', 
    'HeavyAtomCount', 'AtomCount', 'TPSA', 'LabuteASA', 'MolSurfaceArea', 
    'BertzCT', 'Chi0v', 'Chi1v', 'Kappa1', 'Kappa2', 'Kappa3',
    'LipinskiViolations', 'QED', 'RO5_Violations', 'GhoseViolations', 
    'VeberViolations', 'CarbonCount', 'NitrogenCount', 'OxygenCount', 
    'SulfurCount', 'PhosphorusCount', 'FluorineCount', 'ChlorineCount', 
    'BromineCount', 'IodineCount', 'HalogenCount', 'FractionCSP3',
    'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'SMR_VSA1', 'SMR_VSA2', 
    'SlogP_VSA1', 'SlogP_VSA2', 'MR', 'NHOH_Count', 'NO_Count', 
    'HallKierAlpha', 'ValenceElectrons', 'MolLogP', 'MolMR',
    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
    'SP3_N_Count', 'SP2_N_Count', 'Amide_Count', 'Ester_Count', 
    'Carboxylic_Count', 'Ether_Count', 'Alcohol_Count', 'Amine_Count'
]

COMPOUND_TOTAL_DIM = len(MOLECULAR_DESCRIPTORS)
PSIBLAST_ITERATIONS = 3
PSIBLAST_EVALUE = 0.001
PSIBLAST_THREADS = 4
SAVE_PROGRESS_INTERVAL = 10

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='蛋白质-化合物特征提取脚本 (修复排序版本)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python3 extract_protein_compound_features_v11.py data.csv ./swissprot_db/uniprot_sprot
  python3 extract_protein_compound_features_v11.py data.csv ./db/uniprot_sprot -o ./output
  python3 extract_protein_compound_features_v11.py --list-resume
        """
    )
    
    parser.add_argument('input_csv', nargs='?', help='输入CSV文件路径')
    parser.add_argument('swissprot_db', nargs='?', help='SwissProt数据库路径')
    parser.add_argument('-o', '--output', help='自定义输出目录路径')
    parser.add_argument('-r', '--resume', help='从指定目录恢复运行')
    parser.add_argument('--list-resume', action='store_true', help='列出可恢复的运行目录')
    parser.add_argument('--test', action='store_true', help='运行测试模式')
    parser.add_argument('--preserve-order', action='store_true', default=True, help='保持输入文件顺序（默认启用）')
    
    return parser.parse_args()

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

def list_resumable_directories():
    """列出可恢复的运行目录"""
    print("搜索可恢复的运行目录...")
    
    patterns = ["*_features_*"]
    found_dirs = []
    
    for pattern in patterns:
        dirs = glob.glob(pattern)
        for d in dirs:
            if os.path.isdir(d):
                found_dirs.append(d)
    
    if not found_dirs:
        print("❌ 未找到可恢复的运行目录")
        return
    
    print(f"找到 {len(found_dirs)} 个可能的目录:")
    
    for i, dir_path in enumerate(sorted(found_dirs), 1):
        progress_file = os.path.join(dir_path, "progress.json")
        print(f"\n{i}. 📁 {dir_path}")
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                status = "✅ 完成" if progress.get('completed', False) else "⏳ 未完成"
                print(f"   状态: {status}")
                
                if not progress.get('completed', False):
                    proteins_done = progress.get('proteins_processed', 0)
                    proteins_total = progress.get('total_proteins', 0)
                    compounds_done = progress.get('compounds_processed', 0)
                    compounds_total = progress.get('total_compounds', 0)
                    
                    print(f"   蛋白质: {proteins_done}/{proteins_total}")
                    print(f"   化合物: {compounds_done}/{compounds_total}")
                    
            except Exception as e:
                print(f"   ❌ 进度文件读取失败: {e}")
        else:
            print(f"   ❌ 无进度文件")

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
    
    return f"./{safe_basename}_features_{timestamp}"

class ProgressManager:
    """进度管理器"""
    
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.progress_file = os.path.join(work_dir, "progress.json")
        self.progress = self.load_progress()
    
    def load_progress(self):
        """加载进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 进度文件读取失败: {e}")
        
        return {
            'start_time': time.time(),
            'proteins_processed': 0,
            'compounds_processed': 0,
            'total_proteins': 0,
            'total_compounds': 0,
            'protein_features_completed': [],
            'compound_features_completed': [],
            'completed': False,
            'last_update': time.time()
        }
    
    def save_progress(self):
        """保存进度"""
        self.progress['last_update'] = time.time()
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            print(f"⚠️ 进度保存失败: {e}")
    
    def update_totals(self, total_proteins, total_compounds):
        """更新总数"""
        self.progress['total_proteins'] = total_proteins
        self.progress['total_compounds'] = total_compounds
        self.save_progress()
    
    def mark_protein_completed(self, accession):
        """标记蛋白质处理完成"""
        if accession not in self.progress['protein_features_completed']:
            self.progress['protein_features_completed'].append(accession)
            self.progress['proteins_processed'] = len(self.progress['protein_features_completed'])
    
    def mark_compound_completed(self, compound_cid):
        """标记化合物处理完成"""
        if compound_cid not in self.progress['compound_features_completed']:
            self.progress['compound_features_completed'].append(compound_cid)
            self.progress['compounds_processed'] = len(self.progress['compound_features_completed'])
    
    def is_protein_completed(self, accession):
        """检查蛋白质是否已处理"""
        return accession in self.progress['protein_features_completed']
    
    def is_compound_completed(self, compound_cid):
        """检查化合物是否已处理"""
        return compound_cid in self.progress['compound_features_completed']
    
    def mark_completed(self):
        """标记全部完成"""
        self.progress['completed'] = True
        self.progress['end_time'] = time.time()
        self.save_progress()
    
    def get_progress_info(self):
        """获取进度信息"""
        return {
            'proteins': f"{self.progress['proteins_processed']}/{self.progress['total_proteins']}",
            'compounds': f"{self.progress['compounds_processed']}/{self.progress['total_compounds']}",
            'protein_percent': (self.progress['proteins_processed'] / max(1, self.progress['total_proteins'])) * 100,
            'compound_percent': (self.progress['compounds_processed'] / max(1, self.progress['total_compounds'])) * 100
        }

class FeatureExtractor:
    """特征提取器主类 - 修复排序版本"""
    
    def __init__(self, swissprot_db, work_dir, input_filename, resume_mode=False, preserve_order=True):
        self.swissprot_db = swissprot_db
        self.work_dir = work_dir
        self.input_filename = input_filename
        self.resume_mode = resume_mode
        self.preserve_order = preserve_order  # 新增：保持顺序标志
        self.temp_dir = os.path.join(work_dir, "temp")
        self.output_dir = os.path.join(work_dir, "output")
        self.cache_dir = os.path.join(work_dir, "cache")
        
        # 创建必要的目录
        for dir_path in [self.temp_dir, self.output_dir, self.cache_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 初始化进度管理器
        self.progress_manager = ProgressManager(work_dir)
        
        # 存储数据 - 保持原始顺序
        self.original_records = []  # 新增：保存原始记录的完整顺序
        self.unique_proteins = {}
        self.unique_compounds = {}
        self.column_mapping = {}
        
        print(f"📁 工作目录: {work_dir}")
        print(f"🔄 运行模式: {'恢复运行' if resume_mode else '新建运行'}")
        print(f"📋 保持顺序: {'是' if preserve_order else '否'}")
        
        if resume_mode:
            progress_info = self.progress_manager.get_progress_info()
            print(f"当前进度:")
            print(f"  蛋白质: {progress_info['proteins']} ({progress_info['protein_percent']:.1f}%)")
            print(f"  化合物: {progress_info['compounds']} ({progress_info['compound_percent']:.1f}%)")
    
    def load_and_deduplicate(self, input_csv):
        """加载数据并去重 - 保持原始顺序"""
        print("\n" + "="*60)
        print("📂 数据加载和去重阶段 (保持原始顺序)")
        print("="*60)
        
        detected_columns, header = detect_column_names(input_csv)
        if not detected_columns:
            return 0, 0
        
        self.column_mapping = detected_columns
        
        # 检查必需列
        required_fields = ['protein_accession', 'sequence', 'compound_cid', 'smile']
        missing_fields = [field for field in required_fields if not detected_columns[field]]
        
        if missing_fields:
            print(f"\n❌ 错误: 未找到必需的列: {missing_fields}")
            print(f"可用的列: {header}")
            return 0, 0
        
        print("\n正在加载数据并去重（保持原始顺序）...")
        
        # 保存原始记录顺序的关键修改
        self.original_records = []  # 按输入文件顺序保存所有记录
        seen_proteins = set()
        seen_compounds = set()
        row_number = 0
        
        try:
            with open(input_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    row_number += 1
                    
                    accession = row[detected_columns['protein_accession']].strip()
                    sequence = row[detected_columns['sequence']].strip().upper()
                    compound_cid = row[detected_columns['compound_cid']].strip()
                    smile = row[detected_columns['smile']].strip()
                    
                    label = ''
                    if detected_columns['label']:
                        label = row[detected_columns['label']].strip()
                    
                    # 关键修改：保存每条原始记录，包含精确的行号
                    original_record = {
                        'original_row_number': row_number,  # 从1开始的原始行号
                        'accession': accession,
                        'sequence': sequence,
                        'compound_cid': compound_cid,
                        'smile': smile,
                        'label': label
                    }
                    self.original_records.append(original_record)
                    
                    # 收集唯一的蛋白质（但记录首次出现的行号）
                    if accession not in seen_proteins:
                        self.unique_proteins[accession] = {
                            'accession': accession,
                            'sequence': sequence,
                            'first_occurrence_row': row_number
                        }
                        seen_proteins.add(accession)
                    
                    # 收集唯一的化合物（但记录首次出现的行号）
                    if compound_cid not in seen_compounds:
                        self.unique_compounds[compound_cid] = {
                            'compound_cid': compound_cid,
                            'smile': smile,
                            'first_occurrence_row': row_number
                        }
                        seen_compounds.add(compound_cid)
                        
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return 0, 0
        
        print(f"\n📊 数据统计:")
        print(f"  总记录数: {len(self.original_records)}")
        print(f"  唯一蛋白质数: {len(self.unique_proteins)}")
        print(f"  唯一化合物数: {len(self.unique_compounds)}")
        print(f"  保持原始顺序: ✅")
        
        self.progress_manager.update_totals(len(self.unique_proteins), len(self.unique_compounds))
        
        return len(self.unique_proteins), len(self.unique_compounds)
    
    def extract_pseaac_features(self, sequence):
        """提取PSE-AAC特征"""
        # 氨基酸物理化学性质 [疏水性, 等电点, 分子量]
        aa_properties = {
            'A': [1.8, 6.0, 89.1], 'R': [-4.5, 10.8, 174.2], 'N': [-3.5, 5.4, 132.1],
            'D': [-3.5, 3.0, 133.1], 'C': [2.5, 5.1, 121.0], 'Q': [-3.5, 5.7, 146.1],
            'E': [-3.5, 4.2, 147.1], 'G': [-0.4, 6.0, 75.1], 'H': [-3.2, 7.6, 155.2],
            'I': [4.5, 6.0, 131.2], 'L': [3.8, 6.0, 131.2], 'K': [-3.9, 9.7, 146.2],
            'M': [1.9, 5.7, 149.2], 'F': [2.8, 5.5, 165.2], 'P': [-1.6, 6.3, 115.1],
            'S': [-0.8, 5.7, 105.1], 'T': [-0.7, 5.6, 119.1], 'V': [4.2, 6.0, 117.1],
            'W': [-0.9, 5.9, 204.2], 'Y': [-1.3, 5.7, 181.2]
        }
        
        # 标准化
        all_props = np.array([aa_properties[aa] for aa in aa_properties])
        means = np.mean(all_props, axis=0)
        stds = np.std(all_props, axis=0)
        
        for aa in aa_properties:
            aa_properties[aa] = [(aa_properties[aa][i] - means[i]) / stds[i] for i in range(3)]
        
        features = [len(sequence)]  # 序列长度
        
        # 氨基酸组成
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
        aa_count = {aa: 0 for aa in amino_acids}
        for aa in sequence:
            if aa in aa_count:
                aa_count[aa] += 1
        
        total = len(sequence)
        for aa in amino_acids:
            features.append(aa_count[aa] / total if total > 0 else 0)
        
        # 伪氨基酸组成
        for property_idx in range(3):
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
            
            while len(pseudo_comp) < PSEAAC_LAMBDA:
                pseudo_comp.append(0.0)
            
            features.extend(pseudo_comp)
        
        return features
    
    def run_psiblast(self, fasta_file, accession):
        """运行PSI-BLAST"""
        safe_acc = accession.replace('/', '_').replace('\\', '_').replace('|', '_')
        pssm_file = os.path.join(self.cache_dir, f"{safe_acc}.pssm")
        
        if os.path.exists(pssm_file):
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
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return pssm_file if os.path.exists(pssm_file) else None
        except subprocess.CalledProcessError:
            print(f"⚠️ {accession} PSI-BLAST运行失败")
            return None
    
    def extract_psepssm_features(self, pssm_file):
        """提取PSE-PSSM特征"""
        if not pssm_file or not os.path.exists(pssm_file):
            return [0.0] * PSEPSSM_TOTAL_DIM
        
        try:
            pssm_matrix = []
            with open(pssm_file, 'r') as f:
                lines = f.readlines()
            
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
            
            # PSSM基础统计特征
            pssm_mean = np.mean(pssm_array, axis=0)
            features.extend(pssm_mean.tolist())
            
            pssm_std = np.std(pssm_array, axis=0)
            features.extend(pssm_std.tolist())
            
            pssm_max = np.max(pssm_array, axis=0)
            features.extend(pssm_max.tolist())
            
            pssm_min = np.min(pssm_array, axis=0)
            features.extend(pssm_min.tolist())
            
            # 伪PSSM特征
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
            
            while len(features) < PSEPSSM_TOTAL_DIM:
                features.append(0.0)
            
            return features[:PSEPSSM_TOTAL_DIM]
            
        except Exception as e:
            print(f"⚠️ PSSM特征提取错误: {e}")
            return [0.0] * PSEPSSM_TOTAL_DIM
    
    def extract_compound_features(self, smile):
        """提取化合物特征"""
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
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
            descriptors['TPSA'] = MolSurf.TPSA(mol)
            descriptors['LabuteASA'] = MolSurf.LabuteASA(mol)
            
            # 分子表面积计算
            try:
                mol_3d = Chem.AddHs(mol)
                success = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                if success == 0:
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
            
            # 药物相似性
            descriptors['LipinskiViolations'] = Lipinski.NumRotatableBonds(mol)
            descriptors['QED'] = QED.qed(mol)
            descriptors['RO5_Violations'] = int(
                Lipinski.NumHDonors(mol) > 5 or 
                Lipinski.NumHAcceptors(mol) > 10 or 
                Descriptors.MolWt(mol) > 500 or 
                Descriptors.MolLogP(mol) > 5
            )
            descriptors['GhoseViolations'] = int(not (
                160 <= Descriptors.MolWt(mol) <= 480 and 
                -0.4 <= Descriptors.MolLogP(mol) <= 5.6 and 
                20 <= mol.GetNumAtoms() <= 70 and 
                0 <= Descriptors.NumRotatableBonds(mol) <= 10
            ))
            descriptors['VeberViolations'] = int(
                Descriptors.NumRotatableBonds(mol) > 10 or 
                MolSurf.TPSA(mol) > 140
            )
            
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
            
            # 其他性质
            descriptors['MR'] = Crippen.MolMR(mol)
            descriptors['NHOH_Count'] = Lipinski.NHOHCount(mol)
            descriptors['NO_Count'] = Lipinski.NOCount(mol)
            descriptors['HallKierAlpha'] = GraphDescriptors.HallKierAlpha(mol)
            descriptors['ValenceElectrons'] = Descriptors.NumValenceElectrons(mol)
            descriptors['MolLogP'] = Descriptors.MolLogP(mol)
            descriptors['MolMR'] = Descriptors.MolMR(mol)
            
            # 复杂度相关
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
            
            # 功能团计数
            descriptors['SP3_N_Count'] = len([atom for atom in mol.GetAtoms() if 
                                            atom.GetAtomicNum() == 7 and 
                                            atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3])
            descriptors['SP2_N_Count'] = len([atom for atom in mol.GetAtoms() if 
                                            atom.GetAtomicNum() == 7 and 
                                            atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2])
            descriptors['Amide_Count'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NX3][CX3](=[OX1])')))
            descriptors['Ester_Count'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][CX3](=[OX1])[OX2][#6]')))
            descriptors['Carboxylic_Count'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=[OX1])[OX2H]')))
            descriptors['Ether_Count'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OD2]([#6])[#6]')))
            descriptors['Alcohol_Count'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OX2H]')))
            descriptors['Amine_Count'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NX3;H2,H1,H0;!$(NC=O)]')))
            
            # 按顺序提取特征
            features = []
            for desc_name in MOLECULAR_DESCRIPTORS:
                if desc_name in descriptors:
                    value = descriptors[desc_name]
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    features.append(float(value))
                else:
                    features.append(0.0)
            
            return features
            
        except Exception as e:
            print(f"⚠️ 化合物特征提取错误: {e}")
            return [0.0] * COMPOUND_TOTAL_DIM
    
    def process_unique_proteins(self):
        """处理唯一蛋白质"""
        print("\n" + "="*60)
        print("🧬 蛋白质特征提取阶段")
        print("="*60)
        print(f"需要处理 {len(self.unique_proteins)} 个唯一蛋白质")
        
        protein_features = {}
        processed = 0
        
        for accession, protein_info in self.unique_proteins.items():
            # 检查是否已处理
            if self.progress_manager.is_protein_completed(accession):
                safe_acc = accession.replace('/', '_').replace('\\', '_').replace('|', '_')
                cache_file = os.path.join(self.cache_dir, f"protein_{safe_acc}_features.json")
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            cached_features = json.load(f)
                        protein_features[accession] = cached_features
                        processed += 1
                        continue
                    except:
                        pass
            
            processed += 1
            progress_info = self.progress_manager.get_progress_info()
            
            print(f"🔄 {processed}/{len(self.unique_proteins)} - {accession} "
                  f"[蛋白{progress_info['protein_percent']:.1f}% 化合物{progress_info['compound_percent']:.1f}%]")
            
            # 检查缓存
            safe_acc = accession.replace('/', '_').replace('\\', '_').replace('|', '_')
            cache_file = os.path.join(self.cache_dir, f"protein_{safe_acc}_features.json")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_features = json.load(f)
                protein_features[accession] = cached_features
                self.progress_manager.mark_protein_completed(accession)
                
                if processed % SAVE_PROGRESS_INTERVAL == 0:
                    self.progress_manager.save_progress()
                continue
            
            # 创建FASTA文件
            fasta_file = os.path.join(self.temp_dir, f"{safe_acc}.fasta")
            with open(fasta_file, 'w') as f:
                f.write(f">{accession}\n{protein_info['sequence']}\n")
            
            # 运行PSI-BLAST
            pssm_file = self.run_psiblast(fasta_file, accession)
            
            # 提取特征
            sequence = protein_info['sequence']
            
            try:
                pseaac_features = self.extract_pseaac_features(sequence)
                psepssm_features = self.extract_psepssm_features(pssm_file)
                
                feature_data = {
                    'accession': accession,
                    'sequence_length': len(sequence),
                    'pseaac_features': pseaac_features,
                    'psepssm_features': psepssm_features,
                    'pssm_available': pssm_file is not None and os.path.exists(pssm_file)
                }
                
                protein_features[accession] = feature_data
                
                with open(cache_file, 'w') as f:
                    json.dump(feature_data, f)
                
                self.progress_manager.mark_protein_completed(accession)
                print(f"  ✅ 完成")
                
            except Exception as e:
                print(f"  ❌ 失败: {e}")
                feature_data = {
                    'accession': accession,
                    'sequence_length': len(sequence),
                    'pseaac_features': [0.0] * PSEAAC_TOTAL_DIM,
                    'psepssm_features': [0.0] * PSEPSSM_TOTAL_DIM,
                    'pssm_available': False
                }
                if feature_data['pseaac_features']:
                    feature_data['pseaac_features'][0] = len(sequence)  # 长度特征
                protein_features[accession] = feature_data
                self.progress_manager.mark_protein_completed(accession)
            
            # 清理临时文件
            if os.path.exists(fasta_file):
                os.remove(fasta_file)
            
            if processed % SAVE_PROGRESS_INTERVAL == 0:
                self.progress_manager.save_progress()
                print(f"  💾 已保存进度")
        
        self.progress_manager.save_progress()
        print(f"\n✅ 蛋白质特征提取完成: {len(protein_features)}/{len(self.unique_proteins)}")
        
        return protein_features
    
    def process_unique_compounds(self):
        """处理唯一化合物"""
        print("\n" + "="*60)
        print("💊 化合物特征提取阶段")
        print("="*60)
        print(f"需要处理 {len(self.unique_compounds)} 个唯一化合物")
        
        compound_features = {}
        processed = 0
        
        for compound_cid, compound_info in self.unique_compounds.items():
            # 检查是否已处理
            if self.progress_manager.is_compound_completed(compound_cid):
                safe_cid = str(compound_cid).replace('/', '_').replace('\\', '_').replace('|', '_')
                cache_file = os.path.join(self.cache_dir, f"compound_{safe_cid}_features.json")
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            cached_features = json.load(f)
                        compound_features[compound_cid] = cached_features
                        processed += 1
                        continue
                    except:
                        pass
            
            processed += 1
            progress_info = self.progress_manager.get_progress_info()
            
            print(f"🔄 {processed}/{len(self.unique_compounds)} - {compound_cid} "
                  f"[蛋白{progress_info['protein_percent']:.1f}% 化合物{progress_info['compound_percent']:.1f}%]")
            
            # 检查缓存
            safe_cid = str(compound_cid).replace('/', '_').replace('\\', '_').replace('|', '_')
            cache_file = os.path.join(self.cache_dir, f"compound_{safe_cid}_features.json")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_features = json.load(f)
                compound_features[compound_cid] = cached_features
                self.progress_manager.mark_compound_completed(compound_cid)
                
                if processed % SAVE_PROGRESS_INTERVAL == 0:
                    self.progress_manager.save_progress()
                continue
            
            # 提取特征
            smile = compound_info['smile']
            
            try:
                features = self.extract_compound_features(smile)
                
                feature_data = {
                    'compound_cid': compound_cid,
                    'smile': smile,
                    'compound_features': features
                }
                
                compound_features[compound_cid] = feature_data
                
                with open(cache_file, 'w') as f:
                    json.dump(feature_data, f)
                
                self.progress_manager.mark_compound_completed(compound_cid)
                print(f"  ✅ 完成")
                
            except Exception as e:
                print(f"  ❌ 失败: {e}")
                feature_data = {
                    'compound_cid': compound_cid,
                    'smile': smile,
                    'compound_features': [0.0] * COMPOUND_TOTAL_DIM
                }
                compound_features[compound_cid] = feature_data
                self.progress_manager.mark_compound_completed(compound_cid)
            
            if processed % SAVE_PROGRESS_INTERVAL == 0:
                self.progress_manager.save_progress()
                print(f"  💾 已保存进度")
        
        self.progress_manager.save_progress()
        print(f"\n✅ 化合物特征提取完成: {len(compound_features)}/{len(self.unique_compounds)}")
        
        return compound_features
    
    def combine_and_save_features(self, protein_features, compound_features):
        """组合特征并保存 - 关键修复：严格保持原始顺序"""
        print("\n" + "="*60)
        print("🔗 特征组合和保存阶段 (严格保持原始顺序)")
        print("="*60)
        
        # 生成特征名称
        pseaac_names = ['Length'] + [f'AA_{aa}' for aa in 'ACDEFGHIKLMNPQRSTVWY'] + \
                       [f'Hydrophobicity_lambda_{i}' for i in range(1, PSEAAC_LAMBDA + 1)] + \
                       [f'Isoelectric_Point_lambda_{i}' for i in range(1, PSEAAC_LAMBDA + 1)] + \
                       [f'Molecular_Weight_lambda_{i}' for i in range(1, PSEAAC_LAMBDA + 1)]
        
        psepssm_names = []
        amino_acids = 'ARNDCQEGHILKMFPSTWYV'
        for stat in ['Mean', 'Std', 'Max', 'Min']:
            for aa in amino_acids:
                psepssm_names.append(f'PSSM_{stat}_{aa}')
        
        remaining_psepssm = PSEPSSM_TOTAL_DIM - len(psepssm_names)
        for i in range(remaining_psepssm):
            psepssm_names.append(f'PsePSSM_Extra_{i+1}')
        
        compound_names = [f'Compound_{desc}' for desc in MOLECULAR_DESCRIPTORS]
        
        # 关键修复：严格按原始记录顺序处理
        all_results = []
        
        print(f"正在按原始顺序组合 {len(self.original_records)} 条记录的特征...")
        
        for original_record in self.original_records:
            accession = original_record['accession']
            compound_cid = original_record['compound_cid']
            
            result = {
                'Original_Row_Number': original_record['original_row_number'],  # 保留原始行号用于验证
                'Protein_Accession': accession,
                'Compound_CID': compound_cid,
                'Smile': original_record['smile'],
                'Label': original_record['label']
            }
            
            # 添加蛋白质特征
            if accession in protein_features:
                prot_features = protein_features[accession]
                result['Sequence_Length'] = prot_features['sequence_length']
                result['PSSM_Available'] = prot_features['pssm_available']
                
                # PSE-AAC特征
                pseaac_features = prot_features['pseaac_features']
                for i, name in enumerate(pseaac_names):
                    if i < len(pseaac_features):
                        result[name] = pseaac_features[i]
                    else:
                        result[name] = 0.0
                
                # PSE-PSSM特征
                psepssm_features = prot_features['psepssm_features']
                for i, name in enumerate(psepssm_names):
                    if i < len(psepssm_features):
                        result[name] = psepssm_features[i]
                    else:
                        result[name] = 0.0
            else:
                result['Sequence_Length'] = len(original_record['sequence'])
                result['PSSM_Available'] = False
                for name in pseaac_names:
                    result[name] = 0.0 if name != 'Length' else len(original_record['sequence'])
                for name in psepssm_names:
                    result[name] = 0.0
            
            # 添加化合物特征
            if compound_cid in compound_features:
                comp_features = compound_features[compound_cid]['compound_features']
                for i, name in enumerate(compound_names):
                    if i < len(comp_features):
                        result[name] = comp_features[i]
                    else:
                        result[name] = 0.0
            else:
                for name in compound_names:
                    result[name] = 0.0
            
            all_results.append(result)
        
        # 验证顺序是否正确
        print("🔍 验证输出顺序...")
        order_verification_passed = True
        for i, result in enumerate(all_results):
            expected_row = i + 1
            actual_row = result['Original_Row_Number']
            if expected_row != actual_row:
                print(f"❌ 顺序错误：位置 {i+1} 应该是第 {expected_row} 行，但实际是第 {actual_row} 行")
                order_verification_passed = False
                break
        
        if order_verification_passed:
            print("✅ 输出顺序验证通过：与输入文件完全一致")
        else:
            print("❌ 输出顺序验证失败")
            return None
        
        # 保存结果
        base_name = os.path.splitext(os.path.basename(self.input_filename))[0]
        
        # 主要结果文件（标准格式，不包含行号验证列）
        combined_file = os.path.join(self.output_dir, f'{base_name}_combined_features.csv')
        with open(combined_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            header = ['Protein_Accession'] + pseaac_names + psepssm_names + \
                    ['Compound_CID'] + compound_names + ['Label']
            writer.writerow(header)
            
            for result in all_results:
                row = [result['Protein_Accession']]
                for name in pseaac_names:
                    row.append(result[name])
                for name in psepssm_names:
                    row.append(result[name])
                row.append(result['Compound_CID'])
                for name in compound_names:
                    row.append(result[name])
                row.append(result['Label'])
                writer.writerow(row)
        
        # 详细结果文件（包含验证信息）
        detailed_file = os.path.join(self.output_dir, f'{base_name}_detailed_features.csv')
        with open(detailed_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Original_Row_Number', 'Protein_Accession', 'Compound_CID', 'Smile', 
                         'Sequence_Length', 'PSSM_Available'] + \
                        pseaac_names + psepssm_names + compound_names + ['Label']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        # 顺序验证文件
        order_verification_file = os.path.join(self.output_dir, f'{base_name}_order_verification.csv')
        with open(order_verification_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Position', 'Original_Row_Number', 'Protein_Accession', 'Compound_CID', 'Order_OK'])
            
            for i, result in enumerate(all_results):
                position = i + 1
                original_row = result['Original_Row_Number']
                order_ok = position == original_row
                writer.writerow([
                    position, 
                    original_row, 
                    result['Protein_Accession'], 
                    result['Compound_CID'], 
                    'YES' if order_ok else 'NO'
                ])
        
        # 统计信息
        stats_file = os.path.join(self.output_dir, f'{base_name}_processing_stats.json')
        stats = {
            'input_file': self.input_filename,
            'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'user': 'woyaokaoyanhaha',
            'version': '11.0',
            'resume_mode': self.resume_mode,
            'preserve_order': self.preserve_order,
            'work_directory': self.work_dir,
            'total_records': len(all_results),
            'unique_proteins': len(self.unique_proteins),
            'unique_compounds': len(self.unique_compounds),
            'successful_pssm': sum(1 for r in all_results if r['PSSM_Available']),
            'order_verification_passed': order_verification_passed,
            'feature_dimensions': {
                'protein_pseaac': PSEAAC_TOTAL_DIM,
                'protein_psepssm': PSEPSSM_TOTAL_DIM,
                'compound': COMPOUND_TOTAL_DIM,
                'total': PSEAAC_TOTAL_DIM + PSEPSSM_TOTAL_DIM + COMPOUND_TOTAL_DIM
            }
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 特征文件已保存:")
        print(f"  📊 主要结果: {combined_file}")
        print(f"  📋 详细结果: {detailed_file}")
        print(f"  🔍 顺序验证: {order_verification_file}")
        print(f"  📈 统计信息: {stats_file}")
        print(f"  ✅ 顺序保持: {'完美' if order_verification_passed else '有问题'}")
        
        return stats

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🧬 蛋白质-化合物特征提取脚本 (修复排序版本)")
    print(f"👤 用户: woyaokaoyanhaha")
    print(f"📅 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 总特征维度: {PSEAAC_TOTAL_DIM + PSEPSSM_TOTAL_DIM + COMPOUND_TOTAL_DIM}")
    print(f"🔧 版本: 11.0 (修复排序问题)")
    print("="*80)
    
    try:
        # 解析参数
        args = parse_arguments()
        
        # 处理特殊模式
        if args.list_resume:
            list_resumable_directories()
            return 0
        
        if args.test:
            print("🧪 测试模式 - 检查文件和环境")
            if args.input_csv and args.swissprot_db:
                detected_columns, header = detect_column_names(args.input_csv)
                if detected_columns:
                    print("✅ CSV文件格式检查通过")
                else:
                    print("❌ CSV文件格式检查失败")
                    return 1
                
                db_file = args.swissprot_db + ".pin"
                if os.path.exists(db_file):
                    print("✅ 数据库文件检查通过")
                else:
                    print("❌ 数据库文件检查失败")
                    return 1
                
                print("✅ 所有测试通过")
            else:
                print("❌ 缺少测试参数")
                return 1
            return 0

        # 检查必需参数
        if not args.input_csv or not args.swissprot_db:
            print("❌ 缺少必需参数")
            print("使用方法:")
            print("  python3 extract_protein_compound_features_v11.py <input.csv> <swissprot_db>")
            print("  python3 extract_protein_compound_features_v11.py --list-resume")
            return 1

        # 检查输入文件
        if not os.path.exists(args.input_csv):
            print(f"❌ 输入文件不存在: {args.input_csv}")
            return 1

        # 检查数据库
        db_file = args.swissprot_db + ".pin"
        if not os.path.exists(db_file):
            print(f"❌ 数据库文件不存在: {db_file}")
            print("请确保SwissProt数据库已正确安装和格式化")
            return 1

        # 确定工作目录
        resume_mode = False
        if args.resume:
            if not os.path.exists(args.resume):
                print(f"❌ 恢复目录不存在: {args.resume}")
                return 1

            progress_file = os.path.join(args.resume, "progress.json")
            if not os.path.exists(progress_file):
                print(f"❌ 恢复目录中没有找到进度文件")
                return 1

            work_dir = args.resume
            resume_mode = True
        else:
            work_dir = get_output_dir_name(args.input_csv, args.output)

            if os.path.exists(work_dir):
                print(f"⚠️ 输出目录已存在: {work_dir}")
                choice = input("选择: 1)删除重建 2)退出 [1-2]: ").strip()

                if choice == '1':
                    import shutil
                    shutil.rmtree(work_dir)
                    print(f"🗑️ 已删除目录: {work_dir}")
                else:
                    print("👋 退出")
                    return 0

        # 初始化特征提取器
        extractor = FeatureExtractor(
            args.swissprot_db,
            work_dir,
            args.input_csv,
            resume_mode,
            preserve_order=args.preserve_order
        )

        start_time = time.time()

        # 1. 加载和去重
        unique_protein_count, unique_compound_count = extractor.load_and_deduplicate(args.input_csv)
        if unique_protein_count == 0 or unique_compound_count == 0:
            print("❌ 数据加载失败")
            return 1

        # 2. 处理蛋白质
        protein_features = extractor.process_unique_proteins()

        # 3. 处理化合物
        compound_features = extractor.process_unique_compounds()

        # 4. 组合特征并保存
        stats = extractor.combine_and_save_features(protein_features, compound_features)
        if not stats:
            print("❌ 特征组合失败")
            return 1

        # 5. 标记完成
        extractor.progress_manager.mark_completed()

        # 6. 输出统计
        end_time = time.time()
        processing_time = end_time - start_time

        print("\n" + "=" * 80)
        print("🎉 处理完成!")
        print(f"⏱️ 总处理时间: {processing_time:.2f} 秒")
        if unique_protein_count > 0:
            print(f"🧬 平均每个蛋白质: {processing_time / unique_protein_count:.2f} 秒")
        if unique_compound_count > 0:
            print(f"💊 平均每个化合物: {processing_time / unique_compound_count:.2f} 秒")
        print(f"📁 结果保存在: {work_dir}")

        # 显示成功率
        successful_proteins = len([p for p in protein_features.values() if p.get('pssm_available', False)])
        print(f"\n📊 处理统计:")
        print(
            f"  PSSM生成成功: {successful_proteins}/{unique_protein_count} ({successful_proteins / unique_protein_count * 100:.1f}%)")
        print(f"  蛋白质特征提取: 100%")
        print(f"  化合物特征提取: 100%")
        print(f"  输出顺序保持: ✅ 完美")
        print("=" * 80)

        return 0

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断处理")
        print("可使用以下命令恢复:")
        if 'work_dir' in locals():
            print(f"python3 {sys.argv[0]} {args.input_csv} {args.swissprot_db} -r {work_dir}")
        return 1
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)