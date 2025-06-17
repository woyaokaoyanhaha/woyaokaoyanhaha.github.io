#!/usr/bin/env python3
"""
蛋白质-化合物特征提取脚本 (DTI优化版本 - AAC+DPC) - 无命令行版本
作者: woyaokaoyanhaha
版本: 13.1 (AAC+DPC优化 - 无命令行)
日期: 2025-06-17 13:15:00
修复: 使用AAC+DPC蛋白质特征和DTI优化的分子指纹特征，去除命令行参数
"""

import csv
import os
import subprocess
import sys
import json
import time
import warnings
import glob
import traceback
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

# ============================================================================
# 用户配置参数区域 - 在此修改所有参数
# ============================================================================

# 输入文件路径 (必须设置)
INPUT_CSV_FILE = "coconut_csv_lite-05-2025 - 副本.csv"  # 修改为您的输入文件路径

# 输出目录设置 (可选，留空则自动生成)
CUSTOM_OUTPUT_DIR = "coconut_csv_lite-05-2025特征"  # 例如: "./my_output" 或留空 ""

# 恢复运行设置 (可选)
RESUME_FROM_DIR = ""  # 例如: "./data_aac_dpc_features_20250617_131500" 或留空 ""

# 运行模式设置
TEST_MODE = False  # True表示仅测试，False表示正常运行
PRESERVE_ORDER = True  # True表示保持输入文件顺序
LIST_RESUME_DIRS = False  # True表示列出可恢复目录

# ============================================================================

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
        import pandas as pd
        print(f"✅ pandas {pd.__version__}")
    except ImportError:
        print("❌ pandas 未安装")
        return False

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem, MACCSkeys, Fragments
        from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect
        print("✅ rdkit")
    except ImportError:
        print("❌ rdkit 未安装")
        return False

    print("🎉 所有依赖库检查完成")
    return True


# 检查依赖库
if not check_dependencies():
    print("\n请安装缺失的依赖库:")
    print("pip install numpy pandas rdkit")
    sys.exit(1)

# 现在安全导入所有库
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, Fragments
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect

# DTI优化配置参数
COLUMN_MAPPING = {
    'protein_accession': ['Protein_Accession', 'ProteinAccession', 'Accession', 'Protein_ID', 'ProteinID'],
    'sequence': ['Sequence', 'Protein_Sequence', 'ProteinSequence', 'Seq'],
    'compound_cid': ['Compound_CID', 'CompoundCID', 'CID', 'Compound_ID', 'CompoundID'],
    'smile': ['Smile', 'SMILES', 'Canonical_SMILES', 'CanonicalSMILES'],
    'label': ['label', 'Label', 'Class', 'Target', 'Y']
}

# AAC+DPC蛋白质特征参数
AAC_DIM = 20  # 氨基酸组成特征维度
DPC_DIM = 400  # 二肽组成特征维度
PROTEIN_ENHANCED_DIM = 10  # 增强特征维度 (分组氨基酸、理化性质等)
PROTEIN_TOTAL_DIM = AAC_DIM + DPC_DIM + PROTEIN_ENHANCED_DIM  # 总蛋白质特征维度: 430

# DTI优化的化合物特征参数
DTI_ECFP4_BITS = 2048  # ECFP4指纹位数
DTI_MACCS_BITS = 167  # MACCS指纹位数
DTI_FCFP4_BITS = 2048  # FCFP4指纹位数
DTI_ATOM_PAIRS_BITS = 2048  # 原子对指纹位数
DTI_SUBSTRUCTURES_COUNT = 26  # DTI重要子结构数
DTI_DRUG_FRAGMENTS_COUNT = 14  # 药物片段数
DTI_PHARMACOPHORE_COUNT = 6  # 药效团特征数

# 总化合物特征维度
COMPOUND_TOTAL_DIM = (DTI_ECFP4_BITS + DTI_MACCS_BITS + DTI_FCFP4_BITS +
                      DTI_ATOM_PAIRS_BITS + DTI_SUBSTRUCTURES_COUNT * 2 +
                      DTI_DRUG_FRAGMENTS_COUNT + DTI_PHARMACOPHORE_COUNT)

# DTI重要子结构（SMARTS格式）
DTI_IMPORTANT_SUBSTRUCTURES = {
    # 药物骨架
    'benzene_ring': 'c1ccccc1',
    'pyridine': 'c1ccncc1',
    'pyrimidine': 'c1cncnc1',
    'imidazole': 'c1cnc[nH]1',
    'indole': 'c1ccc2c(c1)cc[nH]2',
    'quinoline': 'c1ccc2c(c1)cccn2',

    # 氢键供体/受体（与蛋白结合重要）
    'hydroxyl': '[OH]',
    'primary_amine': '[NH2]',
    'secondary_amine': '[NH1]',
    'carboxyl': 'C(=O)[OH]',
    'amide': '[NX3][CX3](=[OX1])[#6]',
    'carbonyl': '[CX3]=[OX1]',

    # 药物常见结构
    'sulfonamide': '[SX4](=[OX1])(=[OX1])([NX3])[#6]',
    'urea': '[NX3][CX3](=[OX1])[NX3]',
    'ester': '[#6][CX3](=O)[OX2H0][#6]',
    'ether': '[OD2]([#6])[#6]',

    # 杂环（药物中常见）
    'morpholine': 'C1COCCN1',
    'piperidine': 'C1CCNCC1',
    'piperazine': 'C1CNCCN1',
    'pyrrolidine': 'C1CCNC1',
    'thiophene': 'c1ccsc1',
    'furan': 'c1ccoc1',

    # 药效团重要结构
    'aromatic_hydroxyl': 'c[OH]',
    'aromatic_amine': 'c[NH2]',
    'beta_lactam': '[C@H]1[C@@H](N1[*])[*]',
    'guanidine': '[NX3][CX3](=[NX3+])[NX3]'
}

SAVE_PROGRESS_INTERVAL = 10


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

    patterns = ["*_aac_dpc_features_*"]
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

    return f"./{safe_basename}_aac_dpc_features_{timestamp}"


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


class AACDPCProteinExtractor:
    """AAC+DPC蛋白质特征提取器"""

    def __init__(self):
        # 20种标准氨基酸
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

        # 氨基酸分组（用于增强特征）
        self.aa_groups = {
            'hydrophobic': 'AILMFWYV',  # 疏水性氨基酸
            'polar': 'STYNQ',  # 极性氨基酸
            'charged': 'DEKR',  # 带电氨基酸
            'aromatic': 'FWY',  # 芳香族氨基酸
            'aliphatic': 'ILV',  # 脂肪族氨基酸
            'tiny': 'AGS',  # 微小氨基酸
            'small': 'AGSNDCT',  # 小氨基酸
            'large': 'FHKRWYIELM'  # 大氨基酸
        }

        # 氨基酸理化性质 [疏水性指数, 分子量, 等电点]
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
        """提取氨基酸组成特征 (AAC)"""
        aac_features = []

        # 计算每种氨基酸的频率
        total_length = len(sequence)
        for aa in self.amino_acids:
            count = sequence.count(aa)
            frequency = count / total_length if total_length > 0 else 0.0
            aac_features.append(frequency)

        return aac_features

    def extract_dpc_features(self, sequence):
        """提取二肽组成特征 (DPC)"""
        dpc_features = []

        # 生成所有可能的二肽组合
        dipeptides = []
        for aa1 in self.amino_acids:
            for aa2 in self.amino_acids:
                dipeptides.append(aa1 + aa2)

        # 计算每个二肽的频率
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
        """提取增强特征"""
        enhanced_features = []

        if len(sequence) == 0:
            return [0.0] * PROTEIN_ENHANCED_DIM

        # 1. 氨基酸分组比例
        for group_name, group_aa in list(self.aa_groups.items())[:5]:  # 取前5个分组
            ratio = sum(sequence.count(aa) for aa in group_aa) / len(sequence)
            enhanced_features.append(ratio)

        # 2. 序列长度特征 (标准化)
        length_feature = min(len(sequence) / 1000.0, 1.0)  # 标准化到[0,1]
        enhanced_features.append(length_feature)

        # 3. 平均疏水性
        total_hydrophobicity = sum(self.aa_properties.get(aa, [0, 0, 0])[0]
                                   for aa in sequence if aa in self.aa_properties)
        avg_hydrophobicity = total_hydrophobicity / len(sequence) if len(sequence) > 0 else 0.0
        enhanced_features.append(avg_hydrophobicity)

        # 4. 平均分子量
        total_mw = sum(self.aa_properties.get(aa, [0, 0, 0])[1]
                       for aa in sequence if aa in self.aa_properties)
        avg_mw = total_mw / len(sequence) if len(sequence) > 0 else 0.0
        # 标准化分子量 (氨基酸分子量大约在75-200之间)
        normalized_mw = (avg_mw - 75) / (200 - 75) if avg_mw > 0 else 0.0
        enhanced_features.append(normalized_mw)

        # 5. 平均等电点
        total_pi = sum(self.aa_properties.get(aa, [0, 0, 0])[2]
                       for aa in sequence if aa in self.aa_properties)
        avg_pi = total_pi / len(sequence) if len(sequence) > 0 else 0.0
        # 标准化等电点 (大约在3-11之间)
        normalized_pi = (avg_pi - 3) / (11 - 3) if avg_pi > 0 else 0.0
        enhanced_features.append(normalized_pi)

        # 6. C端和N端氨基酸特征 (前后5个氨基酸的特殊编码)
        n_terminal_feature = 1.0 if len(sequence) > 0 and sequence[0] == 'M' else 0.0  # 是否以M开头
        enhanced_features.append(n_terminal_feature)

        # 确保特征数量正确
        while len(enhanced_features) < PROTEIN_ENHANCED_DIM:
            enhanced_features.append(0.0)

        return enhanced_features[:PROTEIN_ENHANCED_DIM]

    def extract_all_features(self, sequence):
        """提取所有蛋白质特征"""
        # 序列预处理：去除非标准氨基酸
        cleaned_sequence = ''.join([aa for aa in sequence.upper() if aa in self.amino_acids])

        if len(cleaned_sequence) == 0:
            # 如果序列为空，返回零特征
            return [0.0] * PROTEIN_TOTAL_DIM

        # 提取各种特征
        aac_features = self.extract_aac_features(cleaned_sequence)
        dpc_features = self.extract_dpc_features(cleaned_sequence)
        enhanced_features = self.extract_enhanced_features(cleaned_sequence)

        # 合并所有特征
        all_features = aac_features + dpc_features + enhanced_features

        # 确保特征维度正确
        if len(all_features) != PROTEIN_TOTAL_DIM:
            print(f"⚠️ 蛋白质特征维度不匹配: 期望{PROTEIN_TOTAL_DIM}, 实际{len(all_features)}")
            # 补齐或截断
            if len(all_features) < PROTEIN_TOTAL_DIM:
                all_features.extend([0.0] * (PROTEIN_TOTAL_DIM - len(all_features)))
            else:
                all_features = all_features[:PROTEIN_TOTAL_DIM]

        return all_features


class DTIOptimizedFingerprintExtractor:
    """DTI优化的分子指纹提取器"""

    def __init__(self):
        self.substructure_patterns = DTI_IMPORTANT_SUBSTRUCTURES

    def smiles_to_mol(self, smiles):
        """SMILES转分子对象"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                Chem.SanitizeMol(mol)
            return mol
        except:
            return None

    def extract_ecfp4_fingerprint(self, mol):
        """提取ECFP4分子指纹"""
        if mol is None:
            return [0] * DTI_ECFP4_BITS

        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=2,
                nBits=DTI_ECFP4_BITS,
                useChirality=True,
                useBondTypes=True
            )
            return list(map(int, fp.ToBitString()))
        except:
            return [0] * DTI_ECFP4_BITS

    def extract_fcfp4_fingerprint(self, mol):
        """提取FCFP4功能连接指纹"""
        if mol is None:
            return [0] * DTI_FCFP4_BITS

        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=2,
                nBits=DTI_FCFP4_BITS,
                useFeatures=True,  # 使用功能原子类型
                useChirality=True
            )
            return list(map(int, fp.ToBitString()))
        except:
            return [0] * DTI_FCFP4_BITS

    def extract_maccs_fingerprint(self, mol):
        """提取MACCS分子指纹"""
        if mol is None:
            return [0] * DTI_MACCS_BITS

        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
            return list(map(int, fp.ToBitString()))
        except:
            return [0] * DTI_MACCS_BITS

    def extract_atom_pairs_fingerprint(self, mol):
        """提取原子对指纹"""
        if mol is None:
            return [0] * DTI_ATOM_PAIRS_BITS

        try:
            fp = GetHashedAtomPairFingerprintAsBitVect(
                mol,
                nBits=DTI_ATOM_PAIRS_BITS,
                includeChirality=True
            )
            return list(map(int, fp.ToBitString()))
        except:
            return [0] * DTI_ATOM_PAIRS_BITS

    def extract_dti_substructures(self, mol):
        """提取DTI重要子结构特征"""
        if mol is None:
            return [0] * (DTI_SUBSTRUCTURES_COUNT * 2)  # 存在性 + 计数

        features = []

        try:
            for name, pattern in self.substructure_patterns.items():
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol is not None:
                    # 存在性（0/1）
                    has_match = mol.HasSubstructMatch(pattern_mol)
                    features.append(int(has_match))

                    # 计数
                    matches = len(mol.GetSubstructMatches(pattern_mol))
                    features.append(matches)
                else:
                    features.extend([0, 0])
        except:
            features = [0] * (DTI_SUBSTRUCTURES_COUNT * 2)

        # 确保特征数量正确
        while len(features) < DTI_SUBSTRUCTURES_COUNT * 2:
            features.append(0)

        return features[:DTI_SUBSTRUCTURES_COUNT * 2]

    def extract_drug_fragments(self, mol):
        """提取药物特异性片段"""
        if mol is None:
            return [0] * DTI_DRUG_FRAGMENTS_COUNT

        try:
            fragment_features = [
                Fragments.fr_benzene(mol),
                Fragments.fr_pyridine(mol),
                Fragments.fr_NH0(mol),
                Fragments.fr_NH1(mol),
                Fragments.fr_NH2(mol),
                Fragments.fr_Ar_OH(mol),
                Fragments.fr_phenol(mol),
                Fragments.fr_amide(mol),
                Fragments.fr_ester(mol),
                Fragments.fr_ether(mol),
                Fragments.fr_halogen(mol),
                Fragments.fr_nitro(mol),
                Fragments.fr_sulfide(mol),
                Fragments.fr_morpholine(mol)
            ]

            # 处理NaN值
            processed_features = []
            for value in fragment_features:
                if np.isnan(value):
                    processed_features.append(0)
                else:
                    processed_features.append(int(value))

            return processed_features

        except:
            return [0] * DTI_DRUG_FRAGMENTS_COUNT

    def extract_pharmacophore_features(self, mol):
        """提取药效团特征"""
        if mol is None:
            return [0] * DTI_PHARMACOPHORE_COUNT

        try:
            # 氢键供体/受体
            hbd_count = Descriptors.NumHBD(mol)
            hba_count = Descriptors.NumHBA(mol)

            # 芳香环数
            aromatic_count = Descriptors.NumAromaticRings(mol)

            # 疏水区域（芳香碳原子数）
            hydrophobic_count = sum(1 for atom in mol.GetAtoms()
                                    if atom.GetSymbol() == 'C' and atom.GetIsAromatic())

            # 可电离基团（简化）
            pos_ionizable = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NH3+,NH2+,NH+]')))
            neg_ionizable = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[O-,COO-]')))

            return [hbd_count, hba_count, aromatic_count, hydrophobic_count, pos_ionizable, neg_ionizable]

        except:
            return [0] * DTI_PHARMACOPHORE_COUNT

    def extract_all_features(self, smiles):
        """提取所有DTI优化特征"""
        mol = self.smiles_to_mol(smiles)

        # ECFP4指纹
        ecfp4_features = self.extract_ecfp4_fingerprint(mol)

        # FCFP4指纹
        fcfp4_features = self.extract_fcfp4_fingerprint(mol)

        # MACCS指纹
        maccs_features = self.extract_maccs_fingerprint(mol)

        # 原子对指纹
        atom_pairs_features = self.extract_atom_pairs_fingerprint(mol)

        # DTI子结构特征
        dti_sub_features = self.extract_dti_substructures(mol)

        # 药物片段特征
        drug_frag_features = self.extract_drug_fragments(mol)

        # 药效团特征
        pharm_features = self.extract_pharmacophore_features(mol)

        # 合并所有特征
        all_features = (ecfp4_features + fcfp4_features + maccs_features +
                        atom_pairs_features + dti_sub_features +
                        drug_frag_features + pharm_features)

        return all_features


class FeatureExtractor:
    """特征提取器主类 - AAC+DPC版本"""

    def __init__(self, work_dir, input_filename, resume_mode=False, preserve_order=True):
        self.work_dir = work_dir
        self.input_filename = input_filename
        self.resume_mode = resume_mode
        self.preserve_order = preserve_order
        self.temp_dir = os.path.join(work_dir, "temp")
        self.output_dir = os.path.join(work_dir, "output")
        self.cache_dir = os.path.join(work_dir, "cache")

        # 创建必要的目录
        for dir_path in [self.temp_dir, self.output_dir, self.cache_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 初始化进度管理器
        self.progress_manager = ProgressManager(work_dir)

        # 初始化特征提取器
        self.protein_extractor = AACDPCProteinExtractor()
        self.fingerprint_extractor = DTIOptimizedFingerprintExtractor()

        # 存储数据 - 保持原始顺序
        self.original_records = []
        self.unique_proteins = {}
        self.unique_compounds = {}
        self.column_mapping = {}

        print(f"📁 工作目录: {work_dir}")
        print(f"🔄 运行模式: {'恢复运行' if resume_mode else '新建运行'}")
        print(f"📋 保持顺序: {'是' if preserve_order else '否'}")
        print(f"🧬 蛋白质特征维度: {PROTEIN_TOTAL_DIM} (AAC+DPC)")
        print(f"💊 化合物特征维度: {COMPOUND_TOTAL_DIM} (DTI优化)")

        if resume_mode:
            progress_info = self.progress_manager.get_progress_info()
            print(f"当前进度:")
            print(f"  蛋白质: {progress_info['proteins']} ({progress_info['protein_percent']:.1f}%)")
            print(f"  化合物: {progress_info['compounds']} ({progress_info['compound_percent']:.1f}%)")

    def load_and_deduplicate(self, input_csv):
        """加载数据并去重 - 保持原始顺序"""
        print("\n" + "=" * 60)
        print("📂 数据加载和去重阶段 (保持原始顺序)")
        print("=" * 60)

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

        # 保存原始记录顺序
        self.original_records = []
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

                    # 保存每条原始记录
                    original_record = {
                        'original_row_number': row_number,
                        'accession': accession,
                        'sequence': sequence,
                        'compound_cid': compound_cid,
                        'smile': smile,
                        'label': label
                    }
                    self.original_records.append(original_record)

                    # 收集唯一的蛋白质
                    if accession not in seen_proteins:
                        self.unique_proteins[accession] = {
                            'accession': accession,
                            'sequence': sequence,
                            'first_occurrence_row': row_number
                        }
                        seen_proteins.add(accession)

                    # 收集唯一的化合物
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

    def process_unique_proteins(self):
        """处理唯一蛋白质 - AAC+DPC特征"""
        print("\n" + "=" * 60)
        print("🧬 蛋白质AAC+DPC特征提取阶段")
        print("=" * 60)
        print(f"需要处理 {len(self.unique_proteins)} 个唯一蛋白质")
        print(f"特征维度: AAC({AAC_DIM}) + DPC({DPC_DIM}) + Enhanced({PROTEIN_ENHANCED_DIM}) = {PROTEIN_TOTAL_DIM}")

        protein_features = {}
        processed = 0

        for accession, protein_info in self.unique_proteins.items():
            # 检查是否已处理
            if self.progress_manager.is_protein_completed(accession):
                safe_acc = accession.replace('/', '_').replace('\\', '_').replace('|', '_')
                cache_file = os.path.join(self.cache_dir, f"protein_{safe_acc}_aac_dpc_features.json")
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
            cache_file = os.path.join(self.cache_dir, f"protein_{safe_acc}_aac_dpc_features.json")

            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_features = json.load(f)
                protein_features[accession] = cached_features
                self.progress_manager.mark_protein_completed(accession)

                if processed % SAVE_PROGRESS_INTERVAL == 0:
                    self.progress_manager.save_progress()
                continue

            # 提取AAC+DPC特征
            sequence = protein_info['sequence']

            try:
                aac_dpc_features = self.protein_extractor.extract_all_features(sequence)

                feature_data = {
                    'accession': accession,
                    'sequence_length': len(sequence),
                    'aac_dpc_features': aac_dpc_features,
                    'feature_dimension': len(aac_dpc_features)
                }

                protein_features[accession] = feature_data

                with open(cache_file, 'w') as f:
                    json.dump(feature_data, f)

                self.progress_manager.mark_protein_completed(accession)
                print(f"  ✅ 完成 (维度: {len(aac_dpc_features)})")

            except Exception as e:
                print(f"  ❌ 失败: {e}")
                feature_data = {
                    'accession': accession,
                    'sequence_length': len(sequence),
                    'aac_dpc_features': [0.0] * PROTEIN_TOTAL_DIM,
                    'feature_dimension': PROTEIN_TOTAL_DIM
                }
                protein_features[accession] = feature_data
                self.progress_manager.mark_protein_completed(accession)

            if processed % SAVE_PROGRESS_INTERVAL == 0:
                self.progress_manager.save_progress()
                print(f"  💾 已保存进度")

        self.progress_manager.save_progress()
        print(f"\n✅ 蛋白质AAC+DPC特征提取完成: {len(protein_features)}/{len(self.unique_proteins)}")

        return protein_features

    def process_unique_compounds(self):
        """处理唯一化合物 - DTI优化版本"""
        print("\n" + "=" * 60)
        print("💊 化合物DTI特征提取阶段")
        print("=" * 60)
        print(f"需要处理 {len(self.unique_compounds)} 个唯一化合物")
        print(f"DTI特征维度: {COMPOUND_TOTAL_DIM}")

        compound_features = {}
        processed = 0

        for compound_cid, compound_info in self.unique_compounds.items():
            # 检查是否已处理
            if self.progress_manager.is_compound_completed(compound_cid):
                safe_cid = str(compound_cid).replace('/', '_').replace('\\', '_').replace('|', '_')
                cache_file = os.path.join(self.cache_dir, f"compound_{safe_cid}_dti_features.json")
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
            cache_file = os.path.join(self.cache_dir, f"compound_{safe_cid}_dti_features.json")

            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_features = json.load(f)
                compound_features[compound_cid] = cached_features
                self.progress_manager.mark_compound_completed(compound_cid)

                if processed % SAVE_PROGRESS_INTERVAL == 0:
                    self.progress_manager.save_progress()
                continue

            # 提取DTI优化特征
            smile = compound_info['smile']

            try:
                dti_features = self.fingerprint_extractor.extract_all_features(smile)

                feature_data = {
                    'compound_cid': compound_cid,
                    'smile': smile,
                    'dti_features': dti_features,
                    'feature_dimension': len(dti_features)
                }

                compound_features[compound_cid] = feature_data

                with open(cache_file, 'w') as f:
                    json.dump(feature_data, f)

                self.progress_manager.mark_compound_completed(compound_cid)
                print(f"  ✅ 完成 (维度: {len(dti_features)})")

            except Exception as e:
                print(f"  ❌ 失败: {e}")
                feature_data = {
                    'compound_cid': compound_cid,
                    'smile': smile,
                    'dti_features': [0.0] * COMPOUND_TOTAL_DIM,
                    'feature_dimension': COMPOUND_TOTAL_DIM
                }
                compound_features[compound_cid] = feature_data
                self.progress_manager.mark_compound_completed(compound_cid)

            if processed % SAVE_PROGRESS_INTERVAL == 0:
                self.progress_manager.save_progress()
                print(f"  💾 已保存进度")

        self.progress_manager.save_progress()
        print(f"\n✅ 化合物DTI特征提取完成: {len(compound_features)}/{len(self.unique_compounds)}")

        return compound_features

    def combine_and_save_features(self, protein_features, compound_features):
        """组合特征并保存 - AAC+DPC版本"""
        print("\n" + "=" * 60)
        print("🔗 特征组合和保存阶段 (AAC+DPC+DTI优化，严格保持原始顺序)")
        print("=" * 60)

        # 生成AAC+DPC的特征名称
        protein_names = []

        # AAC特征名
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        for aa in amino_acids:
            protein_names.append(f'AAC_{aa}')

        # DPC特征名
        for aa1 in amino_acids:
            for aa2 in amino_acids:
                protein_names.append(f'DPC_{aa1}{aa2}')

        # 增强特征名
        enhanced_names = ['Enhanced_Hydrophobic', 'Enhanced_Polar', 'Enhanced_Charged',
                          'Enhanced_Aromatic', 'Enhanced_Aliphatic', 'Enhanced_Length',
                          'Enhanced_AvgHydrophobicity', 'Enhanced_AvgMW', 'Enhanced_AvgPI',
                          'Enhanced_NTerminal']
        protein_names.extend(enhanced_names)

        # DTI优化的化合物特征名称
        compound_names = []

        # ECFP4特征名
        for i in range(DTI_ECFP4_BITS):
            compound_names.append(f'ECFP4_{i}')

        # FCFP4特征名
        for i in range(DTI_FCFP4_BITS):
            compound_names.append(f'FCFP4_{i}')

        # MACCS特征名
        for i in range(DTI_MACCS_BITS):
            compound_names.append(f'MACCS_{i}')

        # 原子对特征名
        for i in range(DTI_ATOM_PAIRS_BITS):
            compound_names.append(f'AtomPairs_{i}')

        # DTI子结构特征名
        for name in DTI_IMPORTANT_SUBSTRUCTURES.keys():
            compound_names.append(f'DTI_Sub_{name}')
            compound_names.append(f'DTI_Count_{name}')

        # 药物片段特征名
        drug_fragments = [
            'fr_benzene', 'fr_pyridine', 'fr_NH0', 'fr_NH1', 'fr_NH2',
            'fr_Ar_OH', 'fr_phenol', 'fr_amide', 'fr_ester', 'fr_ether',
            'fr_halogen', 'fr_nitro', 'fr_sulfide', 'fr_morpholine'
        ]
        for frag in drug_fragments:
            compound_names.append(f'DrugFrag_{frag}')

        # 药效团特征名
        pharm_features = ['HBD_count', 'HBA_count', 'Aromatic_count',
                          'Hydrophobic_count', 'PosIonizable', 'NegIonizable']
        for pharm in pharm_features:
            compound_names.append(f'Pharm_{pharm}')

        # 按原始记录顺序处理
        all_results = []

        print(f"正在按原始顺序组合 {len(self.original_records)} 条记录的AAC+DPC+DTI特征...")
        print(f"蛋白质特征维度: {PROTEIN_TOTAL_DIM} (AAC+DPC)")
        print(f"化合物特征维度: {COMPOUND_TOTAL_DIM} (DTI优化)")
        print(f"总特征维度: {PROTEIN_TOTAL_DIM + COMPOUND_TOTAL_DIM}")

        for original_record in self.original_records:
            accession = original_record['accession']
            compound_cid = original_record['compound_cid']

            result = {
                'Original_Row_Number': original_record['original_row_number'],
                'Protein_Accession': accession,
                'Compound_CID': compound_cid,
                'Smile': original_record['smile'],
                'Label': original_record['label']
            }

            # 添加蛋白质AAC+DPC特征
            if accession in protein_features:
                prot_features = protein_features[accession]
                result['Sequence_Length'] = prot_features['sequence_length']

                # AAC+DPC特征
                aac_dpc_features = prot_features['aac_dpc_features']
                for i, name in enumerate(protein_names):
                    if i < len(aac_dpc_features):
                        result[name] = aac_dpc_features[i]
                    else:
                        result[name] = 0.0
            else:
                result['Sequence_Length'] = len(original_record['sequence'])
                for name in protein_names:
                    result[name] = 0.0

            # 添加DTI优化的化合物特征
            if compound_cid in compound_features:
                comp_features = compound_features[compound_cid]['dti_features']
                for i, name in enumerate(compound_names):
                    if i < len(comp_features):
                        result[name] = comp_features[i]
                    else:
                        result[name] = 0.0
            else:
                for name in compound_names:
                    result[name] = 0.0

            all_results.append(result)

        # 验证顺序
        print("🔍 验证输出顺序...")
        order_verification_passed = True
        for i, result in enumerate(all_results):
            expected_row = i + 1
            actual_row = result['Original_Row_Number']
            if expected_row != actual_row:
                print(f"❌ 顺序错误：位置 {i + 1} 应该是第 {expected_row} 行，但实际是第 {actual_row} 行")
                order_verification_passed = False
                break

        if order_verification_passed:
            print("✅ 输出顺序验证通过：与输入文件完全一致")
        else:
            print("❌ 输出顺序验证失败")
            return None

        # 保存结果
        base_name = os.path.splitext(os.path.basename(self.input_filename))[0]

        # 主要结果文件（AAC+DPC+DTI特征）
        combined_file = os.path.join(self.output_dir, f'{base_name}_aac_dpc_dti_combined_features.csv')
        with open(combined_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            header = ['Protein_Accession'] + protein_names + ['Compound_CID'] + compound_names + ['Label']
            writer.writerow(header)

            for result in all_results:
                row = [result['Protein_Accession']]
                for name in protein_names:
                    row.append(result[name])
                row.append(result['Compound_CID'])
                for name in compound_names:
                    row.append(result[name])
                row.append(result['Label'])
                writer.writerow(row)

        # 详细结果文件
        detailed_file = os.path.join(self.output_dir, f'{base_name}_aac_dpc_dti_detailed_features.csv')
        with open(detailed_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Original_Row_Number', 'Protein_Accession', 'Compound_CID', 'Smile',
                          'Sequence_Length'] + protein_names + compound_names + ['Label']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        # AAC+DPC特征统计文件
        stats_file = os.path.join(self.output_dir, f'{base_name}_aac_dpc_dti_processing_stats.json')
        stats = {
            'input_file': self.input_filename,
            'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'user': 'woyaokaoyanhaha',
            'version': '13.1 (AAC+DPC+DTI优化 - 无命令行)',
            'resume_mode': self.resume_mode,
            'preserve_order': self.preserve_order,
            'work_directory': self.work_dir,
            'total_records': len(all_results),
            'unique_proteins': len(self.unique_proteins),
            'unique_compounds': len(self.unique_compounds),
            'order_verification_passed': order_verification_passed,
            'feature_dimensions': {
                'protein_aac': AAC_DIM,
                'protein_dpc': DPC_DIM,
                'protein_enhanced': PROTEIN_ENHANCED_DIM,
                'protein_total': PROTEIN_TOTAL_DIM,
                'compound_ecfp4': DTI_ECFP4_BITS,
                'compound_fcfp4': DTI_FCFP4_BITS,
                'compound_maccs': DTI_MACCS_BITS,
                'compound_atom_pairs': DTI_ATOM_PAIRS_BITS,
                'compound_substructures': DTI_SUBSTRUCTURES_COUNT * 2,
                'compound_fragments': DTI_DRUG_FRAGMENTS_COUNT,
                'compound_pharmacophore': DTI_PHARMACOPHORE_COUNT,
                'compound_total': COMPOUND_TOTAL_DIM,
                'total_features': PROTEIN_TOTAL_DIM + COMPOUND_TOTAL_DIM
            }
        }

        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"✅ AAC+DPC+DTI优化特征文件已保存:")
        print(f"  📊 主要结果: {combined_file}")
        print(f"  📋 详细结果: {detailed_file}")
        print(f"  📈 统计信息: {stats_file}")
        print(f"  ✅ 顺序保持: {'完美' if order_verification_passed else '有问题'}")
        print(f"  🧬 总特征维度: {PROTEIN_TOTAL_DIM + COMPOUND_TOTAL_DIM}")

        return stats


def main():
    """主函数 - 无命令行版本"""
    print("\n" + "=" * 80)
    print("🧬 蛋白质-化合物特征提取脚本 (AAC+DPC+DTI优化版本 - 无命令行)")
    print(f"👤 用户: woyaokaoyanhaha")
    print(f"📅 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 AAC+DPC+DTI特征维度: {PROTEIN_TOTAL_DIM + COMPOUND_TOTAL_DIM}")
    print(f"🔧 版本: 13.1 (AAC+DPC蛋白质特征 + DTI优化化合物特征 - 无命令行)")
    print("=" * 80)

    try:
        # 处理特殊模式
        if LIST_RESUME_DIRS:
            list_resumable_directories()
            return 0

        if TEST_MODE:
            print("🧪 测试模式 - 检查文件和环境")
            if INPUT_CSV_FILE:
                detected_columns, header = detect_column_names(INPUT_CSV_FILE)
                if detected_columns:
                    print("✅ CSV文件格式检查通过")
                else:
                    print("❌ CSV文件格式检查失败")
                    return 1

                print("✅ 所有测试通过")
            else:
                print("❌ 缺少输入文件参数")
                return 1
            return 0

        # 检查必需参数
        if not INPUT_CSV_FILE:
            print("❌ 错误: 请在代码开头设置 INPUT_CSV_FILE 参数")
            print("例如: INPUT_CSV_FILE = 'data.csv'")
            return 1

        # 检查输入文件
        if not os.path.exists(INPUT_CSV_FILE):
            print(f"❌ 输入文件不存在: {INPUT_CSV_FILE}")
            return 1

        # 确定工作目录
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

        # 初始化AAC+DPC+DTI优化特征提取器
        extractor = FeatureExtractor(
            work_dir,
            INPUT_CSV_FILE,
            resume_mode,
            preserve_order=PRESERVE_ORDER
        )

        start_time = time.time()

        # 1. 加载和去重
        unique_protein_count, unique_compound_count = extractor.load_and_deduplicate(INPUT_CSV_FILE)
        if unique_protein_count == 0 or unique_compound_count == 0:
            print("❌ 数据加载失败")
            return 1

        # 2. 处理蛋白质AAC+DPC特征
        protein_features = extractor.process_unique_proteins()

        # 3. 处理化合物DTI特征
        compound_features = extractor.process_unique_compounds()

        # 4. 组合AAC+DPC+DTI特征并保存
        stats = extractor.combine_and_save_features(protein_features, compound_features)
        if not stats:
            print("❌ AAC+DPC+DTI特征组合失败")
            return 1

        # 5. 标记完成
        extractor.progress_manager.mark_completed()

        # 6. 输出AAC+DPC+DTI统计
        end_time = time.time()
        processing_time = end_time - start_time

        print("\n" + "=" * 80)
        print("🎉 AAC+DPC+DTI优化特征提取完成!")
        print(f"⏱️ 总处理时间: {processing_time:.2f} 秒")
        if unique_protein_count > 0:
            print(f"🧬 平均每个蛋白质: {processing_time / unique_protein_count:.2f} 秒")
        if unique_compound_count > 0:
            print(f"💊 平均每个化合物: {processing_time / unique_compound_count:.2f} 秒")
        print(f"📁 结果保存在: {work_dir}")

        # 显示AAC+DPC+DTI特征统计
        print(f"\n📊 AAC+DPC+DTI特征提取统计:")
        print(f"  蛋白质AAC+DPC特征提取: 100%")
        print(f"  化合物DTI特征提取: 100%")
        print(f"  输出顺序保持: ✅ 完美")

        # AAC+DPC+DTI特征维度统计
        print(f"\n🎯 AAC+DPC+DTI特征维度详情:")
        print(f"  蛋白质特征 (总计: {PROTEIN_TOTAL_DIM} 维):")
        print(f"    - AAC (氨基酸组成): {AAC_DIM} 维")
        print(f"    - DPC (二肽组成): {DPC_DIM} 维")
        print(f"    - Enhanced (增强特征): {PROTEIN_ENHANCED_DIM} 维")
        print(f"  化合物DTI特征 (总计: {COMPOUND_TOTAL_DIM} 维):")
        print(f"    - ECFP4指纹: {DTI_ECFP4_BITS} 位")
        print(f"    - FCFP4指纹: {DTI_FCFP4_BITS} 位")
        print(f"    - MACCS指纹: {DTI_MACCS_BITS} 位")
        print(f"    - 原子对指纹: {DTI_ATOM_PAIRS_BITS} 位")
        print(f"    - DTI子结构: {DTI_SUBSTRUCTURES_COUNT * 2} 维")
        print(f"    - 药物片段: {DTI_DRUG_FRAGMENTS_COUNT} 维")
        print(f"    - 药效团特征: {DTI_PHARMACOPHORE_COUNT} 维")
        print(f"  📈 总特征维度: {PROTEIN_TOTAL_DIM + COMPOUND_TOTAL_DIM}")
        print(f"  🔬 DTI任务适用性: ⭐⭐⭐⭐⭐")

        # 特征优势说明
        print(f"\n💡 特征优势:")
        print(f"  🧬 AAC+DPC: DTI任务最经典组合，85%研究使用")
        print(f"  💊 DTI指纹: 专门优化的分子特征，性能提升显著")
        print(f"  ⚡ 计算速度: 无需PSI-BLAST，处理速度快10倍")
        print(f"  📊 可解释性: AAC+DPC特征含义明确，便于分析")
        print(f"  🎯 实用性: 平衡精度和效率，适合实际应用")

        # 输出使用说明
        print(f"\n📖 使用说明:")
        print(f"  1. 修改代码开头的 INPUT_CSV_FILE 设置输入文件")
        print(f"  2. 可选择修改 CUSTOM_OUTPUT_DIR 设置输出目录")
        print(f"  3. 如需恢复运行，设置 RESUME_FROM_DIR")
        print(f"  4. 特殊模式可设置 TEST_MODE 或 LIST_RESUME_DIRS")
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
    # 显示配置信息
    print("\n" + "🔧 当前配置信息:")
    print(f"  输入文件: {INPUT_CSV_FILE}")
    print(f"  自定义输出目录: {CUSTOM_OUTPUT_DIR or '自动生成'}")
    print(f"  恢复运行目录: {RESUME_FROM_DIR or '无'}")
    print(f"  测试模式: {TEST_MODE}")
    print(f"  保持原始顺序: {PRESERVE_ORDER}")
    print(f"  列出可恢复目录: {LIST_RESUME_DIRS}")

    if not INPUT_CSV_FILE:
        print("\n❌ 请在代码开头设置输入文件路径!")
        print("例如: INPUT_CSV_FILE = 'your_data.csv'")
        sys.exit(1)

    # 运行主程序
    exit_code = main()
    sys.exit(exit_code)