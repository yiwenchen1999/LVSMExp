#!/usr/bin/env python3
"""
批量替换元数据文件中的绝对路径

用法:
    python update_paths.py --old-path "/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LVSMExp/data_samples" \
                           --new-path "/music-shared-disk/group/ct/yiwen/codes/LVSMExp/data_samples" \
                           --root-dir ./data_samples

    # 使用默认值（仅替换 data_samples 部分）
    python update_paths.py --old-path "/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LVSMExp/data_samples" \
                           --new-path "/music-shared-disk/group/ct/yiwen/codes/LVSMExp/data_samples"

    # 预览模式（不实际修改文件）
    python update_paths.py --old-path "..." --new-path "..." --dry-run

    # 指定文件扩展名
    python update_paths.py --old-path "..." --new-path "..." --extensions json txt
"""

import argparse
import os
import json
import re
from pathlib import Path
from typing import List, Set
import shutil
from datetime import datetime


def find_files(root_dir: str, extensions: List[str]) -> List[Path]:
    """递归查找指定扩展名的文件"""
    root = Path(root_dir)
    files = []
    for ext in extensions:
        files.extend(root.rglob(f"*.{ext}"))
    return files


def replace_in_text_file(file_path: Path, old_path: str, new_path: str, dry_run: bool = False) -> tuple[bool, int]:
    """
    在文本文件中替换路径
    返回: (是否修改, 替换次数)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old_path not in content:
            return False, 0
        
        # 执行替换
        new_content = content.replace(old_path, new_path)
        count = content.count(old_path)
        
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        
        return True, count
    except Exception as e:
        print(f"  错误: 处理文件 {file_path} 时出错: {e}")
        return False, 0


def replace_in_json_file(file_path: Path, old_path: str, new_path: str, dry_run: bool = False) -> tuple[bool, int]:
    """
    在 JSON 文件中替换路径（使用字符串替换方法，简单可靠）
    返回: (是否修改, 替换次数)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old_path not in content:
            return False, 0
        
        # 执行替换
        new_content = content.replace(old_path, new_path)
        count = content.count(old_path)
        
        if not dry_run:
            # 验证 JSON 格式是否正确
            try:
                json.loads(new_content)
            except json.JSONDecodeError as e:
                print(f"  警告: 替换后 JSON 格式无效，跳过文件 {file_path}: {e}")
                return False, 0
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        
        return True, count
    except Exception as e:
        print(f"  错误: 处理文件 {file_path} 时出错: {e}")
        return False, 0


def backup_file(file_path: Path, backup_dir: Path) -> bool:
    """备份文件"""
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / file_path.name
        # 如果备份文件已存在，添加时间戳
        if backup_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
        shutil.copy2(file_path, backup_path)
        return True
    except Exception as e:
        print(f"  警告: 无法备份文件 {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="批量替换元数据文件中的绝对路径",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--old-path",
        type=str,
        required=True,
        help="要替换的旧路径"
    )
    
    parser.add_argument(
        "--new-path",
        type=str,
        required=True,
        help="替换后的新路径"
    )
    
    parser.add_argument(
        "--root-dir",
        type=str,
        default="./data_samples",
        help="要搜索的根目录（默认: ./data_samples）"
    )
    
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=["json", "txt"],
        help="要处理的文件扩展名（默认: json txt）"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式：只显示会修改的文件，不实际修改"
    )
    
    parser.add_argument(
        "--backup",
        action="store_true",
        help="修改前备份文件（备份到 .path_update_backup 目录）"
    )
    
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=".path_update_backup",
        help="备份目录（默认: .path_update_backup）"
    )
    
    args = parser.parse_args()
    
    # 验证路径
    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        print(f"错误: 根目录不存在: {root_dir}")
        return 1
    
    old_path = args.old_path
    new_path = args.new_path
    
    print(f"配置:")
    print(f"  根目录: {root_dir}")
    print(f"  旧路径: {old_path}")
    print(f"  新路径: {new_path}")
    print(f"  文件扩展名: {', '.join(args.extensions)}")
    print(f"  预览模式: {'是' if args.dry_run else '否'}")
    print(f"  备份: {'是' if args.backup else '否'}")
    print()
    
    # 查找文件
    print(f"正在查找文件...")
    files = find_files(str(root_dir), args.extensions)
    print(f"找到 {len(files)} 个文件")
    print()
    
    # 处理文件
    modified_files = []
    total_replacements = 0
    
    backup_dir = Path(args.backup_dir) if args.backup else None
    
    for file_path in files:
        # 检查文件是否包含旧路径
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if old_path not in content:
                continue
        except Exception as e:
            print(f"跳过文件 {file_path}（无法读取）: {e}")
            continue
        
        # 备份
        if args.backup and backup_dir:
            if backup_file(file_path, backup_dir):
                print(f"已备份: {file_path}")
        
        # 处理文件
        print(f"处理: {file_path}")
        
        if file_path.suffix.lower() == '.json':
            modified, count = replace_in_json_file(file_path, old_path, new_path, args.dry_run)
        else:
            modified, count = replace_in_text_file(file_path, old_path, new_path, args.dry_run)
        
        if modified:
            modified_files.append(file_path)
            total_replacements += count
            action = "将修改" if args.dry_run else "已修改"
            print(f"  {action}: {count} 处替换")
        else:
            print(f"  未修改")
    
    # 总结
    print()
    print("=" * 60)
    print("总结:")
    print(f"  处理的文件数: {len(files)}")
    print(f"  修改的文件数: {len(modified_files)}")
    print(f"  总替换次数: {total_replacements}")
    
    if args.dry_run:
        print()
        print("这是预览模式，文件未被实际修改。")
        print("要实际执行替换，请移除 --dry-run 选项。")
    
    if args.backup and backup_dir:
        print(f"  备份目录: {backup_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())

