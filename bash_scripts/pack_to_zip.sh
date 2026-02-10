#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=pack_to_zip
#SBATCH --output=pack_to_zip.out
#SBATCH --error=pack_to_zip.err

# Script to pack directories to zip files to save inodes
# Usage: 
#   pack_to_zip.sh <target_dir> [delete_after_pack]
#   If delete_after_pack is "true", original files will be deleted after packing

TARGET_DIR="${1}"
DELETE_AFTER="${2:-false}"

if [ -z "$TARGET_DIR" ]; then
    echo "错误: 请提供目标目录"
    echo "用法: $0 <target_dir> [delete_after_pack]"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 目录不存在: $TARGET_DIR"
    exit 1
fi

# Get absolute path
TARGET_DIR=$(cd "$TARGET_DIR" && pwd)
ZIP_FILE="${TARGET_DIR}.zip"
PARENT_DIR=$(dirname "$TARGET_DIR")
DIR_NAME=$(basename "$TARGET_DIR")

echo "=========================================="
echo "开始打包目录: $TARGET_DIR"
echo "目标 ZIP 文件: $ZIP_FILE"
echo "=========================================="

# Count files before packing
FILE_COUNT=$(find "$TARGET_DIR" -type f | wc -l)
echo "目录中的文件数量: $FILE_COUNT"

# Check if zip file already exists
if [ -f "$ZIP_FILE" ]; then
    echo "警告: ZIP 文件已存在: $ZIP_FILE"
    read -p "是否覆盖? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "取消操作"
        exit 1
    fi
    rm -f "$ZIP_FILE"
fi

# Pack directory to zip
echo "正在打包..."
cd "$PARENT_DIR"
zip -r "$ZIP_FILE" "$DIR_NAME" -q

if [ $? -eq 0 ]; then
    ZIP_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
    echo "✓ 打包成功!"
    echo "ZIP 文件大小: $ZIP_SIZE"
    echo "ZIP 文件位置: $ZIP_FILE"
    
    # Verify zip file
    echo "验证 ZIP 文件..."
    unzip -t "$ZIP_FILE" -q
    if [ $? -eq 0 ]; then
        echo "✓ ZIP 文件验证通过"
        
        # Delete original files if requested
        if [ "$DELETE_AFTER" = "true" ]; then
            echo "正在删除原始文件以释放 inode..."
            find "$TARGET_DIR" -type f -delete
            find "$TARGET_DIR" -type d -empty -delete
            rmdir "$TARGET_DIR" 2>/dev/null
            echo "✓ 原始文件已删除"
            echo "释放的 inode 数量: ~$FILE_COUNT"
        else
            echo "提示: 原始文件保留。如需删除以释放 inode，请运行:"
            echo "  find \"$TARGET_DIR\" -type f -delete"
            echo "  find \"$TARGET_DIR\" -type d -empty -delete"
        fi
    else
        echo "✗ ZIP 文件验证失败，保留原始文件"
        exit 1
    fi
else
    echo "✗ 打包失败"
    exit 1
fi

echo "=========================================="
echo "完成!"
echo "=========================================="

