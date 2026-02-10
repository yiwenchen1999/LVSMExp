#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=unpack_from_zip
#SBATCH --output=unpack_from_zip.out
#SBATCH --error=unpack_from_zip.err

# Script to unpack zip files
# Usage: unpack_from_zip.sh <zip_file> [extract_to_dir]

ZIP_FILE="${1}"
EXTRACT_TO="${2}"

if [ -z "$ZIP_FILE" ]; then
    echo "错误: 请提供 ZIP 文件路径"
    echo "用法: $0 <zip_file> [extract_to_dir]"
    exit 1
fi

if [ ! -f "$ZIP_FILE" ]; then
    echo "错误: ZIP 文件不存在: $ZIP_FILE"
    exit 1
fi

# Get absolute paths
ZIP_FILE=$(cd "$(dirname "$ZIP_FILE")" && pwd)/$(basename "$ZIP_FILE")
PARENT_DIR=$(dirname "$ZIP_FILE")

if [ -z "$EXTRACT_TO" ]; then
    # Extract to same directory as zip file
    EXTRACT_TO="$PARENT_DIR"
else
    EXTRACT_TO=$(cd "$EXTRACT_TO" && pwd)
fi

echo "=========================================="
echo "开始解压 ZIP 文件: $ZIP_FILE"
echo "解压到: $EXTRACT_TO"
echo "=========================================="

# Check available inodes
AVAILABLE_INODES=$(df -i "$EXTRACT_TO" | tail -1 | awk '{print $4}')
echo "可用 inode 数量: $AVAILABLE_INODES"

# Extract zip file
echo "正在解压..."
cd "$EXTRACT_TO"
unzip -q "$ZIP_FILE"

if [ $? -eq 0 ]; then
    echo "✓ 解压成功!"
    echo "解压位置: $EXTRACT_TO"
else
    echo "✗ 解压失败"
    exit 1
fi

echo "=========================================="
echo "完成!"
echo "=========================================="

