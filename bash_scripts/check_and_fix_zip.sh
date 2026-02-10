#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=check_and_fix_zip
#SBATCH --output=check_and_fix_zip.out
#SBATCH --error=check_and_fix_zip.err

# Script to check and fix incomplete zip files
# Usage: check_and_fix_zip.sh <target_dir>

TARGET_DIR="${1}"

if [ -z "$TARGET_DIR" ]; then
    echo "错误: 请提供目标目录"
    echo "用法: $0 <target_dir>"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 目录不存在: $TARGET_DIR"
    exit 1
fi

TARGET_DIR=$(cd "$TARGET_DIR" && pwd)
ZIP_FILE="${TARGET_DIR}.zip"
ZIP_FILE_TMP="${TARGET_DIR}.zip.tmp"
ZIP_FILE_PARTIAL="${TARGET_DIR}.zip.partial"

echo "=========================================="
echo "检查 ZIP 文件状态"
echo "目标目录: $TARGET_DIR"
echo "ZIP 文件: $ZIP_FILE"
echo "=========================================="

# Check if directory still has files
FILE_COUNT=$(find "$TARGET_DIR" -type f 2>/dev/null | wc -l)
echo "目录中的文件数量: $FILE_COUNT"

# Check for various zip file states
if [ -f "$ZIP_FILE" ]; then
    echo "找到 ZIP 文件: $ZIP_FILE"
    ZIP_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
    echo "ZIP 文件大小: $ZIP_SIZE"
    
    # Test zip file integrity
    echo "测试 ZIP 文件完整性..."
    unzip -t "$ZIP_FILE" -q 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ ZIP 文件完整且有效"
        
        # Count files in zip
        FILES_IN_ZIP=$(unzip -l "$ZIP_FILE" 2>/dev/null | tail -1 | awk '{print $2}')
        echo "ZIP 中的文件数量: $FILES_IN_ZIP"
        
        if [ "$FILE_COUNT" -gt 0 ]; then
            echo ""
            echo "警告: 目录中仍有文件，ZIP 可能不完整"
            echo "建议: 删除不完整的 ZIP，重新打包"
            read -p "是否删除 ZIP 文件并重新打包? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -f "$ZIP_FILE"
                echo "已删除 ZIP 文件，可以重新运行打包脚本"
            fi
        else
            echo "✓ 目录已清空，ZIP 文件完整"
        fi
    else
        echo "✗ ZIP 文件损坏或不完整"
        echo "建议删除并重新打包"
        read -p "是否删除损坏的 ZIP 文件? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f "$ZIP_FILE"
            echo "已删除损坏的 ZIP 文件"
        fi
    fi
elif [ -f "$ZIP_FILE_TMP" ]; then
    echo "找到临时 ZIP 文件: $ZIP_FILE_TMP"
    echo "这可能是未完成的打包过程留下的"
    ZIP_SIZE=$(du -h "$ZIP_FILE_TMP" | cut -f1)
    echo "临时文件大小: $ZIP_SIZE"
    read -p "是否删除临时文件? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$ZIP_FILE_TMP"
        echo "已删除临时文件"
    fi
elif [ -f "$ZIP_FILE_PARTIAL" ]; then
    echo "找到部分 ZIP 文件: $ZIP_FILE_PARTIAL"
    echo "这可能是未完成的打包过程留下的"
    ZIP_SIZE=$(du -h "$ZIP_FILE_PARTIAL" | cut -f1)
    echo "部分文件大小: $ZIP_SIZE"
    read -p "是否删除部分文件? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$ZIP_FILE_PARTIAL"
        echo "已删除部分文件"
    fi
else
    echo "未找到 ZIP 文件"
    echo "可能的原因:"
    echo "  1. zip 命令在创建文件之前就被取消了"
    echo "  2. zip 文件被创建在其他位置"
    echo "  3. zip 文件已被删除"
    echo ""
    echo "建议: 重新运行打包脚本"
fi

echo ""
echo "=========================================="
echo "当前状态"
echo "=========================================="
echo "目录文件数: $FILE_COUNT"
if [ "$FILE_COUNT" -gt 0 ]; then
    echo "状态: 目录仍有文件，inode 未释放"
    echo "操作: 需要完成打包并删除原始文件才能释放 inode"
else
    echo "状态: 目录已清空"
fi

# Check for any zip-related files in parent directory
PARENT_DIR=$(dirname "$TARGET_DIR")
echo ""
echo "在父目录中查找相关文件:"
find "$PARENT_DIR" -maxdepth 1 -name "$(basename "$TARGET_DIR")*.zip*" -type f 2>/dev/null | while read f; do
    SIZE=$(du -h "$f" | cut -f1)
    echo "  $f ($SIZE)"
done

echo "=========================================="

