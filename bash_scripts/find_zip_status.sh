#!/bin/bash

# Quick diagnostic script to find what happened to the zip file
# Usage: ./find_zip_status.sh

TARGET_DIR="/projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense"
ZIP_FILE="${TARGET_DIR}.zip"
PARENT_DIR="/projects/vig/Datasets/objaverse/hf-objaverse-v1"

echo "=========================================="
echo "诊断 ZIP 文件状态"
echo "=========================================="
echo "目标目录: $TARGET_DIR"
echo "预期 ZIP 文件: $ZIP_FILE"
echo ""

# 1. Check if target directory exists and has files
echo "1. 检查目标目录状态:"
if [ -d "$TARGET_DIR" ]; then
    FILE_COUNT=$(find "$TARGET_DIR" -type f 2>/dev/null | wc -l)
    DIR_COUNT=$(find "$TARGET_DIR" -type d 2>/dev/null | wc -l)
    echo "   ✓ 目录存在"
    echo "   文件数量: $FILE_COUNT"
    echo "   子目录数量: $DIR_COUNT"
    
    if [ "$FILE_COUNT" -gt 0 ]; then
        echo "   状态: 目录仍有文件，inode 未释放"
        SIZE=$(du -sh "$TARGET_DIR" 2>/dev/null | cut -f1)
        echo "   目录大小: $SIZE"
    else
        echo "   状态: 目录为空"
    fi
else
    echo "   ✗ 目录不存在（可能已被删除）"
fi
echo ""

# 2. Check for the expected zip file
echo "2. 检查预期 ZIP 文件:"
if [ -f "$ZIP_FILE" ]; then
    ZIP_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
    ZIP_SIZE_BYTES=$(stat -f%z "$ZIP_FILE" 2>/dev/null || stat -c%s "$ZIP_FILE" 2>/dev/null)
    ZIP_MOD_TIME=$(stat -f%Sm "$ZIP_FILE" 2>/dev/null || stat -c%y "$ZIP_FILE" 2>/dev/null)
    
    echo "   ✓ ZIP 文件存在"
    echo "   文件大小: $ZIP_SIZE ($ZIP_SIZE_BYTES bytes)"
    echo "   修改时间: $ZIP_MOD_TIME"
    
    # Test if zip is valid
    echo "   验证 ZIP 文件..."
    unzip -t "$ZIP_FILE" -q 2>&1
    if [ $? -eq 0 ]; then
        echo "   ✓ ZIP 文件完整且有效"
        FILES_IN_ZIP=$(unzip -l "$ZIP_FILE" 2>/dev/null | tail -1 | awk '{print $2}')
        echo "   ZIP 中的文件数: $FILES_IN_ZIP"
    else
        echo "   ✗ ZIP 文件损坏或不完整"
        echo "   这可能是中断的打包过程留下的不完整文件"
    fi
else
    echo "   ✗ ZIP 文件不存在"
fi
echo ""

# 3. Search for any zip-related files in parent directory
echo "3. 搜索父目录中的相关文件:"
echo "   搜索模式: rendered.zip*"
FOUND_FILES=$(find "$PARENT_DIR" -maxdepth 1 -name "rendered.zip*" -type f 2>/dev/null)
if [ -n "$FOUND_FILES" ]; then
    echo "$FOUND_FILES" | while read f; do
        SIZE=$(du -h "$f" | cut -f1)
        MOD_TIME=$(stat -f%Sm "$f" 2>/dev/null || stat -c%y "$f" 2>/dev/null)
        echo "   找到: $(basename "$f")"
        echo "         大小: $SIZE, 修改时间: $MOD_TIME"
    done
else
    echo "   未找到相关文件"
fi
echo ""

# 4. Check for temporary or partial files
echo "4. 检查临时/部分文件:"
TEMP_PATTERNS=(
    "${ZIP_FILE}.tmp"
    "${ZIP_FILE}.partial"
    "${ZIP_FILE}.part"
    "${ZIP_FILE}.lock"
    "${PARENT_DIR}/rendered.zip.tmp"
    "${PARENT_DIR}/rendered.zip.partial"
)

for pattern in "${TEMP_PATTERNS[@]}"; do
    if [ -f "$pattern" ]; then
        SIZE=$(du -h "$pattern" | cut -f1)
        echo "   找到临时文件: $pattern ($SIZE)"
    fi
done

if [ -z "$(find "$PARENT_DIR" -maxdepth 1 -name "rendered.zip*" -type f 2>/dev/null)" ]; then
    echo "   未找到临时文件"
fi
echo ""

# 5. Check current working directory for zip files
echo "5. 检查当前目录是否有 zip 文件:"
CURRENT_DIR=$(pwd)
if [ -f "${CURRENT_DIR}/rendered.zip" ]; then
    SIZE=$(du -h "${CURRENT_DIR}/rendered.zip" | cut -f1)
    echo "   找到: ${CURRENT_DIR}/rendered.zip ($SIZE)"
else
    echo "   当前目录无 zip 文件"
fi
echo ""

# 6. Check if zip process might still be running
echo "6. 检查是否有 zip 进程在运行:"
ZIP_PROCESSES=$(ps aux | grep -E "zip.*rendered" | grep -v grep)
if [ -n "$ZIP_PROCESSES" ]; then
    echo "   找到运行中的 zip 进程:"
    echo "$ZIP_PROCESSES"
else
    echo "   无运行中的 zip 进程"
fi
echo ""

# 7. Summary and recommendations
echo "=========================================="
echo "总结和建议"
echo "=========================================="

if [ -f "$ZIP_FILE" ]; then
    unzip -t "$ZIP_FILE" -q 2>&1
    if [ $? -eq 0 ]; then
        if [ -d "$TARGET_DIR" ] && [ "$(find "$TARGET_DIR" -type f 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "情况: ZIP 文件完整，但原始目录仍有文件"
            echo "建议: 可以安全删除原始目录以释放 inode"
            echo "命令: find \"$TARGET_DIR\" -type f -delete && find \"$TARGET_DIR\" -type d -empty -delete"
        else
            echo "情况: ZIP 文件完整，目录已清空"
            echo "状态: ✓ 正常"
        fi
    else
        echo "情况: ZIP 文件存在但损坏或不完整"
        echo "建议: 删除损坏的 ZIP 文件并重新打包"
        echo "命令: rm \"$ZIP_FILE\" && 重新运行打包脚本"
    fi
elif [ -d "$TARGET_DIR" ] && [ "$(find "$TARGET_DIR" -type f 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "情况: ZIP 文件不存在，原始目录仍有文件"
    echo "原因: zip 命令可能在创建文件前就被中断了"
    echo "建议: 重新运行打包脚本"
    echo "命令: sbatch bash_scripts/rm_rayzer_format.sh"
else
    echo "情况: ZIP 文件不存在，目录也不存在或为空"
    echo "可能: 文件已被移动到其他位置或已删除"
fi

echo "=========================================="

