#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=pack_large_dirs
#SBATCH --output=pack_large_dirs.out
#SBATCH --error=pack_large_dirs.err

# Script to pack multiple large directories to save inodes
# Based on the file counts from rm_rayzer_format.sh

BASE_DIR="/projects/vig/Datasets/objaverse/hf-objaverse-v1"
DELETE_AFTER="${1:-false}"  # Set to "true" to delete after packing

# Directories to pack (with approximate file counts from comments)
# Format: "relative_path:description"
DIRS_TO_PACK=(
    "glbs:GLB files (77,755 files)"
    "lvsm_with_envmaps:LVSM with envmaps (4,025,006 files)"
    "lvsm_with_envmaps_rotating_env:Rotating env (68,632 files)"
    "lvsm_with_envmaps_test_split:Test split (18,631 files)"
    "rendered:Rendered images (2,689,870 files)"
    "rendered_dense:Dense rendered (2,557,602 files)"
    "rendered_dense_lightPlus:Light plus (1,496,139 files)"
    "rendered_dense_v0:Version 0 (21,408 files)"
    "rendered_previews:Previews (6,210 files)"
    "rendered_test_split:Test split rendered (18,638 files)"
)

echo "=========================================="
echo "批量打包大目录以释放 inode"
echo "基础目录: $BASE_DIR"
echo "删除原始文件: $DELETE_AFTER"
echo "=========================================="

# Check current inode usage
echo "当前 inode 使用情况:"
df -i "$BASE_DIR" | tail -1

TOTAL_FILES=0
PACKED_COUNT=0
FAILED_COUNT=0

for dir_info in "${DIRS_TO_PACK[@]}"; do
    IFS=':' read -r rel_path desc <<< "$dir_info"
    TARGET_DIR="$BASE_DIR/$rel_path"
    
    if [ ! -d "$TARGET_DIR" ]; then
        echo "跳过 (目录不存在): $TARGET_DIR"
        continue
    fi
    
    echo ""
    echo "----------------------------------------"
    echo "处理: $rel_path"
    echo "描述: $desc"
    echo "----------------------------------------"
    
    # Count files
    FILE_COUNT=$(find "$TARGET_DIR" -type f 2>/dev/null | wc -l)
    if [ "$FILE_COUNT" -eq 0 ]; then
        echo "跳过 (空目录或已打包): $TARGET_DIR"
        continue
    fi
    
    TOTAL_FILES=$((TOTAL_FILES + FILE_COUNT))
    echo "文件数量: $FILE_COUNT"
    
    # Pack directory
    ZIP_FILE="${TARGET_DIR}.zip"
    if [ -f "$ZIP_FILE" ]; then
        echo "ZIP 文件已存在，跳过: $ZIP_FILE"
        continue
    fi
    
    PARENT_DIR=$(dirname "$TARGET_DIR")
    DIR_NAME=$(basename "$TARGET_DIR")
    
    echo "正在打包..."
    cd "$PARENT_DIR"
    zip -r "$ZIP_FILE" "$DIR_NAME" -q
    
    if [ $? -eq 0 ]; then
        # Verify
        unzip -t "$ZIP_FILE" -q > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            ZIP_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
            echo "✓ 打包成功! 大小: $ZIP_SIZE"
            PACKED_COUNT=$((PACKED_COUNT + 1))
            
            # Delete if requested
            if [ "$DELETE_AFTER" = "true" ]; then
                echo "删除原始文件..."
                find "$TARGET_DIR" -type f -delete
                find "$TARGET_DIR" -type d -empty -delete
                rmdir "$TARGET_DIR" 2>/dev/null
                echo "✓ 已删除，释放 ~$FILE_COUNT 个 inode"
            fi
        else
            echo "✗ ZIP 验证失败"
            rm -f "$ZIP_FILE"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    else
        echo "✗ 打包失败"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo ""
echo "=========================================="
echo "总结"
echo "=========================================="
echo "处理的文件总数: $TOTAL_FILES"
echo "成功打包: $PACKED_COUNT 个目录"
echo "失败: $FAILED_COUNT 个目录"
echo ""
echo "当前 inode 使用情况:"
df -i "$BASE_DIR" | tail -1
echo "=========================================="

