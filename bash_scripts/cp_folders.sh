#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=cp_folders
#SBATCH --output=cp_folders.out
#SBATCH --error=cp_folders.err

# Copy first-level folders from dirA to dirB
dirA="dirA"  # Source directory
dirB="dirB"  # Destination directory

# Check if source directory exists
if [ ! -d "$dirA" ]; then
    echo "错误: 源目录 $dirA 不存在"
    exit 1
fi

# Create destination directory if it doesn't exist
if [ ! -d "$dirB" ]; then
    echo "创建目标目录: $dirB"
    mkdir -p "$dirB"
fi

# Copy first-level folders one by one
for dir in "$dirA"/*/; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        echo "正在复制: $dirname -> $dirB/"
        cp -r "$dir" "$dirB/$dirname" || {
            echo "错误: 复制 $dirname 失败"
            exit 1
        }
    fi
done

echo "完成！所有文件夹已复制到 $dirB"




