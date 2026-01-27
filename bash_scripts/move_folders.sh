#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --job-name=move_folders
#SBATCH --output=move_folders.out
#SBATCH --error=move_folders.err

# Copy first-level folders from dirA to dirB
dirA="/projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_with_envmaps/"  # Source directory
dirB="/scratch/chen.yiwe/temp_objaverse/lvsm_with_envmaps/"  # Destination directory

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

# Function to check if folder is complete by comparing file counts
check_folder_complete() {
    local src_dir="$1"
    local dest_dir="$2"
    
    if [ ! -d "$dest_dir" ]; then
        return 1  # Not complete (doesn't exist)
    fi
    
    # Count files in source and destination (excluding .git directories for speed)
    local src_count=$(find "$src_dir" -type f ! -path "*/.git/*" 2>/dev/null | wc -l | tr -d ' ')
    local dest_count=$(find "$dest_dir" -type f ! -path "*/.git/*" 2>/dev/null | wc -l | tr -d ' ')
    
    # Also check directory counts
    local src_dir_count=$(find "$src_dir" -type d ! -path "*/.git/*" 2>/dev/null | wc -l | tr -d ' ')
    local dest_dir_count=$(find "$dest_dir" -type d ! -path "*/.git/*" 2>/dev/null | wc -l | tr -d ' ')
    
    # Consider complete if file and directory counts match (allow small difference for timing)
    if [ "$src_count" -eq "$dest_count" ] && [ "$src_dir_count" -eq "$dest_dir_count" ]; then
        return 0  # Complete
    else
        return 1  # Not complete
    fi
}

# Copy first-level folders one by one
for dir in "$dirA"/*/; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        dest_path="$dirB/$dirname"
        
        # Check if folder exists and is complete
        if [ -d "$dest_path" ]; then
            if check_folder_complete "$dir" "$dest_path"; then
                echo "跳过: $dirname (已完整存在于 $dirB/)"
                continue
            else
                echo "检测到不完整的文件夹: $dirname，将重新复制..."
                rm -rf "$dest_path"
            fi
        fi
        
        echo "正在复制: $dirname -> $dirB/"
        cp -r "$dir" "$dest_path" || {
            echo "错误: 复制 $dirname 失败"
            exit 1
        }
    fi
done

echo "完成！所有文件夹已复制到 $dirB"

