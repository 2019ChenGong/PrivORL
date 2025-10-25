#!/bin/bash

# 删除所有 results_* 目录下的 model-{N}.pt 文件（N = 4, 9, 14, ..., 94）

# 生成要删除的文件编号序列：从 4 到 94，步长为 5
for n in $(seq 4 5 94); do
    pattern="results_*/model-${n}.pt"
    
    # 使用 glob 匹配所有符合条件的文件
    files=( $pattern )
    
    # 检查是否有匹配的文件
    if [ -e "${files[0]}" ]; then
        for file in "${files[@]}"; do
            if [ -f "$file" ]; then
                rm "$file"
                echo "已删除: $file"
            fi
        done
    else
        echo "未找到匹配文件: $pattern"
    fi
done

echo "清理完成。"