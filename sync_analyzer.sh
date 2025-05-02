#!/bin/bash

# 同步腳本：在本地版本控制的腳本和實際使用的文件之間保持同步

echo "開始同步 custom_analyzer.py..."

# 檢查 scripts 目錄是否存在
if [ ! -d "scripts" ]; then
    mkdir -p scripts
    echo "創建 scripts 目錄"
fi

# 檢查目標文件是否存在
RESULTS_FILE="/home/sbplab/JNRLEE/MicDysphagiaFramework/results/custom_analyzer.py"
LOCAL_FILE="scripts/custom_analyzer.py"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "警告：目標文件 $RESULTS_FILE 不存在"
    if [ -f "$LOCAL_FILE" ]; then
        echo "將本地版本複製到目標位置"
        cp "$LOCAL_FILE" "$RESULTS_FILE"
    else
        echo "錯誤：本地文件和目標文件都不存在"
        exit 1
    fi
else
    # 檢查文件修改時間，選擇較新的版本
    if [ -f "$LOCAL_FILE" ]; then
        LOCAL_MOD=$(stat -c %Y "$LOCAL_FILE")
        RESULTS_MOD=$(stat -c %Y "$RESULTS_FILE")
        
        if [ $LOCAL_MOD -gt $RESULTS_MOD ]; then
            echo "本地版本較新，正在更新目標文件..."
            cp "$LOCAL_FILE" "$RESULTS_FILE"
            echo "已更新目標文件"
        elif [ $RESULTS_MOD -gt $LOCAL_MOD ]; then
            echo "目標文件較新，正在更新本地版本..."
            cp "$RESULTS_FILE" "$LOCAL_FILE"
            echo "已更新本地版本"
        else
            echo "文件已同步，無需更新"
        fi
    else
        echo "本地文件不存在，正在從目標位置複製..."
        cp "$RESULTS_FILE" "$LOCAL_FILE"
        echo "已創建本地版本"
    fi
fi

echo "同步完成" 