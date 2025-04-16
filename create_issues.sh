#!/bin/bash

# 此腳本用於從temp_issues目錄中讀取問題文件並創建GitHub Issues
# 需要GitHub個人訪問令牌(PAT)具有"repo"權限

# 設置GitHub仓库信息
REPO_OWNER="JNRLEE"
REPO_NAME="SBP_analyzer"
GITHUB_API="https://api.github.com"

# 請在此處替換為您的GitHub個人訪問令牌
TOKEN="your_github_token_here"

# temp_issues目錄路徑
ISSUES_DIR="/Users/jnrle/Library/CloudStorage/GoogleDrive-jenner.lee.com@gmail.com/My Drive/MicforDysphagia/ProjectDeveloper/temp_issues"

# 創建issue的函數
create_issue() {
    local file=$1
    echo "Processing $file..."
    
    # 從文件中提取標題和內容
    local title=$(grep "^# " "$file" | sed 's/^# //')
    local body=$(tail -n +2 "$file" | sed -e 's/"/\\"/g' -e 's/$/\\n/' | tr -d '\n')
    
    # 創建issue
    curl -s -X POST "$GITHUB_API/repos/$REPO_OWNER/$REPO_NAME/issues" \
        -H "Authorization: token $TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        -d "{\"title\":\"$title\",\"body\":\"$body\",\"labels\":[\"enhancement\"]}"
    
    echo "Created issue: $title"
    
    # 避免GitHub API速率限制
    sleep 1
}

# 按順序從issues_list.md中讀取並創建issues
process_issues_in_order() {
    echo "Processing issues in order from issues_list.md..."
    
    # 從issues_list.md中提取issue文件名
    grep -o "issue_[0-9]_[0-9].md" "$ISSUES_DIR/issues_list.md" | while read -r issue_file; do
        full_path="$ISSUES_DIR/$issue_file"
        if [ -f "$full_path" ]; then
            create_issue "$full_path"
        else
            echo "Warning: Issue file $issue_file not found."
        fi
    done
}

# 主程序
echo "Starting GitHub issue creation..."

# 檢查token是否已設置
if [ "$TOKEN" = "your_github_token_here" ]; then
    echo "Error: Please set your GitHub personal access token first."
    exit 1
fi

# 處理issues
process_issues_in_order

echo "All issues have been created." 