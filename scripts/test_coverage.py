#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
此腳本用於運行單元測試並生成覆蓋率報告，能夠識別未被測試覆蓋的程式碼部分。
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime

def parse_arguments():
    """
    解析命令行參數。
    
    Returns:
        argparse.Namespace: 解析後的參數
    """
    parser = argparse.ArgumentParser(description='運行單元測試並生成覆蓋率報告')
    parser.add_argument('--src-dir', type=str, default='.', 
                        help='源碼目錄路徑')
    parser.add_argument('--omit', type=str, default='tests/*,scripts/*,*/__pycache__/*,*/__init__.py', 
                        help='排除的文件模式,逗號分隔')
    parser.add_argument('--output-dir', type=str, default='coverage_reports', 
                        help='輸出報告目錄')
    parser.add_argument('--html', action='store_true', 
                        help='生成HTML報告')
    parser.add_argument('--xml', action='store_true', 
                        help='生成XML報告')
    parser.add_argument('--json', action='store_true', 
                        help='生成JSON報告')
    return parser.parse_args()

def setup_environment():
    """
    設置測試環境，安裝必要的依賴。
    
    Returns:
        bool: 環境設置是否成功
    """
    try:
        # 檢查 coverage 是否已安裝
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', 'coverage'],
            capture_output=True,
            text=True
        )
        
        # 如果未安裝，則安裝 coverage
        if result.returncode != 0:
            print("安裝 coverage 套件...")
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', 'coverage', 'pytest', 'pytest-cov'],
                check=True
            )
        
        return True
    except subprocess.SubprocessError as e:
        print(f"環境設置失敗: {e}")
        return False

def run_tests_with_coverage(args):
    """
    使用 pytest 和 coverage 運行測試並收集覆蓋率資料。
    
    Args:
        args (argparse.Namespace): 命令行參數
        
    Returns:
        bool: 測試執行是否成功
    """
    try:
        # 確保輸出目錄存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 組建 pytest 命令
        cmd = [
            sys.executable, '-m', 'pytest',
            f'--cov={args.src_dir}',
            f'--cov-report=term',
            f'--cov-config=.coveragerc',
            '--verbose'
        ]
        
        # 添加報告選項
        if args.html:
            cmd.append(f'--cov-report=html:{os.path.join(args.output_dir, "html")}')
        if args.xml:
            cmd.append(f'--cov-report=xml:{os.path.join(args.output_dir, "coverage.xml")}')
        if args.json:
            cmd.append('--cov-report=json:.coverage.json')
        
        # 添加忽略路徑
        cmd.append(f'--cov-omit={args.omit}')
        
        # 運行測試
        print(f"執行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print("測試成功完成")
            return True
        else:
            print(f"測試執行失敗，返回碼: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"運行測試時出錯: {e}")
        return False

def generate_summary_report(args):
    """
    生成測試覆蓋率摘要報告。
    
    Args:
        args (argparse.Namespace): 命令行參數
        
    Returns:
        bool: 報告生成是否成功
    """
    try:
        # 如果生成了JSON報告，則解析它
        if args.json and os.path.exists('.coverage.json'):
            with open('.coverage.json', 'r') as f:
                coverage_data = json.load(f)
            
            # 從覆蓋率數據中提取摘要
            summary = {}
            total_statements = 0
            total_missing = 0
            total_coverage = 0
            
            for file_path, file_data in coverage_data['files'].items():
                # 排除應該忽略的文件
                skip = False
                for omit_pattern in args.omit.split(','):
                    if omit_pattern.endswith('*'):
                        prefix = omit_pattern[:-1]
                        if file_path.startswith(prefix):
                            skip = True
                            break
                    elif file_path == omit_pattern:
                        skip = True
                        break
                
                if skip:
                    continue
                
                statements = len(file_data['statements'])
                missing = len(file_data.get('missing_lines', []))
                coverage_pct = ((statements - missing) / statements) * 100 if statements > 0 else 0
                
                summary[file_path] = {
                    'statements': statements,
                    'missing': missing,
                    'coverage': coverage_pct
                }
                
                total_statements += statements
                total_missing += missing
            
            # 計算整體覆蓋率
            if total_statements > 0:
                total_coverage = ((total_statements - total_missing) / total_statements) * 100
            
            # 生成摘要報告
            report_path = os.path.join(args.output_dir, 'summary_report.txt')
            with open(report_path, 'w') as f:
                f.write(f"測試覆蓋率摘要報告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"總語句數: {total_statements}\n")
                f.write(f"未覆蓋語句數: {total_missing}\n")
                f.write(f"總覆蓋率: {total_coverage:.2f}%\n\n")
                f.write("每個文件的覆蓋率:\n")
                f.write("-" * 80 + "\n")
                
                # 依覆蓋率排序
                sorted_files = sorted(summary.items(), key=lambda x: x[1]['coverage'])
                
                for file_path, data in sorted_files:
                    f.write(f"{file_path}:\n")
                    f.write(f"  語句數: {data['statements']}\n")
                    f.write(f"  未覆蓋語句數: {data['missing']}\n")
                    f.write(f"  覆蓋率: {data['coverage']:.2f}%\n\n")
            
            print(f"摘要報告已生成: {report_path}")
            return True
        else:
            print("未找到JSON覆蓋率數據，無法生成詳細摘要報告")
            return False
    
    except Exception as e:
        print(f"生成摘要報告時出錯: {e}")
        return False

def main():
    """
    主函數，協調整個測試與覆蓋率生成流程。
    """
    args = parse_arguments()
    
    print("設置測試環境...")
    if not setup_environment():
        sys.exit(1)
    
    print("運行測試並收集覆蓋率數據...")
    if not run_tests_with_coverage(args):
        print("測試失敗，但仍嘗試生成報告...")
    
    print("生成覆蓋率摘要報告...")
    generate_summary_report(args)
    
    print("測試覆蓋率分析完成")

if __name__ == "__main__":
    main() 