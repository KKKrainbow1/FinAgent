#!/bin/bash
# 分类型测试 SFT 数据生成，每种类型跑 5 条
# 用法: bash run_type_tests.sh

TYPES=("financial_query" "single_company_simple" "single_company_medium" "company_comparison" "industry_analysis" "risk_analysis")

for t in "${TYPES[@]}"; do
    echo "========== 测试 $t =========="
    python 10_generate_sft_data.py --type "$t" --test 5

    # 把生成结果重命名为 test_<type>.jsonl
    if [ -f "./data/sft/sft_data_v2.jsonl" ]; then
        mv ./data/sft/sft_data_v2.jsonl "./data/sft/test_${t}.jsonl"
        echo "已保存到 data/sft/test_${t}.jsonl"
    else
        echo "⚠ $t 未生成输出文件"
    fi

    # 清理断点文件，避免影响下一轮
    rm -f ./data/sft/checkpoint_v2.json

    echo ""
done

echo "========== 全部测试完成 =========="
ls -lh ./data/sft/test_*.jsonl
