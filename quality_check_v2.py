#!/usr/bin/env python3
"""
SFT 数据 10 维度质量检查脚本（修正版）
逐条检查每条数据，对每条做 10 维度质量评估

修正记录：
  - D4: 补充指标类型匹配检查（问盈利返回偿债等）
  - D7: 补充流动比率标准检查
  - D9: 补充非杜邦场景的时间混用检查
  - D10: 扩展到 answer 中含风险提示段的所有样本
  - 路径: 支持命令行参数指定输入输出
"""

import json
import re
import sys
import argparse
from collections import Counter, defaultdict

# ============ 常量 ============

# 金融行业分析主体关键词（不含研报来源）
FINANCIAL_SUBJECTS = [
    "银行", "保险", "信托", "金融行业",
    "工商银行", "建设银行", "农业银行", "中国银行", "交通银行",
    "招商银行", "浦发银行", "兴业银行", "民生银行", "光大银行",
    "华夏银行", "平安银行", "中信银行", "邮储银行", "北京银行",
    "南京银行", "宁波银行", "江苏银行", "上海银行", "杭州银行",
    "中国人寿", "中国平安", "中国太保", "新华保险", "中国人保",
]

# 盈利指标关键词
PROFITABILITY_KW = ["ROE", "ROA", "净利率", "毛利率", "净资产收益率", "每股收益",
                     "EPS", "净利润增长", "盈利", "利润率"]
# 偿债指标关键词
SOLVENCY_KW = ["资产负债率", "流动比率", "速动比率", "现金比率", "产权比率",
                "利息保障", "偿债"]
# 营运指标关键词
OPERATIONAL_KW = ["周转率", "周转天数", "存货周转", "应收账款周转", "总资产周转",
                   "经营效率", "营运"]


# ============ 辅助函数 ============

def extract_numbers(text):
    """从文本中提取所有数字（包括百分比和小数），同时保留绝对值版本"""
    raw = set(re.findall(r'[-+]?\d+\.?\d*%?', text))
    result = set()
    for n in raw:
        result.add(n)
        if n.startswith('-'):
            result.add(n[1:])
    return result


def is_financial_industry(question, steps):
    """判断问题的分析主体是否为金融行业公司"""
    check_text = question
    for s in steps:
        ai = s.get("action_input", "")
        if s.get("action") in ("search_financial", "search_report") and len(ai) < 30:
            check_text += " " + ai
    return any(kw in check_text for kw in FINANCIAL_SUBJECTS)


def get_finish_answer(steps, final_answer=None):
    """提取最终答案文本。

    V4 schema:plan["final_answer"] 显式传入,优先取。
    V1 schema:从 steps[finish] 抽,作为 fallback 兼容老数据。
    """
    if final_answer:
        return final_answer
    for s in steps:
        if s.get("action") == "finish":
            return s.get("action_input", "")
    return ""


# ============ 10 维度检查函数 ============

def check_thought_coherence(steps):
    """
    D1: Thought 推理链连贯性
    后一步 Thought 是否引用了前一步 Observation 的具体数字，而非笼统地说"已获取数据"
    """
    issues = []
    for i in range(1, len(steps)):
        thought = steps[i].get("thought", "")
        prev_obs = steps[i - 1].get("observation", "")

        if not prev_obs or steps[i - 1].get("action") == "finish":
            continue

        prev_numbers = extract_numbers(prev_obs)
        thought_numbers = extract_numbers(thought)

        # 检查 thought 中是否引用了前一步 observation 的至少1个具体数字
        has_specific_ref = len(prev_numbers & thought_numbers) >= 1

        if not has_specific_ref:
            # 检查是否用了笼统表述
            vague_patterns = [
                r"已获取.*数据",
                r"已检索到.*信息",
                r"已查到.*数据",
                r"数据已获取",
                r"获取了.*指标",
            ]
            is_vague = any(re.search(p, thought) for p in vague_patterns)
            if is_vague:
                issues.append({
                    "type": "thought_coherence",
                    "detail": f"step{i}的Thought笼统引用前步Observation（如'已获取数据'），未提及具体数字"
                })

    return issues


def check_finish_depth(steps, question, final_answer=None):
    """
    D2: Finish 答案深度
    是否引用了德勤框架（盈利/偿债/营运三维度）、杜邦分析等
    """
    issues = []
    answer = get_finish_answer(steps, final_answer)
    if not answer:
        return issues

    # 需要深度分析的问题
    depth_keywords = ["分析", "评估", "风险", "前景", "比较", "对比", "怎么样", "如何",
                      "综合", "全面", "投资价值", "深入", "财务状况"]
    needs_depth = any(kw in question for kw in depth_keywords)

    if needs_depth and len(answer) < 100:
        issues.append({
            "type": "answer_depth",
            "detail": "答案过短，缺乏深度分析"
        })

    # 对于需要多维度分析的长答案，检查是否引用了分析框架
    if needs_depth and len(answer) > 200:
        has_framework = any(kw in answer for kw in [
            "盈利", "偿债", "营运", "杜邦", "三维度", "三个维度",
            "净利率", "周转率", "权益乘数", "德勤",
            "盈利能力", "偿债能力", "营运能力",
        ])
        if not has_framework:
            issues.append({
                "type": "answer_depth",
                "detail": "答案缺少分析框架引用（如德勤三维度、杜邦分析）"
            })

    return issues


def check_assertive_vs_hypothetical(steps, final_answer=None):
    """
    D3: 断言式 vs 假设式
    答案是否用"ROE为36.99%"而非"如果ROE较高则说明..."
    """
    issues = []
    answer = get_finish_answer(steps, final_answer)
    if not answer:
        return issues

    hypothetical_patterns = [
        r"如果.*(?:较高|较低|良好|不错).*(?:则|说明|表明)",
        r"假设.*(?:那么|则)",
        r"若.*(?:可能|则说明)",
    ]

    hypothetical_count = sum(len(re.findall(p, answer)) for p in hypothetical_patterns)

    has_specific_assertion = bool(re.search(
        r'(?:ROE|净利率|资产负债率|毛利率|周转率|EPS|每股收益).*?(?:为|达到|约|仅)?\d+\.?\d*%?',
        answer
    ))

    if hypothetical_count > 0 and not has_specific_assertion:
        issues.append({
            "type": "hypothetical",
            "detail": "答案缺少具体百分比数字，全部为定性描述"
        })
    elif hypothetical_count >= 3:
        issues.append({
            "type": "hypothetical",
            "detail": f"答案包含{hypothetical_count}处假设式表述，应更多使用断言式"
        })

    return issues


def check_observation_match(steps, question):
    """
    D4: Observation 匹配度
    - 问盈利能力时返回的是否是盈利指标（而非偿债指标）
    - 问公司A时返回的是否是公司A的数据
    """
    issues = []

    # 判断问题关注的指标维度
    asks_profitability = any(kw in question for kw in PROFITABILITY_KW)
    asks_solvency = any(kw in question for kw in SOLVENCY_KW)
    asks_operational = any(kw in question for kw in OPERATIONAL_KW)

    for i, step in enumerate(steps):
        if step.get("action") == "finish":
            continue

        obs = step.get("observation", "")
        action = step.get("action", "")
        action_input = step.get("action_input", "")

        if not obs:
            continue

        # 检查1: 公司名匹配——搜索A公司但返回0条
        if action_input and len(action_input) < 30:
            if "找到" in obs and "0 条" in obs:
                issues.append({
                    "type": "obs_match",
                    "detail": f"step{i}: 搜索'{action_input}'但未找到数据"
                })

        # 检查2: 指标类型匹配（仅对 search_financial 检查）
        # 只有当所有非finish步骤都没返回正确类型的指标时才标记
        # （agent在后续步骤补检了正确数据是正常行为）
        if action == "search_financial" and i == 0:
            obs_has_profitability = any(kw in obs for kw in ["ROE", "ROA", "净利率", "毛利率", "每股收益", "EPS"])
            obs_has_solvency = any(kw in obs for kw in ["资产负债率", "流动比率", "速动比率"])

            # 检查后续步骤是否补检了正确类型的数据
            later_has_correct = False
            for j in range(i + 1, len(steps)):
                later_obs = steps[j].get("observation", "")
                if asks_profitability and any(kw in later_obs for kw in ["ROE", "ROA", "净利率", "毛利率", "EPS"]):
                    later_has_correct = True
                    break
                if asks_solvency and any(kw in later_obs for kw in ["资产负债率", "流动比率", "速动比率"]):
                    later_has_correct = True
                    break

            if not later_has_correct:
                if asks_profitability and not asks_solvency and not asks_operational:
                    if obs_has_solvency and not obs_has_profitability:
                        issues.append({
                            "type": "obs_match",
                            "detail": f"step{i}: 问盈利能力但全程未检索到盈利指标数据"
                        })
                if asks_solvency and not asks_profitability and not asks_operational:
                    if obs_has_profitability and not obs_has_solvency:
                        issues.append({
                            "type": "obs_match",
                            "detail": f"step{i}: 问偿债能力但全程未检索到偿债指标数据"
                        })

    return issues


def check_number_consistency(steps, final_answer=None):
    """
    D5: 数字一致性
    finish答案中引用的数字是否确实出现在Observation中
    """
    issues = []
    answer = get_finish_answer(steps, final_answer)
    if not answer:
        return issues

    # 收集所有observation中的数字
    all_obs_numbers = set()
    for s in steps:
        if s.get("action") != "finish":
            all_obs_numbers.update(extract_numbers(s.get("observation", "")))

    # 提取答案中与财务指标紧密关联的数字（排除年份）
    metric_pattern = r'(?:ROE|ROA|净利率|毛利率|资产负债率|流动比率|速动比率|周转率|EPS|每股收益|增长率|净利润增长|营收增长)[^\d]{0,10}?(\d+\.?\d*%)'
    answer_metric_numbers = re.findall(metric_pattern, answer)

    # 排除行业基准/判断标准值
    benchmark_patterns = [
        r'(?:高于|低于|超过|不足|达到|通常|一般|安全线|标准|健康|适宜|警戒|不适用)[^\d]{0,5}\d+\.?\d*%',
        r'\d+\.?\d*%\s*(?:为|是|属于)?\s*(?:安全|健康|良好|优秀|一般|较差|警戒)',
        r'(?:行业|通常|一般|标准|经验值)[^\d]{0,10}\d+\.?\d*%',
    ]
    benchmark_numbers = set()
    for bp in benchmark_patterns:
        for m in re.finditer(bp, answer):
            benchmark_numbers.update(re.findall(r'(\d+\.?\d*%)', m.group()))

    # 常见整数基准值
    COMMON_BENCHMARKS = {5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 1, 1.5, 2}

    inconsistent = []
    for num in answer_metric_numbers:
        if num in benchmark_numbers:
            continue

        if num not in all_obs_numbers:
            num_val = num.replace("%", "")
            found = False
            for obs_num in all_obs_numbers:
                obs_val = obs_num.replace("%", "")
                try:
                    if abs(float(num_val) - float(obs_val)) < 0.05:
                        found = True
                        break
                except ValueError:
                    continue
            if not found:
                try:
                    if float(num_val) in COMMON_BENCHMARKS:
                        continue
                except ValueError:
                    pass
                inconsistent.append(num)

    if len(inconsistent) >= 2:
        issues.append({
            "type": "number_consistency",
            "detail": f"答案中引用的数字 {inconsistent[:3]} 未在Observation中出现"
        })

    return issues


def check_industry_adaptation(steps, question, final_answer=None):
    """
    D6: 行业适配性
    银行/保险等金融行业不应套用制造业的分析标准
    """
    issues = []

    if not is_financial_industry(question, steps):
        return issues

    answer = get_finish_answer(steps, final_answer)
    if not answer:
        return issues

    # 银行资产负债率90%+是正常的，不应判为负面
    if re.search(r'资产负债率.*?[89]\d\.?\d*%.*?(?:偏高|风险|过高|危险|承压)', answer) or \
       (re.search(r'资产负债率.*?[89]\d\.?\d*%.*?警戒', answer) and not re.search(r'不适用.*?警戒', answer)):
        if not re.search(r'(?:银行|金融|保险).*?(?:正常|合理|行业特性|行业特征|行业常态)', answer) and \
           not re.search(r'(?:正常水平|行业常态|不适用)', answer):
            issues.append({
                "type": "industry_adaptation",
                "detail": "银行/金融业资产负债率80-90%+属正常水平，不应判断为'偏高'或'风险'"
            })

    # 银行不应提存货周转率/毛利率的"数据缺失"
    for metric in ["存货周转", "毛利率"]:
        if metric in answer:
            if re.search(r'(?:数据缺失|无此指标|无法获取|缺少.*?' + metric + r')', answer):
                issues.append({
                    "type": "industry_adaptation",
                    "detail": f"金融行业无{metric}指标，不应说'数据缺失'而应解释行业特性"
                })
            if metric == "毛利率" and re.search(r'毛利率.*?(?:偏低|较低|不高)', answer):
                issues.append({
                    "type": "industry_adaptation",
                    "detail": "金融行业盈利模式不同，不应使用毛利率来评判"
                })

    # 银行流动比率标准不同
    if re.search(r'流动比率.*?(?:低于|不足|偏低).*?(?:2|安全)', answer):
        issues.append({
            "type": "industry_adaptation",
            "detail": "银行流动比率标准不同于一般企业，不应以2:1为安全线"
        })

    return issues


def check_standard_consistency(steps, final_answer=None):
    """
    D7: 指标判断标准一致性
    ROE>15%为良好/>20%为优秀、流动比率2:1为安全线、资产负债率40-60%适宜
    """
    issues = []
    answer = get_finish_answer(steps, final_answer)
    if not answer:
        return issues

    other_metrics = r'(?:净利率|毛利率|周转率|负债率|乘数)'

    # --- ROE 标准 ---
    # ROE > 15% 判为一般/偏低 → 自相矛盾
    roe_low = re.findall(r'ROE[为约达\s]*(\d+\.?\d*)%[^，。\n]{0,15}?(?:一般|偏低|较低|不高|中等|较差)', answer)
    for roe_val in roe_low:
        try:
            if float(roe_val) > 15:
                ctx = re.search(r'ROE[为约达\s]*' + re.escape(roe_val) + r'%(.{0,15}?)(?:一般|偏低|较低|不高|中等)', answer)
                if ctx and not re.search(other_metrics, ctx.group(1)):
                    issues.append({
                        "type": "standard_consistency",
                        "detail": f"ROE {roe_val}%>15%应属良好水平，但答案判为一般/偏低"
                    })
        except ValueError:
            pass

    # ROE < 15% 判为优秀 → 自相矛盾
    roe_high = re.findall(r'ROE[为约达\s]*(\d+\.?\d*)%[^，。\n]{0,15}?(?:优秀|出色|极高|卓越)', answer)
    for roe_val in roe_high:
        try:
            if float(roe_val) < 15:
                ctx = re.search(r'ROE[为约达\s]*' + re.escape(roe_val) + r'%(.{0,15}?)(?:优秀|出色|极高|卓越)', answer)
                if ctx and not re.search(other_metrics, ctx.group(1)):
                    issues.append({
                        "type": "standard_consistency",
                        "detail": f"ROE {roe_val}%<15%不应判为优秀"
                    })
        except ValueError:
            pass

    # --- 资产负债率标准 ---（非金融行业）
    if not is_financial_industry("", steps):
        alr_high = re.findall(r'资产负债率[为约达\s]*(\d+\.?\d*)%[^，。\n]{0,15}?(?:偏高|过高|风险较大)', answer)
        for val in alr_high:
            try:
                if float(val) <= 50:
                    issues.append({
                        "type": "standard_consistency",
                        "detail": f"资产负债率{val}%在40-60%适宜范围内，不应判为偏高"
                    })
            except ValueError:
                pass

        alr_low = re.findall(r'资产负债率[为约达\s]*(\d+\.?\d*)%[^，。\n]{0,15}?(?:偏低|保守)', answer)
        for val in alr_low:
            try:
                if float(val) > 60:
                    issues.append({
                        "type": "standard_consistency",
                        "detail": f"资产负债率{val}%>60%不应判为偏低/保守"
                    })
            except ValueError:
                pass

    # --- 流动比率标准 ---（非金融行业）
    if not is_financial_industry("", steps):
        # 流动比率 > 2 判为差/偏低 → 自相矛盾
        # 排除 "低于2:1" 这种引用标准的表述
        cr_low = re.findall(r'流动比率[为约达\s]*(\d+\.?\d*)(?![^，。\n]*?低于)[^，。\n]{0,15}?(?:偏低|较低|不足|差)', answer)
        for val in cr_low:
            try:
                if float(val) >= 2.0:
                    issues.append({
                        "type": "standard_consistency",
                        "detail": f"流动比率{val}≥2属安全水平，不应判为偏低/差"
                    })
            except ValueError:
                pass

        # 流动比率 < 1 判为良好/安全 → 自相矛盾
        # 排除"安全标准""安全线"等参考表述（这是在引用标准，不是判断为安全）
        cr_high = re.findall(r'流动比率[为约达\s]*(\d+\.?\d*)[^，。\n]{0,15}?(?:良好|健康|优秀)', answer)
        # 单独处理"安全"：排除"安全标准""安全线"
        for m in re.finditer(r'流动比率[为约达\s]*(\d+\.?\d*)[^，。\n]{0,15}?安全', answer):
            # 检查"安全"后面是不是"标准""线"等
            end_pos = m.end()
            following = answer[end_pos:end_pos + 5] if end_pos < len(answer) else ""
            if not re.match(r'[标线水值]', following):
                cr_high.append(m.group(1))
        for val in cr_high:
            try:
                if float(val) < 1.0:
                    issues.append({
                        "type": "standard_consistency",
                        "detail": f"流动比率{val}<1偿债压力大，不应判为良好/安全"
                    })
            except ValueError:
                pass

    return issues


def check_dupont_correctness(steps, final_answer=None):
    """
    D8: 杜邦拆解正确性
    如果答案做了杜邦分析，检查三个因子相乘是否约等于ROE值，误差不应超过5个百分点
    """
    issues = []
    answer = get_finish_answer(steps, final_answer)
    if not answer or "杜邦" not in answer:
        return issues

    # 跳过声明无法完成杜邦分析的
    if re.search(r'(?:无法|数据有限|缺乏).*?杜邦', answer) or \
       re.search(r'杜邦.*?(?:无法|数据有限|缺乏)', answer):
        return issues

    # 限定到杜邦段落
    dupont_idx = answer.find("杜邦")
    dupont_section = answer[max(0, dupont_idx - 100):min(len(answer), dupont_idx + 500)]

    roe_match = re.search(r'ROE[为约达\s]*(\d+\.?\d*)%', dupont_section)
    net_margin_match = re.search(r'净利[润]?率[为约达\s]*(\d+\.?\d*)%', dupont_section)
    turnover_match = re.search(r'(?:总资产)?周转率[为约达\s]*(\d+\.?\d*)(?:次|倍)?', dupont_section)
    equity_match = re.search(r'权益乘数[为约达\s]*(\d+\.?\d*)(?:倍)?', dupont_section)

    if all([roe_match, net_margin_match, turnover_match, equity_match]):
        try:
            roe = float(roe_match.group(1))
            net_margin = float(net_margin_match.group(1))
            turnover = float(turnover_match.group(1))
            equity_mult = float(equity_match.group(1))

            if net_margin > 100 or turnover > 50 or equity_mult > 50:
                return issues  # 提取异常

            calculated_roe = (net_margin / 100) * turnover * equity_mult * 100
            diff = abs(calculated_roe - roe)

            if diff > 5:
                issues.append({
                    "type": "dupont",
                    "detail": f"杜邦三因子相乘={calculated_roe:.1f}% vs 报告ROE={roe}%，误差{diff:.1f}个百分点（>5%）"
                })
        except (ValueError, ZeroDivisionError):
            pass

    return issues


def check_time_consistency(steps, final_answer=None):
    """
    D9: 时间一致性
    同一分析框架下应使用同期数据，不应把不同期别数据混在一起
    """
    issues = []
    answer = get_finish_answer(steps, final_answer)
    if not answer:
        return issues

    date_pattern = r'(\d{4}-\d{2}-\d{2})'

    # 检查1: 杜邦分析中是否混用不同年份数据
    if "杜邦" in answer or ("净利率" in answer and "周转率" in answer and "权益乘数" in answer):
        answer_dates = re.findall(date_pattern, answer)
        answer_years = set(d[:4] for d in answer_dates)
        year_mentions = re.findall(r'(\d{4})(?:年报|半年报|H[12]|Q[1-4])', answer)
        answer_years.update(year_mentions)

        if len(answer_years) > 1:
            issues.append({
                "type": "time_consistency",
                "detail": f"杜邦分析中混用了不同年份数据: {', '.join(sorted(answer_years))}"
            })

    # 检查2: 同一段落/表格中混用跨度>=2年的数据做同一维度分析
    # 按段落拆分（以空行或**标题**分段）
    sections = re.split(r'\n\s*\n|\*\*[^*]+\*\*', answer)
    for section in sections:
        if len(section) < 50:
            continue
        # 提取该段落中的年份引用
        sec_years = set(re.findall(r'(\d{4})(?:年报|半年报|H[12]|Q[1-4]|-\d{2}-\d{2})', section))
        sec_years = {y for y in sec_years if 2020 <= int(y) <= 2026}
        if len(sec_years) >= 2:
            min_y, max_y = min(int(y) for y in sec_years), max(int(y) for y in sec_years)
            # 跨度>=2年且是同一维度分析（不是趋势对比）
            if max_y - min_y >= 2:
                # 排除趋势对比语境
                trend_kw = ["趋势", "变化", "对比", "较.*年", "同比", "环比", "增长", "下降", "提升"]
                is_trend = any(kw in section for kw in trend_kw)
                if not is_trend:
                    if not any(i["type"] == "time_consistency" for i in issues):
                        issues.append({
                            "type": "time_consistency",
                            "detail": f"同一分析段落中混用了跨度≥2年的数据: {', '.join(sorted(sec_years))}"
                        })

    return issues


def check_risk_specificity(steps, question, final_answer=None):
    """
    D10: 风险提示合理性
    风险分析类答案的风险提示是否具体（含数字/事件），而非空泛的"存在一定经营风险"
    """
    issues = []
    answer = get_finish_answer(steps, final_answer)
    if not answer:
        return issues

    # 扩展检查范围：question含风险关键词 OR answer中有风险提示段落
    risk_in_question = any(kw in question for kw in ["风险", "隐患", "挑战"])
    risk_in_answer = bool(re.search(r'(?:风险提示|风险分析|风险因素)[：:：]', answer))

    if not risk_in_question and not risk_in_answer:
        return issues

    # 检查空泛风险提示
    vague_risk_patterns = [
        r"存在一定(?:经营|财务|市场)?风险",
        r"需要关注(?:相关)?风险",
        r"有一定的(?:不确定性|波动性)",
        r"面临一定(?:挑战|压力)",
    ]
    vague_count = sum(len(re.findall(p, answer)) for p in vague_risk_patterns)

    # 检查具体风险提示
    specific_risk = bool(re.search(r'(?:风险|隐患|挑战).*?\d+\.?\d*%', answer)) or \
                    bool(re.search(r'(?:毛利率|利润率|增速|营收).*?(?:下降|压缩|收窄|下滑).*?\d', answer)) or \
                    bool(re.search(r'(?:政策|监管|行业|竞争|产能).*?(?:导致|影响|冲击|挤压)', answer))

    if vague_count >= 2 and not specific_risk:
        issues.append({
            "type": "risk_specificity",
            "detail": f"风险提示过于空泛（{vague_count}处），缺少具体数字或事件支撑"
        })

    return issues


# ============ 打分 ============

def score_sample(issues):
    """根据问题类型和数量打分（5分制，0.5步进）"""
    if not issues:
        return 5.0

    deductions = {
        "thought_coherence": 0.5,
        "answer_depth": 0.5,
        "hypothetical": 0.5,
        "obs_match": 1.0,
        "number_consistency": 1.0,
        "industry_adaptation": 1.0,
        "standard_consistency": 0.5,
        "dupont": 1.0,
        "time_consistency": 0.5,
        "risk_specificity": 0.5,
    }

    score = 5.0
    for issue in issues:
        score -= deductions.get(issue["type"], 0.5)

    return max(1.0, round(score * 2) / 2)


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="SFT 数据 10 维度质量检查")
    parser.add_argument("--input", type=str, default="data/sft/sft_data.jsonl",
                        help="输入数据路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出报告路径（默认在输入文件同目录生成 quality_report.jsonl）")
    parser.add_argument("--v1-report", type=str, default=None,
                        help="V1质检报告路径（用于对比）")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.replace(".jsonl", "_quality_report.jsonl")

    print("=" * 60)
    print("SFT 数据 10 维度质量检查")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print("=" * 60)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    print(f"总条数: {total}")

    results = []
    issue_counter = Counter()
    score_counter = Counter()
    low_score_samples = []

    for idx, line in enumerate(lines):
        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError:
            print(f"[WARN] 第{idx}行JSON解析失败，跳过")
            continue

        question = data.get("question", "")
        q_type = data.get("type", "unknown")
        steps = data.get("steps", [])

        all_issues = []
        all_issues.extend(check_thought_coherence(steps))           # D1
        all_issues.extend(check_finish_depth(steps, question))      # D2
        all_issues.extend(check_assertive_vs_hypothetical(steps))   # D3
        all_issues.extend(check_observation_match(steps, question)) # D4
        all_issues.extend(check_number_consistency(steps))          # D5
        all_issues.extend(check_industry_adaptation(steps, question))  # D6
        all_issues.extend(check_standard_consistency(steps))        # D7
        all_issues.extend(check_dupont_correctness(steps))          # D8
        all_issues.extend(check_time_consistency(steps))            # D9
        all_issues.extend(check_risk_specificity(steps, question))  # D10

        score = score_sample(all_issues)

        result = {
            "index": idx,
            "question": question,
            "type": q_type,
            "score": score,
            "issues": all_issues
        }
        results.append(result)

        score_counter[score] += 1
        for iss in all_issues:
            issue_counter[iss["type"]] += 1

        if score <= 2:
            low_score_samples.append((idx, question, score, [i["type"] for i in all_issues]))

        if (idx + 1) % 100 == 0:
            print(f"  已检查 {idx + 1}/{total} 条...")

    print(f"  已检查 {total}/{total} 条 (完成)")

    # 写出报告
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n质检报告已写入: {output_path}")

    # ===== 统计摘要 =====
    print("\n" + "=" * 60)
    print("统计摘要")
    print("=" * 60)

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n平均分: {avg_score:.2f}")

    print(f"\n分数分布:")
    for score in sorted(score_counter.keys(), reverse=True):
        count = score_counter[score]
        bar = "█" * int(count / total * 100)
        print(f"  {score:.1f}分: {count:4d} ({count / total * 100:5.1f}%) {bar}")

    issue_names = {
        "thought_coherence": "D1-Thought连贯性",
        "answer_depth": "D2-答案深度",
        "hypothetical": "D3-假设式表述",
        "obs_match": "D4-Observation匹配",
        "number_consistency": "D5-数字一致性",
        "industry_adaptation": "D6-行业适配性",
        "standard_consistency": "D7-指标标准一致性",
        "dupont": "D8-杜邦拆解正确性",
        "time_consistency": "D9-时间一致性",
        "risk_specificity": "D10-风险提示合理性",
    }

    print(f"\n各类问题出现频率:")
    all_issue_types = ["thought_coherence", "answer_depth", "hypothetical", "obs_match",
                       "number_consistency", "industry_adaptation", "standard_consistency",
                       "dupont", "time_consistency", "risk_specificity"]
    for issue_type in all_issue_types:
        count = issue_counter.get(issue_type, 0)
        name = issue_names.get(issue_type, issue_type)
        print(f"  {name}: {count:4d} ({count / total * 100:5.1f}%)")

    # ===== V1 对比（如果提供） =====
    v1_report_path = args.v1_report or "data/sft/quality_report.jsonl"
    v1_issues = {}
    v1_total = 0
    v1_avg = 0
    try:
        with open(v1_report_path, "r", encoding="utf-8") as f:
            v1_results = [json.loads(l) for l in f]
        v1_total = len(v1_results)
        v1_avg = sum(r["score"] for r in v1_results) / v1_total
        for r in v1_results:
            for iss in r.get("issues", []):
                v1_issues[iss["type"]] = v1_issues.get(iss["type"], 0) + 1
    except FileNotFoundError:
        pass

    if v1_total:
        print("\n" + "=" * 60)
        print("V1 vs V2 对比")
        print("=" * 60)
        print(f"\n{'指标':<25s} {'V1':>10s} {'V2':>10s} {'变化':>10s}")
        print("-" * 60)
        print(f"{'总条数':<25s} {v1_total:>10d} {total:>10d} {total - v1_total:>+10d}")
        print(f"{'平均分':<25s} {v1_avg:>10.2f} {avg_score:>10.2f} {avg_score - v1_avg:>+10.2f}")

        for issue_type in all_issue_types:
            v1_count = v1_issues.get(issue_type, 0)
            v2_count = issue_counter.get(issue_type, 0)
            v1_rate = v1_count / v1_total * 100
            v2_rate = v2_count / total * 100
            name = issue_names.get(issue_type, issue_type)
            print(f"  {name:<23s} {v1_rate:>8.1f}% {v2_rate:>8.1f}% {v2_rate - v1_rate:>+8.1f}pp")

        v1_d1 = v1_issues.get("thought_coherence", 0) / v1_total * 100
        v2_d1 = issue_counter.get("thought_coherence", 0) / total * 100
        print(f"\n★ 核心指标 - D1 Thought连贯性")
        print(f"  V1: {v1_d1:.1f}% → V2: {v2_d1:.1f}% (降{v1_d1 - v2_d1:.1f}pp)")

    # ===== 建议移除清单 =====
    print(f"\n{'=' * 60}")
    print(f"建议移除清单 (score <= 2)")
    print(f"{'=' * 60}")
    if low_score_samples:
        print(f"共 {len(low_score_samples)} 条:")
        for idx, q, score, itypes in low_score_samples:
            print(f"  [{idx:3d}] score={score:.1f} | {q[:40]}... | issues: {', '.join(itypes)}")
    else:
        print("  无 (所有样本分数 > 2)")


if __name__ == "__main__":
    main()
