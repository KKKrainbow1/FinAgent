"""
MinerU content_list.json 清洗脚本

输入: data/processed/mineru_compare/{stem}/vlm/{stem}_content_list.json (默认)
输出: data/processed/mineru_compare/{stem}/vlm/{stem}_content_list_cleaned.json
     (保留原文件不删,只加 cleaned 版本)

清洗规则（8 条,按优先级排序）:
  [P0-1] filter_trivial_types    : 丢 footer / page_number / header (页眉,不是 section heading)
  [P0-2] filter_short_text       : 丢 type=text 且长度 < 20 的噪声
  [P0-3] remove_repeated_runners : 统计重复文本 > 50% 页面,识别 MinerU 没标出的页眉页脚
  [P0-4] truncate_disclaimer     : 文档后半部分遇"免责声明"等 text_level=1 header,全部截断
  [P1-1] remove_toc              : 连续 3+ 行"标题...页码"样式,整段丢(目录页)
  [P1-2] merge_broken_blocks     : 上下两 block 被 PDF 换行切断时,合成完整句
  [P2-1] merge_image_caption     : 图/表 caption 独立成 text block 时,附到前一个 image/chart
  [P2-2] fix_ocr_errors          : 数字上下文 o/O -> 0, l/I -> 1（保守,仅纯数字段内）

用法:
    # 单份
    python finagent_repo/03d_clean_content_list.py \\
        --input data/processed/mineru_compare/000001_20251028_0/vlm/000001_20251028_0_content_list.json

    # 批量
    python finagent_repo/03d_clean_content_list.py \\
        --input-dir data/processed/mineru_compare
"""
import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============ 常量 ============

# 一定丢的类型(MinerU 已经分类,直接过滤)
TRIVIAL_TYPES = {'footer', 'page_number', 'header'}

# 最短保留阈值(text 类型)
# 从 20 降到 12: v1 下发现 "买入(维持)"/"Rating: OUTPERFORM"/发布日期等边界长度信息被误删
MIN_TEXT_LEN = 12

# 重复文本在多少比例页面上出现时判定为页眉页脚
RUNNER_PAGE_RATIO = 0.5
RUNNER_MAX_LEN = 50   # 页眉通常短

# 免责声明触发词(遇到 text_level=1 且文本含这些词 → 从这里截断)
# v1 残留: 分析师简介(剑桥/南开履历) / 投资者适当性匹配 / 评级定义 / 海通 Disclaimer — 这里补齐
DISCLAIMER_HEADERS = [
    # 声明/免责类
    '免责声明', '分析师声明', '分析师承诺', '证券分析师承诺',
    '法律声明', '特别声明', '重要声明', '一般声明', '合规声明',
    '风险提示及免责声明', '重要免责声明',
    # 评级/适当性
    '评级说明', '评级定义', '行业评级体系', '公司投资评级',
    '股票评级', '行业评级', '投资评级',
    '投资者适当性', '投资者适当性匹配',
    # 分析师简介(履历章节)
    '分析师简介', '分析师介绍', '研究员简介', '研究员介绍',
    # 英文
    'Disclaimer', 'Disclosure', 'Ratings Definitions', 'Analyst Stock Ratings',
]

# 免责声明最早可能出现的位置（按页占比）— 避免误杀前半部分
DISCLAIMER_EARLIEST_PAGE_RATIO = 0.5

# 目录行模式:
#   经典:  "XX章节......页码" / "XX… 3"(≥3 个 .…·\s 分隔)
#   编号:  "1. 全球... 4" / "5.风险提示 21"(以数字或中文序号开头,空格分隔尾部页码)
#   锚点:  "目录" / "插图目录" / "表格目录" 这种段头本身
RE_TOC_CLASSIC = re.compile(r'^.{2,40}?\s*[\.…·\s]{3,}\s*\d{1,3}\s*$')
RE_TOC_NUMBERED = re.compile(
    r'^(\d+[\.、]?|[一二三四五六七八九十]、|[IVX]+\.?)\s*.{2,40}?\s+\d{1,3}\s*$'
)
RE_TOC_ANCHOR = re.compile(r'^(目录|插图目录|表格目录|图表目录|附录|Contents)$')

# 句末标点
SENTENCE_END = set('。！？!?；;"\'」』】)）》〉〕')
# 上一句末尾如果是这些,说明句子在后面继续(强续写信号)
BROKEN_TAIL = set('，、：:,"\'“（(《〈〔-—')
# 当前句开头若以这些打头,说明是上句续写
BROKEN_HEAD_CHARS = set('0123456789,，.。;；:：)）】'']」}》〉〕%')

# Image caption 前缀模式
RE_IMG_CAPTION = re.compile(r'^(图\s*\d+|表\s*\d+|Figure\s*\d+|Table\s*\d+|资料来源|数据来源)[:：\s]')

# OCR 修复: 数字中间的 o/O/l/I
RE_OCR_O = re.compile(r'(?<=\d)[oO](?=\d)')
RE_OCR_L = re.compile(r'(?<=\d)[lI](?=\d)')

# Markdown blockquote 前缀 (`> ` 开头),MinerU VLM 会把 PDF 里缩进/强调段落标成 blockquote
# 只处理行首 > 标记,不影响公式里的 > 比较符(公式在 $...$ 里)
RE_MD_QUOTE_PREFIX = re.compile(r'^\s*>\s*', re.MULTILINE)

# 首页作者信息识别 (2026-04-20 加)
# 典型:"分析师:林瑾璐" / "021-25102905 linjl@dxzq.net.cn" / "SAC 执业证书编号:S0340520060001"
RE_AUTHOR_TITLE = re.compile(r'^(分析师|研究员|联系人|证券分析师|首席分析师)\s*[:：]\s*\S{2,6}$')
RE_CONTACT_INFO = re.compile(
    r'^[\d\-\s()（）]{7,25}[\s\u4e00-\u9fff]*$'                     # 电话(可能后缀"(主)"等)
    r'|^[\w\.\-\+]+@[\w\.\-]+$'                                     # 邮箱
    r'|^SAC\s*(执业证书)?.{0,6}$'                                  # SAC 证书号 header
    r'|^S\d{10,}$'                                                  # SAC 纯编号
    r'|^(电话|邮箱|执业证书|登记编号|手机|地址|电子邮箱)\s*[:：]\s*.*$'  # 联系方式键值对(value 可为空)
    r'|^(电话|邮箱|SAC|联系方式)\s*[:：]?\s*$'                        # 孤立前缀(MinerU 偶尔切成独立 block)
)


# ============ 工具函数 ============

def count_pages(blocks: list[dict]) -> int:
    """从 blocks 里算总页数"""
    if not blocks:
        return 0
    return max((b.get('page_idx', 0) for b in blocks), default=0) + 1


# ============ P0-0: Markdown 残留清理 ============

def strip_markdown_artifacts(blocks: list[dict]) -> tuple[list[dict], int]:
    """
    清 markdown 残留(目前只有 blockquote 前缀 `>` )
    对所有 text/caption/footnote 字段做 in-place normalize
    """
    fixed = 0

    def _clean(s: str) -> tuple[str, int]:
        new, n = RE_MD_QUOTE_PREFIX.subn('', s)
        return new.strip(), n

    for b in blocks:
        if b.get('text'):
            new, n = _clean(b['text'])
            if n > 0:
                b['text'] = new
                fixed += n
        # table/chart 的 caption/footnote 是 list[str]
        for field in ('table_caption', 'table_footnote', 'chart_caption', 'chart_footnote'):
            if field in b and isinstance(b[field], list):
                new_list = []
                for s in b[field]:
                    new_s, n = _clean(s)
                    new_list.append(new_s)
                    fixed += n
                b[field] = new_list
    return blocks, fixed


# ============ P0-1: 丢 trivial types ============

def filter_trivial_types(blocks: list[dict]) -> tuple[list[dict], int]:
    kept, removed = [], 0
    for b in blocks:
        if b.get('type') in TRIVIAL_TYPES:
            removed += 1
        else:
            kept.append(b)
    return kept, removed


# ============ P0-2: 丢过短 text block ============

def filter_short_text(blocks: list[dict], min_len: int = MIN_TEXT_LEN) -> tuple[list[dict], int]:
    """注意: text_level 非空的永远保留(是章节标题,即便短)"""
    kept, removed = [], 0
    for b in blocks:
        if b.get('type') == 'text' and not b.get('text_level'):
            text = b.get('text', '').strip()
            if len(text) < min_len:
                removed += 1
                continue
        kept.append(b)
    return kept, removed


# ============ P0-2.5: 首页作者信息(分析师签名 + 电话邮箱 SAC) ============

def filter_author_info(blocks: list[dict]) -> tuple[list[dict], int]:
    """
    删"分析师:XX"签名块 + 其后紧邻的联系方式块(电话/邮箱/SAC 编号)

    触发条件:text block,text 匹配 RE_AUTHOR_TITLE
    向后扫最多 8 个 block(覆盖最坏情况:分析师名 + SAC头 + SAC号 + 电话 + 邮箱头 + 邮箱值
    + 兼容 1-2 个中间碎片),每步遇到非 contact 长文本立即 break。
    """
    drop = set()
    n = len(blocks)
    for i, b in enumerate(blocks):
        if i in drop or b.get('type') != 'text':
            continue
        text = (b.get('text') or '').strip()
        if not RE_AUTHOR_TITLE.match(text):
            continue
        drop.add(i)
        # 向后扫 8 个 block(保护:遇到长正文立即 break,不会过度删除)
        for j in range(i + 1, min(i + 9, n)):
            nb = blocks[j]
            if nb.get('type') != 'text':
                continue
            nt = (nb.get('text') or '').strip()
            if not nt:
                continue
            # 遇到下一个作者标题,停(下一轮会处理)
            if RE_AUTHOR_TITLE.match(nt):
                break
            # 只删命中 contact pattern 的,不依赖 len 兜底(避免误删短正文章节)
            if RE_CONTACT_INFO.match(nt):
                drop.add(j)
            else:
                # 非 contact pattern 的长正文/未知块,立即 break
                break

    return [b for i, b in enumerate(blocks) if i not in drop], len(drop)


# ============ P0-3: 统计型页眉页脚去除 ============

def remove_repeated_runners(blocks: list[dict], total_pages: int) -> tuple[list[dict], int]:
    """
    统计 text 短块的重复出现次数,出现在 > RUNNER_PAGE_RATIO 比例页面的,视为 MinerU 没标出的页眉页脚
    只处理: type=text, 无 text_level, 长度 <= RUNNER_MAX_LEN
    """
    if total_pages < 2:
        return blocks, 0

    # 统计 (text, page_idx) 组合
    candidates: dict[str, set[int]] = {}
    for b in blocks:
        if b.get('type') != 'text' or b.get('text_level'):
            continue
        text = b.get('text', '').strip()
        if not text or len(text) > RUNNER_MAX_LEN:
            continue
        candidates.setdefault(text, set()).add(b.get('page_idx', -1))

    threshold = max(2, int(total_pages * RUNNER_PAGE_RATIO))
    runner_set = {t for t, pgs in candidates.items() if len(pgs) >= threshold}

    if not runner_set:
        return blocks, 0

    kept, removed = [], 0
    for b in blocks:
        if (b.get('type') == 'text'
                and not b.get('text_level')
                and b.get('text', '').strip() in runner_set):
            removed += 1
        else:
            kept.append(b)
    return kept, removed


# ============ P0-4: 免责声明后截断 ============

def truncate_disclaimer(blocks: list[dict], total_pages: int) -> tuple[list[dict], int]:
    """
    找到第一个 text_level 非空且 text 命中 DISCLAIMER_HEADERS 的 block,且
    其 page_idx >= total_pages * DISCLAIMER_EARLIEST_PAGE_RATIO,从那里往后全部丢弃
    """
    earliest_page = int(total_pages * DISCLAIMER_EARLIEST_PAGE_RATIO)
    cut_at: Optional[int] = None
    for i, b in enumerate(blocks):
        if b.get('type') != 'text' or not b.get('text_level'):
            continue
        if b.get('page_idx', 0) < earliest_page:
            continue
        text = b.get('text', '').strip()
        if any(kw in text for kw in DISCLAIMER_HEADERS):
            cut_at = i
            break

    if cut_at is None:
        return blocks, 0
    removed = len(blocks) - cut_at
    return blocks[:cut_at], removed


# ============ P1-1: 目录页识别 ============

def remove_toc(blocks: list[dict]) -> tuple[list[dict], int]:
    """
    连续 3+ TOC 样式的 block 整段丢。
    TOC 识别 3 种模式:classic(...页码) / numbered(编号+空格+页码) / anchor(目录/插图目录)
    连续检测允许中间夹空 list 或空 text(MinerU 偶尔在目录项之间插入 list 空块)
    """
    is_toc = [False] * len(blocks)
    for i, b in enumerate(blocks):
        if b.get('type') != 'text':
            continue
        text = (b.get('text') or '').strip()
        if (RE_TOC_CLASSIC.match(text)
                or RE_TOC_NUMBERED.match(text)
                or RE_TOC_ANCHOR.match(text)):
            is_toc[i] = True

    def _is_filler(b):
        """空 list 或空 text 允许被计入 TOC 连续段,但不算 TOC 行本身"""
        if b.get('type') == 'list':
            return True
        if b.get('type') == 'text' and not (b.get('text') or '').strip():
            return True
        return False

    mark_drop = [False] * len(blocks)
    i = 0
    n = len(blocks)
    while i < n:
        if not is_toc[i]:
            i += 1
            continue
        j = i
        toc_count = 0
        while j < n:
            if is_toc[j]:
                toc_count += 1
                j += 1
            elif _is_filler(blocks[j]):
                j += 1
            else:
                break
        if toc_count >= 3:
            for k in range(i, j):
                mark_drop[k] = True
        i = j

    kept = [b for b, drop in zip(blocks, mark_drop) if not drop]
    return kept, sum(mark_drop)


# ============ P1-2: 断句合并 ============

def _should_merge(prev: dict, curr: dict) -> bool:
    """判断 prev 和 curr 是否应该合并（都是 text，被 PDF 换行切断）"""
    if prev.get('type') != 'text' or curr.get('type') != 'text':
        return False
    # 其中任何一个是章节标题,不合并
    if prev.get('text_level') or curr.get('text_level'):
        return False
    # 跨越太多页(>1 页)不合并
    if curr.get('page_idx', 0) - prev.get('page_idx', 0) > 1:
        return False

    prev_text = prev.get('text', '').strip()
    curr_text = curr.get('text', '').strip()
    if not prev_text or not curr_text:
        return False

    prev_last = prev_text[-1]
    curr_first = curr_text[0]

    # 条件: prev 末尾不是句末标点 AND (prev 末尾像逗号/冒号/破折号,或 curr 开头是数字/标点/续字)
    prev_incomplete = prev_last not in SENTENCE_END
    strong_continue = (prev_last in BROKEN_TAIL) or (curr_first in BROKEN_HEAD_CHARS)
    return prev_incomplete and strong_continue


def merge_broken_blocks(blocks: list[dict]) -> tuple[list[dict], int]:
    if not blocks:
        return blocks, 0
    merged = [blocks[0]]
    merge_count = 0
    for b in blocks[1:]:
        prev = merged[-1]
        if _should_merge(prev, b):
            prev['text'] = prev.get('text', '').rstrip() + b.get('text', '').lstrip()
            # bbox 扩展到两个 block 的外包矩形(若都有)
            pb = prev.get('bbox')
            cb = b.get('bbox')
            if pb and cb and len(pb) == 4 and len(cb) == 4:
                prev['bbox'] = [
                    min(pb[0], cb[0]), min(pb[1], cb[1]),
                    max(pb[2], cb[2]), max(pb[3], cb[3]),
                ]
            merge_count += 1
        else:
            merged.append(b)
    return merged, merge_count


# ============ P2-1: 图表 caption 归属 ============

def merge_image_caption(blocks: list[dict]) -> tuple[list[dict], int]:
    """
    某些版面下 MinerU 把图片 caption 误切成独立 text,这里贴回前一个 image/chart/table block 的 caption 字段
    条件: 前一个 block 是 image/chart/table, 当前是 text, 长度 <=80 且匹配 RE_IMG_CAPTION
    """
    if len(blocks) < 2:
        return blocks, 0

    keep_mask = [True] * len(blocks)
    attached = 0
    for i in range(1, len(blocks)):
        curr = blocks[i]
        if curr.get('type') != 'text' or curr.get('text_level'):
            continue
        text = curr.get('text', '').strip()
        if not text or len(text) > 80:
            continue
        if not RE_IMG_CAPTION.match(text):
            continue

        prev = blocks[i - 1]
        ptype = prev.get('type')
        if ptype not in ('image', 'chart', 'table'):
            continue

        # 选 caption 还是 footnote
        target_field = f'{ptype}_footnote' if text.startswith(('资料来源', '数据来源')) else f'{ptype}_caption'
        # image 类型没有 image_caption 字段,统一用 caption
        if ptype == 'image':
            target_field = 'image_footnote' if text.startswith(('资料来源', '数据来源')) else 'image_caption'

        if target_field not in prev:
            prev[target_field] = []
        if isinstance(prev[target_field], list):
            prev[target_field].append(text)
        else:
            prev[target_field] = [prev[target_field], text]
        keep_mask[i] = False
        attached += 1

    kept = [b for b, keep in zip(blocks, keep_mask) if keep]
    return kept, attached


# ============ P2-2: OCR 错识修复 ============

def fix_ocr_errors(blocks: list[dict]) -> tuple[list[dict], int]:
    """
    保守地修复数字中间的 o/O → 0, l/I → 1
    只改 text 字段和 table_body / chart content 等文本内容
    """
    fixed = 0
    for b in blocks:
        # 处理 text
        if b.get('type') == 'text' and b.get('text'):
            new_text, n1 = RE_OCR_O.subn('0', b['text'])
            new_text, n2 = RE_OCR_L.subn('1', new_text)
            if n1 + n2 > 0:
                b['text'] = new_text
                fixed += n1 + n2
        # table_body
        if b.get('type') == 'table' and b.get('table_body'):
            new_body, n1 = RE_OCR_O.subn('0', b['table_body'])
            new_body, n2 = RE_OCR_L.subn('1', new_body)
            if n1 + n2 > 0:
                b['table_body'] = new_body
                fixed += n1 + n2
        # chart content (markdown 表格)
        if b.get('type') == 'chart' and b.get('content'):
            new_content, n1 = RE_OCR_O.subn('0', b['content'])
            new_content, n2 = RE_OCR_L.subn('1', new_content)
            if n1 + n2 > 0:
                b['content'] = new_content
                fixed += n1 + n2
    return blocks, fixed


# ============ 主流程 ============

def clean_one(input_path: Path, output_path: Path) -> dict:
    """返回单份文件的清洗统计"""
    with open(input_path, encoding='utf-8') as f:
        blocks = json.load(f)

    original_count = len(blocks)
    total_pages = count_pages(blocks)
    stats: dict[str, int] = {'original': original_count, 'total_pages': total_pages}

    blocks, n = strip_markdown_artifacts(blocks);       stats['p0_0_md_strip'] = n
    blocks, n = filter_trivial_types(blocks);           stats['p0_1_trivial'] = n
    # 2026-04-20: 必须在 filter_short_text 之前!"分析师:XX"长度 < 12,
    # 若先跑 short_text 会把锚点删掉,author_info 就找不到触发点
    blocks, n = filter_author_info(blocks);             stats['p0_15_author_info'] = n
    blocks, n = filter_short_text(blocks);              stats['p0_2_short_text'] = n
    blocks, n = remove_repeated_runners(blocks, total_pages); stats['p0_3_runners'] = n
    blocks, n = truncate_disclaimer(blocks, total_pages);     stats['p0_4_disclaimer'] = n
    blocks, n = remove_toc(blocks);                     stats['p1_1_toc'] = n
    blocks, n = merge_broken_blocks(blocks);            stats['p1_2_merged'] = n
    blocks, n = merge_image_caption(blocks);            stats['p2_1_caption'] = n
    blocks, n = fix_ocr_errors(blocks);                 stats['p2_2_ocr_fixes'] = n

    stats['final'] = len(blocks)
    stats['reduction_pct'] = round((original_count - len(blocks)) / max(original_count, 1) * 100, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(blocks, f, ensure_ascii=False, indent=2)

    return stats


def find_content_lists(root_dir: Path) -> list[Path]:
    """递归找所有 *_content_list.json(排除已 cleaned 的)"""
    results = []
    for p in root_dir.rglob('*_content_list.json'):
        if '_cleaned' in p.stem:
            continue
        results.append(p)
    return sorted(results)


def batch_run(input_dir: Path, suffix: str = '_cleaned') -> None:
    files = find_content_lists(input_dir)
    if not files:
        logger.warning(f"在 {input_dir} 下未找到 *_content_list.json")
        return

    logger.info(f"开始批量清洗 {len(files)} 份文件")
    all_stats = []
    errors = []

    for i, path in enumerate(files, 1):
        out_path = path.with_name(f"{path.stem}{suffix}.json")
        try:
            st = clean_one(path, out_path)
            st['file'] = path.name
            all_stats.append(st)
            logger.info(
                f"[{i}/{len(files)}] {path.stem}: "
                f"{st['original']} -> {st['final']} blocks ({st['reduction_pct']}% 减少), "
                f"pages={st['total_pages']}, "
                f"md_strip={st['p0_0_md_strip']}, "
                f"trivial={st['p0_1_trivial']}, author={st['p0_15_author_info']}, "
                f"short={st['p0_2_short_text']}, "
                f"runners={st['p0_3_runners']}, disclaim={st['p0_4_disclaimer']}, "
                f"toc={st['p1_1_toc']}, merged={st['p1_2_merged']}, "
                f"cap={st['p2_1_caption']}, ocr={st['p2_2_ocr_fixes']}"
            )
        except Exception as e:
            logger.error(f"[{i}/{len(files)}] {path.name} 失败: {e}")
            errors.append({'file': path.name, 'error': str(e)})

    # 汇总
    if all_stats:
        keys = ['p0_0_md_strip', 'p0_1_trivial', 'p0_15_author_info', 'p0_2_short_text',
                'p0_3_runners', 'p0_4_disclaimer', 'p1_1_toc', 'p1_2_merged',
                'p2_1_caption', 'p2_2_ocr_fixes']
        totals = {k: sum(s[k] for s in all_stats) for k in keys}
        total_orig = sum(s['original'] for s in all_stats)
        total_final = sum(s['final'] for s in all_stats)
        logger.info("=" * 60)
        logger.info(f"汇总: {len(all_stats)} 份成功, {len(errors)} 份失败")
        logger.info(f"  blocks: {total_orig} -> {total_final} "
                    f"({round((total_orig - total_final) / max(total_orig, 1) * 100, 1)}% 减少)")
        for k in keys:
            logger.info(f"  {k}: {totals[k]}")

        # 写汇总 json
        summary_path = input_dir / 'clean_stats_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'per_file': all_stats,
                'totals': totals,
                'errors': errors,
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"汇总统计已存: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='MinerU content_list.json 清洗工具')
    parser.add_argument('--input', type=str, default=None,
                        help='单份 content_list.json 路径')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='批量模式: 递归扫描此目录下的 *_content_list.json')
    parser.add_argument('--output-suffix', type=str, default='_cleaned',
                        help='输出文件名后缀(默认 _cleaned)')
    args = parser.parse_args()

    if args.input:
        in_path = Path(args.input).resolve()
        out_path = in_path.with_name(f"{in_path.stem}{args.output_suffix}.json")
        st = clean_one(in_path, out_path)
        logger.info(f"单份完成: {in_path.name}")
        logger.info(f"  stats: {st}")
        logger.info(f"  输出: {out_path}")
    elif args.input_dir:
        batch_run(Path(args.input_dir).resolve(), args.output_suffix)
    else:
        parser.error('必须指定 --input 或 --input-dir')


if __name__ == '__main__':
    main()
