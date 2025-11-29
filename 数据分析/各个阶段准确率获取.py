import re

def extract_stage_max_acc(log_file_path, stage_order):
    # 存储结果：key=阶段，value=(max_acc列表, 最大值, 记录条数, 最小值, 数据范围)
    results = {
        stage: ([], "未找到数据", 0, None, None)
        for stage in stage_order
    }
    current_stage_idx = -1  # 当前阶段索引（-1表示未启动任何阶段）
    stage_start_lines = {stage: None for stage in stage_order}  # 记录每个阶段的启动行号

    # 正则匹配：提取max_acc和阶段名称
    max_acc_pattern = re.compile(r'max_acc:(\d+\.\d+)')
    stage_pattern = re.compile(r'Learning on (\d+-\d+)')

    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line_num, line in enumerate(file, 1):
            # 1. 检测阶段启动，更新当前阶段和启动行号
            stage_match = stage_pattern.search(line)
            if stage_match:
                stage_name = f"Learning on {stage_match.group(1)}"
                if stage_name in stage_order:
                    current_stage_idx = stage_order.index(stage_name)
                    stage_start_lines[stage_name] = line_num
                    print(f"\n=== 进入阶段：{stage_name}（日志行号：{line_num}）===")

            # 2. 提取max_acc，按阶段顺序和行号精准归属
            max_acc_match = max_acc_pattern.search(line)
            if max_acc_match:
                max_acc = float(max_acc_match.group(1))
                target_stage = None

                # 核心逻辑：按阶段启动顺序和行号区间判断归属
                if current_stage_idx >= 0:
                    # 当前阶段已启动，先判断是否属于当前阶段
                    current_stage_name = stage_order[current_stage_idx]
                    if stage_start_lines[current_stage_name] <= line_num:
                        target_stage = current_stage_name
                    # 若数据行在当前阶段启动前，归属上一个已启动的阶段
                    elif current_stage_idx > 0:
                        prev_stage_name = stage_order[current_stage_idx - 1]
                        if stage_start_lines[prev_stage_name] is not None:
                            target_stage = prev_stage_name
                # 若还未启动任何阶段，归属第一个阶段（默认日志第一个阶段先启动）
                else:
                    first_stage = stage_order[0]
                    target_stage = first_stage

                # 存储并输出归属结果
                if target_stage in results:
                    results[target_stage][0].append(max_acc)
                    print(f"日志行{line_num} - 归属阶段：{target_stage} - max_acc: {max_acc:.4f}")
                else:
                    print(f"日志行{line_num} - 无归属阶段 - max_acc: {max_acc:.4f}（未统计）")

    # 3. 计算每个阶段的统计指标
    overall_max = 0.0
    overall_max_stage = ""
    for stage in stage_order:
        max_acc_list = results[stage][0]
        if max_acc_list:
            stage_max = max(max_acc_list)
            stage_min = min(max_acc_list)
            stage_count = len(max_acc_list)
            stage_range = f"{stage_min:.4f}% - {stage_max:.4f}%"
            results[stage] = (max_acc_list, stage_max, stage_count, stage_min, stage_range)
            if stage_max > overall_max:
                overall_max = stage_max
                overall_max_stage = stage

    return results, overall_max, overall_max_stage, stage_start_lines

if __name__ == "__main__":
    log_file = "main.log"  # 务必改为你的日志文件实际路径
    # 关键：替换为你日志中实际的6个阶段顺序（按日志中出现的先后排序！）
    # 示例格式（请根据你的真实阶段修改，比如）：
    stage_order = [
        "Learning on 0-50",
        "Learning on 50-60",
        "Learning on 60-70",
        "Learning on 70-80",
        "Learning on 80-90",
        "Learning on 90-100"
    ]
    # 运行函数
    final_results, global_max, global_max_stage, stage_starts = extract_stage_max_acc(log_file, stage_order)

    # 4. 输出6阶段详细汇总表
    print("\n" + "="*100)
    print("6阶段max_acc精准统计汇总表（按日志实际顺序排列）")
    print("="*100)
    print(f"{'阶段':<20} {'最大值':<12} {'记录条数':<10} {'最小值':<12} {'数据范围':<20} {'阶段启动行号':<15}")
    print("-"*100)
    for stage in stage_order:
        _, stage_max, stage_count, stage_min, stage_range = final_results[stage]
        start_line = stage_starts[stage] or "未检测到启动"
        if isinstance(stage_max, float):
            print(f"{stage:<20} {stage_max:.4f}% {'':<4}{stage_count:<6} {stage_min:.4f}% {'':<4}{stage_range:<20} {start_line:<15}")
        else:
            print(f"{stage:<20} {stage_max:<12} {'':<4}{stage_count:<6} {'-':<12} {'':<4}{'-':<20} {start_line:<15}")

    # 输出全局最大值
    print("\n" + "="*100)
    if global_max > 0:
        print(f"全局最大max_acc：{global_max:.4f}%")
        print(f"对应阶段：{global_max_stage}")
    else:
        print("未找到有效max_acc数据")
    print("="*100)
