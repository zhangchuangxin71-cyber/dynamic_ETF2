import re


def extract_global_max_metrics(log_file):
    # 强制UTF-8编码，确保特殊字符（如小数点、空格）正常读取
    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # ---------------------- 1. 提取所有 incremental_train 列表的全局最大值 ----------------------
    all_inc_values = []
    inc_pattern = r"\[memo.py\] => incremental_train:\s*\[([\d.,\s-]+?)\]"
    inc_matches = re.findall(inc_pattern, content)
    if inc_matches:
        for inc_list_str in inc_matches:
            for num_str in inc_list_str.split(','):
                num_str = num_str.strip()
                if num_str and num_str.replace('.', '').isdigit():
                    all_inc_values.append(float(num_str))
    global_inc_max = max(all_inc_values) if all_inc_values else "未找到 incremental_train 数据"

    # ---------------------- 2. 提取所有 max_acc 字段的全局最大值（交叉验证） ----------------------
    all_max_acc = []
    max_acc_pattern = r"\[memo.py\] => max_acc:(\d+\.\d+)"
    max_acc_matches = re.findall(max_acc_pattern, content)
    if max_acc_matches:
        all_max_acc = [float(acc) for acc in max_acc_matches]
    global_max_acc = max(all_max_acc) if all_max_acc else "未找到 max_acc 数据"

    # ---------------------- 3. 提取所有 init_train 列表的全局最大值 ----------------------
    all_init_values = []
    init_pattern = r"\[memo.py\] => init_train:\s*\[([\d.,\s-]+?)\]"
    init_matches = re.findall(init_pattern, content)
    if init_matches:
        for init_list_str in init_matches:
            for num_str in init_list_str.split(','):
                num_str = num_str.strip()
                if num_str and num_str.replace('.', '').isdigit():
                    all_init_values.append(float(num_str))
    global_init_max = max(all_init_values) if all_init_values else "未找到 init_train 数据"

    # ---------------------- 4. 提取 lrate、distill_weight、init_lr（日志中无则标注） ----------------------
    # 提取 lrate
    lrate_pattern = r"lrate:\s*(\d+\.\d+)\b"
    lrate = re.findall(lrate_pattern, content)[-1] if re.findall(lrate_pattern, content) else "未找到 lrate"

    # 提取 distill_weight
    distill_pattern = r"distill_weight:\s*(\d+\.\d+)\b"
    distill_weight = re.findall(distill_pattern, content)[-1] if re.findall(distill_pattern, content) else "未找到 distill_weight"

    # 新增：提取 init_lr（匹配日志中 "init_lr: 数值" 格式）
    init_lr_pattern = r"init_lr:\s*(\d+\.\d+)\b"
    init_lr = re.findall(init_lr_pattern, content)[-1] if re.findall(init_lr_pattern, content) else "未找到 init_lr"

    return {
        "lrate": lrate,
        "distill_weight": distill_weight,
        "init_lr": init_lr,  # 新增 init_lr 字段
        "global_init_train_max": global_init_max,
        "global_incremental_train_max": global_inc_max,
        "global_max_acc": global_max_acc
    }


def print_global_results(metrics):
    print("=" * 70)
    print("日志全局最大值提取结果：")
    print(f"1. lrate: {metrics['lrate']}")
    print(f"2. distill_weight: {metrics['distill_weight']}")
    print(f"3. init_lr: {metrics['init_lr']}")  # 新增 init_lr 输出
    # 格式化输出，保留4位小数（若为数字）
    if isinstance(metrics['global_init_train_max'], float):
        print(f"4. init_train 全局最大值: {metrics['global_init_train_max']:.4f}%")
    else:
        print(f"4. init_train 全局最大值: {metrics['global_init_train_max']}")

    if isinstance(metrics['global_incremental_train_max'], float):
        print(f"5. incremental_train 全局最大值: {metrics['global_incremental_train_max']:.4f}%")
    else:
        print(f"5. incremental_train 全局最大值: {metrics['global_incremental_train_max']}")

    if isinstance(metrics['global_max_acc'], float):
        print(f"6. max_acc 全局最大值: {metrics['global_max_acc']:.4f}%")
    else:
        print(f"6. max_acc 全局最大值: {metrics['global_max_acc']}")
    print("=" * 70)


if __name__ == "__main__":
    # 请替换为你的日志文件实际路径（相对路径/绝对路径均可）
    LOG_FILE = "main.log"  # 示例：LOG_FILE = "/xxx/xxx/main.log"
    try:
        results = extract_global_max_metrics(LOG_FILE)
        print_global_results(results)
    except Exception as e:
        print(f"运行错误：{str(e)}（请先检查日志文件路径是否正确）")
