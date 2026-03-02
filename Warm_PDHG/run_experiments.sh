#!/bin/bash

# --- 1. 路径与日志配置 ---
PYTHON_EXEC="/home/qidong/anaconda3/envs/pdmcf/bin/python"
SCRIPT_PATH="/home/qidong/dd/FlowShrink-PDHG/Warm_PDHG/main.py"
LOG_FILE="log.txt"

# 获取当前时间
START_TIME=$(date)

# 打印开始时间到日志
echo "========================================================" >> "$LOG_FILE"
echo "Batch Test Started at $START_TIME" >> "$LOG_FILE"
echo "========================================================" >> "$LOG_FILE"

# --- 2. 定义测试用例 ---
# 格式: "N k K warm_start(0/1) pdhg(0/1)"
# 0 表示 False, 1 表示 True
test_cases=(
    # N | k | K | warm_start(0/1) | pdhg(0/1)
    "500 20 500 1 1"
    "1000 20 1000 1 1"

    "500 20 500 0 1"
    "1000 20 1000 0 1"

    # "100 10 100 0 0"
    # "500 10 500 0 0"

    # "100 20 50 0 0"
    # "500 20 250 0 0"

    # "100 20 100 0 0"
    # "500 20 500 0 0"

    # # 单独测试warm的情况（给greedy准备）
    # "100 10 100 1 0"
    # "500 10 500 1 0"

    # "100 20 50 1 0"
    # "500 20 250 1 0"

    # "100 20 100 1 0"
    # "500 20 500 1 0"

    # # 下面使用pdhg求的值代替最优值
    # "100 10 100 1 1"
    # "500 10 500 1 1"

    # "100 20 50 1 1"
    # "500 20 250 1 1"

    # "100 20 100 1 1"
    # "500 20 500 1 1"
)

# --- 3. 循环执行 ---
for params in "${test_cases[@]}"; do
    # 读取参数 (将字符串拆分为变量)
    set -- $params
    VAL_N=$1
    VAL_k=$2
    VAL_K=$3
    VAL_WARM=$4
    VAL_PDHG=$5

    # 为了日志好看，构造一个 tag
    TAG="N=$VAL_N, k=$VAL_k, K=$VAL_K, Warm=$VAL_WARM"

    echo "Running Test: [ $TAG ] ..." >> "$LOG_FILE"
    
    # 执行 Python 脚本
    # 使用 -u 参数让 python 禁用输出缓冲，保证日志实时写入
    $PYTHON_EXEC -u $SCRIPT_PATH \
        --N $VAL_N \
        --k $VAL_k \
        --K $VAL_K \
        --warm_start $VAL_WARM \
        --pdhg $VAL_PDHG \
        >> "$LOG_FILE" 2>&1
    
    # 检查执行状态
    if [ $? -eq 0 ]; then
        echo ">>> Success: [ $TAG ]" >> "$LOG_FILE"
    else
        echo ">>> Failed:  [ $TAG ]" >> "$LOG_FILE"
    fi
    
    # 添加分隔符
    echo -e "\n--------------------------------------------------------\n" >> "$LOG_FILE"

    # 稍微休息一下，防止 GPU/CPU 过热或文件写入冲突
    sleep 2
done

# 记录结束时间
END_TIME=$(date)
echo "Batch Test Finished at $END_TIME" >> "$LOG_FILE"