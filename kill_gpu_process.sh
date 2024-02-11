#!/bin/bash

# 获取与/dev/nvidia*相关的进程号
pids=$(fuser -v /dev/nvidia* 2>/dev/null | awk '{print $2}')

# 检查是否存在相关进程
if [ -z "$pids" ]; then
  echo "没有找到与/dev/nvidia*相关的进程."
  exit 0
fi

# 打印找到的进程号
echo "找到以下与/dev/nvidia*相关的进程:"
echo "$pids"

# 杀死进程
echo "杀死这些进程..."
kill -9 $pids

echo "进程已杀死."
