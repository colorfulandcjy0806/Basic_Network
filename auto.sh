#!/bin/bash

# 进入你的Git仓库目录
# cd /path/to/your/git/repository

# 执行git add .
git add .

# 询问用户输入commit message
read -p "请输入commit message: " commit_message

# 检查commit message是否为空
if [ -z "$commit_message" ]; then
  echo "commit message不能为空，提交取消。"
  exit 1
fi

# 执行git commit
git commit -m "$commit_message"

# 推送到远程仓库（假设远程分支为main，你可以根据实际情况修改）
git push

