#!/bin/bash

echo "=========================================="
echo "  淘天播术-电商直播话术军师 - 直播话术智能分析系统"
echo "=========================================="
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到Python3，请先安装Python 3.8或更高版本"
    exit 1
fi

echo "[1/3] 正在检查依赖..."
# 检查并安装依赖
pip3 install -q -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[警告] 部分依赖安装失败，尝试继续启动..."
fi

echo "[2/3] 正在启动系统..."
echo ""
echo "系统将在浏览器中打开，请稍候..."
echo "如果浏览器没有自动打开，请手动访问：http://localhost:8501"
echo ""
echo "按Ctrl+C可以停止服务"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 启动Streamlit
streamlit run app.py --server.port=8501 --server.address=localhost
