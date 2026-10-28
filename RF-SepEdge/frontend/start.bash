# 获取项目根目录 RF-SepEdge
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 启动前端 HTTP 服务
cd "$PROJECT_DIR" || exit 1

echo "前端访问地址：http://127.0.0.1:8080/frontend/"

python -m http.server 8080 --bind 127.0.0.1