#!/bin/bash

# ACD-TCM模型 部署脚本

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 项目配置
PROJECT_NAME="ACD-TCM"
PROJECT_DIR="/root/autodl-tmp/ACD-TCM"
MODEL_DIR="${PROJECT_DIR}/models"
SRC_DIR="${PROJECT_DIR}/src"
LOGS_DIR="${PROJECT_DIR}/logs"
CONFIG_DIR="${PROJECT_DIR}/configs"
DATA_DIR="${PROJECT_DIR}/data"

# 服务配置
SERVICE_PORT=7861
SERVICE_HOST="0.0.0.0"
WORKERS=1

# 检查系统要求
check_system_requirements() {
    log_info "检查系统要求..."
    
    # 检查Python版本
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装，请先安装Python3.8+"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python版本: $PYTHON_VERSION"
    
    # 检查CUDA
    if command -v nvidia-smi &> /dev/null; then
        log_info "检测到NVIDIA GPU:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    else
        log_warning "未检测到NVIDIA GPU，将使用CPU模式"
    fi
    
    # 检查内存
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    log_info "系统内存: ${TOTAL_MEM}GB"
    
    if [ "$TOTAL_MEM" -lt 16 ]; then
        log_warning "系统内存不足16GB，可能影响模型性能"
    fi
    
    # 检查磁盘空间
    DISK_SPACE=$(df -h $PROJECT_DIR | awk 'NR==2{print $4}')
    log_info "可用磁盘空间: $DISK_SPACE"
    
    log_success "系统要求检查完成"
}

# 创建必要目录
create_directories() {
    log_info "创建项目目录结构..."
    
    mkdir -p "$LOGS_DIR"
    mkdir -p "$DATA_DIR/uploads"
    mkdir -p "$DATA_DIR/results"
    
    # 设置目录权限
    chmod 755 "$PROJECT_DIR"
    chmod 755 "$LOGS_DIR"
    chmod 755 "$DATA_DIR"
    
    log_success "目录结构创建完成"
}

# 安装Python依赖
install_dependencies() {
    log_info "安装Python依赖包..."
    
    cd "$PROJECT_DIR"
    
    # 检查pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 未安装，请先安装pip3"
        exit 1
    fi
    
    # 升级pip
    python3 -m pip install --upgrade pip
    
    # 安装依赖
    if [ -f "requirements.txt" ]; then
        log_info "从requirements.txt安装依赖..."
        pip3 install -r requirements.txt
    else
        log_warning "requirements.txt不存在，手动安装核心依赖..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip3 install transformers accelerate peft gradio pillow numpy pandas
    fi
    
    log_success "依赖安装完成"
}

# 验证模型文件
validate_model() {
    log_info "验证模型文件..."
    
    MODEL_PATH="$MODEL_DIR/checkpoint-669"
    
    if [ ! -d "$MODEL_PATH" ]; then
        log_error "模型文件不存在: $MODEL_PATH"
        log_info "请确保模型文件已正确放置在models目录下"
        exit 1
    fi
    
    # 检查关键模型文件
    REQUIRED_FILES=("config.json" "pytorch_model.bin" "tokenizer.json")
    
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$MODEL_PATH/$file" ]; then
            log_warning "模型文件可能不完整，缺少: $file"
        fi
    done
    
    log_success "模型文件验证完成"
}

# 配置环境变量
setup_environment() {
    log_info "配置环境变量..."
    
    # 创建环境配置文件
    cat > "$PROJECT_DIR/.env" << EOF
# ACD-TCM 环境配置
PROJECT_DIR=$PROJECT_DIR
MODEL_DIR=$MODEL_DIR
LOGS_DIR=$LOGS_DIR
DATA_DIR=$DATA_DIR

# 服务配置
SERVICE_PORT=$SERVICE_PORT
SERVICE_HOST=$SERVICE_HOST
WORKERS=$WORKERS

# 模型配置
MODEL_NAME=checkpoint-669
MAX_LENGTH=2048
TEMPERATURE=0.7

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=$LOGS_DIR/acd_tcm.log
EOF
    
    # 设置PYTHONPATH
    export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
    
    log_success "环境变量配置完成"
}

# 运行测试
run_tests() {
    log_info "运行系统测试..."
    
    cd "$PROJECT_DIR"
    
    if [ -f "tests/run_tests.py" ]; then
        python3 tests/run_tests.py
        if [ $? -eq 0 ]; then
            log_success "所有测试通过"
        else
            log_error "测试失败，请检查系统配置"
            exit 1
        fi
    else
        log_warning "测试文件不存在，跳过测试"
    fi
}

# 启动服务
start_service() {
    log_info "启动ACD-TCM服务..."
    
    cd "$PROJECT_DIR"
    
    # 检查端口是否被占用
    if lsof -Pi :$SERVICE_PORT -sTCP:LISTEN -t >/dev/null ; then
        log_warning "端口 $SERVICE_PORT 已被占用，尝试终止现有进程..."
        pkill -f "python.*acne_diagnosis_web.py" || true
        sleep 2
    fi
    
    # 创建启动脚本
    cat > "$PROJECT_DIR/start_service.sh" << EOF
#!/bin/bash
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:\$PYTHONPATH"
source .env
python3 src/acne_diagnosis_web.py
EOF
    
    chmod +x "$PROJECT_DIR/start_service.sh"
    
    # 启动服务
    log_info "在后台启动服务..."
    nohup bash "$PROJECT_DIR/start_service.sh" > "$LOGS_DIR/service.log" 2>&1 &
    
    SERVICE_PID=$!
    echo $SERVICE_PID > "$PROJECT_DIR/service.pid"
    
    # 等待服务启动
    sleep 5
    
    # 检查服务状态
    if ps -p $SERVICE_PID > /dev/null; then
        log_success "服务启动成功！"
        log_info "服务PID: $SERVICE_PID"
        log_info "访问地址: http://$SERVICE_HOST:$SERVICE_PORT"
        log_info "日志文件: $LOGS_DIR/service.log"
    else
        log_error "服务启动失败，请检查日志: $LOGS_DIR/service.log"
        exit 1
    fi
}

# 停止服务
stop_service() {
    log_info "停止ACD-TCM服务..."
    
    if [ -f "$PROJECT_DIR/service.pid" ]; then
        SERVICE_PID=$(cat "$PROJECT_DIR/service.pid")
        if ps -p $SERVICE_PID > /dev/null; then
            kill $SERVICE_PID
            log_success "服务已停止"
        else
            log_warning "服务进程不存在"
        fi
        rm -f "$PROJECT_DIR/service.pid"
    else
        log_warning "未找到服务PID文件"
    fi
    
    # 强制终止相关进程
    pkill -f "python.*acne_diagnosis_web.py" || true
}

# 查看服务状态
check_status() {
    log_info "检查服务状态..."
    
    if [ -f "$PROJECT_DIR/service.pid" ]; then
        SERVICE_PID=$(cat "$PROJECT_DIR/service.pid")
        if ps -p $SERVICE_PID > /dev/null; then
            log_success "服务正在运行 (PID: $SERVICE_PID)"
            log_info "访问地址: http://$SERVICE_HOST:$SERVICE_PORT"
        else
            log_error "服务未运行"
        fi
    else
        log_error "服务未启动"
    fi
}

# 显示帮助信息
show_help() {
    echo "ACD-TCM 痤疮诊断系统部署脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  install     完整安装和部署系统"
    echo "  start       启动服务"
    echo "  stop        停止服务"
    echo "  restart     重启服务"
    echo "  status      查看服务状态"
    echo "  test        运行测试"
    echo "  help        显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 install   # 完整部署"
    echo "  $0 start     # 启动服务"
    echo "  $0 status    # 查看状态"
}

# 主函数
main() {
    echo "======================================"
    echo "    ACD-TCM 痤疮诊断系统部署脚本"
    echo "======================================"
    echo ""
    
    case "${1:-install}" in
        "install")
            log_info "开始完整部署..."
            check_system_requirements
            create_directories
            install_dependencies
            validate_model
            setup_environment
            run_tests
            start_service
            log_success "部署完成！"
            ;;
        "start")
            start_service
            ;;
        "stop")
            stop_service
            ;;
        "restart")
            stop_service
            sleep 2
            start_service
            ;;
        "status")
            check_status
            ;;
        "test")
            run_tests
            ;;
        "help")
            show_help
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"