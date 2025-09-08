#!/bin/bash

# Face Swap Advanced - Git Installation Script
# This script clones the repository and sets up the environment

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/teguh87/faceswap.git"
PROJECT_NAME="faceswap"
PYTHON_MIN_VERSION="3.8"
DEFAULT_INSTALL_DIR="$HOME/faceswap"

# Global variables
INSTALL_DIR=""
PYTHON_CMD=""
CREATE_VENV=true
INSTALL_GPU=false
INSTALL_DEV=false
DOWNLOAD_MODELS=false
SKIP_DEPS_CHECK=false
FORCE_INSTALL=false
VERBOSE=false

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_info() {
    print_color $BLUE "[INFO] $1"
}

print_success() {
    print_color $GREEN "[SUCCESS] $1"
}

print_warning() {
    print_color $YELLOW "[WARNING] $1"
}

print_error() {
    print_color $RED "[ERROR] $1"
}

print_header() {
    print_color $PURPLE "=================================="
    print_color $PURPLE "$1"
    print_color $PURPLE "=================================="
}

# Function to show usage
show_usage() {
    cat << EOF
Face Swap Advanced - Git Installation Script

Usage: $0 [OPTIONS]

Options:
    -d, --dir DIR           Installation directory (default: $DEFAULT_INSTALL_DIR)
    -p, --python PYTHON     Python command to use (default: auto-detect)
    --no-venv               Don't create virtual environment
    --gpu                   Install with GPU support
    --dev                   Install development dependencies
    --models                Download pre-trained models
    --skip-deps             Skip system dependencies check
    --force                 Force installation (overwrite existing)
    -v, --verbose           Verbose output
    -h, --help              Show this help message

Examples:
    $0                                    # Basic installation
    $0 --gpu --models                     # Install with GPU support and models
    $0 --dev --dir /opt/face-swap         # Development install in custom directory
    $0 --python python3.9 --no-venv      # Use specific Python without venv

EOF
}

# Function to parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--dir)
                INSTALL_DIR="$2"
                shift 2
                ;;
            -p|--python)
                PYTHON_CMD="$2"
                shift 2
                ;;
            --no-venv)
                CREATE_VENV=false
                shift
                ;;
            --gpu)
                INSTALL_GPU=true
                shift
                ;;
            --dev)
                INSTALL_DEV=true
                shift
                ;;
            --models)
                DOWNLOAD_MODELS=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPS_CHECK=true
                shift
                ;;
            --force)
                FORCE_INSTALL=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Set default installation directory if not provided
    if [ -z "$INSTALL_DIR" ]; then
        INSTALL_DIR="$DEFAULT_INSTALL_DIR"
    fi
}

# Function to detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to compare version numbers
version_compare() {
    if [[ $1 == $2 ]]; then
        return 0
    fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    # fill empty fields in ver1 with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do
        ver1[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++)); do
        if [[ -z ${ver2[i]} ]]; then
            # fill empty fields in ver2 with zeros
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]})); then
            return 1
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]})); then
            return 2
        fi
    done
    return 0
}

# Function to find suitable Python command
find_python() {
    local python_candidates=("python3" "python" "python3.11" "python3.10" "python3.9" "python3.8")
    
    if [ -n "$PYTHON_CMD" ]; then
        if command_exists "$PYTHON_CMD"; then
            echo "$PYTHON_CMD"
            return 0
        else
            print_error "Specified Python command '$PYTHON_CMD' not found"
            exit 1
        fi
    fi
    
    for cmd in "${python_candidates[@]}"; do
        if command_exists "$cmd"; then
            local version=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            version_compare "$version" "$PYTHON_MIN_VERSION"
            if [[ $? -eq 1 ]] || [[ $? -eq 0 ]]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    
    print_error "No suitable Python ($PYTHON_MIN_VERSION+) found"
    exit 1
}

# Function to install system dependencies
install_system_dependencies() {
    local os=$(detect_os)
    
    print_info "Installing system dependencies for $os..."
    
    case $os in
        linux)
            if command_exists apt-get; then
                # Ubuntu/Debian
                print_info "Installing dependencies with apt-get..."
                sudo apt-get update
                sudo apt-get install -y \
                    git \
                    python3-dev \
                    python3-pip \
                    python3-venv \
                    libgl1-mesa-glx \
                    libglib2.0-0 \
                    libsm6 \
                    libxext6 \
                    libxrender-dev \
                    libgomp1 \
                    wget \
                    curl
            elif command_exists yum; then
                # CentOS/RHEL/Fedora
                print_info "Installing dependencies with yum..."
                sudo yum update -y
                sudo yum install -y \
                    git \
                    python3-devel \
                    python3-pip \
                    mesa-libGL \
                    glib2 \
                    libSM \
                    libXext \
                    libXrender \
                    libgomp \
                    wget \
                    curl
            elif command_exists pacman; then
                # Arch Linux
                print_info "Installing dependencies with pacman..."
                sudo pacman -Syu --noconfirm \
                    git \
                    python \
                    python-pip \
                    mesa \
                    glib2 \
                    libsm \
                    libxext \
                    libxrender \
                    wget \
                    curl
            else
                print_warning "Unknown Linux distribution. Please install dependencies manually."
            fi
            ;;
        macos)
            if command_exists brew; then
                print_info "Installing dependencies with Homebrew..."
                brew update
                brew install git python wget curl
            else
                print_warning "Homebrew not found. Please install Homebrew first or install dependencies manually."
                print_info "Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            fi
            ;;
        windows)
            print_warning "Windows detected. Please ensure you have:"
            print_info "  - Git for Windows"
            print_info "  - Python $PYTHON_MIN_VERSION+"
            print_info "  - Visual Studio Build Tools"
            ;;
        *)
            print_warning "Unknown OS. Please install dependencies manually:"
            print_info "  - Git"
            print_info "  - Python $PYTHON_MIN_VERSION+"
            print_info "  - Build tools for your system"
            ;;
    esac
}

# Function to check system dependencies
check_dependencies() {
    if [ "$SKIP_DEPS_CHECK" = true ]; then
        print_info "Skipping dependencies check..."
        return 0
    fi
    
    print_info "Checking system dependencies..."
    
    local missing_deps=()
    
    if ! command_exists git; then
        missing_deps+=("git")
    fi
    
    if [ -z "$(find_python 2>/dev/null)" ]; then
        missing_deps+=("python$PYTHON_MIN_VERSION+")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_warning "Missing dependencies: ${missing_deps[*]}"
        read -p "Do you want to install system dependencies? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_system_dependencies
        else
            print_error "Please install missing dependencies and run the script again."
            exit 1
        fi
    else
        print_success "All system dependencies are available"
    fi
}

# Function to clone repository
clone_repository() {
    print_info "Cloning repository from $REPO_URL..."
    
    if [ -d "$INSTALL_DIR" ]; then
        if [ "$FORCE_INSTALL" = true ]; then
            print_warning "Removing existing directory: $INSTALL_DIR"
            rm -rf "$INSTALL_DIR"
        else
            print_error "Directory already exists: $INSTALL_DIR"
            print_info "Use --force to overwrite or choose a different directory with --dir"
            exit 1
        fi
    fi
    
    if [ "$VERBOSE" = true ]; then
        git clone --progress "$REPO_URL" "$INSTALL_DIR"
    else
        git clone --quiet "$REPO_URL" "$INSTALL_DIR"
    fi
    
    if [ ! -d "$INSTALL_DIR" ]; then
        print_error "Failed to clone repository"
        exit 1
    fi
    
    print_success "Repository cloned to: $INSTALL_DIR"
}

# Function to setup virtual environment
setup_virtual_environment() {
    if [ "$CREATE_VENV" = false ]; then
        print_info "Skipping virtual environment creation..."
        return 0
    fi
    
    print_info "Creating virtual environment..."
    
    cd "$INSTALL_DIR"
    
    local venv_dir="venv"
    $PYTHON_CMD -m venv "$venv_dir"
    
    # Activate virtual environment
    source "$venv_dir/bin/activate" 2>/dev/null || source "$venv_dir/Scripts/activate" 2>/dev/null
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    print_success "Virtual environment created and activated"
}

# Function to install Python dependencies
install_python_dependencies() {
    print_info "Installing Python dependencies..."
    
    cd "$INSTALL_DIR"
    
    # Determine which requirements to install
    local requirements_files=("requirements.txt")
    
    if [ "$INSTALL_GPU" = true ]; then
        print_info "Including GPU support..."
        if [ -f "requirements-gpu.txt" ]; then
            requirements_files+=("requirements-gpu.txt")
        fi
    fi
    
    if [ "$INSTALL_DEV" = true ]; then
        print_info "Including development dependencies..."
        if [ -f "requirements-dev.txt" ]; then
            requirements_files+=("requirements-dev.txt")
        fi
    fi
    
    # Install requirements
    for req_file in "${requirements_files[@]}"; do
        if [ -f "$req_file" ]; then
            print_info "Installing from $req_file..."
            if [ "$VERBOSE" = true ]; then
                pip install -r "$req_file"
            else
                pip install -r "$req_file" --quiet
            fi
        else
            print_warning "Requirements file not found: $req_file"
        fi
    done
    
    # Install package in development mode
    print_info "Installing Face Swap Advanced in development mode..."
    if [ "$VERBOSE" = true ]; then
        pip install -e .
    else
        pip install -e . --quiet
    fi
    
    print_success "Python dependencies installed"
}

# Function to download models
download_models() {
    if [ "$DOWNLOAD_MODELS" = false ]; then
        return 0
    fi
    
    print_info "Downloading pre-trained models..."
    
    cd "$INSTALL_DIR"
    
    # Create models directory
    mkdir -p models
    
    # Download inswapper model (example URL - replace with actual)
    local model_url="https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx"
    local model_file="models/inswapper_128.onnx"
    
    if [ ! -f "$model_file" ]; then
        print_info "Downloading inswapper_128.onnx..."
        if command_exists wget; then
            wget -O "$model_file" "$model_url"
        elif command_exists curl; then
            curl -L -o "$model_file" "$model_url"
        else
            print_warning "Neither wget nor curl found. Please download models manually:"
            print_info "  Model URL: $model_url"
            print_info "  Save to: $model_file"
            return 0
        fi
        
        if [ -f "$model_file" ]; then
            print_success "Model downloaded: $model_file"
        else
            print_error "Failed to download model"
        fi
    else
        print_info "Model already exists: $model_file"
    fi
}

# Function to run tests
run_tests() {
    print_info "Running basic tests..."
    
    cd "$INSTALL_DIR"
    
    # Test import
    if $PYTHON_CMD -c "import face_swap_advanced; print('✓ Package import successful')" 2>/dev/null; then
        print_success "Package import test passed"
    else
        print_error "Package import test failed"
        return 1
    fi
    
    # Test CLI
    if $PYTHON_CMD -c "from face_swap_advanced.cli import main; print('✓ CLI import successful')" 2>/dev/null; then
        print_success "CLI import test passed"
    else
        print_error "CLI import test failed"
        return 1
    fi
    
    # Test CLI command availability
    if command_exists face-swap; then
        print_success "CLI command 'face-swap' is available"
    else
        print_warning "CLI command 'face-swap' not found in PATH"
        print_info "You may need to restart your shell or source the virtual environment"
    fi
}

# Function to create activation script
create_activation_script() {
    if [ "$CREATE_VENV" = false ]; then
        return 0
    fi
    
    print_info "Creating activation script..."
    
    cd "$INSTALL_DIR"
    
    cat > activate_face_swap.sh << 'EOF'
#!/bin/bash
# Face Swap Advanced - Environment Activation Script

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "Face Swap Advanced environment activated!"
    echo "Use 'face-swap --help' to get started"
elif [ -f "$SCRIPT_DIR/venv/Scripts/activate" ]; then
    source "$SCRIPT_DIR/venv/Scripts/activate"
    echo "Face Swap Advanced environment activated!"
    echo "Use 'face-swap --help' to get started"
else
    echo "Virtual environment not found!"
    exit 1
fi
EOF
    
    chmod +x activate_face_swap.sh
    
    print_success "Activation script created: activate_face_swap.sh"
}

# Function to show post-installation instructions
show_post_install_instructions() {
    print_header "Installation Complete!"
    
    print_success "Face Swap Advanced has been installed successfully!"
    print_info "Installation directory: $INSTALL_DIR"
    
    echo
    print_color $CYAN "Getting Started:"
    
    if [ "$CREATE_VENV" = true ]; then
        print_info "1. Activate the environment:"
        print_color $YELLOW "   cd $INSTALL_DIR"
        print_color $YELLOW "   source activate_face_swap.sh"
        echo
    fi
    
    print_info "2. Test the installation:"
    print_color $YELLOW "   face-swap --help"
    echo
    
    print_info "3. Basic usage example:"
    print_color $YELLOW "   face-swap --src source.jpg --ref reference.jpg --tgt target.mp4 --model models/inswapper_128.onnx"
    echo
    
    if [ "$DOWNLOAD_MODELS" = false ]; then
        print_warning "Models not downloaded. You'll need to download the inswapper_128.onnx model manually."
        print_info "See README.md for model download instructions."
        echo
    fi
    
    print_info "4. For more information:"
    print_color $YELLOW "   cat $INSTALL_DIR/README.md"
    print_color $YELLOW "   cat $INSTALL_DIR/INSTALLATION.md"
    echo
    
    print_color $GREEN "Enjoy using Face Swap Advanced!"
}

# Main installation function
main() {
    print_header "Face Swap Advanced - Git Installation"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Find Python command
    PYTHON_CMD=$(find_python)
    print_info "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"
    
    # Show installation summary
    print_info "Installation Summary:"
    print_info "  Directory: $INSTALL_DIR"
    print_info "  Python: $PYTHON_CMD"
    print_info "  Virtual Environment: $CREATE_VENV"
    print_info "  GPU Support: $INSTALL_GPU"
    print_info "  Development Mode: $INSTALL_DEV"
    print_info "  Download Models: $DOWNLOAD_MODELS"
    echo
    
    # Ask for confirmation
    if [ "$FORCE_INSTALL" = false ]; then
        read -p "Proceed with installation? (Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            print_info "Installation cancelled"
            exit 0
        fi
    fi
    
    # Run installation steps
    check_dependencies
    clone_repository
    setup_virtual_environment
    install_python_dependencies
    download_models
    create_activation_script
    run_tests
    show_post_install_instructions
}

# Run main function with all arguments
main "$@"