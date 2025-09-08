# Git Installation Guide

This guide covers installation of Face Swap Advanced directly from Git repository using shell scripts.

## Quick Installation (Recommended)

### One-liner Installation

```bash
# Basic installation
curl -fsSL https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/quick-install.sh | bash

# Or with wget
wget -O- https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/quick-install.sh | bash
```

### Quick Installation with Options

```bash
# With GPU support and models
curl -fsSL https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/quick-install.sh | bash -s -- --gpu --models

# Development installation
curl -fsSL https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/quick-install.sh | bash -s -- --dev --models

# Custom directory
curl -fsSL https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/quick-install.sh | bash -s -- --dir /opt/face-swap
```

## Manual Installation

### Step 1: Download the Installation Script

```bash
# Download the installer
wget https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/install.sh
# Or
curl -O https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/install.sh

# Make it executable
chmod +x install.sh
```

### Step 2: Run the Installation

```bash
# Basic installation
./install.sh

# View all options
./install.sh --help
```

## Installation Options

### Basic Options

```bash
# Basic installation (default location: ~/face-swap-advanced)
./install.sh

# Custom installation directory
./install.sh --dir /path/to/install

# Use specific Python version
./install.sh --python python3.9

# Skip virtual environment creation
./install.sh --no-venv
```

### Feature Options

```bash
# Install with GPU support
./install.sh --gpu

# Install development dependencies
./install.sh --dev

# Download pre-trained models
./install.sh --models

# Complete installation (GPU + Dev + Models)
./install.sh --gpu --dev --models
```

### Advanced Options

```bash
# Force installation (overwrite existing)
./install.sh --force

# Skip system dependencies check
./install.sh --skip-deps

# Verbose output
./install.sh --verbose

# Combine options
./install.sh --gpu --models --dir /opt/face-swap --verbose
```

## Installation Examples

### 1. Standard User Installation

```bash
# Download and run installer
curl -fsSL https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/install.sh | bash

# Activate environment
cd ~/face-swap-advanced
source activate_face_swap.sh

# Test installation
face-swap --help
```

### 2. Developer Installation

```bash
# Download installer
wget https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/install.sh
chmod +x install.sh

# Install with development tools
./install.sh --dev --models --verbose

# Activate environment
cd ~/face-swap-advanced
source activate_face_swap.sh

# Run tests
make test
```

### 3. Server Installation

```bash
# Install in system directory (requires sudo for dependencies)
sudo ./install.sh --dir /opt/face-swap-advanced --skip-deps --no-venv

# Create systemd service (example)
sudo systemctl enable face-swap-service
```

### 4. GPU Workstation Installation

```bash
# Full installation with GPU support
./install.sh --gpu --models --dev --dir /workspace/face-swap

# Verify GPU support
cd /workspace/face-swap
source activate_face_swap.sh
python -c "import onnxruntime; print('GPU Available:', 'CUDAExecutionProvider' in onnxruntime.get_available_providers())"
```

## System Requirements

### Minimum Requirements

- **OS**: Linux, macOS, Windows (WSL)
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB+ recommended)
- **Storage**: 2GB free space
- **Internet**: Required for downloading dependencies and models

### Recommended Requirements

- **OS**: Ubuntu 20.04+ / macOS 10.15+ / Windows 10+ with WSL2
- **Python**: 3.9 or higher
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for GPU acceleration)
- **Storage**: 10GB+ free space

## Platform-Specific Instructions

### Ubuntu/Debian

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install basic dependencies (optional - script will handle this)
sudo apt install -y git python3 python3-pip python3-venv curl wget

# Run installer
curl -fsSL https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/install.sh | bash
```

### CentOS/RHEL/Fedora

```bash
# Update system
sudo yum update -y  # or sudo dnf update -y for Fedora

# Install basic dependencies (optional)
sudo yum install -y git python3 python3-pip curl wget

# Run installer
curl -fsSL https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/install.sh | bash
```

### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install basic dependencies (optional)
brew install git python wget

# Run installer
curl -fsSL https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/install.sh | bash
```

### Windows (WSL)

```bash
# In WSL terminal
# Update system
sudo apt update && sudo apt upgrade -y

# Install Windows-specific dependencies
sudo apt install -y build-essential

# Run installer
curl -fsSL https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/install.sh | bash
```

## Post-Installation

### Verification

```bash
# Activate environment (if using virtual environment)
cd ~/face-swap-advanced
source activate_face_swap.sh

# Test CLI
face-swap --help

# Test Python import
python -c "import face_swap_advanced; print('✓ Installation successful')"

# Check GPU support (if installed with --gpu)
python -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('Available providers:', providers)
print('GPU Support:', 'CUDAExecutionProvider' in providers)
"
```

### Basic Usage Test

```bash
# Download test images (replace with actual test data)
mkdir -p test_data
# Add your test images: source.jpg, reference.jpg, target.jpg/mp4

# Download model (if not done during installation)
wget -P models/ https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx

# Test face swap
face-swap \
  --src test_data/source.jpg \
  --ref test_data/reference.jpg \
  --tgt test_data/target.jpg \
  --model models/inswapper_128.onnx \
  --debug
```

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   chmod +x install.sh
   ```

2. **Python Not Found**
   ```bash
   # Install Python first
   sudo apt install python3 python3-pip  # Ubuntu
   brew install python  # macOS
   ```

3. **Git Not Found**
   ```bash
   # Install Git
   sudo apt install git  # Ubuntu
   brew install git  # macOS
   ```

4. **Network Issues**
   ```bash
   # Download manually and run
   wget https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/install.sh
   ./install.sh --verbose
   ```

5. **CUDA Issues**
   ```bash
   # Check CUDA installation
   nvidia-smi
   
   # Reinstall with CPU only
   ./install.sh --no-gpu
   ```

### Getting Help

1. **Check the logs**: Use `--verbose` flag for detailed output
2. **Read error messages**: The script provides detailed error information
3. **Check system requirements**: Ensure your system meets minimum requirements
4. **Manual installation**: Try installing dependencies manually first
5. **Report issues**: Create an issue on GitHub with full error output

## Uninstallation

### Download and Run Uninstaller

```bash
# Quick uninstall
curl -fsSL https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/uninstall.sh | bash

# Or download first
wget https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/uninstall.sh
chmod +x uninstall.sh
./uninstall.sh
```

### Uninstaller Options

```bash
# Keep models for future use
./uninstall.sh --keep-models

# Force removal without confirmation
./uninstall.sh --force

# Custom installation directory
./uninstall.sh --dir /path/to/installation
```

## Advanced Installation

### Custom Configuration

```bash
# Create custom configuration
cat > custom_config.json << EOF
{
  "install_dir": "/opt/face-swap",
  "python_version": "3.9",
  "create_venv": true,
  "install_gpu": true,
  "install_dev": false,
  "download_models": true
}
EOF

# Install with configuration
./install.sh --config custom_config.json
```

### Batch Installation

```bash
#!/bin/bash
# batch_install.sh - Install on multiple machines

machines=("server1" "server2" "server3")

for machine in "${machines[@]}"; do
  echo "Installing on $machine..."
  ssh "$machine" "curl -fsSL https://raw.githubusercontent.com/teguh87/face-swap-advanced/main/install.sh | bash -s -- --gpu --models"
done
```

### Docker Installation

```bash
# Clone repository
git clone https://github.com/teguh87/face-swap-advanced.git
cd face-swap-advanced

# Build Docker image
docker build -t face-swap-advanced .

# Run container
docker run -it --gpus all -v $(pwd)/data:/app/data face-swap-advanced
```

## Updates

### Update Installation

```bash
# Navigate to installation directory
cd ~/face-swap-advanced

# Pull latest changes
git pull origin main

# Update dependencies
source activate_face_swap.sh
pip install -r requirements.txt --upgrade

# Or reinstall completely
./install.sh --force
```

This Git installation method provides a complete, automated setup process with extensive customization options!