#!/bin/bash

# Face Swap Advanced - Quick Installation Script
# One-liner installation for easy deployment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO] $1${NC}"; }
print_success() { echo -e "${GREEN}[SUCCESS] $1${NC}"; }
print_error() { echo -e "${RED}[ERROR] $1${NC}"; }

# Configuration
REPO_URL="https://raw.githubusercontent.com/teguh87/faceswap/main/install.sh"
TEMP_INSTALLER="/tmp/face_swap_installer.sh"

print_info "Face Swap Advanced - Quick Installer"
print_info "Downloading and running full installer..."

# Download the main installer
if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$REPO_URL" -o "$TEMP_INSTALLER"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$TEMP_INSTALLER" "$REPO_URL"
else
    print_error "Neither curl nor wget found. Please install one of them first."
    exit 1
fi

# Make executable and run
chmod +x "$TEMP_INSTALLER"
"$TEMP_INSTALLER" "$@"

# Cleanup
rm -f "$TEMP_INSTALLER"

print_success "Quick installation completed!"