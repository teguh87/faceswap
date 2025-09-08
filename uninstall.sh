#!/bin/bash

# Face Swap Advanced - Uninstallation Script
# This script removes the Face Swap Advanced installation

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
DEFAULT_INSTALL_DIR="$HOME/face-swap-advanced"

# Global variables
INSTALL_DIR=""
REMOVE_MODELS=false
REMOVE_CACHE=true
FORCE_REMOVE=false
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
Face Swap Advanced - Uninstallation Script

Usage: $0 [OPTIONS]

Options:
    -d, --dir DIR           Installation directory (default: $DEFAULT_INSTALL_DIR)
    --keep-models           Keep downloaded models
    --no-cache              Don't remove pip cache
    --force                 Force removal without confirmation
    -v, --verbose           Verbose output
    -h, --help              Show this help message

Examples:
    $0                      # Standard uninstallation
    $0 --keep-models        # Keep models for future use
    $0 --force              # No confirmation prompts
    $0 --dir /opt/face-swap # Custom installation directory

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
            --keep-models)
                REMOVE_MODELS=false
                shift
                ;;
            --no-cache)
                REMOVE_CACHE=false
                shift
                ;;
            --force)
                FORCE_REMOVE=true
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get installation size
get_installation_size() {
    if [ -d "$INSTALL_DIR" ]; then
        if command_exists du; then
            du -sh "$INSTALL_DIR" 2>/dev/null | cut -f1 || echo "Unknown"
        else
            echo "Unknown"
        fi
    else
        echo "0"
    fi
}

# Function to find and deactivate virtual environment
deactivate_venv() {
    # Check if we're in a virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        # Check if it's our virtual environment
        if [[ "$VIRTUAL_ENV" == *"$INSTALL_DIR"* ]]; then
            print_info "Deactivating virtual environment..."
            if command_exists deactivate; then
                deactivate 2>/dev/null || true
            fi
            unset VIRTUAL_ENV
        fi
    fi
}

# Function to uninstall pip packages
uninstall_pip_packages() {
    print_info "Uninstalling Face Swap Advanced from pip..."
    
    # Try to find Python in the virtual environment first
    local python_cmd=""
    if [ -f "$INSTALL_DIR/venv/bin/python" ]; then
        python_cmd="$INSTALL_DIR/venv/bin/python"
    elif [ -f "$INSTALL_DIR/venv/Scripts/python.exe" ]; then
        python_cmd="$INSTALL_DIR/venv/Scripts/python.exe"
    else
        # Fallback to system Python
        for cmd in python3 python; do
            if command_exists "$cmd"; then
                python_cmd="$cmd"
                break
            fi
        done
    fi
    
    if [ -n "$python_cmd" ]; then
        # Uninstall face-swap-advanced package
        $python_cmd -m pip uninstall face-swap-advanced -y 2>/dev/null || true
        
        # Also try to uninstall if installed in development mode
        if [ -f "$INSTALL_DIR/setup.py" ] || [ -f "$INSTALL_DIR/pyproject.toml" ]; then
            cd "$INSTALL_DIR"
            $python_cmd -m pip uninstall . -y 2>/dev/null || true
        fi
        
        print_success "Pip packages uninstalled"
    else
        print_warning "Could not find Python to uninstall pip packages"
    fi
}

# Function to remove CLI commands
remove_cli_commands() {
    print_info "Removing CLI commands..."
    
    local commands=("face-swap" "faceswap")
    local removed_count=0
    
    for cmd in "${commands[@]}"; do
        if command_exists "$cmd"; then
            local cmd_path=$(which "$cmd" 2>/dev/null)
            if [ -n "$cmd_path" ]; then
                # Check if it's in a virtual environment path
                if [[ "$cmd_path" == *"$INSTALL_DIR"* ]]; then
                    rm -f "$cmd_path" 2>/dev/null || true
                    ((removed_count++))
                    if [ "$VERBOSE" = true ]; then
                        print_info "Removed: $cmd_path"
                    fi
                fi
            fi
        fi
    done
    
    if [ $removed_count -gt 0 ]; then
        print_success "Removed $removed_count CLI command(s)"
    else
        print_info "No CLI commands found to remove"
    fi
}

# Function to clean pip cache
clean_pip_cache() {
    if [ "$REMOVE_CACHE" = false ]; then
        return 0
    fi
    
    print_info "Cleaning pip cache..."
    
    # Find Python command
    local python_cmd=""
    for cmd in python3 python; do
        if command_exists "$cmd"; then
            python_cmd="$cmd"
            break
        fi
    done
    
    if [ -n "$python_cmd" ]; then
        $python_cmd -m pip cache purge 2>/dev/null || true
        print_success "Pip cache cleaned"
    else
        print_warning "Could not find Python to clean pip cache"
    fi
}

# Function to backup models
backup_models() {
    if [ "$REMOVE_MODELS" = true ] || [ ! -d "$INSTALL_DIR/models" ]; then
        return 0
    fi
    
    local backup_dir="$HOME/face-swap-models-backup-$(date +%Y%m%d_%H%M%S)"
    
    print_info "Backing up models to: $backup_dir"
    mkdir -p "$backup_dir"
    
    if cp -r "$INSTALL_DIR/models/"* "$backup_dir/" 2>/dev/null; then
        print_success "Models backed up to: $backup_dir"
        echo "  You can use these models in future installations"
    else
        print_warning "Failed to backup models"
        rmdir "$backup_dir" 2>/dev/null || true
    fi
}

# Function to remove installation directory
remove_installation() {
    if [ ! -d "$INSTALL_DIR" ]; then
        print_warning "Installation directory not found: $INSTALL_DIR"
        return 0
    fi
    
    print_info "Removing installation directory: $INSTALL_DIR"
    
    # Backup models if requested
    backup_models
    
    # Remove the directory
    if [ "$VERBOSE" = true ]; then
        rm -rf "$INSTALL_DIR"
    else
        rm -rf "$INSTALL_DIR" 2>/dev/null || true
    fi
    
    if [ ! -d "$INSTALL_DIR" ]; then
        print_success "Installation directory removed"
    else
        print_error "Failed to remove installation directory"
        print_info "You may need to remove it manually with: rm -rf \"$INSTALL_DIR\""
    fi
}

# Function to clean up shell configurations
cleanup_shell_configs() {
    print_info "Checking shell configurations for references..."
    
    local shell_configs=(
        "$HOME/.bashrc"
        "$HOME/.bash_profile" 
        "$HOME/.zshrc"
        "$HOME/.profile"
    )
    
    local cleaned_count=0
    
    for config_file in "${shell_configs[@]}"; do
        if [ -f "$config_file" ]; then
            # Look for references to the installation directory
            if grep -q "$INSTALL_DIR" "$config_file" 2>/dev/null; then
                print_warning "Found references to Face Swap Advanced in: $config_file"
                print_info "Please manually remove any PATH modifications or aliases"
                
                if [ "$VERBOSE" = true ]; then
                    echo "  Matching lines:"
                    grep -n "$INSTALL_DIR" "$config_file" 2>/dev/null | head -5 || true
                fi
                ((cleaned_count++))
            fi
        fi
    done
    
    if [ $cleaned_count -eq 0 ]; then
        print_success "No shell configuration cleanup needed"
    fi
}

# Function to show uninstallation summary
show_uninstall_summary() {
    print_info "Uninstallation Summary:"
    print_info "  Directory: $INSTALL_DIR"
    print_info "  Remove Models: $REMOVE_MODELS"
    print_info "  Clean Cache: $REMOVE_CACHE"
    
    local size=$(get_installation_size)
    print_info "  Installation Size: $size"
    echo
}

# Function to confirm uninstallation
confirm_uninstallation() {
    if [ "$FORCE_REMOVE" = true ]; then
        return 0
    fi
    
    show_uninstall_summary
    
    print_warning "This will permanently remove Face Swap Advanced and all its data."
    
    if [ "$REMOVE_MODELS" = false ]; then
        print_info "Models will be backed up before removal."
    else
        print_warning "Models will also be deleted!"
    fi
    
    echo
    read -p "Are you sure you want to uninstall Face Swap Advanced? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Uninstallation cancelled"
        exit 0
    fi
}

# Function to perform complete uninstallation
perform_uninstallation() {
    print_header "Starting Uninstallation"
    
    # Step 1: Deactivate virtual environment
    deactivate_venv
    
    # Step 2: Uninstall pip packages
    uninstall_pip_packages
    
    # Step 3: Remove CLI commands
    remove_cli_commands
    
    # Step 4: Remove installation directory
    remove_installation
    
    # Step 5: Clean pip cache
    clean_pip_cache
    
    # Step 6: Check shell configurations
    cleanup_shell_configs
    
    print_header "Uninstallation Complete"
}

# Function to show post-uninstall information
show_post_uninstall_info() {
    print_success "Face Swap Advanced has been successfully uninstalled!"
    echo
    
    print_info "What was removed:"
    print_info "  ✓ Installation directory: $INSTALL_DIR"
    print_info "  ✓ Python packages"
    print_info "  ✓ CLI commands"
    
    if [ "$REMOVE_CACHE" = true ]; then
        print_info "  ✓ Pip cache"
    fi
    
    echo
    print_info "What remains:"
    print_info "  • System dependencies (Python, Git, etc.)"
    print_info "  • Global pip packages (if any)"
    
    if [ "$REMOVE_MODELS" = false ]; then
        print_info "  • Model backups (check your home directory)"
    fi
    
    echo
    print_color $GREEN "Thank you for using Face Swap Advanced!"
    print_info "If you decide to reinstall, you can use the install.sh script again."
}

# Main uninstallation function
main() {
    print_header "Face Swap Advanced - Uninstallation"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Check if installation exists
    if [ ! -d "$INSTALL_DIR" ]; then
        print_warning "Installation directory not found: $INSTALL_DIR"
        print_info "Face Swap Advanced may already be uninstalled or installed elsewhere."
        
        # Ask if user wants to specify different directory
        if [ "$FORCE_REMOVE" = false ]; then
            echo
            read -p "Do you want to specify a different installation directory? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                read -p "Enter installation directory: " user_dir
                if [ -d "$user_dir" ]; then
                    INSTALL_DIR="$user_dir"
                else
                    print_error "Directory not found: $user_dir"
                    exit 1
                fi
            else
                print_info "Exiting uninstaller"
                exit 0
            fi
        else
            exit 0
        fi
    fi
    
    # Confirm uninstallation
    confirm_uninstallation
    
    # Perform uninstallation
    perform_uninstallation
    
    # Show post-uninstall information
    show_post_uninstall_info
}

# Run main function with all arguments
main "$@"