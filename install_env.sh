#!/bin/bash -e

set -o errexit
set -o nounset
set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

CONDA_ENV=${1:-"vomp"}
CUDA_VERSION=${CUDA_VERSION:-"11.8.0"}

export CLI_COLOR=1
RED='\033[1;31m'
GREEN='\033[1;32m'
NOCOLOR='\033[0m'

USAGE="${GREEN}Usage: bash $0 [CONDA_ENV] [AUTO_CONFIRM]${NOCOLOR}
\n\n
Arguments:
  CONDA_ENV     : Name of conda environment (default: vomp)
  AUTO_CONFIRM  : YES to skip all confirmation prompts, NO to ask for each (default: NO)
\n\n
Examples:
  bash $0                          # Use default env name, ask for confirmations
  bash $0 my_env                   # Use 'my_env' as environment name, ask for confirmations  
  bash $0 vomp YES       # Use default env name, auto-confirm all prompts
\n\n
"

if [ $# -gt "2" ]; then
    echo -e "${RED}Error:${NOCOLOR} Wrong argument number"
    echo
    echo -e "$USAGE"
    exit 1
fi

case "$CUDA_VERSION" in
  11.8|11.8.*)
    TORCH_CUDA_VERSION="11.8"
    TORCH_CUDA_TAG="cu118"
    FLASH_CUDA_TAG="cu118"
    SPCONV_PACKAGE="spconv-cu118"
    ;;
  12.1|12.1.*)
    TORCH_CUDA_VERSION="12.1"
    TORCH_CUDA_TAG="cu121"
    # FlashAttention uses its CUDA 12.2 build for CUDA 12.x.
    FLASH_CUDA_TAG="cu122"
    SPCONV_PACKAGE="spconv-cu120"
    ;;
  *)
    echo "This PyTorch 2.4.0 setup supports CUDA 11.8 or 12.1. Set CUDA_VERSION accordingly."
    exit 1
    ;;
esac
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

AUTOCONFIRM="NO"
if [ $# -ge 2 ]; then
    if [ "$2" == "YES" ]; then
        AUTOCONFIRM="YES"
    fi
fi

if [ "$AUTOCONFIRM" == "YES" ]; then
  echo -e "Auto-confirm: ${RED}ENABLED${NOCOLOR} - script will not ask for confirmation"
  echo ""
else
  echo -e "Auto-confirm: ${GREEN}DISABLED${NOCOLOR} - script will ask for confirmation"
  echo ""
fi

ask_yes_no() {
    local prompt="$1"
    local answer=""

    if [ "$AUTOCONFIRM" == "YES" ]; then
      echo ""
      echo -e "${GREEN}$prompt${NOCOLOR} (Y/N): ${GREEN}Y (auto-confirmed)${NOCOLOR}"
      return 0
    else
      echo ""
      while true; do
          echo -ne "${GREEN}$prompt${NOCOLOR} (Y/N): "
          read answer
          case $answer in
              [Yy]* ) echo -e "${GREEN}Y${NOCOLOR}"; return 0;;
              [Nn]* ) echo -e "${RED}N${NOCOLOR}"; return 1;;
              * ) echo -e "${RED}Please answer Y or N.${NOCOLOR}";;
          esac
      done
    fi
}

# Create and activate conda environment
eval "$(conda shell.bash hook)"

# Finds the path of the environment if the environment already exists
CONDA_ENV_PATH=$(conda env list | sed -E -n "s/^${CONDA_ENV}[[:space:]]+\*?[[:space:]]*(.*)$/\1/p")
if [ -z "${CONDA_ENV_PATH}" ]; then
  if ask_yes_no "Conda environment '${CONDA_ENV}' not found. Create new environment with Python 3.11?"; then
    echo -e "${GREEN}Creating conda environment '${CONDA_ENV}'${NOCOLOR}"
    conda create --name "${CONDA_ENV}" -y python=3.11 || { echo -e "${RED}Failed to create conda environment${NOCOLOR}"; exit 1; }
  else
    echo -e "${RED}Environment creation cancelled. Exiting.${NOCOLOR}"
    exit 1
  fi
else
  echo -e "${GREEN}NOTE: Conda environment '${CONDA_ENV}' already exists at ${CONDA_ENV_PATH}${NOCOLOR}"
  if ask_yes_no "Ok to install packages in existing conda environment '${CONDA_ENV}'?"; then
    echo -e "${GREEN}Proceeding with installation in existing environment${NOCOLOR}"
  else
    echo -e "${RED}Installation cancelled. Exiting.${NOCOLOR}"
    exit 1
  fi
fi
conda activate "$CONDA_ENV"
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "VoMP requires Python 3.11"'

echo "--------------------------------------------------------------------"
echo -e "${GREEN}Installing System-Wide Dependencies${NOCOLOR}"
echo "--------------------------------------------------------------------"

if ask_yes_no "Ok to install OpenGL (libglib2.0-dev libgl1) required only if you use material visualization GUI?"; then
    echo -e "${GREEN}...Running OpenGL installation${NOCOLOR}"
    sudo apt-get install -y libglib2.0-dev libgl1 || { echo -e "${RED}Failed to install OpenGL dependencies${NOCOLOR}"; exit 1; }
    conda install -c conda-forge mesa-libgl-devel-cos7-x86_64 -y || { echo -e "${RED}Failed to install mesa-libgl${NOCOLOR}"; exit 1; }
else
    echo -e "${RED}...Skipping OpenGL installation${NOCOLOR}"
fi

if ask_yes_no "Ok to install EGL (libegl1) required only if you use material visualization GUI without a display?"; then
    echo -e "${GREEN}...Running EGL installation${NOCOLOR}"
    sudo apt-get install -y libegl1 || { echo -e "${RED}Failed to install EGL${NOCOLOR}"; exit 1; }
else
    echo -e "${RED}...Skipping EGL installation${NOCOLOR}"
fi


if ask_yes_no "Install/reinstall NVIDIA CUDA ${CUDA_VERSION} in conda environment? (Skip if already installed)"; then
  echo -e "${GREEN}...Installing NVIDIA Cuda (in Conda env; keeping systems settings intact)${NOCOLOR}"
  conda install -y cuda="${CUDA_VERSION}" -c nvidia/label/cuda-"${CUDA_VERSION}" || { echo -e "${RED}Failed to install CUDA${NOCOLOR}"; exit 1; }
  conda install -y cmake ninja cuda-toolkit -c nvidia/label/cuda-"${CUDA_VERSION}" || { echo -e "${RED}Failed to install build tools${NOCOLOR}"; exit 1; }
  export CUDA_HOME="$CONDA_PREFIX"
else
  echo -e "${RED}...Skipping CUDA installation${NOCOLOR}"
fi

echo -e "${GREEN}...Setting up CUDA environment${NOCOLOR}"
export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda env config vars set CUDA_HOME="$CUDA_HOME"
conda env config vars set PATH="$PATH"
conda env config vars set LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

echo "--------------------------------------------------------------------"
echo -e "${GREEN}Installing Pip/Conda Packages${NOCOLOR}"
echo "--------------------------------------------------------------------"

echo -e "${GREEN}...Installing utilities${NOCOLOR}"
pip install --upgrade pip wheel setuptools pytest packaging ninja || { echo -e "${RED}Failed to install pip utilities${NOCOLOR}"; exit 1; }

if ask_yes_no "Install/reinstall PyTorch 2.4.0 and related packages? (Skip if already installed with correct version)"; then
  echo -e "${GREEN}...Installing PyTorch${NOCOLOR}"
  pip install torch==2.4.0 torchvision==0.19.0 xformers==0.0.27.post2 --index-url "$TORCH_INDEX_URL" || { echo -e "${RED}Failed to install PyTorch${NOCOLOR}"; exit 1; }
else
  echo -e "${RED}...Skipping PyTorch installation${NOCOLOR}"
fi

python - "$TORCH_CUDA_VERSION" <<'PYTHON'
import sys
import torch

if torch.__version__.split("+")[0] != "2.4.0" or torch.version.cuda != sys.argv[1]:
    raise SystemExit(f"Install PyTorch 2.4.0 with CUDA {sys.argv[1]} before continuing.")
PYTHON

if ask_yes_no "Install/reinstall torch-tensorrt? (Optional; only needed for use_trt=True)"; then
  echo -e "${GREEN}...Installing torch-tensorrt${NOCOLOR}"
  pip install "torch-tensorrt==2.4.0+${TORCH_CUDA_TAG}" --extra-index-url "$TORCH_INDEX_URL" || { echo -e "${RED}Failed to install torch-tensorrt${NOCOLOR}"; exit 1; }
else
  echo -e "${RED}...Skipping torch-tensorrt installation${NOCOLOR}"
fi

if ask_yes_no "Install/reinstall diff-gaussian-rasterization? (Skip if already installed)"; then
  echo -e "${GREEN}...Installing diff-gaussian-rasterization${NOCOLOR}"
  pip install --no-build-isolation git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d || { echo -e "${RED}Failed to install diff-gaussian-rasterization${NOCOLOR}"; exit 1; }
else
  echo -e "${RED}...Skipping diff-gaussian-rasterization installation${NOCOLOR}"
fi


echo -e "${GREEN}...Installing flash_attn${NOCOLOR}"
if ask_yes_no "Install prebuilt flash_attn binary (faster) instead of building from source?"; then
  echo -e "${GREEN}Installing prebuilt flash_attn binary${NOCOLOR}"
  pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+${FLASH_CUDA_TAG}torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" --no-build-isolation || { echo -e "${RED}Failed to install prebuilt flash_attn${NOCOLOR}"; exit 1; }
else
  echo -e "${GREEN}Building flash_attn from source${NOCOLOR}"
  pip install flash_attn==2.5.9.post1 --no-build-isolation || { echo -e "${RED}Failed to build flash_attn from source${NOCOLOR}"; exit 1; }
fi

if ask_yes_no "Install/reinstall spconv? (Skip if already installed with correct CUDA version)"; then
  pip install "${SPCONV_PACKAGE}==2.3.8" || { echo -e "${RED}Failed to install spconv${NOCOLOR}"; exit 1; }
else
  echo -e "${RED}...Skipping spconv installation${NOCOLOR}"
fi

if ask_yes_no "Install/reinstall Kaolin 0.18.0? (Skip if already installed with correct CUDA version)"; then
  echo -e "${GREEN}...Installing Kaolin${NOCOLOR}"
  pip install kaolin==0.18.0 -f "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_${TORCH_CUDA_TAG}.html" || { echo -e "${RED}Failed to install Kaolin${NOCOLOR}"; exit 1; }
else
  echo -e "${RED}...Skipping Kaolin installation${NOCOLOR}"
fi

if ask_yes_no "Install base inference requirements from requirements_inference.txt?"; then
  echo -e "${GREEN}Installing from requirements_inference.txt${NOCOLOR}"
  pip install --no-build-isolation -r requirements_inference.txt || { echo -e "${RED}Failed to install from requirements_inference.txt${NOCOLOR}"; exit 1; }
else
  echo -e "${RED}Skipping base inference requirements${NOCOLOR}"
fi

if ask_yes_no "Install dev requirements from requirements_dev.txt (only required for training or finetuning)?"; then
  echo -e "${GREEN}Installing from requirements_dev.txt${NOCOLOR}"
  pip install --no-build-isolation -r requirements_dev.txt || { echo -e "${RED}Failed to install from requirements_dev.txt${NOCOLOR}"; exit 1; }
else
  echo -e "${RED}Skipping dev requirements${NOCOLOR}"
fi

echo "--------------------------------------------------------------------"
echo -e "${GREEN}Installing VoMP${NOCOLOR}"
echo "--------------------------------------------------------------------"

echo -e "${GREEN}Installing vox2seq extension${NOCOLOR}"
pip install extensions/vox2seq/ --no-build-isolation || { echo -e "${RED}Failed to install vox2seq${NOCOLOR}"; exit 1; }
echo -e "${GREEN}Installing VoMP in editable mode${NOCOLOR}"
pip install -e . || { echo -e "${RED}Failed to install VoMP${NOCOLOR}"; exit 1; }

echo "--------------------------------------------------------------------"
echo -e "${GREEN}Downloading Model Weights${NOCOLOR}"
echo "--------------------------------------------------------------------"

echo -e "If you just want to use the model for inference, follow the instructions in the README.md file to download the weights."

if ask_yes_no "Download initialization weights? (for training from scratch)"; then
  echo -e "${GREEN}Downloading initialization weights${NOCOLOR}"
  chmod +x weights/download.sh
  ./weights/download.sh || { echo -e "${RED}Failed to download initialization weights${NOCOLOR}"; exit 1; }
  echo -e "${GREEN}Initialization weights downloaded successfully${NOCOLOR}"
else
  echo -e "${RED}Skipping initialization weights download${NOCOLOR}"
fi
