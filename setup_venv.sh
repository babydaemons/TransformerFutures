#!/bin/bash

# ==========================================
# Micromamba Local Environment Setup Script (Windows Git Bash)
# Target: Driver 515.x+ (CUDA 11.8)
# ==========================================

# 設定 (Windows形式のパスを扱いやすくするため、MSYS2/Git Bashのパス形式を使用)
PROJECT_DIR="$(pwd)"
MAMBA_ROOT="$PROJECT_DIR/.venv"
MAMBA_EXE="$MAMBA_ROOT/micromamba.exe"
ENV_PREFIX="$MAMBA_ROOT/envs/futures-gpu"

# ディレクトリ作成
mkdir -p "$MAMBA_ROOT/envs"

echo ">>> 1. Downloading micromamba for Windows..."
if [ ! -f "$MAMBA_EXE" ] || [ ! -s "$MAMBA_EXE" ]; then
    # Windowsバイナリを直接ダウンロード (exe形式)
    # 安定性を高めるため、GitHubの公式リリースから直接取得します
    curl -Ls https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64 -o "$MAMBA_EXE"
    
    [ -f "$MAMBA_EXE" ] && chmod +x "$MAMBA_EXE"
else
    echo "micromamba already exists."
fi

echo ">>> 2. Creating environment in $ENV_PREFIX ..."
echo "    Target: PyTorch + CUDA 11.8 + Python 3.11"

# 環境変数の設定 (micromamba実行に必要)
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"

# クリーンインストールのために既存環境を削除
if [ -d "$ENV_PREFIX" ]; then
    echo "Removing existing environment..."
    rm -rf "$ENV_PREFIX"
fi

# micromambaが存在することを確認
if [ ! -f "$MAMBA_EXE" ]; then
    echo "Error: micromamba.exe could not be downloaded."
    exit 1
fi

# 環境作成コマンド (Windows環境の依存関係に最適化)
# 注: Windowsでは pytorch-cuda の代わりに直接 cuda-toolkit を指定する場合が多いですが、
# 元の構成を維持しつつ、Windowsで動作するパッケージを指定します。
"$MAMBA_EXE" create -y -p "$ENV_PREFIX" \
    -c pytorch -c nvidia -c conda-forge \
    python=3.11 \
    pandas \
    "polars>=1.6" \
    scikit-learn \
    tqdm \
    joblib \
    pytorch::pytorch \
    pytorch::pytorch-cuda=11.8 \
    "numpy<2.0" \
    ipykernel \
    numba

echo ">>> 3. Creating activation script..."

# 起動用スクリプトの作成 (Windows Git Bash用)
cat <<EOF > "$PROJECT_DIR/activate_venv.sh"
#!/bin/bash
# usage: source ./activate_venv.sh

export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"

# Windows版micromambaのシェル初期化
eval "\$("$MAMBA_EXE" shell hook --shell bash)"

# 環境の有効化
micromamba activate "$ENV_PREFIX"

# WindowsではLD_LIBRARY_PATHの代わりにPATHにbinを追加するのが一般的
export PATH="$ENV_PREFIX/Library/bin:$ENV_PREFIX/Scripts:\$PATH"

echo "----------------------------------------"
echo "Environment activated: $ENV_PREFIX"
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "----------------------------------------"
EOF

chmod +x "$PROJECT_DIR/activate_venv.sh"

echo "=========================================="
echo "Setup Complete!"
echo "Please run: source ./activate_venv.sh"
echo "=========================================="
