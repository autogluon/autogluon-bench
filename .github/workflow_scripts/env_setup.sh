function setup_build_env {
    python3 -m pip install --upgrade pip
    python3 -m pip install "black>=22.3,<23.0"
    python3 -m pip install isort>=5.10
}
