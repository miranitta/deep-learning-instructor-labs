#!/usr/bin/env bash
set -e

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name deep-learning-instructor-labs --display-name "Python (DL Instructor Labs)"

echo "Setup complete. Activate with: source .venv/bin/activate"
