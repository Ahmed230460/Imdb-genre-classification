#!/bin/bash
apt-get update && apt-get install -y unzip python3-distutils python3-dev build-essential
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
unzip -o rf_model.zip  # Force overwrite if exists
