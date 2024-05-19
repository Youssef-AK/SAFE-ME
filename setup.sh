#!/bin/bash

# Install tesseract-ocr and its required libraries
apt-get update
apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev

# Install pip packages
pip install pytesseract
