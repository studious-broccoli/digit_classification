#!/bin/bash

# Locale fix (prevents fatal Python errors)
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# Ensure Python can find your code
export PYTHONPATH="$(pwd)/src"

echo "✅ Environment configured:"
echo "LC_ALL=$LC_ALL"
echo "LANG=$LANG"
echo "PYTHONPATH=$PYTHONPATH"

