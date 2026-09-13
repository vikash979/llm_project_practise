#!/bin/bash

export OPENAI_API_KEY=""

source ../env/bin/activate
echo $OPENAI_API_KEY

python main.py
