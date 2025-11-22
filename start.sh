#!/bin/bash

source venv/bin/activate

if [[ "$1" == "--notrain" ]]; then
  echo "Saltandose Entrenamiento..."
else
  python train_hand_model.py
fi

python realtime_hand_counter.py
