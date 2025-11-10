# BeMeLy
Using foundation models and Multiple Instance Learning to classify non-cancerous, metastatic and lymphomatous H&amp;E lymph nodes

## 1. Installation
This repository is intented to work with in combination with the TRIDENT repository https://github.com/mahmoodlab/TRIDENT. It is usable with the same environment. For installation:
- Create an environment: conda create -n "trident" python=3.10, and activate it conda activate trident.
- Cloning: git clone https://github.com/mahmoodlab/trident.git && cd trident.
- Local installation: pip install -e .

Then clone the present repo with:
- cd ..
- git clone https://github.com/streilla/BeMeLy.git

## 2. Extract patch features with TRIDENT
See https://github.com/mahmoodlab/TRIDENT to extract patch features with UNI-2h and Virchow2 (or any other available foundation model)

## 3. Train models
To train UNI-2h and Virchow2 models, ensure you have the correct credentials (follow error messages for instructions) and then run the following piece of code:
```
python train_abmil_models.py --data_root_dir /path/to/mag_dir --results_dir ./eval_metrics --model_dir ./trained_models --splits_dir ./splits --df_path ./dummy.csv
```

