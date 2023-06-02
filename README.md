# Arxiv Table Baselines

Firstly you need to clone dataset from the [official website](https://gonito.net/challenge/arxiv-tables) following
"Get your repo" from [the instruction](https://gonito.net/challenge-how-to/arxiv-tables).
To run T5 and LayoutLM baselines you need out of OCR engine. We provide it in our [storage](https://figshare.com/account/articles/23284160).
Unpack it to your directory with cloned challenge. Now you can train both baseline models and evaluate results.

### LayoutLMv3
Install requirements:

```
pip install -r LayoutLM/requirements.txt
```

To run LayoutLMv3 baseline you need to set proper paths in lines 17 and 18 of script `LayoutLM/loop.py` and run it:

```
sh LayoutLM/loop.py
```

Output will be stored in `LOG_DIR/'predictions.tsv` (see line 18 of the script).

### T5
We use training pipeline from DUE benchmark. Firstly set up an environment as described
[there](https://github.com/due-benchmark/baselines#install-benchmark-related-repositories).
Now you need to convert the challenge to DUE-compatible format:
```
export DUE_DATASET_PATH=/where/to/store/chalenge/in/due/format
python T5/to_due.sh /path/to/cloned/challenge $DUE_DATASET_PATH
```
Now you need to download Base T5 model from [here](https://duebenchmark.com/data) and decompress it.

The next step is binarization of the dataset:
```
export T5_MODEL_PATH=/path/to/downloaded/model
export BINARIZATION_OUT=/where/to/store/binarized/dataset
sh T5/create_memmaps.sh
```
Now you can finally train the model:
```
export TRAINING_OUT=/where/to/store/model_out
sh T5/train.sh
```
Model output will be stored in `$TRAINING_OUT/test_generations.txt`

### Evaluation
Both outputs from T5 and LayoutLM are converted the same way:
```
python out_to_gonito.sh /path/to/model/out /path/to/cloned/challenge/test-A/in.tsv /path/to/cloned/challenge/test-A/out.tsv
```

When model out is in `test-A/out.tsv`, follow the evaluation instructions from
[the website](https://gonito.net/challenge-how-to/arxiv-tables).
