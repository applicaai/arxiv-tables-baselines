
python ./benchmarker/cli/l5/create_memmaps.py \
    --dataset_path_or_name $DUE_DATASET_PATH \
    --model_path $T5_MODEL_PATH \
    --memmap_path $BINARIZATION_OUT \
    --max_encoder_length 2048 \
    --segment_levels "(tokens,pages)" \
    --processes 20 \
    --ocr_engine microsoft_cv \
    --train_strategy concat \
    --dev_strategy concat \
    --test_strategy concat \
    --use_fast_tokenizer
