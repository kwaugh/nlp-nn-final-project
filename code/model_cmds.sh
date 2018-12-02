python main.py --batch_size 40 --num_workers 0 --epochs 1000 --train_path_input data/english_parse_train.txt --train_path_output data/french_parse_train.txt --dev_path_input data/english_parse_dev.txt --dev_path_output data/french_parse_dev.txt --decoder_len_limit 206
    # english_parse->french_parse

# python main.py --batch_size 100 --num_workers 3 --epochs 1000
    # english->french
