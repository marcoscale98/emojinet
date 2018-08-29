import os
import argparse
from keras.preprocessing import sequence
from keras.utils import to_categorical

import numpy as np
import logging

from models import get_model
from preprocessing.embeddings import restore_from_file
from preprocessing.reader import SemEvalDatasetReader, EvalitaDatasetReader
from preprocessing.text import Tokenizer
from utils.callbacks import EvalCallback, ValidationEarlyStopping
from utils.fileprovider import FileProvider

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    """##### Parameter parsing"""

    parser = argparse.ArgumentParser(description='Train the emoji task')
    parser.add_argument('--embeddings', default=None,
                        help='The directory with the precomputed embeddings')
    parser.add_argument('--workdir', required=True,
                        help='Work path')
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to be used [0, 1, -1]")
    parser.add_argument('--base-model', required=True,
                        help='Model to be trained', choices=["base_cnn", "base_lstm"])
    parser.add_argument('--evalita', default=False, action='store_true', help='Train and test on EVALITA2018 dataset')
    parser.add_argument('--semeval', default=False, action='store_true', help='Train and test on SemEval2018 dataset')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='The size of a mini-batch')
    parser.add_argument('--embeddings-only', default=False, action='store_true',
                        help='Only use words from the embeddings vocabulary')
    parser.add_argument('--embeddings-size', type=int, default=300,
                        help='Default size of the embeddings if precomputed ones are omitted')
    parser.add_argument('--max-dict', type=int, default=300000,
                        help='Maximum dictionary size')
    parser.add_argument('--max-epoch', type=int, default=20,
                        help='The maximum epoch number')
    parser.add_argument('--max-seq-length', type=int, default=40,
                        help='Maximum sequence length')
    parser.add_argument('--embeddings-skip-first-line', default=False, action='store_true', help='Skip first line of the embeddings')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)

    files = FileProvider(args.workdir)
    logging.info("Starting training with parameters: {0}".format(vars(args)))

    """##### Loading the dataset"""

    if args.semeval:
        raw_train = SemEvalDatasetReader(files.semeval_train)
        raw_test = SemEvalDatasetReader(files.semeval_test)
    else:
        if not path.exists(files.evalita_train):
            raw_train, raw_test = EvalitaDatasetReader(files.evalita).split()
        else:
            raw_train = EvalitaDatasetReader(files.evalita_train)
            raw_test = EvalitaDatasetReader(files.evalita_test)
    raw_train, raw_val = raw_train.split(test_size=0.1)

    tokenizer = Tokenizer(num_words=args.max_dict, lower=True)
    tokenizer.fit_on_texts(raw_train.X)
    vocabulary_size = min(len(tokenizer.word_index), args.max_dict)
    logging.info("Vocabulary size: %d, Total words: %d" % (vocabulary_size, len(tokenizer.word_counts)))

    X_train = tokenizer.texts_to_sequences(raw_train.X)
    Y_train = raw_train.Y
    X_val = tokenizer.texts_to_sequences(raw_val.X)
    Y_val = raw_val.Y
    X_test = tokenizer.texts_to_sequences(raw_test.X)
    Y_test = raw_test.Y
    Y_dictionary = raw_train.Y_dictionary
    Y_class_weights = len(Y_train) / np.power(np.bincount(Y_train), 1.1)
    Y_class_weights *= 1.0 / np.min(Y_class_weights)
    logging.info("Class weights: %s" % str(Y_class_weights))

    del raw_train
    del raw_val
    del raw_test

    logging.info("Padding train and test")

    max_seq_length = 0
    for seq in X_train:
        if len(seq) > max_seq_length:
            max_seq_length = len(seq)
    logging.info("Max sequence length in training set: %d" % max_seq_length)
    max_seq_length = min(max_seq_length, args.max_seq_length)
    X_train = sequence.pad_sequences(X_train, maxlen=max_seq_length)
    X_val = sequence.pad_sequences(X_val, maxlen=max_seq_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_seq_length)

    """##### Initializing embeddings"""
    logging.info("Initializing embeddings")

    embedding_size = args.embeddings_size
    embeddings = None
    if args.embeddings:
        # Init embeddings here
        words = set(tokenizer.word_index.keys())
        embeddings, embedding_size = restore_from_file(args.embeddings, words, lower=True, skip_first_line = args.embeddings_skip_first_line)

    if embeddings is not None and args.embeddings_only:
        resolved = []
        for word in embeddings:
            word_id = tokenizer.resolve_word(word)
            if word_id is not None:
                resolved.append(embeddings[word])
        logging.info("Restored %d embeddings" % len(resolved))
        embedding_matrix = np.vstack(resolved)
        vocabulary_size = len(resolved)
        del resolved
    else:
        # ReLU Xavier initialization
        embedding_matrix = np.random.randn(vocabulary_size, embedding_size).astype(np.float32) # * np.sqrt(2.0/vocabulary_size)

        if embeddings is not None:
            restored = 0
            for word in embeddings:
                word_id = tokenizer.resolve_word(word)
                if word_id is not None:
                    embedding_matrix[word_id] = embeddings[word]
                    restored += 1
            logging.info("Restored %d (%.2f%%) embeddings" % (restored, (float(restored) / vocabulary_size) * 100))
    del embeddings

    # Rescaling embeddings
    means = np.mean(embedding_matrix, axis=0)
    variance = np.sqrt(np.mean((embedding_matrix - means) ** 2, axis=0))
    embedding_matrix = (embedding_matrix - means) / variance

    """##### Model definition"""
    logging.info("Initializing model")

    params = {
        "vocabulary_size": vocabulary_size,
        "embedding_size": embedding_size,
        "max_seq_length": max_seq_length,
        "embedding_matrix": embedding_matrix,
        "y_dictionary": Y_dictionary
    }
    model = get_model(args.base_model).apply(params)

    """##### Load model"""

    # needs also storing&restoring of the current epoch, also not sure Adam weights are preserved
    #if path.exists(files.model):
    #    model.load_weights(files.model)

    """##### Continue with model"""

    print(model.summary())

    Y_train_one_hot = to_categorical(Y_train, num_classes=len(Y_dictionary))

    callbacks = {
        "test": EvalCallback("test", X_test, Y_test),
        "train": EvalCallback("train", X_train, Y_train, period=3),
        "val": EvalCallback("validation", X_val, Y_val)
    }
    callbacks["stop"] = ValidationEarlyStopping(monitor=callbacks["val"])
    model.fit(X_train,
              Y_train_one_hot,
              class_weight=Y_class_weights,
              epochs=args.max_epoch,
              batch_size=args.batch_size,
              shuffle=True,
              callbacks=[callback for callback in callbacks.values()])

    logging.info("Saving model to json")

    model_json = model.to_json()
    with open(files.model_json, "w", encoding="utf-8") as json_file:
        json_file.write(model_json)

    logging.info("Saving model weights")

    model.save_weights(files.model)

    logging.info("Evaluating")

    callbacks["test"].evaluate()
