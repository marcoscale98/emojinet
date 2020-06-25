from keras import Model, Input
from keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM, Lambda
from keras import regularizers
import numpy as np
import keras.backend as K
from keras.optimizers import RMSprop


def base_lstm_subword(vocabulary_size: int, embedding_size: int, max_char_length: int, max_seq_length: int, embedding_matrix: np.array, y_dictionary: dict) -> Model:
    input = Input(shape=(max_char_length,), name='main_input')
    mask = Input(shape=(max_seq_length, max_char_length,), name='mask_input')

    embedding = Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_size,
        weights=[embedding_matrix],
        input_length=max_char_length,
        trainable=True,
        embeddings_regularizer=regularizers.l2(0.000001))(input)
    embedding = Dropout(0.4)(embedding)
    model = Lambda(lambda values: K.batch_dot(values[0], values[1]))([mask, embedding])

    #model = Dropout(0.4)(model)
    model = Bidirectional(LSTM(256))(model)
    model = Dense(len(y_dictionary), activation='softmax')(model)
    model = Model([input, mask], model)

    optimizer = RMSprop(lr=0.001, decay=0.00005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model