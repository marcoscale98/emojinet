from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, Flatten
from keras import regularizers
import numpy as np
from keras.optimizers import Adam


def base_cnn(vocabulary_size: int, embedding_size: int, max_seq_length: int, embedding_matrix: np.array, y_dictionary: dict) -> Model:
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_size,
                        output_dim=embedding_size,
                        weights=[embedding_matrix],
                        input_length=max_seq_length,
                        trainable=True,
                        embeddings_regularizer=regularizers.l2(0.000001)))
    model.add(Dropout(0.4))
    model.add(Conv1D(filters=512, kernel_size=5, activation='relu', padding="same", kernel_regularizer=regularizers.l2(0.00001)))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(len(y_dictionary), activation='softmax'))

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model