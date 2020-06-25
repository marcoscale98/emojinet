from keras import Model, Input
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, Flatten, Average, \
    GlobalAveragePooling1D, Dot, Lambda
from keras import regularizers
import numpy as np
import keras.backend as K
from keras.optimizers import Adam, RMSprop


def ensemble_cnn_subword(vocabulary_size: int, embedding_size: int, max_char_length: int, max_seq_length: int, embedding_matrix: np.array, y_dictionary: dict) -> Model:
    input = Input(shape=(max_char_length,), name='main_input')
    mask = Input(shape=(max_seq_length, max_char_length,), name='mask_input')

    embedding = Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_size,
        weights=[embedding_matrix],
        input_length=max_char_length,
        trainable=True,
        embeddings_regularizer=regularizers.l2(0.000001))(input)
    print(mask.shape)
    print(embedding.shape)
    embedding = Lambda(lambda values: K.batch_dot(values[0], values[1]))([mask, embedding])
    print(embedding.shape)

    # Convolutions
    models = []
    for i in range(3, 7):
        model = Dropout(0.4)(embedding)
        model = Conv1D(filters=512, kernel_size=i, activation='relu', padding="same",
                       kernel_regularizer=regularizers.l2(0.00001))(model)
        model = MaxPooling1D(pool_size=5)(model)
        model = Flatten()(model)
        models.append(model)

    # Dense layers
    model = GlobalAveragePooling1D()(embedding)
    for i in range(1):
        model = Dense(4096, activation='tanh', kernel_regularizer=regularizers.l2(0.00001))(model)
    models.append(model)

    model = Average()(models)
    model = Dropout(0.4)(model)
    model = Dense(len(y_dictionary), activation='softmax')(model)
    model = Model([input, mask], model)

    optimizer = RMSprop(lr=0.001, decay=0.00005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model