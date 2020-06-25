from keras import Model, Input
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, Flatten, Average, GlobalAveragePooling1D
from keras import regularizers
import numpy as np
from keras.optimizers import Adam


def ensemble_cnn(vocabulary_size: int, embedding_size: int, max_seq_length: int, embedding_matrix: np.array, y_dictionary: dict) -> Model:
    input = Input(shape=(max_seq_length,))
    embedding = Embedding(input_dim=vocabulary_size,
                        output_dim=embedding_size,
                        weights=[embedding_matrix],
                        input_length=max_seq_length,
                        trainable=True,
                        embeddings_regularizer=regularizers.l2(0.00001))(input)

    # Convolutions
    models = []
    for i in range(3, 7):
        model = Dropout(0.4)(embedding)
        model = Conv1D(filters=512, kernel_size=i, activation='relu', padding="same", kernel_regularizer=regularizers.l2(0.00001))(model)
        model = MaxPooling1D(pool_size=5)(model)
        model = Flatten()(model)
        models.append(model)

    # Dense layers
    model = GlobalAveragePooling1D()(embedding)
    for i in range(5):
        model = Dense(4096, activation='tanh', kernel_regularizer=regularizers.l2(0.00001))(model)
    models.append(model)

    model = Average()(models)
    model = Dropout(0.4)(model)
    model = Dense(len(y_dictionary), activation='softmax')(model)
    model = Model(input, model)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model