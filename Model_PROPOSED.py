from keras import Sequential
from keras.applications import EfficientNetB3
import numpy as np
from Evaluation import evaluation
from keras.layers import Dense
from keras.layers import LSTM

def Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target):
    IMG_SIZE = [32, 32, 3]
    Feat1 = np.zeros((Train_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Train_Data.shape[0]):
        Feat1[i, :] = np.resize(Train_Data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    Train_Data = Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat2 = np.zeros((Test_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Test_Data.shape[0]):
        Feat2[i, :] = np.resize(Test_Data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    Test_Data = Feat2.reshape(Feat2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])
    efficient_net = EfficientNetB3(
        weights='imagenet',
        input_shape=(32, 32, 3),
        include_top=False,
        pooling='max'
    )
    model = Sequential()
    model.add(efficient_net)
    model.add(Dense(units=Train_Target.shape[1], activation='relu'))
    model.add(Dense(units=Train_Target.shape[1], activation='relu'))
    model.add(LSTM(10, input_shape=(Train_Data.shape[1], Train_Data.shape[-1])))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(Train_Data, Train_Target, epochs=200, batch_size=32, validation_data=(Test_Data, Test_Target))
    pred = model.predict(Test_Data)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, Test_Target)
    pred = pred.astype('int')
    return Eval, pred

