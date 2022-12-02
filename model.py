import numpy as np
import os

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


from sklearn.metrics import multilabel_confusion_matrix, accuracy_score




def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


actions = np.array(['1', '2','3','4','5'])
label_map = {label:num for num, label in enumerate(actions)}


sequences, labels = [], []



# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Database') 

# Actions that we try to detect
actions = np.array(['1', '2','3','4','5'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Thirty videos worth of data
no_sequences = 30



for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

np.array(sequences).shape

print("Companing")

X = np.array(sequences)
X.shape
print(X.shape)
y = to_categorical(labels).astype(int)



print("Data Capture Successful")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
X_train.shape
X_test.shape
y_test.shape


print("Traning Model")
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


res = [.7, 0.2, 0.1]
actions[np.argmax(res)]



model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=20, callbacks=[tb_callback])



print(model.summary())


# end

res = model.predict(X_test)
actions[np.argmax(res[2])]
actions[np.argmax(y_test[2])]
actions[np.argmax(res[1])]
actions[np.argmax(y_test[1])]

print("Saving the model")

model.save('action.h5')
# yhat = model.predict(X_test)
# ytrue = np.argmax(y_test, axis=1).tolist()
# yhat = np.argmax(yhat, axis=1).tolist()
# multilabel_confusion_matrix(ytrue, yhat)

# colors = [(245,117,16), (117,245,16), (16,117,245)]








