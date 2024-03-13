import flwr as fl
import tensorflow as tf
import numpy as np

NUM_CLIENTS = 5

# Define a simple model using TensorFlow
def generate_ann():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def split_index(a, n):
    s = np.array_split(np.arange(len(a)), n)
    return s


# Code to load the dataset
def load_datasets(num_clients: int):
    # Distribute it to train and test set
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Normalize data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train, y_train = x_train[:10_000], y_train[:10_000]
    x_test, y_test = x_test[:1000], y_test[:1000]

    # Randomize the datasets
    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    x_test, y_test = unison_shuffled_copies(x_test, y_test)

    # Split training set into NUM_CLIENTS partitions to simulate the individual dataset
    train_index = split_index(x_train, num_clients)
    test_index = split_index(x_test, num_clients)

    # Split each partition
    train_ds = []
    val_ds = []
    test_ds = []
    for cid in range(num_clients):
        val_size = len(train_index[cid]) // 10
        train_input_data, train_output_data = x_train[train_index[cid]], y_train[train_index[cid]]
        val_input_data, val_output_data = train_input_data[:val_size], train_output_data[:val_size]
        train_input_data, train_output_data = train_input_data[val_size:], train_output_data[val_size:]
        train_dataset = (train_input_data, train_output_data)
        val_dataset = (val_input_data, val_output_data)
        test_dataset = (x_test[test_index[cid]], y_test[test_index[cid]])
        train_ds.append(train_dataset)
        val_ds.append(val_dataset)
        test_ds.append(test_dataset)
    return train_ds, val_ds, test_ds


trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)

#Create a class to contain the details of the client and be the interface
class MyClient(fl.client.NumPyClient):
    def __init__(self, net, train_dataset, test_dataset):
        self.model = net
        self.trainloader = train_dataset
        self.valloader = test_dataset
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.trainloader[0],self.trainloader[1], epochs=10, batch_size=64)
        return self.model.get_weights(), len(self.trainloader[0]), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.valloader[0], self.valloader[1])
        return loss, len(self.valloader[0]), {"accuracy": float(accuracy)}
