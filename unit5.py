import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from typing import List, Dict, Optional, Tuple, Union
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
import flwr as fl
import numpy as np
from flwr.common import EvaluateRes, FitRes, Scalar
from flwr.common import Metrics
from flwr.server.client_proxy import ClientProxy
global trainloaders, valloaders, testloader, model 


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def split_index(a, n):
    s = np.array_split(np.arange(len(a)), n)
    return s

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

    # Split training set into 'num_clients' partitions to simulate the individual dataset
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

def get_parameters(net) -> List[np.array]:
    return net.get_weights()

def set_parameters(net, parameters: List[np.ndarray]):
    net.set_weights(parameters)
    return net

def train(net, trainloader, batch_size: int, epochs: int):
    net.fit(trainloader[0], trainloader[1],
            epochs=epochs, batch_size=batch_size, steps_per_epoch=1)
    return net

def test(net, testloader):
    loss, accuracy = net.evaluate(testloader[0], testloader[1])
    return loss, accuracy

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        self.net = set_parameters(self.net, parameters)
        self.net = train(self.net, self.trainloader, batch_size=config['batch_size'], epochs=config['epochs'])
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        self.net = set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        print(f"[Client {self.cid}] loss:{loss}, Client {self.cid} accuracy:{accuracy}")
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        return aggregated_loss, {"accuracy": aggregated_accuracy}
    
    
    
def evaluate_fn(
        server_round: int, 
        parameters: fl.common.NDArrays, 
        config: Dict[str, fl.common.Scalar]
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    dataset = testloader[0]
    set_parameters(model, parameters)  # Update model with the latest parameters
    loss, accuracy = test(model, dataset)
    print(f"Server-side evaluation round {server_round} with loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


