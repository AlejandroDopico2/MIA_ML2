from unit5 import *


NUM_CLIENTS = 100
NUM_ROUNDS = 10
MODEL_IDEN = 1
FRACTION_FIT = 1

   
def generate_ann(iden: int = MODEL_IDEN, lr: float = 1e-4):
    if iden == 0:
        model = tf.keras.Sequential([
            Flatten(input_shape=(32, 32, 3)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])
    elif iden == 1:
        model = tf.keras.Sequential([
            Conv2D(32, 7, activation='relu', input_shape=(32, 32, 3)),
            Conv2D(64, 5, activation='relu'),
            BatchNormalization(1),
            Conv2D(128, 5, activation='relu'),
            Conv2D(128, 5, activation='relu'),
            Dropout(0.2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(10, activation='softmax')
        ])
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=['accuracy']
    )
    return model
    

def client_fn(cid) -> FlowerClient:
    net = generate_ann()
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)


def fit_config(server_round: int):
    return dict(server_round=server_round, epochs=10, batch_size=100)

if __name__ == '__main__':
    model = generate_ann()
    params = get_parameters(model)
    del model 

    trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)
    
    strategy = AggregateCustomMetricStrategy(
        fraction_fit=FRACTION_FIT, 
        fraction_evaluate=0.5, 
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=fl.common.ndarrays_to_parameters(params),
        on_fit_config_fn=fit_config,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS), 
        strategy=strategy,
    )
