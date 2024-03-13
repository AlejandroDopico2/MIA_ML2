from clientUtils import *

# Start the client
model=generate_ann()

#The following line is deprecated in the current flower version
# fl.client.start_numpy_client(server_address="localhost:8080", client=MyClient(model,trainloaders[0],valloaders[0]))

fl.client.start_client(server_address=f"localhost:8080", client=MyClient(model,trainloaders[0],valloaders[0]).to_client())
