import requests
import pprint

# from simulations.json_file_hotel import JsonfileHotel as app
# from simulations.json_file_sockshop import JsonfileSockshop as app
# from simulations.json_file_complex import JsonfileComplex as app
from simulations.json_file_test import JsonfileTest as app

input = app.input
def get_input():
    return input

input["cores_matrix"] = [[1] * len(input["node_memories"])] * len(input["function_names"])
input["workload_on_destination_matrix"] = [[1] * len(input["node_memories"])] * len(input["function_names"])

response = requests.request(method='get', url="http://localhost:5003/", json=input)

pprint.pprint(response.json())
