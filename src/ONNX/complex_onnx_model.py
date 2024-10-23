import onnx
from onnx import helper, TensorProto

# create input, weights and output tensors
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
weight1_tensor = helper.make_tensor_value_info('weight1', TensorProto.FLOAT, [4, 3])  # First FC layer weights
weight2_tensor = helper.make_tensor_value_info('weight2', TensorProto.FLOAT, [3, 2])  # Second FC layer weights
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

# create layers - matmul1, relu1, matmul2, relu2

#matmul1: input * weight1
matmul1_node = helper.make_node(
    'MatMul',      # Operation type
    ['input', 'weight1'],  # Inputs
    ['hidden1'],   # Output of this node
    name='MatMul1'
)

#relu1: ReLU(matmul1)
relu1_node = helper.make_node(
    'Relu',        # Operation type
    ['hidden1'],   # Input
    ['relu1'],     # Output
    name='ReLU1'
)

#matmul2: relu1 * weight2
matmul2_node = helper.make_node(
    'MatMul',      # Operation type
    ['relu1', 'weight2'],  # Inputs
    ['hidden2'],    # Output of this node
    name='MatMul2'
)

#relu2: ReLU(hidden2)
relu2_node = helper.make_node(
    'Relu',        # Operation type
    ['hidden2'],   # Input
    ['output'],    # Final output
    name='ReLU2'
)

#create the computation graph
graph_def = helper.make_graph(
    [matmul1_node, relu1_node, matmul2_node, relu2_node],  # List of nodes in the graph
    'two_matmuls_relu_model',  
    [input_tensor, weight1_tensor, weight2_tensor],  # Graph inputs
    [output_tensor]  
)

# create the model
model_def = helper.make_model(graph_def, producer_name='onnx-complex-example')

#save
onnx.save(model_def, 'complex_model.onnx')

