import onnx
from onnx import helper, TensorProto

#define inut and output tensors
input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1000])

#create matmul layer
node_def = helper.make_node(
    'MatMul', 
    ['input', 'weight'],  
    ['output'],  
)

# create the graph
graph_def = helper.make_graph(
    [node_def],
    'simple_model',  
    [input],  
    [output],  
)

#create the model 
model_def = helper.make_model(graph_def, producer_name='simple_model_example')

# Save the model
onnx.save(model_def, '../data/simple_model.onnx')