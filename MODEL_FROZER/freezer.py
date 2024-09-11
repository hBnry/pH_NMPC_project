# to create frozen model
# from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def model_frost(model, show_graph, file_path, file_name):
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    # # inspect the layers operations inside your frozen graph definition and see the name of its input and output
    # tensors layers = [op.name for op in frozen_func.graph.get_operations()]

    # Save frozen graph from frozen ConcreteFunction to hard drive
    # serialize the frozen graph and its text representation to disk.
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=file_path,
                      name=file_name,
                      as_text=False,)


def model_defrost(file_path, show_graph):
    # Frozen model is loaded as graph def
    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile(file_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph

        layers = [op.name for op in import_graph.get_operations()]
        if print_graph:
            for layer in layers:
                print(layer)

        return wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, inputs),
            tf.nest.map_structure(import_graph.as_graph_element, outputs))

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["x:0"],
                                    outputs=["Identity:0"],
                                    print_graph=show_graph)

    return frozen_func


# if __name__ == '__main__':
#     d = model_defrost(file_path="C:\\Users\\Henry\\PycharmProjects\\MPC "
#                                 "project\\MODEL_FROZER\\frozen_models\\lstm_frozen_graph.pb",
#                       show_graph=True)
