from torch import nn

class LastMLP(nn.Module):

    """
    A PyTorch implementation for a multi-layer perceptron (MLP) that 
    also provides a means to generate last linear layer output for use 
    in embeddings.
    """

    def __init__(self, full_mlp_op_list):

        """
        Constructs the MLP by applying the operations given sequentially.
        """

        # We assume first that the MLP's last operation is the last linear layer.
        lin_layer_output = len(full_mlp_op_list) - 1

        # However, we must check if the output is being squashed by nn.Tanh()
        if isinstance(full_mlp_op_list[lin_layer_output], nn.Tanh):

            # By Stable-baslines3's definition of create_mlp, the linear layer must be 
            # the next to last.
            lin_layer_output -= 1

        # Break up the full operation list into two parts
        list_of_ops_up_to_lin_layer = full_mlp_op_list[:lin_layer_output]
        list_of_ops_after_lin_layer = full_mlp_op_list[lin_layer_output:]

        # Form two nn.Sequential modules to use in forward passes
        self.ops_up_to_lin_layer = nn.Sequential(*list_of_ops_up_to_lin_layer)
        self.ops_after_lin_layer = nn.Sequential(*list_of_ops_after_lin_layer)

    def forward(self, module_input, last=False):

        # Apply a normal forward pass by sequentially doing the operations of the MLP.
        output_up_to_lin_layer = self.ops_up_to_lin_layer(module_input)
        output_of_module = self.ops_after_lin_layer(output_up_to_lin_layer)

        # If the forward pass should also give last linear layer embeddings, return them 
        # as well. Otherwise, just return the output of the MLP.
        if last:
            return output_of_module, output_up_to_lin_layer
        else:
            return output_of_module