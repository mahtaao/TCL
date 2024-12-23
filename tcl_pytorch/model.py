import torch
import torch.nn as nn
import torch.nn.functional as F

class TCL(nn.Module):
    def __init__(self, input_size, list_hidden_nodes, num_class, wd=1e-4, maxout_k=2, MLP_trainable=True, feature_nonlinearity='abs'):
        super(TCL, self).__init__()

        self.num_layer = len(list_hidden_nodes)
        self.maxout_k = maxout_k
        self.MLP_trainable = MLP_trainable
        self.feature_nonlinearity = feature_nonlinearity

        self.hidden_layers = nn.ModuleList()
        for ln in range(self.num_layer):
            in_dim = list_hidden_nodes[ln-1] if ln > 0 else input_size
            out_dim = list_hidden_nodes[ln]

            if ln < self.num_layer - 1:
                out_dim = maxout_k * out_dim

            # Inner product
            layer = nn.Linear(in_dim, out_dim, bias=True)
            setattr(self, f'layer{ln+1}', layer)
            self.hidden_layers.append(layer)
        self.layerMLR = nn.Linear(list_hidden_nodes[-1], num_class, bias=True)

    def forward(self, x):
        with torch.set_grad_enabled(self.MLP_trainable):
            for ln in range(self.num_layer):
                x = self.hidden_layers[ln](x)

                if ln < self.num_layer - 1:
                    x = self.maxout(x, self.maxout_k)
                else:  # The last layer (feature value)
                    if self.feature_nonlinearity == 'abs':
                        x = torch.abs(x)
                    else:
                        raise ValueError

        feats = x

        # MLR layer
        logits = self.layerMLR(feats)

        return logits, feats

    def maxout(self, y, k):
        input_shape = y.size()
        ndim = len(input_shape)
        ch = input_shape[-1]
        assert ndim == 4 or ndim == 2
        assert ch is not None and ch % k == 0

        if ndim == 4:
            y = y.view(-1, input_shape[1], input_shape[2], ch // k, k)
        else:
            y = y.view(-1, int(ch // k), k)

        y, _ = torch.max(y, dim=ndim)
        return y

class TCL_new(nn.Module):
    def __init__(self, input_size, list_hidden_nodes, num_class, wd=1e-4, maxout_k=2, MLP_trainable=True, feature_nonlinearity='abs'):
        super(TCL_new, self).__init__()

        self.num_layer = len(list_hidden_nodes)
        self.maxout_k = maxout_k
        self.MLP_trainable = MLP_trainable
        self.feature_nonlinearity = feature_nonlinearity

        self.layer1 = nn.Linear(input_size, 40, bias=True)
        # self.layer2 = nn.Linear(40, 40, bias=True)
        # self.layer3 = nn.Linear(40, 40, bias=True)
        # self.layer4 = nn.Linear(40, 40, bias=True)
        self.layer5= nn.Linear(40, 20, bias=True)


        # self.hidden_layers = nn.ModuleList()
        # for ln in range(self.num_layer):
        #     in_dim = list_hidden_nodes[ln-1] if ln > 0 else input_size
        #     out_dim = list_hidden_nodes[ln]

        #     # if ln < self.num_layer - 1:
        #     #     out_dim = maxout_k * out_dim

        #     # Inner product
        #     layer = nn.Linear(in_dim, out_dim, bias=True)
        #     setattr(self, f'layer{ln+1}', layer)
        #     self.hidden_layers.append(layer)
        self.layerMLR = nn.Linear(list_hidden_nodes[-1], num_class, bias=True)

    def forward(self, x):
        # with torch.set_grad_enabled(self.MLP_trainable):
        #     for ln in range(self.num_layer):
                # x = torch.relu(self.hidden_layers[ln](x))
                # x = torch.relu(self.layer1(x))
                # x = torch.relu(self.layer2(x))
                # x = torch.relu(self.layer3(x))
                # x = torch.relu(self.layer4(x))
                # x = torch.relu(self.layer5(x))

                # if ln < self.num_layer - 1:
                #     x = self.maxout(x, self.maxout_k)
                # else:  # The last layer (feature value)
                #     if self.feature_nonlinearity == 'abs':
                #         x = torch.abs(x)
                #     else:
                #         raise ValueError

        x = torch.relu(self.layer1(x))
        # x = torch.relu(self.layer2(x))
        # x = torch.relu(self.layer3(x))
        # x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        feats = x

        # MLR layer
        logits = self.layerMLR(feats)

        return logits, feats