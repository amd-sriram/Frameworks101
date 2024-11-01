import torchvision
import torch 
from torch import nn
from torchvision.models.resnet import Bottleneck


class ModelLayers:


    def __init__(self):
        self.counts = {
            "conv2d": 0,
            "linear": 0,
            "batchnorm": 0,
            "relu": 0,
            "maxpool" : 0,
            "avgpool" : 0
                }


    def flatten_model(self, modules):
        """
        This method flattens a hierarchical model into single level array
        """
        #Code from https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model/69544742#69544742
        def flatten_list(_2d_list):
            flat_list = []
            # Iterate through the outer list
            for element in _2d_list:
                if type(element) is list:
                    # If the element is of type list, iterate through the sublist
                    for item in element:
                        flat_list.append(item)
                else:
                    flat_list.append(element)
            return flat_list

        ret = []
        try:
            for _, n in modules:
                ret.append(flatten_model(n))
        except:
            try:
                if str(modules._modules.items()) == "odict_items([])":
                    ret.append(modules)
                else:
                    for _, n in modules._modules.items():
                        ret.append(self.flatten_model(n))
            except:
                ret.append(modules)
        return flatten_list(ret)


    def find_top_layers(self, model, top_count):
        
        #named_layers = dict(model.modules())
        #names = named_layers.keys()

        named_layers = self.flatten_model(model)
        names = range(len(named_layers))

        #print (named_layers)
        


        for name in names:
            layer = named_layers[name]
            if isinstance(layer, nn.Conv2d):
                self.counts["conv2d"] += 1
            elif isinstance(layer, nn.Linear):
                self.counts["linear"] += 1
            elif isinstance(layer, nn.BatchNorm2d):
                self.counts["batchnorm"] += 1
            elif isinstance(layer, nn.ReLU):
                self.counts["relu"] += 1
            elif isinstance(layer, nn.MaxPool2d):
                self.counts["maxpool"] += 1
            elif isinstance(layer, nn.Sequential) or isinstance(layer, Bottleneck):
                #get children
                children_nodes = layer.children()
                self.find_top_layers(children_nodes, top_count)
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                self.counts["avgpool"] += 1
            else:
                print ("not supported", layer)

        #sort based on count
        sorted_count = dict(sorted(self.counts.items(), key=lambda item: -item[1]))

        for key in sorted_count:
            print (key, ",", sorted_count[key])


if __name__ == "__main__":


    model = torchvision.models.resnet50(pretrained=True)
    top_count = 5

    model_layers = ModelLayers()
    model_layers.find_top_layers(model, top_count)