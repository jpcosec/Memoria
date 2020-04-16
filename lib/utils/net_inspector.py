import torch
import torch.nn as nn
from collections import OrderedDict

import logging


logging.basicConfig(filename='inspection.log', level=logging.DEBUG)

class inspector():
    def __init__(self, model):
        self.model=model
        self.model_keys = [29]
        device = 'cuda'

    def inspect(self):
        self.model_features = OrderedDict()
        self.model_keys = OrderedDict()
        self.model_layers = 1
        hooks=[]

        def register_model_hook(module):
            #global self.model_layers
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            #m_key = "%s-%i" % (class_name, self.model_layers)
            #self.model_layers += 1

            module_idx = len(self.model_keys)

            m_key = "%s-%i" % (class_name, module_idx + 1)

            self.model_features[m_key] = 1

            def hook(mod, inp, out):
                #logging.debug(m_key+":"+str(out.shape[-1]))

                self.model_features[m_key] = out

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == self.model)
            ):
                # logging.debug(str(module))
                #logging.debug(str(m_key))
                # logging.debug(str())
                # print(m_key)
                if self.model_layers in self.model_keys:
                    module.register_forward_hook(hook)

        self.model.apply(register_model_hook)

        inp = torch.rand(1, 3, 224, 224).to(device)

        self.model.eval()

        _ = self.model(inp)

        print(self.model_features)

        sizes = [(k,tensor.shape[-1]) for k,tensor in self.model_features.items()]

        print(sizes)

def summary(model, input_size, key,batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):

            #print(m_key)
            summary[m_key] = output.shape
            #if key==m_key:
            #    summary[m_key] = output
            #else:
            #    summary[m_key] = None

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            #hooks.append(module.register_forward_hook(hook))
            module.register_forward_hook(hook)


    summary = OrderedDict()

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)

    x = [torch.rand(2,3,224,224)+1]
    model(*x)

    # remove these hooks
    #for h in hooks:
    #    h.remove()
    #print(summary["ReLU-39"].mean())

    print(summary)

    return model,summary


if __name__ == '__main__':
    from lib.utils.imagenet.utils import load_model


    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    model=load_model("MobileNetV2")

    model,s=summary(model,(3,224,224),"ReLU-39")
    print(s["ReLU-39"].mean())

    x = [torch.rand(2,3,224,224)+1]
    # print(type(x[0]))

    # make a forward pass
    # print(x.shape)
    model(*x)

    print(s["ReLU-39"].mean())