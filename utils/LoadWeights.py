import torch


def load_preweights(model, preweights):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(preweights)
    # for key, value in model_dict.items():
    #     print("model key {}, value {}".format(key, model_dict[key].size()))
    for key, value in pretrained_dict.items():
        print("pretrained_dict key {}, value {}".format(
            key, pretrained_dict[key].size()))
    state_dict = {}
    for key, value in model_dict.items():
        if key == "features.0.weight":
            state_dict[key] = pretrained_dict["features.0.weight"]
            print("loading the parameters {}".format(key))
        elif key == "features.0.bias":
            state_dict[key] = pretrained_dict["features.0.bias"]
            print("loading the parameters {}".format(key))
        elif key == "features.3.weight":
            state_dict[key] = pretrained_dict["features.3.weight"]
            print("loading the parameters {}".format(key))
        elif key == "features.3.bias":
            state_dict[key] = pretrained_dict["features.3.bias"]
            print("loading the parameters {}".format(key))
        elif key == "features.6.weight":
            state_dict[key] = pretrained_dict["features.6.weight"]
            print("loading the parameters {}".format(key))
        elif key == "features.6.bias":
            state_dict[key] = pretrained_dict["features.6.bias"]
            print("loading the parameters {}".format(key))
        elif key == "features.8.weight":
            state_dict[key] = pretrained_dict["features.8.weight"]
            print("loading the parameters {}".format(key))
        elif key == "features.8.bias":
            state_dict[key] = pretrained_dict["features.8.bias"]
            print("loading the parameters {}".format(key))
        elif key == "conv5.0.weight":
            state_dict[key] = pretrained_dict["features.10.weight"]
            print("loading the parameters {}".format(key))
        elif key == "conv5.0.bias":

            state_dict[key] = pretrained_dict["features.10.bias"]
            print("loading the parameters {}".format(key))

        elif key == "classifier.1.weight":
            state_dict[key] = pretrained_dict["classifier.1.weight"]
            print("loading the parameters {}".format(key))
        elif key == "classifier.1.bias":
            state_dict[key] = pretrained_dict["classifier.1.bias"]
            print("loading the parameters {}".format(key))
        elif key == "classifier.4.weight":
            state_dict[key] = pretrained_dict["classifier.4.weight"]
            print("loading the parameters {}".format(key))
        elif key == "classifier.4.bias":
            state_dict[key] = pretrained_dict["classifier.1.bias"]
            print("loading the parameters {}".format(key))
        else:
            state_dict[key] = model_dict[key]
    return state_dict
