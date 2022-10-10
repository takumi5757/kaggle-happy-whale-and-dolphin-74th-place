import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class Classifier(nn.Module):
    def __init__(
        self,
        in_dim,
        num_classes,
        bias=False,
        scale=1.0,
        learn_scale=False,
        cls_type="linear",
    ):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.classifier_type = cls_type

        if self.classifier_type == "linear":
            self.layers = nn.Linear(in_dim, num_classes)

        else:
            raise ValueError(
                "Not implemented / recognized classifier type {}".format(
                    self.classifier_type
                )
            )

    def forward(self, features):
        scores = self.layers(features)
        return scores


class PoolArcface(nn.Module):
    def __init__(self, in_dim, num_classes, s, m, ls_eps):
        super(PoolArcface, self).__init__()
        self.pooling = GeM()
        self.drop = nn.Dropout(p=0.2, inplace=False)
        self.fc = nn.Linear(in_dim, 512)
        self.arc = ArcMarginProduct(
            512,
            num_classes,
            s=s,
            m=m,
            easy_margin=False,
            ls_eps=ls_eps,
        )

    def forward(self, features, labels):
        pooled_features = self.pooling(features).flatten(1)
        pooled_drop = self.drop(pooled_features)
        emb = self.fc(pooled_drop)
        output = self.arc(emb, labels)
        return output

    def extract(self, features):
        pooled_features = self.pooling(features).flatten(1)
        pooled_drop = self.drop(pooled_features)
        emb = self.fc(pooled_drop)
        return emb


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


# class Arcface(nn.Module):
#     def __init__(self, in_dim, num_classes, s, m, ls_eps):
#         super(Arcface, self).__init__()
#         self.fc = nn.Linear(in_dim, 512)
#         self.arc = ArcMarginProduct(
#             512,
#             num_classes,
#             s=s,
#             m=m,
#             easy_margin=False,
#             ls_eps=ls_eps,
#         )

#     def forward(self, features, labels):
#         emb = self.fc(features)
#         output = self.arc(emb, labels)
#         return output

#     def extract(self, features):
#         emb = self.fc(features)
#         return emb


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp((1.0 - torch.pow(cosine, 2)), 1e-9, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            #             phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # support amp
            phi = torch.where(cosine.float() > self.th, phi, cosine.float() - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # TODO can change device
        one_hot = torch.zeros(cosine.size(), device="cuda")
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


def get_encoder(name, pretrained=True, **kwargs):
    model = timm.create_model(
        name, pretrained=pretrained, num_classes=0, drop_rate=0.0, drop_path_rate=0.1
    )
    return model


def get_encoder_from_config(config):
    return get_encoder(**config)


def get_classifier(name):
    classifiers_map = {"base": Classifier, "poolarcface": PoolArcface}

    if name not in classifiers_map:
        raise ValueError("Name of network unknown %s" % name)

    def get_network_fn(**kwargs):
        return classifiers_map[name](**kwargs)

    return get_network_fn


def get_classifier_from_config(config):

    cls_type = config.get("cls_type", False)
    bias = config.get("bias", False)
    scale = config.get("scale", 1.0)
    learn_scale = config.get("learn_scale", False)

    if cls_type:
        if config["name"] == "base":
            return get_classifier(config["name"])(
                in_dim=config["in_dim"],
                num_classes=config["num_classes"],
                bias=bias,
                scale=scale,
                learn_scale=learn_scale,
                cls_type=cls_type,
            )
        elif config["name"] == "poolarcface":
            return get_classifier(config["name"])(
                in_dim=config["in_dim"],
                num_classes=config["num_classes"],
                s=config["s"],
                m=config["m"],
                ls_eps=config["ls_eps"],
            )
    else:
        print("No cls_type in the config")
