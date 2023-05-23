import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

class CosineLinear(nn.Module):
    def __init__(self,in_features, out_features, requires_grad=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requires_grad = requires_grad
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=requires_grad)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def add_classes(self, num_new_classes):
        cos = CosineLinear(
            self.in_features, num_new_classes, requires_grad=self.requires_grad
        )
        self.weight = nn.Parameter(
            torch.cat(
                [
                    self.weight, cos.weight.to(self.weight.device)
                ], dim=0
            ),
            requires_grad=self.requires_grad
        ).to(self.weight.device)
        del cos

    def forward(self, cls_features):
        return F.linear(F.normalize(cls_features, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
    

class MeanHead(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cos = CosineLinear(in_features, out_features)
        self.seen_tasks = set()

    def before_train(self, task_id):
        """
        Should be called before training (1 epoch).
        Retuns True if training should continue, False otherwise.
        """
        if task_id in self.seen_tasks:
            return False
        self.seen_tasks.add(task_id)
        self.embedding_list = []
        self.label_list = []
        return True

    def forward(self, cls_features, target=None):
        if target is None:
            return self.cos(cls_features)
        # store examples
        self.embedding_list.append(cls_features.cpu())
        self.label_list.append(target.cpu())

    def after_train(self):
        """
        Should be called after training (1 epoch).
        """
        embedding_list = torch.cat(self.embedding_list, dim=0)
        label_list = torch.cat(self.label_list, dim=0)

        class_list=np.unique(label_list)

        self.cos.add_classes(len(class_list))

        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            proto=embedding.mean(0).to(self.cos.weight.device)
            self.cos.weight.data[class_index]=proto
        
    