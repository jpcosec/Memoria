import torch.nn as nn
import torch.nn.functional as F


def dist_loss_gen(T=8):
    def dist_loss(student_scores, teacher_scores, T=T):
        return nn.KLDivLoss()(F.log_softmax(student_scores / T, dim=1), F.softmax(teacher_scores / T, dim=1))

    return dist_loss


# class distiller()

def train_op(distiller, teacher, data, target, device):
    x_train = data.to(device)
    y_train = target.to(device)

    distiller["optimizer"].zero_grad()
    # Forward pass

    # Predecir
    S_y_pred = distiller["model"](x_train.view(-1, 784))
    T_y_pred = teacher(x_train)

    # Compute Loss
    loss = distiller["criterion"](S_y_pred, T_y_pred)
    # Backward pass
    loss.backward()
    distiller["optimizer"].step()
    return loss
