import torch.nn as nn
import torch
import torch.nn.functional as F


class StudentLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(StudentLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()

    def forward(self, student_output, ground_truth, teacher_predictions, lin_comb_alpha):
        """Compute loss for model.
        Args:
            teacher_predictions: vector of shape [bsz, n_classes], already normalized
            ground_truth: array of shape [bsz, n_classes].
            student_output: vector of shape [bsz, n_classes].
        Returns:
            A loss scalar.
        """
        # Student loss: crossentropy (student output (logits -> temperature softmax)
        # compared to dot products of ground truth and teacher prediction
        # (logits -> temperature softmax temperature > 1) (output of last linear teacher layer)
        # do dot product
        # alternatively, linear combination
        # dot_prod = ground_truth * teacher_predictions
        dot_prod = ((1 - lin_comb_alpha) * ground_truth) + (lin_comb_alpha * F.softmax(teacher_predictions *
                                                                                       self.temperature, dim=1)).cuda()
        # pred_similarities = torch.matmul(student_output, torch.t(teacher_predictions))  # shape [bsz, bsz]
        # todo kill gradients if teacher and student output are too similar
        loss = self.loss_fn(input=student_output * self.temperature, target=dot_prod)
        return loss


class TeacherLoss(nn.Module):
    def __init__(self):
        super(TeacherLoss, self).__init__()
        self.temperature = 1.
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()

    def forward(self, teacher_predictions, ground_truth):
        # Teacher loss: Crossentropy of ground truth and l2-normalized teacher prediction
        # (output linear layer 3)
        loss = self.loss_fn(input=teacher_predictions * self.temperature, target=ground_truth)
        return loss

