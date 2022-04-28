import torch
import math
import os
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import numpy as np
from utils.buffer import Buffer
from utils.triplet import merge
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
from pytorch_metric_learning.miners import TripletMarginMiner

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--gamma', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--sigmoid', type=float, required=True,
                        help='Penalty weight.')
    return parser
miner = TripletMarginMiner(margin=0.2, type_of_triplets='semihard')
loss_function = nn.KLDivLoss(reduction='batchmean')
class Correlation(nn.Module):
    """Correlation Congruence for Knowledge Distillation, ICCV 2019.
    The authors nicely shared the code with me. I restructured their code to be
    compatible with my running framework. Credits go to the original author"""
    def __init__(self):
        super(Correlation, self).__init__()

    def forward(self, f_s, f_t):
        delta = torch.abs(f_s - f_t)
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss
class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss
class CC(nn.Module):

	def __init__(self, gamma, P_order):
		super(CC, self).__init__()
		self.gamma = gamma
		self.P_order = P_order

	def forward(self, feat_s, feat_t):
		corr_mat_s = self.get_correlation_matrix(feat_s)
		corr_mat_t = self.get_correlation_matrix(feat_t)

		loss = F.mse_loss(corr_mat_s, corr_mat_t)

		return loss

	def get_correlation_matrix(self, feat):
		feat = F.normalize(feat, p=2, dim=-1)
		sim_mat  = torch.matmul(feat, feat.t())
		corr_mat = torch.zeros_like(sim_mat)

		for p in range(self.P_order+1):
			corr_mat += math.exp(-2*self.gamma) * (2*self.gamma)**p / \
						math.factorial(p) * torch.pow(sim_mat, p)

		return corr_mat

kdloss = KDLoss(3.5)
closs = Correlation()
ccloss = CC(2, 3)

class Hcr(ContinualModel):
    NAME = 'hcr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Hcr, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)     #LC

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss_rehearsal1 = ccloss(buf_outputs, buf_logits)+closs(buf_outputs, buf_logits)
            #L3
            loss += self.args.alpha * loss_rehearsal1

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss_rehearsal2 = self.loss(buf_outputs, buf_labels)    #ER
            loss += self.args.beta * loss_rehearsal2

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            buf_embedding = F.normalize(buf_outputs, p=2, dim=1)
            anchor_id, positive_id, negative_id = miner(buf_embedding, buf_labels)
            anchor = buf_embedding[anchor_id]
            positive = buf_embedding[positive_id]
            negative = buf_embedding[negative_id]
            ap_dist = torch.norm(anchor - positive, p=2, dim=1)
            an_dist = torch.norm(anchor - negative, p=2, dim=1)
            loss_rehearsal4 = -torch.log(torch.exp(-ap_dist) / (torch.exp(-an_dist) + torch.exp(-ap_dist))).mean()
            #L2 distance
            loss += self.args.sigmoid * loss_rehearsal4

            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            teacher_embedding = F.normalize(buf_logits, p=2, dim=1)
            student_embedding = F.normalize(buf_outputs, p=2, dim=1)

            # if not self.args.flag_merge:
                # generate triplets
            with torch.no_grad():
                anchor_id, positive_id, negative_id = miner(student_embedding, buf_labels)

            # get teacher embedding in tuples
            teacher_anchor = teacher_embedding[anchor_id]
            teacher_positive = teacher_embedding[positive_id]
            teacher_negative = teacher_embedding[negative_id]
            # get student embedding in triplets
            student_anchor = student_embedding[anchor_id]
            student_positive = student_embedding[positive_id]
            student_negative = student_embedding[negative_id]
            # get a-p dist and a-n dist in teacher embedding
            teacher_ap_dist = torch.norm(teacher_anchor - teacher_positive, p=2, dim=1)
            teacher_an_dist = torch.norm(teacher_anchor - teacher_negative, p=2, dim=1)
            # get a-p dist and a-n dist in student embedding
            student_ap_dist = torch.norm(student_anchor - student_positive, p=2, dim=1)
            student_an_dist = torch.norm(student_anchor - student_negative, p=2, dim=1)
            # get probability of triplets in teacher embedding
            teacher_prob = torch.sigmoid((teacher_an_dist - teacher_ap_dist) / 4)
            teacher_prob_aug = torch.cat([teacher_prob.unsqueeze(1), 1 - teacher_prob.unsqueeze(1)])
            # get probability of triplets in student embedding
            student_prob = torch.sigmoid((student_an_dist - student_ap_dist) / 4)
            student_prob_aug = torch.cat([student_prob.unsqueeze(1), 1 - student_prob.unsqueeze(1)])
            # compute loss function
            loss_value = 1000 * loss_function(torch.log(student_prob_aug), teacher_prob_aug)

            loss_rehearsal3 = torch.mean(torch.sum(loss_value, dim=0))  #L2 distribution
            loss += self.args.gamma * loss_rehearsal3.cpu().item() * student_prob.size()[0]


        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data
                             )

        return loss.item()
