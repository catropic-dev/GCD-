import torch
import numpy as np
import torch
import torch.nn.functional as F

def get_topk_mask(logits, k=1):
    pass
    probs = F.softmax(logits, dim=1)
    _, topk_indices = torch.topk(probs, k=k, dim=0)
    mask = torch.zeros_like(logits[:, 0], dtype=torch.bool)
    for class_idx in range(probs.size(1)):
        mask[topk_indices[:, class_idx]] = True
    return mask.unsqueeze(1).long()

def shoot_infs(inp_tensor):
    pass
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

class SinkhornKnopp(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.num_iters = args.num_iters_sk
        self.epsilon = args.epsilon_sk
        self.imb_factor = args.imb_factor

    def _get_row_constraint(self, Q):
        if self.imb_factor > 1:
            marginals_argsort = torch.argsort(Q.sum(1))
            marginals_argsort = marginals_argsort.detach()
            r = []
            for i in range(Q.shape[0]):
                r.append((1 / self.imb_factor) ** (i / (Q.shape[0] - 1.0)))
            r = np.array(r)
            r = r * (Q.shape[1] / Q.shape[0])
            r = torch.from_numpy(r).to(Q.device)
            r[marginals_argsort] = torch.sort(r)[0]
            r = torch.clamp(r, min=1)
            r /= r.sum()
        else:
            r = torch.ones(Q.shape[0]).to(Q.device) / Q.shape[0]
        return r

    @torch.no_grad()
    def iterate(self, Q_main):
        Q_main = shoot_infs(Q_main)
        sum_Q = torch.sum(Q_main)
        Q_main /= sum_Q
        c = torch.ones(Q_main.shape[1]).to(Q_main.device) / Q_main.shape[1]
        r = self._get_row_constraint(Q_main)
        for it in range(self.num_iters):
            u = r / torch.sum(Q_main, dim=1)
            u = shoot_infs(u)
            Q_main *= u.unsqueeze(1)
            Q_main *= (c / torch.sum(Q_main, dim=0)).unsqueeze(0)
        return (Q_main / torch.sum(Q_main, dim=0, keepdim=True)).t().float()

    @torch.no_grad()
    def forward(self, logits):
        q = logits / self.epsilon
        M = torch.max(q)
        q -= M
        q = torch.exp(q).t()
        return self.iterate(q)
from types import SimpleNamespace
if __name__ == '__main__':
    args_dict = {'num_iters_sk': 25, 'epsilon_sk': 0.07, 'imb_factor': 100}
    args = SimpleNamespace(**args_dict)
    transporter = CalibratedSinkhorn(args).cuda()
    X = torch.randn(77, 128).cuda()
    Y = torch.randn(77, 128).cuda()
    Z = torch.randn(77, 128).cuda()
    logits_dict = {'disc': X, 'gen': Y, 'inv': Z}
    Q = transporter([logits_dict['disc'], logits_dict['gen'], logits_dict['inv']])
    labels = Q.argmax(dim=1)
