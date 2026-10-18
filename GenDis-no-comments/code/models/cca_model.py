import torch
import torch.nn.functional as F
import torch.nn as nn

class CCAProjection(nn.Module):
    pass

    def __init__(self, in_dim1, in_dim2, k):
        super().__init__()
        self.proj1 = nn.Linear(in_dim1, k, bias=False)
        self.proj2 = nn.Linear(in_dim2, k, bias=False)
        self.outdim_size = k

    def forward(self, h1, h2):
        z1 = self.proj1(h1)
        z2 = self.proj2(h2)
        com = torch.concat([z1, z2], dim=-1)
        return (z1, z2, com)

    def calculate_loss(self, H1, H2, r1=0.001, r2=0.001, eps=1e-09, mode=0):
        H1, H2 = (H1.t(), H2.t())
        o1 = o2 = H1.size(0)
        m = H1.size(1)
        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        SigmaHat11 = H1bar @ H1bar.T / (m - 1) + r1 * torch.eye(o1, device=H1.device)
        SigmaHat22 = H2bar @ H2bar.T / (m - 1) + r2 * torch.eye(o2, device=H1.device)
        SigmaHat12 = H1bar @ H2bar.T / (m - 1)
        [D1, V1] = torch.linalg.eigh(SigmaHat11)
        [D2, V2] = torch.linalg.eigh(SigmaHat22)
        D1 = torch.clamp(D1, min=1e-06).to(H1.device)
        D2 = torch.clamp(D2, min=1e-06).to(H1.device)
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1].to(H1.device)
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2].to(H1.device)
        SigmaHat11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** (-0.5))), V1.t())
        SigmaHat22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** (-0.5))), V2.t())
        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv, SigmaHat12.to(torch.float32)), SigmaHat22RootInv)
        trace_TT = torch.matmul(Tval.t(), Tval)
        trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0]) * r1).to(H1.device))
        U, V = torch.linalg.eigh(trace_TT)
        U = torch.where(U > eps, U, (torch.ones(U.shape).double() * eps).to(H1.device))
        U = torch.clamp(U, min=1e-06)
        U = U.topk(self.outdim_size)[0]
        if mode == 2:
            corr = torch.mean(torch.log(U))
            return -corr
        elif mode == 1:
            corr = torch.mean(torch.sqrt(U))
            return -corr
        elif mode == 0:
            corr = torch.sum(torch.sqrt(U))
            return -corr
        else:
            assert False

    def Soft_DCCA_loss(self, view_t1: torch.Tensor, view_t2: torch.Tensor, eta1=0.01, eta2=0.001, rho=0.9, S1_prev=None, S2_prev=None):
        pass
        corr_square = (view_t1 - view_t2) ** 2
        corr_loss = torch.sqrt(torch.sum(corr_square, dim=-1) + 1e-08)
        corr_loss = eta1 * torch.mean(corr_loss)
        N1, d1 = view_t1.shape
        N2, d2 = view_t2.shape
        S_t1 = view_t1.T @ view_t1 / (N1 - 1)
        S_t2 = view_t2.T @ view_t2 / (N2 - 1)
        if S1_prev is None:
            S1_prev = torch.zeros_like(S_t1, device=view_t1.device)
        if S2_prev is None:
            S2_prev = torch.zeros_like(S_t2, device=view_t2.device)
        S1_updated = rho * S1_prev + (1 - rho) * S_t1
        S2_updated = rho * S2_prev + (1 - rho) * S_t2
        decov_loss_t1 = torch.sum(torch.abs(S1_updated)) - torch.sum(torch.diag(S1_updated))
        decov_loss_t2 = torch.sum(torch.abs(S2_updated)) - torch.sum(torch.diag(S2_updated))
        decov_loss = eta2 * torch.mean(decov_loss_t1 + decov_loss_t2)
        return (corr_loss, decov_loss, S1_updated.detach(), S2_updated.detach())
