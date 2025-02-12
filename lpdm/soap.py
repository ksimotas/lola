r"""PyTorch implementation of SOAP

Adapted from https://github.com/nikhilvyas/SOAP

References:
    | SOAP: Improving and Stabilizing Shampoo using Adam (Vyas et al., 2024)
    | https://arxiv.org/abs/2409.11321
"""

import torch

from typing import Iterable, Tuple


class SOAP(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.9, 0.99, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        precondition_frequency: int = 16,
        precondition_warmup: int = 0,
        precondition_1d: bool = False,
        max_precond_dim: int = 4096,
        merge_dims: bool = False,
    ):
        defaults = {
            "lr": lr,
            "betas": betas[:3],
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "precondition_warmup": precondition_warmup,
            "precondition_1d": precondition_1d,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
        }
        super().__init__(params, defaults)

    @staticmethod
    def merge_shape(shape, max_precond_dim=4096):
        new_shape = []
        cum_size = 1

        for s in shape[::-1]:
            temp_size = cum_size * s
            if temp_size > max_precond_dim:
                if cum_size > 1:
                    new_shape.append(cum_size)
                    cum_size = s
                else:
                    new_shape.append(s)
                    cum_size = 1
            else:
                cum_size = temp_size

        if cum_size > 1:
            new_shape.append(cum_size)

        new_shape = new_shape[::-1]

        return new_shape

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            lists = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                state = self.state[p]
                state["step"] = step = state.get("step", -1) + 1

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(g)
                    state["exp_avg_sq"] = torch.zeros_like(g)

                    self.init_preconditioner(
                        g,
                        state,
                        precondition_1d=group["precondition_1d"],
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                    )

                    continue

                # Project gradients to the eigenbasis of Shampoo's preconditioner
                g_sq = self.project(g, state).square()

                lists.append((p, g, g_sq, state["exp_avg"], state["exp_avg_sq"]))

            if not lists:
                continue

            params, grads, grads_sq, exp_avg, exp_avg_sq = zip(*lists, strict=True)

            # Moments
            beta1, beta2, beta3 = group["betas"]

            beta1_ = 1 - (1 - beta1) / (1 - beta1**step)
            beta2_ = 1 - (1 - beta2) / (1 - beta2**step)
            beta3_ = 1 - (1 - beta3) / (1 - beta3 ** (step + 1))

            torch._foreach_lerp_(exp_avg, grads, 1 - beta1_)
            torch._foreach_lerp_(exp_avg_sq, grads_sq, 1 - beta2_)

            del grads_sq

            # Denominator
            denom = torch._foreach_sqrt(exp_avg_sq)
            torch._foreach_add_(denom, group["eps"])

            # Update
            updates = []

            for p, g, u, d in zip(params, grads, exp_avg, denom, strict=True):
                state = self.state[p]

                # Adam in the eigenbasis of Shampoo's preconditioner
                u = self.project(u, state)
                u = u / d
                u = self.project(u, state, back=True)

                updates.append(u)

                # Update GG and Q
                self.update_preconditioner(
                    g,
                    state,
                    shampoo_beta=beta3_,
                    precondition_frequency=group["precondition_frequency"],
                    precondition_warmup=group["precondition_warmup"],
                )

            torch._foreach_add_(params, updates, alpha=-group["lr"])

            if group["weight_decay"] > 0:
                torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])

        return loss

    def init_preconditioner(
        self,
        grad,
        state,
        precondition_1d=False,
        max_precond_dim=4096,
        merge_dims=False,
    ):
        """Initializes the preconditioner matrices."""

        state["GG"] = []
        state["Q"] = None

        grad = grad.squeeze()

        if merge_dims:
            state["precond_shape"] = self.merge_shape(grad.shape, max_precond_dim)
        else:
            state["precond_shape"] = grad.shape

        if grad.numel() > 1 and (grad.ndim > 1 or precondition_1d):
            for s in state["precond_shape"]:
                if s > max_precond_dim or s == 1:
                    state["GG"].append(None)
                else:
                    state["GG"].append(torch.zeros(s, s, device=grad.device))
        else:
            state["GG"].append(None)

        self.update_preconditioner(grad, state, shampoo_beta=0.0)

    def update_preconditioner(
        self,
        grad,
        state,
        shampoo_beta=0.99,
        precondition_frequency=16,
        precondition_warmup=0,
    ):
        """Updates the preconditioner matrices and the eigenbases."""

        grad = grad.reshape(state["precond_shape"])

        for i, m in enumerate(state["GG"]):
            if m is not None:
                outer = torch.tensordot(
                    grad,
                    grad,
                    dims=[(*range(i), *range(i + 1, grad.ndim))] * 2,
                )
                m.lerp_(outer, 1 - shampoo_beta)

        if state["Q"] is None:
            state["Q"] = self.get_orthogonal_matrix(state)
        elif state["step"] < precondition_warmup or state["step"] % precondition_frequency == 0:
            state["Q"] = self.get_orthogonal_matrix_QR(state)

    def project(self, grad, state, back: bool = False):
        """Projects the gradient to/from the eigenbasis of the preconditioner."""

        grad_shape = grad.shape
        grad = grad.reshape(state["precond_shape"])

        for q in state["Q"]:
            if q is None:
                grad = grad.movedim(0, -1)
            else:
                grad = torch.tensordot(
                    grad,
                    q,
                    dims=[[0], [1 if back else 0]],
                )

        return grad.reshape(grad_shape)

    def get_orthogonal_matrix(self, state):
        """Computes the eigenbasis of the preconditioner using torch.linalg.eigh decomposition."""

        Q = []

        for m in state["GG"]:
            if m is None:
                Q.append(None)
            else:
                _, q = torch.linalg.eigh(
                    m.to(torch.float32) + 1e-12 * torch.eye(*m.shape, device=m.device)
                )
                Q.append(torch.fliplr(q))

        return Q

    def get_orthogonal_matrix_QR(self, state):
        """Computes the eigenbasis of the preconditioner using one round of power
        iteration followed by torch.linalg.qr decomposition."""

        Q = []

        for m, q in zip(state["GG"], state["Q"], strict=True):
            if m is None:
                Q.append(None)
            else:
                m, q = m.to(torch.float32), q.to(torch.float32)
                mq = m @ q
                eigen = torch.einsum("ij,ij->j", q, mq)
                order = torch.argsort(eigen, descending=True)
                q[:, order], _ = torch.linalg.qr(mq[:, order])
                Q.append(q)

        return Q
