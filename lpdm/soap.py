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
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        precondition_frequency: int = 16,
        precondition_1d: bool = False,
        max_precond_dim: int = 4096,
        merge_dims: bool = False,
    ):
        defaults = {
            "lr": lr,
            "betas": (betas[0], betas[1]),
            "shampoo_beta": betas[2],
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
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
            vals = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                p.grad = None

                state = self.state[p]
                state["step"] = step = state.get("step", -1) + 1

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    self.init_preconditioner(
                        grad,
                        state,
                        precondition_1d=group["precondition_1d"],
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                    )
                    self.update_preconditioner(
                        grad,
                        state,
                        shampoo_beta=group["shampoo_beta"],
                        precondition_frequency=group["precondition_frequency"],
                    )

                    continue  # skip first gradient step

                # Project gradients to the eigenbases of Shampoo's preconditioner
                grad_projected = self.project(grad, state)

                vals.append((p, grad, grad_projected, state["exp_avg"], state["exp_avg_sq"]))

            if not vals:
                continue

            params, grad, grad_projected, exp_avg, exp_avg_sq = zip(*vals, strict=True)
            beta1, beta2 = group["betas"]

            debiased1 = (1 - beta1) / (1 - beta1**step)
            debiased2 = (1 - beta2) / (1 - beta2**step)

            # Decay the first and second moment running average
            torch._foreach_lerp_(exp_avg, grad, debiased1)
            torch._foreach_mul_(exp_avg_sq, 1 - debiased2)
            torch._foreach_addcmul_(exp_avg_sq, grad_projected, grad_projected, value=debiased2)

            del grad_projected

            updates = []

            for p, g, ea, ea_sq in zip(params, grad, exp_avg, exp_avg_sq, strict=True):
                state = self.state[p]

                # Project the exponential moving average of gradients to the eigenbases of Shampoo's preconditioner
                u = self.project(ea, state)

                # Normalize with respect to the sqrt of the exponential moving average of the squared projected gradients
                u = u / torch.clip(torch.sqrt(ea_sq), min=group["eps"])

                # Project back
                u = self.project_back(u, state)

                updates.append(u)

                # Update GG and Q
                self.update_preconditioner(
                    g,
                    state,
                    shampoo_beta=group["shampoo_beta"],
                    precondition_frequency=group["precondition_frequency"],
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
                if s > max_precond_dim:
                    state["GG"].append(None)
                else:
                    state["GG"].append(torch.zeros(s, s, device=grad.device))
        else:
            state["GG"].append(None)

    def update_preconditioner(
        self,
        grad,
        state,
        shampoo_beta=0.999,
        precondition_frequency=16,
    ):
        """Updates the preconditioner matrices and the eigenbases."""

        grad = grad.reshape(state["precond_shape"])

        for i, m in enumerate(state["GG"]):
            if m is not None:
                outer_product = torch.tensordot(
                    grad,
                    grad,
                    dims=[(*range(i), *range(i + 1, grad.ndim))] * 2,
                )
                m.lerp_(outer_product, 1 - shampoo_beta)

        if state["Q"] is None:
            state["Q"] = self.get_orthogonal_matrix(state)

        if state["step"] > 0 and state["step"] % precondition_frequency == 0:
            state["Q"] = self.get_orthogonal_matrix_QR(state)

    def project(self, grad, state):
        """Projects the gradient to the eigenbases of the preconditioner."""

        grad_shape = grad.shape
        grad = grad.reshape(state["precond_shape"])

        for mat in state["Q"]:
            if mat is None:
                grad = grad.movedim(0, -1)
            else:
                grad = torch.tensordot(
                    grad,
                    mat,
                    dims=[[0], [0]],
                )

        return grad.reshape(grad_shape)

    def project_back(self, grad, state):
        """Projects the gradient back to the original space."""

        grad_shape = grad.shape
        grad = grad.reshape(state["precond_shape"])

        for mat in state["Q"]:
            if mat is None:
                grad = grad.movedim(0, -1)
            else:
                grad = torch.tensordot(
                    grad,
                    mat,
                    dims=[[0], [1]],
                )

        return grad.reshape(grad_shape)

    def get_orthogonal_matrix(self, state):
        """Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition."""

        results = []

        for m in state["GG"]:
            if m is None:
                results.append(None)
            else:
                try:
                    _, Q = torch.linalg.eigh(
                        m.data.to(torch.float32) + 1e-30 * torch.eye(m.shape[0], device=m.device)
                    )
                except:  # noqa: E722
                    _, Q = torch.linalg.eigh(
                        m.data.to(torch.float64) + 1e-30 * torch.eye(m.shape[0], device=m.device)
                    )
                    Q = Q.to(torch.float32)

                Q = torch.flip(Q, [1])
                results.append(Q)

        return results

    def get_orthogonal_matrix_QR(self, state):
        """Computes the eigenbases of the preconditioner using one round of power
        iteration followed by torch.linalg.qr decomposition."""

        exp_avg_sq = state["exp_avg_sq"].reshape(state["precond_shape"])

        results = []

        for dim, (m, o) in enumerate(zip(state["GG"], state["Q"], strict=True)):
            if m is None:
                results.append(None)
            else:
                m, o = m.data.to(torch.float32), o.data.to(torch.float32)
                mo = m @ o
                est_eig = torch.einsum("ij,ij->j", o, mo)
                sort_idx = torch.argsort(est_eig, descending=True)
                exp_avg_sq = exp_avg_sq.index_select(dim, sort_idx)
                Q, _ = torch.linalg.qr(mo[:, sort_idx])
                results.append(Q)

        state["exp_avg_sq"] = exp_avg_sq.reshape(state["exp_avg_sq"].shape)

        return results
