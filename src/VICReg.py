import torch
from torch import nn
from lightly.models.modules.heads import VICRegProjectionHead
import torch.distributed as dist
from lightly.utils.dist import gather


class VICReg(nn.Module):
    def __init__(self, backbone, projection_head_dims):
        super().__init__()
        self.backbone = backbone
        self.projection_head = VICRegProjectionHead(
            input_dim=projection_head_dims[0],
            hidden_dim=projection_head_dims[1],
            output_dim=projection_head_dims[2],
            num_layers=len(projection_head_dims),
        )

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


class UnbiasedVICRegLoss(nn.Module):
    def __init__(
        self,
        sim_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        gather_distributed: bool = False,
    ):
        """
        Initialize UnbiasedVICRegLoss.

        Args:
            sim_coeff: Coefficient for similarity loss (default: 25.0)
            cov_coeff: Coefficient for covariance loss (default: 1.0)
            gather_distributed: Whether to gather tensors across distributed processes
        """
        super(UnbiasedVICRegLoss, self).__init__()

        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

        self.sim_coeff = sim_coeff
        self.cov_coeff = cov_coeff
        self.gather_distributed = gather_distributed

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Compute the unbiased VICReg loss between two embeddings.

        Args:
            z_a: First embedding tensor of shape (batch_size, dim)
            z_b: Second embedding tensor of shape (batch_size, dim)

        Returns:
            Combined loss incorporating similarity and covariance terms

        Raises:
            AssertionError: If inputs have invalid batch size or incompatible shapes
        """
        assert (
            z_a.shape[0] > 1 and z_b.shape[0] > 1
        ), f"z_a and z_b must have batch size > 1 but found {z_a.shape[0]} and {z_b.shape[0]}"
        assert (
            z_a.shape == z_b.shape
        ), f"z_a and z_b must have same shape but found {z_a.shape} and {z_b.shape}."

        # Get batch size and dimension
        N, D = z_a.size()

        # Gather all batches if in distributed mode
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                z_a = torch.cat(gather(z_a), dim=0)
                z_b = torch.cat(gather(z_b), dim=0)
                N = z_a.size(0)  # Update batch size after gathering

        # Compute representation (similarity) loss
        repr_loss = nn.functional.mse_loss(z_a, z_b)

        # Combine embeddings and split randomly for covariance computation
        combined = torch.cat([z_a, z_b], dim=0)
        indices = torch.randperm(N, device=z_a.device)
        z1 = combined[indices[: N // 2]]
        z2 = combined[indices[N // 2 :]]

        # Compute covariance matrices
        cov_z1 = sum([z.unsqueeze(1) @ z.unsqueeze(0) for z in z1]) / (N // 2 - 1)
        cov_z2 = sum([z.unsqueeze(1) @ z.unsqueeze(0) for z in z2]) / (N // 2 - 1)

        # Compute covariance loss using Frobenius norm
        I = torch.eye(D, device=cov_z1.device)
        cov_diff = (cov_z1 - I) @ (cov_z2 - I)
        cov_loss = torch.norm(cov_diff, p="fro")

        # Combine losses with coefficients
        total_loss = self.sim_coeff * repr_loss + self.cov_coeff * cov_loss

        return total_loss
