import torch
from torch import nn
import torch.distributed as dist
from lightly.utils.dist import gather

class Projector(nn.Module):
    def __init__(self, encoder_dim, projector_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(encoder_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim)
        )
            
    def forward(self, x):
        return self.network(x)

class VICReg(nn.Module):
    def __init__(self, backbone, projection_head_dims):
        super().__init__()
        self.backbone = backbone
        
        #for resnet18 from SIMCLR paper:
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=1)
        self.backbone.maxpool = nn.Identity()

        self.projection_head = Projector(projection_head_dims[0], projection_head_dims[1])

    def forward(self, x):
        y = self.backbone(x)
        return self.projection_head(y)


class UnbiasedVICRegLoss(nn.Module):
    def __init__(
        self,
        sim_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        gather_distributed: bool = False,
    ):
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
        assert (
            z_a.shape[0] > 1 and z_b.shape[0] > 1
        ), f"z_a and z_b must have batch size > 1 but found {z_a.shape[0]} and {z_b.shape[0]}"
        assert (
            z_a.shape == z_b.shape
        ), f"z_a and z_b must have same shape but found {z_a.shape} and {z_b.shape}."

        N, D = z_a.size()

        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                z_a = torch.cat(gather(z_a), dim=0)
                z_b = torch.cat(gather(z_b), dim=0)
                N = z_a.size(0)

        repr_loss = nn.functional.mse_loss(z_a, z_b)

        combined = torch.cat([z_a, z_b], dim=0)
        indices = torch.randperm(N, device=z_a.device)
        z1 = combined[indices[: N // 2]]
        z2 = combined[indices[N // 2 :]]

        cov_z1 = sum([z.unsqueeze(1) @ z.unsqueeze(0) for z in z1]) / (N // 2 - 1)
        cov_z2 = sum([z.unsqueeze(1) @ z.unsqueeze(0) for z in z2]) / (N // 2 - 1)

        I = torch.eye(D, device=cov_z1.device)
        cov_diff = (cov_z1 - I) @ (cov_z2 - I)
        cov_loss = torch.norm(cov_diff, p="fro")

        total_loss = self.sim_coeff * repr_loss + self.cov_coeff * cov_loss

        return total_loss
