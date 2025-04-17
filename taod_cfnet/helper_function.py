import torch 
import einops 

def compute_partial(M):
    """
    M2: tensor có shape (B, C, H, W)
    Trả về: tensor shape (B, 1, 1, 1) chứa giá trị ∂ cho từng batch
    """
    B, C, H, W = M.shape
    device = M.device

    # Tạo chỉ số i, j theo không gian H, W
    i = torch.arange(H, device=device).view(H, 1).expand(H, W) + 1
    j = torch.arange(W, device=device).view(1, W).expand(H, W) + 1

    # Tính ma trận i * j
    ij = (i * j).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)

    # Áp dụng công thức tổng
    weighted = M / ij  # chia từng pixel cho i * j
    partial = weighted.view(B, C, -1).sum(dim=2).view(B, C, 1, 1)  # tổng theo từng ảnh

    return partial


def compute_eqs2(Light):
    """
    Light: tensor shape (B, C, H, W)
    Output: Eqs2, shape (B, C, 1, 1)
    """
    B, C, H, W = Light.shape
    device = Light.device

    # Tạo chỉ số i và j
    i_idx = torch.arange(H, device=device).view(H, 1).expand(H, W)
    j_idx = torch.arange(W, device=device).view(1, W).expand(H, W)

    # Tính index theo công thức: i * n + j
    pos = (i_idx * W + j_idx).float() + 1e-6  # tránh chia cho 0
    # pos = pos.view(1, 1, H, W)  # reshape để broadcast với Light

    # Tính giá trị Eqs2(I)
    weighted = Light / pos        # shape (B, C, H, W)
    eqs2 = weighted.sum(dim=(2, 3)) 
    eqs2 = eqs2.view(B, C, 1, 1)  
    return eqs2


def tuple_partitions(image: torch.Tensor, mask: torch.Tensor): 
    image_patchs = einops.rearrange(image, 'b c (h ph) (w pw) -> b (h w) c ph pw', ph=146, pw=219)
    mask_patchs  = einops.rearrange(mask,  'b c (h ph) (w pw) -> b (h w) c ph pw', ph=146, pw=219)
    return image_patchs, mask_patchs 


def tuple_unpartitions(image_patches: torch.Tensor, mask_patches: torch.Tensor, height: int, width: int):
    image = einops.rearrange(image_patches, 'b (h w) c ph pw -> b c (h ph) (w pw)', h=height, w=width)
    mask = einops.rearrange(mask_patches, 'b (h w) c ph pw -> b c (h ph) (w pw)', h=height, w=width)

    return image, mask
    