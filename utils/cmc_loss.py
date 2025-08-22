import torch
from warnings import warn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

class ContrastiveLoss(_Loss):
    """
    Compute the Contrastive loss defined in:

        Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International
        conference on machine learning. PMLR, 2020. (http://proceedings.mlr.press/v119/chen20j.html)

    Adapted from:
        https://github.com/Sara-Ahmed/SiT/blob/1aacd6adcd39b71efc903d16b4e9095b97dda76f/losses.py#L5

    """

    def __init__(self, temperature: float = 0.5, batch_size: int = -1) -> None:
        """
        Args:
            temperature: Can be scaled between 0 and 1 for learning from negative samples, ideally set to 0.5.

        Raises:
            ValueError: When an input of dimension length > 2 is passed
            ValueError: When input and target are of different shapes

        """
        super().__init__()
        self.temperature = temperature

        if batch_size != -1:
            warn("batch_size is no longer required to be set. It will be estimated dynamically in the forward call")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be B[F].
            target: the shape should be B[F].
        """
        if len(target.shape) > 2 or len(input.shape) > 2:
            raise ValueError(
                f"Either target or input has dimensions greater than 2 where target "
                f"shape is ({target.shape}) and input shape is ({input.shape})"
            )

        if target.shape != input.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        temperature_tensor = torch.as_tensor(self.temperature).to(input.device)
        batch_size = input.shape[0]

        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)
        negatives_mask = torch.clone(negatives_mask.type(torch.float)).to(input.device)

        repr = torch.cat([input, target], dim=0)
        sim_matrix = F.cosine_similarity(repr.unsqueeze(1), repr.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / temperature_tensor)
        denominator = negatives_mask * torch.exp(sim_matrix / temperature_tensor)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
      
        return torch.sum(loss_partial) / (2 * batch_size)



def CAC_loss_func(pred1, pred2, similarity='cosine'):
    """
    Compute CAC loss
    """
    if torch.sum(pred1) == 0 and torch.sum(pred2) == 0:
        return torch.tensor(1.0, device=pred1.device)
    smooth = 1e-6
    dim_len = len(pred1.size())
    if dim_len == 5:
       dim=(2,3,4)
    elif dim_len == 4:
       dim=(2,3)
    intersect = torch.sum(pred1 * pred2,dim=dim)
    y_sum = torch.sum(pred1 * pred1,dim=dim)
    z_sum = torch.sum(pred2 * pred2,dim=dim)
    dice_sim = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    dice_sim = dice_sim.mean()
    if torch.isnan(dice_sim):
        dice_sim = torch.tensor(1.0, device=dice_sim.device, requires_grad=True)
    return dice_sim

def CSC_loss_func(pred1,pred2):
    channel_losses = 0.0
    lens = pred1.shape[0]
    cl_loss = ContrastiveLoss(temperature=0.5)
    for c in range(pred1.shape[0]):
        # pred1 [8,512,10,10]
        # pred1_output_channel [512,10,10]
        pred1_output_channel = pred1[c, :, :, :]  # select the c_th channel of predi
        pred2_output_channel = pred2[c, :, :, :]  # select the c_th channel of pred2
        # print("pred1_output_channel", pred1_output_channel.shape)
        #pred1_2d_flat [10*10,512]
        pred1_2d_flat = pred1_output_channel.reshape(-1, pred1_output_channel.shape[0])  # resize shape
        pred2_2d_flat = pred2_output_channel.reshape(-1, pred2_output_channel.shape[0])  # resize shape
        # print("pred1_2d_flat",pred1_2d_flat.shape)
        # compute the ContrastiveLoss from each channel
        cl_value = cl_loss(pred1_2d_flat, pred2_2d_flat)
        # print(cl_value)
        channel_losses = channel_losses + cl_value
    mean_loss = channel_losses/ lens
    if torch.isnan(mean_loss):
        mean_loss = torch.tensor(1.0, device=mean_loss.device, requires_grad=True)
    return mean_loss


if __name__ == "__main__": 
    x = torch.rand(4,512,3,3)
    # y = torch.rand(4,512,10,10)
    loss = CSC_loss_func(x,x)
    print(loss) # tensor(4.7964)