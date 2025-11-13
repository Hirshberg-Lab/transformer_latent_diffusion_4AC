from rnd_spectra.ontheflydataset import pointwise_G_normalization
import torch
import numpy as np

enc_G = pointwise_G_normalization(mean=torch.Tensor([ 0.3319, -1.8394,  0.0023,  0.5002]), std=torch.Tensor([0.4170, 1.5327, 1.1582, 0.2884]))

class SoftmaxMean:
    def __init__(
            self,
            pred_spectra: np.array, # (samples, bs, length)
            ground_truth_G: torch.Tensor, # (bs, channels, length)
            tau: np.array,
            omega: np.array
            ) -> None:
        
        self.ground_truth_G = ground_truth_G
        self.ground_truth_G_norm = (ground_truth_G**2).mean(-1) # should have shape of (bs,channels)
        self.pred_spectra = torch.Tensor(pred_spectra)

        kernel = ( np.exp( -omega * tau[:,None] ) + np.exp( omega * ( tau[:,None] -1 ) ) )/2./np.pi
        G_preds_array = np.zeros((pred_spectra.shape[0],pred_spectra.shape[1],len(tau)))
        for i_data in range(pred_spectra.shape[1]):
            for i_sample in range(pred_spectra.shape[0]):
                G_preds_array[i_sample,i_data,:] = np.trapz(y=kernel*pred_spectra[i_sample,i_data,:],x=omega)
        
        # G_preds_array has shape of (samples, bs, len(tau))
        self.G_preds_tensor = enc_G(torch.Tensor(G_preds_array)) # (samples, bs, channels, len(tau))


    def L2_distance_G(self) -> torch.Tensor:
        """Compute the L2 distance between ground_truth_G and G_preds_tensor."""
        L2 = ( ( self.G_preds_tensor - self.ground_truth_G.unsqueeze(0) )**2 ).mean(dim=-1) # should have shape of (samples,bs,channels)
        norm_L2_per_channel =  L2 / self.ground_truth_G_norm.unsqueeze(0)
        return norm_L2_per_channel.mean(dim=-1) # average over channels --> should have shape of (samples, bs)
    
    def softmax_mean_spectra(self, beta: float = 1.0) -> torch.Tensor:
        G_L2_distances = self.L2_distance_G() # (samples, bs)
        weights = torch.exp( - beta * G_L2_distances ) # (samples, bs)
        weights = weights / weights.sum(dim=0) # (samples, bs)
        weighted_spec = (weights.unsqueeze(-1) * self.pred_spectra).sum(dim=0) # (bs, length)
        return weighted_spec