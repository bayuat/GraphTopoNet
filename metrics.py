from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import numpy as np

def compute_metrics(pred, target):
    """
    Compute evaluation metrics for predictions.
    """
    # Convert to NumPy arrays for metrics requiring NumPy inputs
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(pred_np - target_np))

    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((pred_np - target_np) ** 2))

    # R² (Coefficient of Determination)
    ss_total = np.sum((target_np - target_np.mean()) ** 2)
    ss_residual = np.sum((target_np - pred_np) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    # SSIM (Structural Similarity Index Measure)
    ssim_value, _ = ssim(target_np, pred_np, full=True, data_range=target_np.max() - target_np.min())

    # PSNR (Peak Signal-to-Noise Ratio)
    psnr_value = psnr(target_np, pred_np, data_range=target_np.max() - target_np.min())

    return mae, rmse, r2, ssim_value, psnr_value