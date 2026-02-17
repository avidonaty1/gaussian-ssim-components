import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

def gaussian_kernel(win_size, sigma):

    """
    Generate a 2D Gaussian kernel using OpenCV.

    The kernel is constructed by computing a 1D Gaussian vector of length
    `win_size` with standard deviation `sigma`, and forming the outer
    product to obtain a separable 2D Gaussian kernel.

    Parameters
    ----------
    win_size : int
        Size of the Gaussian window (must be odd).
    sigma : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    kernel_2d : ndarray of shape (win_size, win_size)
        Normalized 2D Gaussian kernel suitable for convolution.
    """

    kernel = cv2.getGaussianKernel(win_size, sigma)
    kernel_2d = kernel @ kernel.T

    return kernel_2d


def get_mask(image, foreground=False, threshold=1.5):
    
    """
    Compute a foreground or background mask for an image with a solid background.

    The mask is computed in LAB color space using only the a and b channels
    to ensure invariance to luminance variations. The background color is
    estimated as the median chromatic value of the image border pixels.
    Pixels whose chromatic distance from the estimated background is below
    `threshold` are classified as background.

    Morphological closing and opening are applied to remove noise and fill
    holes, and only the largest connected component is retained.

    Parameters
    ----------
    image : ndarray of shape (H, W, 3)
        Input RGB image (uint8).
    foreground : bool, optional
        If True, return a foreground mask instead of a background mask.
        Default is False.
    threshold : float, optional
        Euclidean distance threshold in LAB a–b space for background
        classification. Default is 1.5.

    Returns
    -------
    mask : ndarray of shape (H, W), dtype=bool
        Boolean mask where True indicates selected region (foreground or
        background depending on `foreground` argument).
    """

    # convert to LAB color
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # get border pixels and their LAB values
    border_pixels = np.concatenate([
        lab_image[0, :, :], # top row
        lab_image[-1, :, :], # bottom row
        lab_image[:, 0, :], # left column
        lab_image[:, -1, :] # right column
    ], axis=0)

    # get median background color in a,b of LAB
    background_ab = np.median(border_pixels[:, 1:3], axis=0)

    # gets a, b of every pixel in image and measure distance from backgkround color
    ab = lab_image[:, :, 1:3]
    dist = np.linalg.norm(ab - background_ab, axis=2)

    # gets mask of every pixel that is close within threshold to background color
    mask = dist < threshold

    # inverts to be foreground mask if necessary
    if foreground:
        mask = ~mask

    mask = mask.astype(np.uint8) * 255

    # cleans up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # fill holes inside object
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # remove noise dots

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.uint8) * 255

    return mask.astype(bool)


def ssim_single_channel(x, y, kernel, data_range, k1, k2):

    """
    Compute SSIM and its component maps for a single image channel.

    This function implements the structural similarity (SSIM) index using
    Gaussian-weighted local statistics. Local means, variances, and
    covariance are computed via convolution with the provided kernel.

    The SSIM index is decomposed into luminance, contrast, and structure
    components, and both per-pixel maps and global mean scores are returned.

    Parameters
    ----------
    x : ndarray of shape (H, W)
        First image channel (float64).
    y : ndarray of shape (H, W)
        Second image channel (float64).
    kernel : ndarray
        2D Gaussian kernel used for local statistics.
    data_range : float
        Dynamic range of the input images (e.g., 255 for 8-bit images).
    k1 : float
        Stability constant for luminance term.
    k2 : float
        Stability constant for contrast/structure terms.

    Returns
    -------
    results : dict
        Dictionary containing:
            - luminance_map
            - contrast_map
            - structure_map
            - ssim_map
            - luminance_score (mean)
            - contrast_score (mean)
            - structure_score (mean)
            - ssim_score (mean)
    """

    C1 = (k1 * data_range)**2
    C2 = (k2 * data_range)**2
    C3 = C2 / 2

    # Get pixel sample mean
    muX = cv2.filter2D(x, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)
    muY = cv2.filter2D(y, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)

    # Variance
    sigmaX_sq = cv2.filter2D(x * x, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT) - muX**2
    sigmaY_sq = cv2.filter2D(y * y, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT) - muY**2

    # Covariance
    sigmaXY = cv2.filter2D(x * y, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT) - muX * muY

    sigmaX = np.sqrt(np.maximum(sigmaX_sq, 0))
    sigmaY = np.sqrt(np.maximum(sigmaY_sq, 0))


    # SSIM Components
    luminance_map = (2*muX*muY + C1) / (muX**2 + muY**2 + C1)

    contrast_map = (2*sigmaX*sigmaY + C2) / (sigmaX_sq + sigmaY_sq + C2)

    structure_map = (sigmaXY + C3) / (sigmaX * sigmaY + C3)

    SSIM_map = luminance_map * contrast_map * structure_map


    # Returns results dictionary for a single channel of RGB
    results = {
        "luminance_map": luminance_map,
        "contrast_map": contrast_map,
        "structure_map": structure_map,
        "ssim_map": SSIM_map,
        "luminance_score": luminance_map.mean(),
        "contrast_score": contrast_map.mean(),
        "structure_score": structure_map.mean(),
        "ssim_score": SSIM_map.mean()
    }

    return results


def ssim_rgb(x, y, win_size=11, data_range=255, sigma=1.5, k1=0.01, k2=0.03, mask_out_background=False, visualize_mask=False, visualize_mask_path=None):

    """
    Compute the structural similarity (SSIM) index for RGB images.

    SSIM is computed independently for each color channel using
    Gaussian-weighted local statistics and then averaged across channels.
    Both per-pixel maps and global mean scores are returned.

    Optionally, SSIM can be recomputed over the union of foreground regions
    extracted from both images.

    Parameters
    ----------
    x : ndarray of shape (H, W, 3)
        First RGB image.
    y : ndarray of shape (H, W, 3)
        Second RGB image.
    win_size : int, optional
        Size of the Gaussian window. Default is 11.
    data_range : float, optional
        Dynamic range of input images. Default is 255.
    sigma : float, optional
        Standard deviation of Gaussian window. Default is 1.5.
    k1 : float, optional
        Stability constant for luminance term. Default is 0.01.
    k2 : float, optional
        Stability constant for contrast/structure terms. Default is 0.03.
    mask_out_background : bool, optional
        If True, compute additional SSIM scores restricted to the
        foreground region determined by color-based masking.
    visualize_mask : bool, optional
        If True, saves a .jpg of the mask using matplotlib. Plot shows
        foreground mask of x on image x, foreground mask of y on image y,
        and the union mask of x and y on both images
    visualize_mask_path : string, optional
        Path string to save mask .jpg

    Returns
    -------
    results : dict
        Dictionary containing:
            - luminance_map
            - contrast_map
            - structure_map
            - ssim_map
            - luminance_score
            - contrast_score
            - structure_score
            - ssim_score

        If `mask_out_background=True`, also includes:
            - foreground_luminance_score
            - foreground_contrast_score
            - foreground_structure_score
            - foreground_ssim_score
    """

    # Make sure images are same shape
    assert x.shape == y.shape
    assert x.ndim == 3 and x.shape[2] == 3

    # Changes to float for filtering
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    # Create kernel
    kernel = gaussian_kernel(win_size, sigma)

    # This results in 3 dictionaries of ssim results, 1 for each channel
    channel_results = []
    for i in range(3):
        channel_results.append(
            ssim_single_channel(x[:, :, i], y[:, :, i], kernel, data_range, k1, k2)
        )
    
    # Average maps across RGB channels -
    # The list comprehension turns each element into (3, H, W), so we average
    # across the first dimension while keeping (H, W) shape intact
    luminance_map = np.mean([map["luminance_map"] for map in channel_results], axis=0)
    contrast_map = np.mean([map["contrast_map"] for map in channel_results], axis=0)
    structure_map = np.mean([map["structure_map"] for map in channel_results], axis=0)
    ssim_map = np.mean([map["ssim_map"] for map in channel_results], axis=0)

    # Return results as a dictionary for the RGB image
    results = {
        "luminance_map": luminance_map,
        "contrast_map": contrast_map,
        "structure_map": structure_map,
        "ssim_map": ssim_map,

        # Average of the 3 per-channel scores
        "luminance_score": np.mean([r["luminance_score"] for r in channel_results]),
        "contrast_score": np.mean([r["contrast_score"] for r in channel_results]),
        "structure_score": np.mean([r["structure_score"] for r in channel_results]),
        "ssim_score": np.mean([r["ssim_score"] for r in channel_results])
    }
    
    if mask_out_background:

        x = x.astype(np.uint8)
        y = y.astype(np.uint8)

        maskX = get_mask(x, foreground=True, threshold=1)
        maskY = get_mask(y, foreground=True, threshold=1)

        mask_fg = maskX | maskY

        adjusted_luminance = luminance_map[mask_fg].mean()
        adjusted_contrast = contrast_map[mask_fg].mean()
        adjusted_structure = structure_map[mask_fg].mean()
        adjusted_ssim = ssim_map[mask_fg].mean()

        # adds adjusted scores to the results dictionary
        results.update({
            "foreground_luminance_score": adjusted_luminance,
            "foreground_contrast_score": adjusted_contrast,
            "foreground_structure_score": adjusted_structure,
            "foreground_ssim_score": adjusted_ssim,
        })

        # Saves visualizations of masks to .jpg
        if visualize_mask:
            x_mask = x.copy()
            y_mask = y.copy()

            x_mask[mask_fg] = [255, 255, 0]
            y_mask[mask_fg] = [255, 255, 0]
            x[maskX] = [255, 255, 0]
            y[maskY] = [255, 255, 0]

            fig, axes = plt.subplots(1, 4, figsize=(14, 7))
            axes[0].imshow(x)
            axes[1].imshow(y)
            axes[2].imshow(x_mask)
            axes[3].imshow(y_mask)

            plt.savefig(visualize_mask_path)

    return results

if __name__ == "__main__":

    img1 = cv2.imread(sys.argv[1])
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    img2 = cv2.imread(sys.argv[2])
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    results = ssim_rgb(img1, img2, mask_out_background=True)

    print(f"Base SSIM Score: {results["ssim_score"]}")
    print(f"Foreground SSIM Score: {results["foreground_ssim_score"]}")
    print("-------------------")
    print(f"Base Luminance Score: {results["luminance_score"]}")
    print(f"Foreground Luminance Score: {results["foreground_luminance_score"]}")
    print("-------------------")
    print(f"Base Contrast Score: {results["contrast_score"]}")
    print(f"Foreground Contrast Score: {results["foreground_contrast_score"]}")
    print("-------------------")
    print(f"Base Structure Score: {results["structure_score"]}")
    print(f"Foreground Structure Score: {results["foreground_structure_score"]}")
    print("-------------------")