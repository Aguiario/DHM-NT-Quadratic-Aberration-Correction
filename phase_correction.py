# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from scipy.optimize import differential_evolution

# Main function to perform automatic compensation
def phase_correction(file_name, Lambda, dx, dy, generations, population_size):
    # Load and display the hologram image
    holo = imageio.imread(file_name) 
    plt.figure()
    plt.imshow(holo, cmap='gray')
    plt.title('Hologram')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Get dimensions of the hologram
    M_holo, N_holo = holo.shape
    # Create a meshgrid of pixel coordinates
    X_holo, Y_holo = np.meshgrid(np.arange(-N_holo//2, N_holo//2), np.arange(-M_holo//2, M_holo//2))

    # Define the wavenumber
    k = 2 * np.pi / Lambda

    # Perform Fourier Transform
    FT_holo = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(holo)))

    # Calculate intensity and remove DC term
    I_holo = np.sqrt(np.abs(FT_holo))
    px = 30
    I_holo[M_holo//2 - px:M_holo//2 + px, N_holo//2 - px:N_holo//2 + px] = 0  # Zero intensity around the center

    # Normalize intensity for visualization
    I_holo = 255 * (I_holo - np.min(I_holo)) / (np.max(I_holo) - np.min(I_holo))

    # Binarize image using Otsu's threshold
    hist, bins = np.histogram(I_holo, bins=16)
    threshold = threshold_otsu(bins)
    BW_holo = np.where(I_holo > threshold, 1, 0)

    # Isolate diffraction orders
    cc = label(BW_holo, connectivity=1)
    numPixels = np.array([len(cc[cc == i]) for i in range(1, cc.max() + 1)])
    max_index = np.unravel_index(numPixels.argmax(), numPixels.shape)
    numPixels[max_index] = 0
    second_max_index = np.unravel_index(numPixels.argmax(), numPixels.shape)

    # Filter out non-relevant regions
    M_bw, N_bw = BW_holo.shape
    for i in range(M_bw):
        for j in range(N_bw):
            if cc[i, j] != max_index[0] + 1 and cc[i, j] != second_max_index[0] + 1:
                BW_holo[i, j] = 0
                cc[i, j] = 0

    terms = regionprops(cc)
    plus1 = terms[0].bbox
    plus_coor_bw = [(plus1[0] + plus1[2]) / 2, (plus1[1] + plus1[3]) / 2]

    # Visualize +1 diffraction order
    fig, ax = plt.subplots()
    ax.imshow(BW_holo, cmap='gray')
    rect = Rectangle((plus1[1], plus1[0]), np.abs(plus1[1] - plus1[3]), np.abs(plus1[0] - plus1[2]), linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.title('+1 Diffraction Order Location')
    plt.show()

    # Calculate dimensions and coordinates
    m_bw = np.abs(plus1[0] - plus1[2])
    n_bw = np.abs(plus1[1] - plus1[3])

    # Compensate the tilting angle
    M_holoCompensate, N_holoCompensate = FT_holo.shape
    Filter = np.zeros((M_holoCompensate, N_holoCompensate))
    Filter[int(plus_coor_bw[0] - m_bw):int(plus_coor_bw[0] + m_bw), int(plus_coor_bw[1] - n_bw):int(plus_coor_bw[1] + n_bw)] = 1
    FT_FilteringH = FT_holo * Filter

    # Calculate angles
    ThetaXM_holoCompensate = np.arcsin((M_holoCompensate / 2 - plus_coor_bw[0]) * Lambda / (M_holoCompensate * dx))
    ThetaYM_holoCompensate = np.arcsin((N_holoCompensate / 2 - plus_coor_bw[1]) * Lambda / (N_holoCompensate * dy))
    Reference_holoCompensate = np.exp(1j * k * (np.sin(ThetaXM_holoCompensate) * X_holo * dx + np.sin(ThetaYM_holoCompensate) * Y_holo * dy))
    holo_filtered_holoCompensate = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(FT_FilteringH)))
    holoCompensate = holo_filtered_holoCompensate * Reference_holoCompensate

    # Binarized spherical aberration
    R_SA = np.real(holoCompensate)
    mi_SA = np.min(R_SA)
    mx_SA = np.max(R_SA)
    R_SA = 255 * (R_SA - mi_SA) / (mx_SA - mi_SA)
    threshold_SA = threshold_otsu(R_SA)
    BW_holo_SA = np.where(R_SA > threshold_SA, 1, 0)

    

    # Segment and filter regions for compensation
    cc = label(BW_holo_SA, connectivity=1)
    numPixels = np.array([len(cc[cc == i]) for i in range(1, cc.max() + 1)])
    max_index = np.unravel_index(numPixels.argmax(), numPixels.shape)
    numPixels[max_index] = 0
    second_max_index = np.unravel_index(numPixels.argmax(), numPixels.shape)
    numPixels[second_max_index] = 0
    third_max_index = np.unravel_index(numPixels.argmax(), numPixels.shape)

    M_BW_holo_SA, N_BW_holo_SA = BW_holo_SA.shape
    for i in range(M_BW_holo_SA):
        for j in range(N_BW_holo_SA):
            if cc[i, j] != max_index[0] + 1 and cc[i, j] != second_max_index[0] + 1 and cc[i, j] != third_max_index[0] + 1:
                BW_holo_SA[i, j] = 0
                cc[i, j] = 0

    terms = regionprops(BW_holo_SA)
    best_term_idx = np.argmin([term.eccentricity for term in terms])
    best_term = terms[best_term_idx]
    block_center = ((best_term.bbox[1] + best_term.bbox[3]) / 2, (best_term.bbox[0] + best_term.bbox[2]) / 2)
    true_center = (N_BW_holo_SA / 2, M_BW_holo_SA / 2)
    g, h = (np.array(block_center) - np.array(true_center)) / 1

    # Visualize remaining spherical phase factor
    fig, ax = plt.subplots()
    ax.imshow(BW_holo_SA, cmap='gray')
    rect = Rectangle((best_term.bbox[1], best_term.bbox[0]), np.abs(best_term.bbox[3] - best_term.bbox[1]), np.abs(best_term.bbox[2] - best_term.bbox[0]), linewidth=3, edgecolor='r', facecolor='none')
    ax.scatter(block_center[0], block_center[1], color='red', marker='x', s=100)
    ax.scatter(true_center[0], true_center[1], color='blue', marker='x', s=100)
    ax.add_patch(rect)
    plt.title('Remaining Spherical Phase Factor Location')
    plt.show(block=False)

    # Define new reference wave for compensation
    Cx = np.power((M_holo * dx), 2) / (Lambda * m_bw)
    Cy = np.power((N_holo * dy), 2) / (Lambda * n_bw)
    cur = (Cx + Cy) / 2
    sign = True
    phi_spherical = ((np.pi / Lambda) * ((((X_holo - g)**2) * (dx**2) / cur) + (((Y_holo - h)**2) * (dy**2) / cur)))
    phase_mask = np.exp((1j) * phi_spherical) if sign else np.exp((-1j) * phi_spherical)
    corrected_image = holoCompensate * phase_mask

    # Visualize phase compensation
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(np.angle(phase_mask))
    ax1.set_title('Phase Mask')
    ax2.imshow(np.angle(holoCompensate))
    ax2.set_title('HoloCompensate')
    ax3.imshow(np.angle(corrected_image))
    ax3.set_title('Corrected Image')
    plt.show()

    # Show the final corrected image
    plt.figure()
    plt.imshow(np.angle(corrected_image), cmap='gray')
    plt.title('Non-optimized Compensated Image')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Define spherical phase compensation function
    phi_spherical_C_opt = lambda C: ((np.pi / Lambda) * ((((X_holo - g)**2) * (dx**2) / C) + (((Y_holo - h)**2) * (dy**2) / C)))

    # Define binary cost function
    minfunc = lambda t: M_holo * N_holo - np.sum(np.where(np.angle(holoCompensate * (np.exp(1j * phi_spherical_C_opt(t)) if sign else np.exp(-1j * phi_spherical_C_opt(t)))) + np.pi > 0.5, 1, 0))

    # Determine the optimal parameter for accurate phase compensation
    # Define lower and upper bounds for optimization
    lb = -0.5
    ub = 0.5
    lb = cur + cur * lb
    ub = cur + cur * ub

    """Here goes the heuristic method"""
    # Global optimization using differential evolution
    bounds = [(lb, ub)]

    # Define the cost function
    result = differential_evolution(minfunc, bounds, strategy='best1bin', maxiter=generations, popsize=population_size, tol=1e-6, mutation=(0.5, 1.5), recombination=0.7)

    # Extract the optimized value
    Cy_opt_heuristic = result.x[0]

    phi_spherical = phi_spherical_C_opt(Cy_opt_heuristic)

    print('Optimized C: ', Cy_opt_heuristic)

    """til here"""
    # Apply the optimized phase compensation
    phase_mask = np.exp((1j) * phi_spherical) if sign else np.exp((-1j) * phi_spherical)
    corrected_image = holoCompensate * phase_mask  # Final compensated image

    plt.figure(); plt.imshow(np.angle(corrected_image), cmap='gray'); plt.title('Compensated imaged after optimization'); 
    plt.gca().set_aspect('equal', adjustable='box'); plt.show()

    return corrected_image