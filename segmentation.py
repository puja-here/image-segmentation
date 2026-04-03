"""
=============================================================
  PSO-Based Image Segmentation
  Final Year Project — Beginner Friendly Version
=============================================================
  Algorithm  : Particle Swarm Optimization (PSO)
  Method     : Multi-level Otsu Thresholding via PSO
  Metrics    : PSNR, SSIM, Fitness Value
  Usage      : Replace IMAGE_PATH with your image file path
=============================================================
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import time

# ─────────────────────────────────────────────
#  CONFIGURATION — Change these as needed
# ─────────────────────────────────────────────

# IMAGE_PATH      = "C:\Users\USER\Downloads\pexels-cubir-128756.jpg"   # <-- Replace with your image path
IMAGE_PATH = "C:\\Users\\USER\\Downloads\\pexels-cubir-128756.jpg"
NUM_THRESHOLDS  = 3                  # Number of segmentation levels (try 2, 3, or 4)
SWARM_SIZE      = 30                 # Number of particles (more = slower but better)
MAX_ITERATIONS  = 100                # How many PSO rounds to run
W               = 0.7               # Inertia weight (controls particle momentum)
C1              = 1.5               # Cognitive coefficient (trust in personal best)
C2              = 1.5               # Social coefficient (trust in global best)

# ─────────────────────────────────────────────
#  STEP 1: Load and Prepare the Image
# ─────────────────────────────────────────────

def load_image(path):
    """Load image, convert to grayscale, and return both versions."""
    img_color = cv2.imread(path)
    if img_color is None:
        raise FileNotFoundError(f"Image not found at: {path}\nPlease check the file path.")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    print(f"[OK] Image loaded: {img_gray.shape[1]}x{img_gray.shape[0]} pixels")
    return img_color, img_gray


# ─────────────────────────────────────────────
#  STEP 2: Fitness Function (Otsu's Criterion)
# ─────────────────────────────────────────────

def otsu_fitness(thresholds, histogram, total_pixels):
    """
    Compute Otsu's between-class variance for a given set of thresholds.
    PSO will MAXIMIZE this value to find the best thresholds.

    How it works:
    - Split the image histogram into regions using the thresholds
    - Compute the weighted variance between each region
    - Higher variance = better separation between segments
    """
    thresholds = sorted([int(t) for t in thresholds])
    thresholds = [0] + thresholds + [256]  # Add boundaries

    between_class_variance = 0.0

    for i in range(len(thresholds) - 1):
        low  = thresholds[i]
        high = thresholds[i + 1]

        # Get histogram slice for this region
        region = histogram[low:high]
        weight = np.sum(region) / total_pixels  # Proportion of pixels in region

        if weight == 0:
            continue

        # Mean intensity of this region
        intensity_values = np.arange(low, high)
        mean = np.sum(intensity_values * region) / (np.sum(region) + 1e-10)

        between_class_variance += weight * (mean ** 2)

    return between_class_variance


# ─────────────────────────────────────────────
#  STEP 3: The PSO Algorithm
# ─────────────────────────────────────────────

def run_pso(histogram, total_pixels, num_thresholds,
            swarm_size, max_iter, w, c1, c2):
    """
    Particle Swarm Optimization to find the best image thresholds.

    Each particle = one candidate set of threshold values
    Particles fly through the search space, guided by:
        - Their own best position found so far (personal best)
        - The best position found by any particle (global best)
    """

    print(f"\n[PSO] Starting with {swarm_size} particles, {max_iter} iterations...")
    print(f"[PSO] Searching for {num_thresholds} optimal thresholds (0–255)\n")

    # --- Initialize Particles Randomly ---
    # Each particle is a set of threshold values between 1 and 254
    positions  = np.random.randint(1, 255, size=(swarm_size, num_thresholds)).astype(float)
    velocities = np.random.uniform(-10, 10, size=(swarm_size, num_thresholds))

    # Personal best = each particle's own best position
    personal_best_pos = positions.copy()
    personal_best_fit = np.array([
        otsu_fitness(p, histogram, total_pixels) for p in positions
    ])

    # Global best = best position found across ALL particles
    global_best_idx = np.argmax(personal_best_fit)
    global_best_pos = personal_best_pos[global_best_idx].copy()
    global_best_fit = personal_best_fit[global_best_idx]

    fitness_history = []  # Track progress over iterations

    # --- PSO Main Loop ---
    for iteration in range(max_iter):

        for i in range(swarm_size):
            # Random coefficients for exploration
            r1 = np.random.random(num_thresholds)  # Random for personal best
            r2 = np.random.random(num_thresholds)  # Random for global best

            # Update velocity:
            # New velocity = inertia + cognitive pull + social pull
            velocities[i] = (
                w  * velocities[i]                                  # Inertia (keep moving)
                + c1 * r1 * (personal_best_pos[i] - positions[i])  # Pull toward own best
                + c2 * r2 * (global_best_pos     - positions[i])   # Pull toward swarm best
            )

            # Update position (move the particle)
            positions[i] += velocities[i]

            # Keep thresholds within valid range [1, 254]
            positions[i] = np.clip(positions[i], 1, 254)

            # Evaluate new position
            fitness = otsu_fitness(positions[i], histogram, total_pixels)

            # Update personal best if this position is better
            if fitness > personal_best_fit[i]:
                personal_best_fit[i] = fitness
                personal_best_pos[i] = positions[i].copy()

            # Update global best if this is the best across all particles
            if fitness > global_best_fit:
                global_best_fit = fitness
                global_best_pos = positions[i].copy()

        fitness_history.append(global_best_fit)

        # Print progress every 20 iterations
        if (iteration + 1) % 20 == 0:
            thresholds_so_far = sorted([int(t) for t in global_best_pos])
            print(f"  Iteration {iteration+1:3d}/{max_iter} | "
                  f"Best Fitness: {global_best_fit:.4f} | "
                  f"Thresholds: {thresholds_so_far}")

    # Final optimal thresholds
    optimal_thresholds = sorted([int(t) for t in global_best_pos])
    print(f"\n[PSO Done] Optimal Thresholds Found: {optimal_thresholds}")
    print(f"[PSO Done] Best Fitness Value: {global_best_fit:.4f}")

    return optimal_thresholds, global_best_fit, fitness_history


# ─────────────────────────────────────────────
#  STEP 4: Apply Thresholds to Segment Image
# ─────────────────────────────────────────────

def apply_thresholds(img_gray, thresholds):
    """
    Divide the grayscale image into regions using the threshold values.
    Each region gets a different intensity value (0, 85, 170, 255, etc.)
    """
    segmented = np.zeros_like(img_gray)
    levels = [0] + thresholds + [256]
    num_levels = len(levels) - 1

    # Assign equal-spaced intensity values to each segment
    fill_values = np.linspace(0, 255, num_levels, dtype=np.uint8)

    for i in range(num_levels):
        mask = (img_gray >= levels[i]) & (img_gray < levels[i + 1])
        segmented[mask] = fill_values[i]

    return segmented


# ─────────────────────────────────────────────
#  STEP 5: Evaluate Results (PSNR & SSIM)
# ─────────────────────────────────────────────

def evaluate(original, segmented):
    """
    Measure how good the segmentation is using two standard metrics:
    - PSNR: Peak Signal-to-Noise Ratio (higher = better quality)
    - SSIM: Structural Similarity Index (0 to 1, higher = more similar)
    """
    psnr_value = psnr(original, segmented, data_range=255)
    ssim_value = ssim(original, segmented, data_range=255)
    return psnr_value, ssim_value


# ─────────────────────────────────────────────
#  STEP 6: Visualize Results
# ─────────────────────────────────────────────

def visualize_results(img_color, img_gray, segmented_pso,
                       otsu_thresh, thresholds, fitness_history,
                       psnr_pso, ssim_pso, psnr_otsu, ssim_otsu):
    """Display original, Otsu baseline, and PSO result side by side."""

    # Otsu's single threshold (baseline)
    _, otsu_result = cv2.threshold(img_gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("PSO-Based Image Segmentation", fontsize=16, fontweight='bold', y=1.01)

    # 1. Original Image
    axes[0, 0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image", fontsize=12)
    axes[0, 0].axis('off')

    # 2. Grayscale Image
    axes[0, 1].imshow(img_gray, cmap='gray')
    axes[0, 1].set_title("Grayscale Image", fontsize=12)
    axes[0, 1].axis('off')

    # 3. Image Histogram with PSO thresholds marked
    axes[0, 2].plot(cv2.calcHist([img_gray], [0], None, [256], [0, 256]),
                    color='steelblue', linewidth=1.2)
    for t in thresholds:
        axes[0, 2].axvline(x=t, color='red', linestyle='--', linewidth=1.5,
                           label=f'Threshold: {t}')
    axes[0, 2].set_title("Histogram with PSO Thresholds", fontsize=12)
    axes[0, 2].set_xlabel("Pixel Intensity")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].legend(fontsize=8)

    # 4. Otsu's Baseline Result
    axes[1, 0].imshow(otsu_result, cmap='gray')
    axes[1, 0].set_title(
        f"Otsu's Thresholding (Baseline)\nPSNR: {psnr_otsu:.2f} dB | SSIM: {ssim_otsu:.4f}",
        fontsize=10)
    axes[1, 0].axis('off')

    # 5. PSO Segmentation Result
    axes[1, 1].imshow(segmented_pso, cmap='gray')
    axes[1, 1].set_title(
        f"PSO Segmentation ({len(thresholds)} thresholds)\nPSNR: {psnr_pso:.2f} dB | SSIM: {ssim_pso:.4f}",
        fontsize=10)
    axes[1, 1].axis('off')

    # 6. PSO Convergence Curve
    axes[1, 2].plot(fitness_history, color='darkorange', linewidth=2)
    axes[1, 2].set_title("PSO Convergence Curve", fontsize=12)
    axes[1, 2].set_xlabel("Iteration")
    axes[1, 2].set_ylabel("Fitness Value")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pso_segmentation_result.png", dpi=150, bbox_inches='tight')
    print("\n[Saved] Result image saved as: pso_segmentation_result.png")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN — Run Everything
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("   PSO-Based Image Segmentation")
    print("=" * 55)

    # Step 1: Load image
    img_color, img_gray = load_image(IMAGE_PATH)

    # Step 2: Compute histogram
    histogram = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    total_pixels = img_gray.size

    # Step 3: Run PSO
    start_time = time.time()
    optimal_thresholds, best_fitness, fitness_history = run_pso(
        histogram, total_pixels,
        num_thresholds=NUM_THRESHOLDS,
        swarm_size=SWARM_SIZE,
        max_iter=MAX_ITERATIONS,
        w=W, c1=C1, c2=C2
    )
    elapsed = time.time() - start_time

    # Step 4: Apply thresholds to segment the image
    segmented_pso = apply_thresholds(img_gray, optimal_thresholds)

    # Step 5: Otsu's baseline (single threshold for comparison)
    otsu_val, otsu_result = cv2.threshold(img_gray, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 6: Evaluate both methods
    psnr_pso,  ssim_pso  = evaluate(img_gray, segmented_pso)
    psnr_otsu, ssim_otsu = evaluate(img_gray, otsu_result)

    # Step 7: Print summary
    print("\n" + "=" * 55)
    print("   RESULTS SUMMARY")
    print("=" * 55)
    print(f"  Optimal Thresholds (PSO) : {optimal_thresholds}")
    print(f"  Otsu's Threshold         : {int(otsu_val)}")
    print(f"  PSO Fitness Value        : {best_fitness:.4f}")
    print(f"  Time Taken               : {elapsed:.2f} seconds")
    print()
    print(f"  {'Metric':<12} {'PSO':>12} {'Otsu (baseline)':>18}")
    print(f"  {'-'*44}")
    print(f"  {'PSNR (dB)':<12} {psnr_pso:>12.4f} {psnr_otsu:>18.4f}")
    print(f"  {'SSIM':<12} {ssim_pso:>12.4f} {ssim_otsu:>18.4f}")
    print("=" * 55)

    if psnr_pso > psnr_otsu:
        print("  [RESULT] PSO achieved BETTER PSNR than Otsu's method!")
    else:
        print("  [RESULT] Otsu's method achieved better PSNR (try more particles/iterations).")

    # Step 8: Visualize
    visualize_results(img_color, img_gray, segmented_pso,
                      otsu_val, optimal_thresholds, fitness_history,
                      psnr_pso, ssim_pso, psnr_otsu, ssim_otsu)

    # Step 9: Save segmented image
    cv2.imwrite("segmented_output.png", segmented_pso)
    print("[Saved] Segmented image saved as: segmented_output.png")


if __name__ == "__main__":
    main()