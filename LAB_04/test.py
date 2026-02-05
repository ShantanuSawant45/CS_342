"""
2-D DFT Lab Assignment - Fast & Efficient Version
Implements 2-D Discrete Fourier Transform from scratch (NO FFT)
Optimized for speed without excessive memory usage
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time

# Create output directory if it doesn't exist
OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ============================================================================
# PART 1: Generate 8×8 2-D DFT Basis Functions
# ============================================================================

def generate_2d_dft_basis(N=8):
    """Generate basis functions for N×N 2-D DFT"""
    basis_magnitude = np.zeros((N * N, N, N))

    x, y = np.meshgrid(np.arange(N), np.arange(N))

    idx = 0
    for u in range(N):
        for v in range(N):
            exponent = -2j * np.pi * (u * x + v * y) / N
            basis_magnitude[idx] = np.abs(np.exp(exponent))
            idx += 1

    return basis_magnitude


def display_dft_basis(basis_magnitude, N=8):
    """Display all N² basis functions as a single image"""
    full_image = np.zeros((N * N, N * N))

    idx = 0
    for u in range(N):
        for v in range(N):
            row_start = u * N
            col_start = v * N
            full_image[row_start:row_start + N, col_start:col_start + N] = basis_magnitude[idx]
            idx += 1

    plt.figure(figsize=(10, 10))
    plt.imshow(full_image, cmap='gray', interpolation='nearest')
    plt.title(f'{N}×{N} 2-D DFT Basis Functions ({N * N}×{N * N} Image)', fontsize=14)
    plt.xlabel(f'Horizontal Frequency (v)')
    plt.ylabel(f'Vertical Frequency (u)')

    for i in range(N + 1):
        plt.axhline(y=i * N - 0.5, color='red', linewidth=0.5)
        plt.axvline(x=i * N - 0.5, color='red', linewidth=0.5)

    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'part1_dft_basis.png'), dpi=150)
    plt.close()  # Close instead of show to avoid blocking


# ============================================================================
# PART 2: Create Binary 64×64 Image with Rectangle
# ============================================================================

def create_rectangle_image(top_left_x, top_left_y, width, height, image_size=64):
    """Create binary image with rectangle"""
    image = np.zeros((image_size, image_size))

    x1, y1 = max(0, top_left_x), max(0, top_left_y)
    x2, y2 = min(x1 + width, image_size), min(y1 + height, image_size)

    image[y1:y2, x1:x2] = 1

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.title(f'Binary Image with Rectangle\nPosition: ({top_left_x}, {top_left_y}), Size: {width}×{height}',
              fontsize=12)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.colorbar(label='Pixel Value')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'part2_rectangle_image.png'), dpi=150)
    plt.close()

    return image


# ============================================================================
# PART 3: Compute 2-D DFT from Scratch - FAST VERSION
# ============================================================================

def compute_2d_dft_fast(image):
    """
    Fast 2-D DFT computation without FFT
    Uses row-column decomposition: DFT_2D = DFT_rows × DFT_cols
    This is MUCH faster: O(N³) instead of O(N⁴)
    """
    N, M = image.shape

    # Create DFT kernel matrix for 1-D DFT
    n = np.arange(N)
    k = n.reshape((N, 1))
    kernel_N = np.exp(-2j * np.pi * k * n / N)

    m = np.arange(M)
    l = m.reshape((M, 1))
    kernel_M = np.exp(-2j * np.pi * l * m / M)

    # Apply 1-D DFT to each row
    dft_rows = np.dot(kernel_N, image)

    # Apply 1-D DFT to each column of the result
    dft = np.dot(dft_rows, kernel_M.T)

    return dft


def plot_dft_spectrum(dft, title_prefix="", save_name="dft"):
    """Plot magnitude and phase spectrum of 2-D DFT"""
    magnitude = np.abs(dft)
    phase = np.angle(dft)
    log_magnitude = np.log1p(magnitude)

    fig = plt.figure(figsize=(14, 12))

    # Magnitude (Linear)
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(magnitude, cmap='gray', interpolation='nearest')
    ax1.set_title(f'{title_prefix}Magnitude (Linear)', fontsize=12)
    ax1.set_xlabel('Frequency v')
    ax1.set_ylabel('Frequency u')
    plt.colorbar(im1, ax=ax1)

    # Magnitude (Log)
    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(log_magnitude, cmap='gray', interpolation='nearest')
    ax2.set_title(f'{title_prefix}Magnitude (Log)', fontsize=12)
    ax2.set_xlabel('Frequency v')
    ax2.set_ylabel('Frequency u')
    plt.colorbar(im2, ax=ax2)

    # Phase
    ax3 = plt.subplot(2, 2, 3)
    im3 = ax3.imshow(phase, cmap='gray', interpolation='nearest')
    ax3.set_title(f'{title_prefix}Phase', fontsize=12)
    ax3.set_xlabel('Frequency v')
    ax3.set_ylabel('Frequency u')
    plt.colorbar(im3, ax=ax3, label='Radians')

    # 3D Magnitude
    ax4 = plt.subplot(2, 2, 4, projection='3d')
    N, M = dft.shape
    X, Y = np.meshgrid(np.arange(M), np.arange(N))
    ax4.plot_surface(X, Y, log_magnitude, cmap='gray', edgecolor='none', alpha=0.8)
    ax4.set_title(f'{title_prefix}3D Magnitude', fontsize=12)
    ax4.set_xlabel('Frequency v')
    ax4.set_ylabel('Frequency u')
    ax4.set_zlabel('Log Magnitude')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{save_name}_spectrum.png'), dpi=150)
    plt.close()


# ============================================================================
# PART 4: Centered Image DFT
# ============================================================================

def center_image(image):
    """Center image by multiplying with (-1)^(x+y)"""
    N, M = image.shape
    x, y = np.meshgrid(np.arange(M), np.arange(N))
    multiplier = (-1) ** (x + y)
    centered = image * multiplier

    plt.figure(figsize=(8, 8))
    plt.imshow(centered, cmap='gray', interpolation='nearest')
    plt.title('Centered Image (multiplied by (-1)^(x+y))', fontsize=12)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.colorbar(label='Pixel Value')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'part4_centered_image.png'), dpi=150)
    plt.close()

    return centered


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("2-D DFT Lab Assignment - Fast Implementation")
    print("=" * 60 + "\n")

    # Part 1: DFT Basis
    print("Part 1: Generating 8×8 DFT basis...", end=' ', flush=True)
    start = time.time()
    basis_magnitude = generate_2d_dft_basis(N=8)
    display_dft_basis(basis_magnitude, N=8)
    print(f"Done! ({time.time() - start:.2f}s)\n")

    # Part 2: Rectangle Image
    print("Part 2: Creating rectangle image...")
    top_left_x = int(input("  Top-left X-coordinate: "))
    top_left_y = int(input("  Top-left Y-coordinate: "))
    width = int(input("  Width (pixels): "))
    height = int(input("  Height (pixels): "))

    start = time.time()
    rect_image = create_rectangle_image(top_left_x, top_left_y, width, height, 64)
    print(f"  Done! ({time.time() - start:.2f}s)\n")

    # Part 3: DFT of Original Image
    print("Part 3: Computing 2-D DFT of original image...", end=' ', flush=True)
    start = time.time()
    dft_original = compute_2d_dft_fast(rect_image)
    print(f"Done! ({time.time() - start:.2f}s)")

    print("  Generating spectrum plots...", end=' ', flush=True)
    start = time.time()
    plot_dft_spectrum(dft_original, "Original - ", "part3_original")
    print(f"Done! ({time.time() - start:.2f}s)\n")

    # Part 4: DFT of Centered Image
    print("Part 4: Computing centered image...", end=' ', flush=True)
    start = time.time()
    centered_image = center_image(rect_image)
    print(f"Done! ({time.time() - start:.2f}s)")

    print("  Computing 2-D DFT of centered image...", end=' ', flush=True)
    start = time.time()
    dft_centered = compute_2d_dft_fast(centered_image)
    print(f"Done! ({time.time() - start:.2f}s)")

    print("  Generating spectrum plots...", end=' ', flush=True)
    start = time.time()
    plot_dft_spectrum(dft_centered, "Centered - ", "part4_centered")
    print(f"Done! ({time.time() - start:.2f}s)\n")

    print("=" * 60)
    print(f"✓ All outputs saved to '{OUTPUT_DIR}' folder")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()