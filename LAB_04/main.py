"""
2-D DFT Lab Assignment - Windows Compatible Version
Implements 2-D Discrete Fourier Transform from scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Create output directory if it doesn't exist
OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ============================================================================
# PART 1: Generate 8×8 2-D DFT Basis Functions
# ============================================================================

def generate_2d_dft_basis(N=8):
    """Generate basis functions for N×N 2-D DFT"""
    basis_real = np.zeros((N*N, N, N))
    basis_imag = np.zeros((N*N, N, N))
    basis_magnitude = np.zeros((N*N, N, N))

    idx = 0
    for u in range(N):
        for v in range(N):
            x = np.arange(N)
            y = np.arange(N)
            X, Y = np.meshgrid(x, y)

            exponent = -2j * np.pi * (u * X + v * Y) / N
            basis_func = np.exp(exponent)

            basis_real[idx] = np.real(basis_func)
            basis_imag[idx] = np.imag(basis_func)
            basis_magnitude[idx] = np.abs(basis_func)
            idx += 1

    return basis_real, basis_imag, basis_magnitude


def display_dft_basis(basis_magnitude, N=8):
    """Display all N² basis functions as a single image"""
    full_image = np.zeros((N*N, N*N))

    idx = 0
    for u in range(N):
        for v in range(N):
            row_start = u * N
            col_start = v * N
            full_image[row_start:row_start+N, col_start:col_start+N] = basis_magnitude[idx]
            idx += 1

    plt.figure(figsize=(10, 10))
    plt.imshow(full_image, cmap='gray', interpolation='nearest')
    plt.title(f'{N}×{N} 2-D DFT Basis Functions ({N*N}×{N*N} Image)', fontsize=14)
    plt.xlabel(f'Horizontal Frequency (v)')
    plt.ylabel(f'Vertical Frequency (u)')

    for i in range(N+1):
        plt.axhline(y=i*N-0.5, color='red', linewidth=0.5)
        plt.axvline(x=i*N-0.5, color='red', linewidth=0.5)

    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'part1_dft_basis.png'), dpi=150)
    plt.show()


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
    plt.title(f'Binary Image with Rectangle\nPosition: ({top_left_x}, {top_left_y}), Size: {width}×{height}', fontsize=12)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.colorbar(label='Pixel Value')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'part2_rectangle_image.png'), dpi=150)
    plt.show()

    return image


# ============================================================================
# PART 3: Compute 2-D DFT from Scratch (No FFT)
# ============================================================================

def compute_2d_dft(image):
    """Compute 2-D DFT without using any built-in FFT/DFT functions"""
    N, M = image.shape
    dft = np.zeros((N, M), dtype=complex)

    for u in range(N):
        for v in range(M):
            sum_val = 0
            for x in range(N):
                for y in range(M):
                    exponent = -2j * np.pi * (u * x / N + v * y / M)
                    kernel = np.exp(exponent)
                    sum_val += image[x, y] * kernel
            dft[u, v] = sum_val

    return dft


def plot_dft_spectrum(dft, title_prefix="", save_name="dft"):
    """Plot magnitude and phase spectrum of 2-D DFT"""
    magnitude = np.abs(dft)
    phase = np.angle(dft)
    log_magnitude = np.log1p(magnitude)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Magnitude (Linear)
    im1 = axes[0, 0].imshow(magnitude, cmap='hot', interpolation='nearest')
    axes[0, 0].set_title(f'{title_prefix}Magnitude (Linear)', fontsize=12)
    axes[0, 0].set_xlabel('Frequency v')
    axes[0, 0].set_ylabel('Frequency u')
    plt.colorbar(im1, ax=axes[0, 0])

    # Magnitude (Log)
    im2 = axes[0, 1].imshow(log_magnitude, cmap='hot', interpolation='nearest')
    axes[0, 1].set_title(f'{title_prefix}Magnitude (Log)', fontsize=12)
    axes[0, 1].set_xlabel('Frequency v')
    axes[0, 1].set_ylabel('Frequency u')
    plt.colorbar(im2, ax=axes[0, 1])

    # Phase
    im3 = axes[1, 0].imshow(phase, cmap='twilight', interpolation='nearest')
    axes[1, 0].set_title(f'{title_prefix}Phase', fontsize=12)
    axes[1, 0].set_xlabel('Frequency v')
    axes[1, 0].set_ylabel('Frequency u')
    plt.colorbar(im3, ax=axes[1, 0], label='Radians')

    # 3D Magnitude
    ax4 = fig.add_subplot(224, projection='3d')
    N, M = dft.shape
    X, Y = np.meshgrid(np.arange(M), np.arange(N))
    ax4.plot_surface(X, Y, log_magnitude, cmap='hot', edgecolor='none')
    ax4.set_title(f'{title_prefix}3D Magnitude', fontsize=12)
    ax4.set_xlabel('Frequency v')
    ax4.set_ylabel('Frequency u')
    ax4.set_zlabel('Log Magnitude')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{save_name}_spectrum.png'), dpi=150)
    plt.show()


# ============================================================================
# PART 4: Centered Image DFT
# ============================================================================

def center_image(image):
    """Center image by multiplying with (-1)^(x+y)"""
    N, M = image.shape
    x = np.arange(N)
    y = np.arange(M)
    X, Y = np.meshgrid(y, x)
    multiplier = (-1) ** (X + Y)
    centered = image * multiplier

    plt.figure(figsize=(8, 8))
    plt.imshow(centered, cmap='gray', interpolation='nearest')
    plt.title('Centered Image (multiplied by (-1)^(x+y))', fontsize=12)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.colorbar(label='Pixel Value')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'part4_centered_image.png'), dpi=150)
    plt.show()

    return centered


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("\n2-D DFT Lab Assignment\n")

    # Part 1: DFT Basis
    print("Part 1: Generating 8×8 DFT basis...")
    basis_real, basis_imag, basis_magnitude = generate_2d_dft_basis(N=8)
    display_dft_basis(basis_magnitude, N=8)
    print("✓ Done\n")

    # Part 2: Rectangle Image
    print("Part 2: Creating rectangle image...")
    top_left_x = int(input("  Top-left X-coordinate: "))
    top_left_y = int(input("  Top-left Y-coordinate: "))
    width = int(input("  Width (pixels): "))
    height = int(input("  Height (pixels): "))

    rect_image = create_rectangle_image(top_left_x, top_left_y, width, height, 64)
    print("✓ Done\n")

    # Part 3: DFT of Original Image
    print("Part 3: Computing 2-D DFT of original image...")
    dft_original = compute_2d_dft(rect_image)
    plot_dft_spectrum(dft_original, "Original - ", "part3_original")
    print("✓ Done\n")

    # Part 4: DFT of Centered Image
    print("Part 4: Computing 2-D DFT of centered image...")
    centered_image = center_image(rect_image)
    dft_centered = compute_2d_dft(centered_image)
    plot_dft_spectrum(dft_centered, "Centered - ", "part4_centered")
    print("✓ Done\n")

    print(f"All outputs1 saved to '{OUTPUT_DIR}' folder\n")


if __name__ == "__main__":
    main()