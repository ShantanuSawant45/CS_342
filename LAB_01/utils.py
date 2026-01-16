
import numpy as np
import cv2


def crop_to_square(img):
    height, width = img.shape
    min_dim = min(height, width)
    start_y = (height - min_dim) // 2
    start_x = (width - min_dim) // 2
    square_img = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    return square_img


def spatial_sampling(img, resolution):
    return cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)


def quantize_image(img, bit_depth):
    levels = 2 ** bit_depth
    quantized = np.floor(img / 256 * levels)
    quantized = np.clip(quantized, 0, levels - 1).astype(np.uint8)
    quantized_display = (quantized / (levels - 1) * 255).astype(np.uint8)
    return quantized, quantized_display


def pack_pixels(img_flat, bit_depth):
    pixels_per_byte = 8 // bit_depth
    packed = []
    for i in range(0, len(img_flat), pixels_per_byte):
        byte_val = 0
        for j in range(pixels_per_byte):
            if i + j < len(img_flat):
                byte_val = (byte_val << bit_depth) | int(img_flat[i + j])
            else:
                byte_val = byte_val << bit_depth
        packed.append(byte_val)
    return bytes(packed)


def unpack_pixels(packed_data, bit_depth, total_pixels):
    pixels_per_byte = 8 // bit_depth
    mask = (1 << bit_depth) - 1  # Create bit mask
    unpacked = []
    for byte_val in packed_data:
        for j in range(pixels_per_byte):
            shift = (pixels_per_byte - 1 - j) * bit_depth
            pixel = (byte_val >> shift) & mask
            unpacked.append(pixel)
            if len(unpacked) >= total_pixels:
                break
        if len(unpacked) >= total_pixels:
            break
    return np.array(unpacked[:total_pixels], dtype=np.uint8)


def create_header(spatial_idx, intensity_idx):
    header = (spatial_idx << 2) | intensity_idx
    return header


def parse_header(header_byte):
    spatial_idx = (header_byte >> 2) & 0b11
    intensity_idx = header_byte & 0b11
    return spatial_idx, intensity_idx



SPATIAL_MAP = {100: 0, 200: 1, 400: 2, 800: 3}
SPATIAL_REVERSE_MAP = {0: 100, 1: 200, 2: 400, 3: 800}

INTENSITY_MAP = {1: 0, 2: 1, 4: 2, 8: 3}
INTENSITY_REVERSE_MAP = {0: 1, 1: 2, 2: 4, 3: 8}