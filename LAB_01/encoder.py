
import numpy as np
import cv2
from utils import (crop_to_square, spatial_sampling, quantize_image,
                   pack_pixels, create_header, SPATIAL_MAP, INTENSITY_MAP)


def encode_image(img, spatial_res, bit_depth, output_file):
    print("Step 1: Cropping to square...")
    square_img = crop_to_square(img)

    print(f"Step 2: Resizing to {spatial_res}x{spatial_res}...")
    resized_img = spatial_sampling(square_img, spatial_res)

    print(f"Step 3: Quantizing to {bit_depth} bits...")
    quantized_data, quantized_display = quantize_image(resized_img, bit_depth)

    print("Step 4: Creating header...")
    spatial_idx = SPATIAL_MAP[spatial_res]
    intensity_idx = INTENSITY_MAP[bit_depth]
    header = create_header(spatial_idx, intensity_idx)

    print("Step 5: Packing and saving to file...")
    with open(output_file, 'wb') as f:
        f.write(header.to_bytes(1, byteorder='big'))
        if bit_depth < 8:
            flat_data = quantized_data.flatten()
            packed_data = pack_pixels(flat_data, bit_depth)
            f.write(packed_data)
        else:
            f.write(quantized_data.tobytes())

    import os
    file_size = os.path.getsize(output_file)
    print(f"Encoding complete! File size: {file_size} bytes")
    return quantized_display, file_size

def encode_image_verbose(img_path, spatial_res, bit_depth, output_file):
    try:
        # Load image
        print(f"Loading image from {img_path}...")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Error: Could not load image!")
            return False, None

        print(f"Original image size: {img.shape}")

        # Encode
        encoded_img, file_size = encode_image(img, spatial_res, bit_depth, output_file)

        print("\n" + "=" * 50)
        print("ENCODING SUMMARY")
        print("=" * 50)
        print(f"Output resolution: {spatial_res}x{spatial_res}")
        print(f"Bit depth: {bit_depth} bits ({2 ** bit_depth} gray levels)")
        print(f"Output file: {output_file}")
        print(f"File size: {file_size} bytes")
        print("=" * 50)

        return True, encoded_img

    except Exception as e:
        print(f"Error during encoding: {e}")
        return False, None


if __name__ == "__main__":
    # Test encoding
    test_img_path = "test_image.jpg"
    success, encoded = encode_image_verbose(
        test_img_path,
        spatial_res=200,
        bit_depth=4,
        output_file="encoded.bin"
    )

    if success:
        cv2.imshow("Encoded Image", encoded)
        cv2.waitKey(0)
        cv2.destroyAllWindows()