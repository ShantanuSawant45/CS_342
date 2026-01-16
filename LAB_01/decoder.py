
import numpy as np
import cv2
from utils import (unpack_pixels, parse_header,
                   SPATIAL_REVERSE_MAP, INTENSITY_REVERSE_MAP)


def decode_image(input_file):
    print(f"Decoding image from {input_file}...")
    try:
        print("Step 1: Reading encoded file...")
        with open(input_file, 'rb') as f:
            header_byte = int.from_bytes(f.read(1), byteorder='big')
            pixel_data = f.read()

        print("Step 2: Parsing header...")
        spatial_idx, intensity_idx = parse_header(header_byte)

        resolution = SPATIAL_REVERSE_MAP[spatial_idx]
        bit_depth = INTENSITY_REVERSE_MAP[intensity_idx]

        print(f" Resolution: {resolution}x{resolution}")
        print(f" Bit depth: {bit_depth} bits")

        print("Step 3: Unpacking pixel data...")
        total_pixels = resolution * resolution

        if bit_depth < 8:
            img_data = unpack_pixels(pixel_data, bit_depth, total_pixels)
        else:
            img_data = np.frombuffer(pixel_data, dtype=np.uint8)


        print("Step 4: Reconstructing image...")
        img = img_data.reshape((resolution, resolution))

        levels = 2 ** bit_depth
        img_display = (img.astype(np.float32) / (levels - 1) * 255).astype(np.uint8)

        # Metadata
        metadata = {
            'resolution': resolution,
            'bit_depth': bit_depth,
            'gray_levels': levels,
            'spatial_index': spatial_idx,
            'intensity_index': intensity_idx
        }

        print("Decoding complete!")

        return img_display, metadata

    except Exception as e:
        print(f"Error during decoding: {e}")
        return None, None


def decode_and_display(input_file, save_output=None):

    decoded_img, metadata = decode_image(input_file)

    if decoded_img is not None:
        print("\n" + "=" * 50)
        print("DECODING SUMMARY")
        print("=" * 50)
        print(f"Resolution: {metadata['resolution']}x{metadata['resolution']}")
        print(f"Bit depth: {metadata['bit_depth']} bits")
        print(f"Gray levels: {metadata['gray_levels']}")
        print("=" * 50)


        cv2.imshow(f"Decoded: {metadata['resolution']}x{metadata['resolution']}, "
                   f"{metadata['bit_depth']}-bit", decoded_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if save_output:
            cv2.imwrite(save_output, decoded_img)
            print(f"Decoded image saved to {save_output}")

        return decoded_img
    else:
        print("Decoding failed!")
        return None


if __name__ == "__main__":
    decoded = decode_and_display("temp_encoded.bin", save_output="decoded.png")