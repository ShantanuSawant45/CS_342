
import cv2
import numpy as np
import os
from encoder import encode_image_verbose
from decoder import decode_and_display, decode_image


def display_menu():
    """Display main menu"""
    print("\n" + "=" * 60)
    print("IMAGE ENCODER-DECODER SYSTEM")
    print("=" * 60)
    print("1. Encode Image")
    print("2. Decode Image")
    print("3. Encode & Decode (Test Full Pipeline)")
    print("4. Exit")
    print("=" * 60)


def get_spatial_resolution():
    """Get spatial resolution from user"""
    print("\nSelect Spatial Resolution:")
    print("1. 100 x 100")
    print("2. 200 x 200")
    print("3. 400 x 400")
    print("4. 800 x 800")

    while True:
        try:
            choice = int(input("Enter choice (1-4): "))
            if 1 <= choice <= 4:
                resolutions = [100, 200, 400, 800]
                return resolutions[choice - 1]
            else:
                print("Invalid choice! Please enter 1-4.")
        except ValueError:
            print("Invalid input! Please enter a number.")


def get_intensity_depth():
    """Get intensity depth from user"""
    print("\nSelect Intensity Depth:")
    print("1. 1 bit  (2 gray levels)")
    print("2. 2 bits (4 gray levels)")
    print("3. 4 bits (16 gray levels)")
    print("4. 8 bits (256 gray levels)")

    while True:
        try:
            choice = int(input("Enter choice (1-4): "))
            if 1 <= choice <= 4:
                bit_depths = [1, 2, 4, 8]
                return bit_depths[choice - 1]
            else:
                print("Invalid choice! Please enter 1-4.")
        except ValueError:
            print("Invalid input! Please enter a number.")


def encode_mode():
    print("\n" + "-" * 60)
    print("ENCODING MODE")
    print("-" * 60)
    img_path = input("Enter input image path: ").strip()
    if not os.path.exists(img_path):
        print(f"Error: File '{img_path}' not found!")
        return


    spatial_res = get_spatial_resolution()
    bit_depth = get_intensity_depth()

    default_output = f"encoded_{spatial_res}x{spatial_res}_{bit_depth}bit.bin"
    output_file = input(f"Enter output file name (default: {default_output}): ").strip()
    if not output_file:
        output_file = default_output

    # Encode
    print("\nStarting encoding process...")
    success, encoded_img = encode_image_verbose(img_path, spatial_res, bit_depth, output_file)

    if success:
        # Display result
        cv2.imshow("Encoded Image Preview", encoded_img)
        print("\nPress any key to close preview...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def decode_mode():
    """Decoding mode"""
    print("\n" + "-" * 60)
    print("DECODING MODE")
    print("-" * 60)

    # Get encoded file path
    encoded_file = input("Enter encoded file path: ").strip()

    if not os.path.exists(encoded_file):
        print(f"Error: File '{encoded_file}' not found!")
        return

    # Get output path (optional)
    save_decoded = input("Save decoded image? (y/n): ").strip().lower()
    output_path = None

    if save_decoded == 'y':
        output_path = input("Enter output image path (e.g., decoded.png): ").strip()

    # Decode
    print("\nStarting decoding process...")
    decode_and_display(encoded_file, save_output=output_path)


def test_full_pipeline():
    """Test encoding and decoding together"""
    print("\n" + "-" * 60)
    print("FULL PIPELINE TEST")
    print("-" * 60)

    # Get input
    img_path = input("Enter input image path: ").strip()

    if not os.path.exists(img_path):
        print(f"Error: File '{img_path}' not found!")
        return

    # Get parameters
    spatial_res = get_spatial_resolution()
    bit_depth = get_intensity_depth()

    temp_encoded = "temp_encoded.bin"

    # Load original for comparison
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Encode
    print("\n--- ENCODING ---")
    success, encoded_img = encode_image_verbose(img_path, spatial_res, bit_depth, temp_encoded)

    if not success:
        return

    # Decode
    print("\n--- DECODING ---")
    decoded_img, metadata = decode_image(temp_encoded)

    if decoded_img is not None:
        # Display comparison
        print("\nDisplaying results...")

        # Resize original for fair comparison
        from utils import crop_to_square, spatial_sampling
        original_cropped = crop_to_square(original)
        original_resized = spatial_sampling(original_cropped, spatial_res)

        # Create side-by-side comparison
        comparison = np.hstack([original_resized, encoded_img, decoded_img])

        cv2.imshow("Comparison: Original | Encoded | Decoded", comparison)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Cleanup
        if os.path.exists(temp_encoded):
            os.remove(temp_encoded)


def main():
    while True:
        display_menu()
        try:
            choice = input("\nEnter your choice: ").strip()
            if choice == '1':
                encode_mode()
            elif choice == '2':
                decode_mode()
            elif choice == '3':
                test_full_pipeline()
            elif choice == '4':
                print("\nExiting... Goodbye!")
                break
            else:
                print("Invalid choice! Please enter 1-4.")

        except KeyboardInterrupt:
            print("\n\nProgram interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()