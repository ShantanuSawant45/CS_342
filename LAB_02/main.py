import numpy as np
from PIL import Image
import math

def create_affine_matrix(sx, sy, angle, tx, ty, shx, shy):
    """
    Create affine transformation matrix from individual transformation parameters.
    
    Parameters:
    sx, sy: Scaling factors (horizontal, vertical)
    angle: Rotation angle in degrees
    tx, ty: Translation (horizontal, vertical)
    shx, shy: Shear factors (horizontal, vertical)
    
    Returns:
    3x3 affine transformation matrix
    """
    # Convert angle to radians
    theta = math.radians(angle)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    # Scaling matrix
    S = [[sx, 0, 0],
         [0, sy, 0],
         [0, 0, 1]]
    
    # Rotation matrix
    R = [[cos_theta, -sin_theta, 0],
         [sin_theta, cos_theta, 0],
         [0, 0, 1]]
    
    # Translation matrix
    T = [[1, 0, tx],
         [0, 1, ty],
         [0, 0, 1]]
    
    # Shearing matrix
    Sh = [[1, shx, 0],
          [shy, 1, 0],
          [0, 0, 1]]
    
    # Combine transformations: T * R * Sh * S
    # Matrix multiplication done manually
    result = matrix_multiply_3x3(T, matrix_multiply_3x3(R, matrix_multiply_3x3(Sh, S)))
    
    return result

def matrix_multiply_3x3(A, B):
    """Manually multiply two 3x3 matrices"""
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

def apply_transformation(point, matrix):
    """
    Apply affine transformation to a point.
    
    Parameters:
    point: (x, y) coordinates
    matrix: 3x3 affine transformation matrix
    
    Returns:
    Transformed (x, y) coordinates
    """
    x, y = point
    
    # Homogeneous coordinates [x, y, 1]
    new_x = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]
    new_y = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]
    
    return (new_x, new_y)

def bilinear_interpolation(image, x, y):
    """
    Perform bilinear interpolation to get pixel value at non-integer coordinates.
    
    Parameters:
    image: Input image array
    x, y: Floating point coordinates
    
    Returns:
    Interpolated pixel value
    """
    height, width = image.shape[:2]
    
    # Get the four surrounding pixels
    x1 = int(math.floor(x))
    x2 = x1 + 1
    y1 = int(math.floor(y))
    y2 = y1 + 1
    
    # Check boundaries
    if x1 < 0 or x2 >= width or y1 < 0 or y2 >= height:
        return None
    
    # Get fractional parts
    fx = x - x1
    fy = y - y1
    
    # Handle grayscale and color images
    if len(image.shape) == 2:  # Grayscale
        # Bilinear interpolation formula
        value = (image[y1, x1] * (1 - fx) * (1 - fy) +
                image[y1, x2] * fx * (1 - fy) +
                image[y2, x1] * (1 - fx) * fy +
                image[y2, x2] * fx * fy)
        return int(value)
    else:  # Color
        result = []
        for channel in range(image.shape[2]):
            value = (image[y1, x1, channel] * (1 - fx) * (1 - fy) +
                    image[y1, x2, channel] * fx * (1 - fy) +
                    image[y2, x1, channel] * (1 - fx) * fy +
                    image[y2, x2, channel] * fx * fy)
            result.append(int(value))
        return tuple(result)

def invert_matrix_3x3(matrix):
    """
    Manually compute the inverse of a 3x3 matrix using the adjugate method.
    """
    # Calculate determinant
    det = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
    
    if abs(det) < 1e-10:
        raise ValueError("Matrix is not invertible")
    
    # Calculate adjugate matrix
    adj = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    adj[0][0] = matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]
    adj[0][1] = -(matrix[0][1] * matrix[2][2] - matrix[0][2] * matrix[2][1])
    adj[0][2] = matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]
    
    adj[1][0] = -(matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
    adj[1][1] = matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]
    adj[1][2] = -(matrix[0][0] * matrix[1][2] - matrix[0][2] * matrix[1][0])
    
    adj[2][0] = matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]
    adj[2][1] = -(matrix[0][0] * matrix[2][1] - matrix[0][1] * matrix[2][0])
    adj[2][2] = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    # Inverse = adjugate / determinant
    inverse = [[adj[i][j] / det for j in range(3)] for i in range(3)]
    
    return inverse

def transform_image(input_image, transformation_matrix):
    """
    Apply affine transformation to an image using backward mapping.
    
    Parameters:
    input_image: Input image as numpy array
    transformation_matrix: 3x3 affine transformation matrix
    
    Returns:
    Transformed image as numpy array
    """
    height, width = input_image.shape[:2]
    
    # Create output image with same dimensions
    if len(input_image.shape) == 2:  # Grayscale
        output_image = np.zeros((height, width), dtype=np.uint8)
    else:  # Color
        output_image = np.zeros((height, width, input_image.shape[2]), dtype=np.uint8)
    
    # Invert the transformation matrix for backward mapping
    try:
        inverse_matrix = invert_matrix_3x3(transformation_matrix)
    except ValueError:
        print("Error: Transformation matrix is not invertible")
        return input_image
    
    # Apply transformation using backward mapping
    for y_out in range(height):
        for x_out in range(width):
            # Find corresponding point in input image
            x_in, y_in = apply_transformation((x_out, y_out), inverse_matrix)
            
            # Use bilinear interpolation to get pixel value
            pixel_value = bilinear_interpolation(input_image, x_in, y_in)
            
            if pixel_value is not None:
                output_image[y_out, x_out] = pixel_value
    
    return output_image

def main():
    """Main function to execute affine transformation"""
    
    print("=" * 60)
    print("AFFINE TRANSFORMATION ON DIGITAL IMAGE")
    print("=" * 60)
    
    # Get input image path
    input_path = input("\nEnter input image path: ")
    
    try:
        # Load image
        img = Image.open(input_path)
        image_array = np.array(img)
        
        print(f"\nImage loaded successfully!")
        print(f"Image dimensions: {image_array.shape}")
        
        # Get transformation parameters from user
        print("\n" + "-" * 60)
        print("Enter Transformation Parameters:")
        print("-" * 60)
        
        sx = float(input("1. Horizontal scaling factor (e.g., 1.5): "))
        sy = float(input("2. Vertical scaling factor (e.g., 1.5): "))
        angle = float(input("3. Rotation angle in degrees (e.g., 45): "))
        tx = float(input("4. Horizontal translation in pixels (e.g., 50): "))
        ty = float(input("5. Vertical translation in pixels (e.g., 30): "))
        shx = float(input("6. Horizontal shear factor (e.g., 0.2): "))
        shy = float(input("7. Vertical shear factor (e.g., 0.1): "))
        
        print("\n" + "-" * 60)
        print("Applying transformation...")
        print("-" * 60)
        
        # Create affine transformation matrix
        affine_matrix = create_affine_matrix(sx, sy, angle, tx, ty, shx, shy)
        
        print("\nAffine Transformation Matrix:")
        for row in affine_matrix:
            print([f"{val:8.4f}" for val in row])
        
        # Apply transformation
        transformed_image = transform_image(image_array, affine_matrix)
        
        # Save output image
        output_path ="output.png"
        output_img = Image.fromarray(transformed_image)
        output_img.save(output_path)
        
        print(f"\nTransformation completed successfully!")
        print(f"Output saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"\nError: Input file '{input_path}' not found!")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()