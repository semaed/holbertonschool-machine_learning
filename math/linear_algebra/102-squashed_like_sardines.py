#!/usr/bin/env python3
"""Module Concatenates two matrices along a specific axis"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Args:
    - mat1: First matrix.
    - mat2: Second matrix.
    - axis: Axis along which to concatenate.

    Returns:
    - Concatenated matrix or None if concatenation is not possible.
    """

    # Helper function to get the shape of a matrix
    def get_shape(matrix):
        if not isinstance(matrix, list) or not matrix:
            return []
        return [len(matrix)] + get_shape(matrix[0])

    # Get shapes of both matrices
    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    # Check if shapes are compatible for concatenation
    if len(shape1) != len(shape2):
        return None  # Different dimensions

    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            return None  # Incompatible dimensions

    # Concatenate matrices
    if axis == 0:
        return mat1 + mat2
    else:
        # Recursively concatenate along the specified axis
        return [cat_matrices(row1, row2, axis-1) if axis <= max(len(row1), len(row2))
                else row1 + row2 for row1, row2 in zip(mat1, mat2)]


# Example high-dimensional matrices
mat3_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
mat4_3d = [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]

# Test concatenation of high-dimensional matrices
concat_3d_axis0 = cat_matrices(mat3_3d, mat4_3d)
concat_3d_axis1 = cat_matrices(mat3_3d, mat4_3d, axis=1)
concat_3d_axis2 = cat_matrices(mat3_3d, mat4_3d, axis=2)

(concat_3d_axis0, concat_3d_axis1, concat_3d_axis2)
