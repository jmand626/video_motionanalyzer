import numpy as np
from skimage.transform import pyramid_gaussian
from traitlets import dlink


def lucas_kanade(img1, img2, keypoints, window_size=5):
    """Estimate flow vector at each keypoint using Lucas-Kanade method.

    Args:
        img1 - Grayscale image of the current frame. Flow vectors are computed
            with respect to this frame.
        img2 - Grayscale image of the next frame.
        keypoints - Keypoints to track. Numpy array of shape (N, 2).
        window_size - Window size to determine the neighborhood of each keypoint.
            A window is centered around the current keypoint location.
            You may assume that window_size is always an odd number.
    Returns:
        flow_vectors - Estimated flow vectors for keypoints. flow_vectors[i] is
            the flow vector for keypoint[i]. Numpy array of shape (N, 2).

    Hints:
        - You may use np.linalg.inv to compute inverse matrix
    """
    assert window_size % 2 == 1, "window_size must be an odd number"

    flow_vectors = []
    w = window_size // 2
    img_height, img_width = img1.shape

    # Compute partial derivatives
    Iy, Ix = np.gradient(img1)
    It = img2 - img1

    # For each [y, x] in keypoints, estimate flow vector [vy, vx]
    # using Lucas-Kanade method and append it to flow_vectors.
    for y, x in keypoints:
        # Keypoints can be located between integer pixels (subpixel locations).
        # For simplicity, we round the keypoint coordinates to nearest integer.
        # In order to achieve more accurate results, image brightness at subpixel
        # locations can be computed using bilinear interpolation.
        y, x = int(round(y)), int(round(x))

        # initialize flow values to be zero
        vx = vy = 0 
        if img1[y, x] == 0 or y < w or y > img_height-w-1 or x < w or x > img_width-w-1:
            continue

        ### YOUR CODE HERE
        
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # A bit easy to forget but remember that we have an window size so we must 
        # caluclate our values under that assumption!
        y_start, y_end = y-w, y+w+1

        x_start, x_end = x-w, x+w+1

        # Next we build up A_T * A
        # First we must limit our gradients, which have already been calculated
        Iy_windowed = Iy[y_start:y_end, x_start:x_end].flatten().astype(np.float64)
        Ix_windowed = Ix[y_start:y_end, x_start:x_end].flatten().astype(np.float64)
        It_windowed = It[y_start:y_end, x_start:x_end].flatten().astype(np.float64)

        join_xy = np.sum(Ix_windowed*Iy_windowed)
        # Build up A and b vectors as we are solving for v, not constructing it
        A = np.stack((Ix_windowed, Iy_windowed),axis=1)
        b = -It_windowed
        ata_matrix = A.T @ A
        atb_matrix = A.T @ b

        v = np.linalg.inv(ata_matrix) @ atb_matrix

        flow_vectors.append([v[1], v[0]])

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ### END YOUR CODE

    flow_vectors = np.array(flow_vectors)

    return flow_vectors


def iterative_lucas_kanade(img1, img2, keypoints, window_size=9, num_iters=7, g=None):
    """Estimate flow vector at each keypoint using iterative Lucas-Kanade method.

    Args:
        img1 - Grayscale image of the current frame. Flow vectors are computed
            with respect to this frame.
        img2 - Grayscale image of the next frame.
        keypoints - Keypoints to track. Numpy array of shape (N, 2).
        window_size - Window size to determine the neighborhood of each keypoint.
            A window is centered around the current keypoint location.
            You may assume that window_size is always an odd number.
        num_iters - Number of iterations to update flow vector.
        g - Flow vector guessed from previous pyramid level.
    Returns:
        flow_vectors - Estimated flow vectors for keypoints. flow_vectors[i] is
            the flow vector for keypoint[i]. Numpy array of shape (N, 2).
    """
    assert window_size % 2 == 1, "window_size must be an odd number"
    img_height, img_width = img1.shape

    # Initialize g as zero vector if not provided
    if g is None:
        g = np.zeros(keypoints.shape)

    flow_vectors = []
    w = window_size // 2

    # Compute spatial gradients
    Iy, Ix = np.gradient(img1)

    for y, x, gy, gx in np.hstack((keypoints, g)):
        v = np.zeros(2)  # Initialize flow vector as zero vector
        y1 = int(round(y))
        x1 = int(round(x))
        if img1[y1, x1] == 0 or y1 < w or y1 > img_height-w-1 or x1 < w or x1 > img_width-w-1:
            continue

        # TODO: Compute inverse of G at point (x1, y1)
        ### YOUR CODE HERE
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Should be pretty similar to the original lukas kanade method
        y_start, y_end = y1 - w, y1 + w + 1
        x_start, x_end = x1 - w, x1 + w + 1

        # First we must limit our gradients, which have already been calculated
        Iy_windowed = Iy[y_start:y_end, x_start:x_end].flatten().astype(np.float64)
        Ix_windowed = Ix[y_start:y_end, x_start:x_end].flatten().astype(np.float64)

        join_xy = np.sum(Ix_windowed*Iy_windowed)
        G = np.array([[np.sum(Ix_windowed**2), join_xy], [join_xy, np.sum(Iy_windowed**2)]], dtype=np.float64)
        G_inv = np.linalg.inv(G)
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ### END YOUR CODE

        # Iteratively update flow vector
        for k in range(num_iters):
            vx, vy = v
            # Refined position of the point in the next frame
            y2 = int(round(y + gy + vy))
            x2 = int(round(x + gx + vx))

            if (y2 < w or y2 > img_height-w-1 or x2 < w or x2 > img_width-w-1):
                continue

            # TODO: Compute bk and vk = inv(G) x bk
            ### YOUR CODE HERE
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # Now we want to compute the patch around the point in the next frame
            patch1_I_flat = img1[y_start:y_end, x_start:x_end].flatten().astype(np.float64)
            # We repeat the code from the first block for the next image too, as the whole
            # point of iteration is to look at the patch difference
            y2_start, y2_end = y2 - w, y2 + w + 1
            x2_start, x2_end = x2 - w, x2 + w + 1
            patch2_I_flat = img2[y2_start:y2_end, x2_start:x2_end].flatten().astype(np.float64)

            # Now we can compute the gradient of the difference
            I_k = patch1_I_flat - patch2_I_flat

            # And multiply it by the windowed deratives from the first block!
            bk = np.array([np.sum(I_k * Ix_windowed), np.sum(I_k * Iy_windowed)])
            vk = G_inv @ bk
            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            ### END YOUR CODE

            # Update flow vector by vk
            v += vk

        vx, vy = v
        flow_vectors.append([vy, vx])

    return np.array(flow_vectors)


def pyramid_lucas_kanade(
    img1, img2, keypoints, window_size=9, num_iters=7, level=2, scale=2
):

    """Pyramidal Lucas Kanade method

    Args:
        img1 - same as lucas_kanade
        img2 - same as lucas_kanade
        keypoints - same as lucas_kanade
        window_size - same as lucas_kanade
        num_iters - number of iterations to run iterative LK method
        level - Max level in image pyramid. Original image is at level 0 of
            the pyramid.
        scale - scaling factor of image pyramid.

    Returns:
        d - final flow vectors
    """

    # Build image pyramids of img1 and img2
    pyramid1 = tuple(pyramid_gaussian(img1, max_layer=level, downscale=scale))
    pyramid2 = tuple(pyramid_gaussian(img2, max_layer=level, downscale=scale))

    # Initialize pyramidal guess
    g = np.zeros(keypoints.shape)

    for L in range(level, -1, -1):
        ### YOUR CODE HERE
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        scale_factor = scale**L
        # For each piece in our pyramid, compute how much we have been scaled / how many
        # times the scaling factor has been applied

        # "Compute location of p on I^L"
        p_l = keypoints.astype(np.float64)/scale_factor
        # Run the iterative algo
        d_L = iterative_lucas_kanade(pyramid1[L], pyramid2[L], p_l, window_size, num_iters, g)
        # Now we either end or repeat
        # End:
        if L == 0:
          d = d_L
        else:
          g = scale*(g + d_L)

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ### END YOUR CODE

    d = g + d
    return d


def compute_error(patch1, patch2):
    """Compute MSE between patch1 and patch2

        - Normalize patch1 and patch2 each to zero mean, unit variance
        - Compute mean square error between patch1 and patch2

    Args:
        patch1 - Grayscale image patch of shape (patch_size, patch_size)
        patch2 - Grayscale image patch of shape (patch_size, patch_size)
    Returns:
        error - Number representing mismatch between patch1 and patch2
    """
    assert patch1.shape == patch2.shape, "Different patch shapes"
    error = 0
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    patch_1normalized = (patch1 - np.mean(patch1))/np.std(patch1)
    patch_2normalized = (patch2 - np.mean(patch2))/np.std(patch2)

    error = np.mean((patch_1normalized - patch_2normalized)**2)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE
    return error


def track_features(
    frames,
    keypoints,
    error_thresh=1.5,
    optflow_fn=pyramid_lucas_kanade,
    exclude_border=5,
    **kwargs
):

    """Track keypoints over multiple frames

    Args:
        frames - List of grayscale images with the same shape.
        keypoints - Keypoints in frames[0] to start tracking. Numpy array of
            shape (N, 2).
        error_thresh - Threshold to determine lost tracks.
        optflow_fn(img1, img2, keypoints, **kwargs) - Optical flow function.
        kwargs - keyword arguments for optflow_fn.

    Returns:
        trajs - A list containing tracked keypoints in each frame. trajs[i]
            is a numpy array of keypoints in frames[i]. The shape of trajs[i]
            is (Ni, 2), where Ni is number of tracked points in frames[i].
    """

    kp_curr = keypoints
    trajs = [kp_curr]
    patch_size = 3  # Take 3x3 patches to compute error
    w = patch_size // 2  # patch_size//2 around a pixel

    for i in range(len(frames) - 1):
        I = frames[i]
        J = frames[i + 1]
        flow_vectors = optflow_fn(I, J, kp_curr, **kwargs)
        kp_next = kp_curr + flow_vectors

        new_keypoints = []
        for yi, xi, yj, xj in np.hstack((kp_curr, kp_next)):
            # Declare a keypoint to be 'lost' IF:
            # 1. the keypoint falls outside the image J
            # 2. the error between points in I and J is larger than threshold

            yi = int(round(yi))
            xi = int(round(xi))
            yj = int(round(yj))
            xj = int(round(xj))
            # Point falls outside the image
            if (
                yj > J.shape[0] - exclude_border - 1
                or yj < exclude_border
                or xj > J.shape[1] - exclude_border - 1
                or xj < exclude_border
            ):
                continue

            # Compute error between patches in image I and J
            patchI = I[yi - w : yi + w + 1, xi - w : xi + w + 1]
            patchJ = J[yj - w : yj + w + 1, xj - w : xj + w + 1]
            error = compute_error(patchI, patchJ)
            if error > error_thresh:
                continue

            new_keypoints.append([yj, xj])

        kp_curr = np.array(new_keypoints)
        trajs.append(kp_curr)

    return trajs


def IoU(bbox1, bbox2):
    """Compute IoU of two bounding boxes

    Args:
        bbox1 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
        bbox2 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
    Returns:
        score - IoU score
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    score = 0

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    area_bbox1 = w1*h1
    area_bbox2 = w2*h2

    # Get bottom right corners of each box, which together with the top left corners will
    # help us find the intersections
    x1_4 = x1+w1
    x2_4 = x2+w2
    y1_4 = y1+h1
    y2_4 = y2+h2

    # Considering what we are doing, finding the intersection should be pretty obviously useful
    x_intersect_tl = max(x1, x2)
    y_intersect_tl = max(y1, y2)

    x_intersect_br = min(x1_4, x2_4)
    y_intersect_br = min(y1_4, y2_4)

    # The difference in the intersection of the shape with the top left and bottom right corner
    intersection = max(0, x_intersect_br-x_intersect_tl)*max(0,y_intersect_br-y_intersect_tl)
    union = area_bbox1 + area_bbox2 -intersection

    score = intersection/union


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return score
