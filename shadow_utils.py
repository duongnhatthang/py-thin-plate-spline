import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

from PIL import Image
import thinplate as tps

import os, sys
fpath = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'AA_deform')
sys.path.append(fpath)

import time
from scipy.spatial.distance import cdist
from lib.interfaces import Mesh
from lib.mc.mc import TriangleMeshCreator
from lib.md.deform import ARAPDeformation
from utils import image_utils as im_utils
from utils import render_utils
from main import save_obj_format

def get_tranformation_matrix(h, w, alpha, beta, gamma, dx, dy, dz, f):
    """
    ONLY WORK WITH SQUARE MATRIX
    Source: http://jepsonsblog.blogspot.com/2012/11/rotation-in-3d-using-opencvs.html
    90 degrees being the "normal" position.

    alpha: the rotation around the x axis
    beta: the rotation around the y axis
    gamma: the rotation around the z axis (basically a 2D rotation)
    dx: translation along the x axis
    dy: translation along the y axis
    dz: translation along the z axis (distance to the image)
    f: focal distance (distance between camera and image, a smaller number exaggerates the effect)
    """

    alpha = (alpha - 90.)*np.pi/180.
    beta = (beta - 90.)*np.pi/180.
    gamma = (gamma - 90.)*np.pi/180.
    # Projection 2D -> 3D matrix
    A1 = np.array([[1, 0, -w/2],
          [0, 1, -h/2],
          [0, 0,    0],
          [0, 0,    1]])
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([[1,          0,           0, 0],
          [0, np.cos(alpha), -np.sin(alpha), 0],
          [0, np.sin(alpha),  np.cos(alpha), 0],
          [0,          0,           0, 1]])
    RY = np.array([[np.cos(beta), 0, -np.sin(beta), 0],
          [0, 1,          0, 0],
          [np.sin(beta), 0,  np.cos(beta), 0],
          [0, 0,          0, 1]])
    RZ = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
          [np.sin(gamma),  np.cos(gamma), 0, 0],
          [0,          0,           1, 0],
          [0,          0,           0, 1]])
    # Composed rotation matrix with (RX, RY, RZ)
    R = np.matmul(np.matmul(RX, RY), RZ)
    # Translation matrix
    T = np.array([[1, 0, 0, dx],
         [0, 1, 0, dy],
         [0, 0, 1, dz],
         [0, 0, 0, 1]])
    # 3D -> 2D matrix
    A2 = np.array([[f, 0, w/2, 0],
          [0, f, h/2, 0],
          [0, 0,   1, 0]])
    # Final transformation matrix
    return np.matmul(A2, np.matmul(T, np.matmul(R, A1)))

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def get_mask_bbox(img):
    sum_y = np.sum(img, axis=0)
    sum_x = np.sum(img, axis=1)
    l = first_nonzero(sum_y, 0)[0]
    r = last_nonzero(sum_y, 0)[0]
    t = first_nonzero(sum_x, 0)[0]
    b = last_nonzero(sum_x, 0)[0]
    return (l, r, t, b)

def matrix_padding(m):
    #Pad 3x3 => 4x4 for matrix multiplication
    out = np.zeros((4,4))
    out[:3,:3] = m.copy()
    out[3,3] = 1
    return out

def _affine_transform(mask, fixed_points, rd):
    h, w = mask.shape[0], mask.shape[1]

    srcTri = np.array( [[0,0], [w-1,0], [w//2, h//2]]).astype(np.float32)
    dstTri = np.array( [[0,0], [w-1,0], [w//2+rd[0], h//2+rd[1]]]).astype(np.float32)
    warp_mat = cv2.getAffineTransform(srcTri, dstTri)
    affine_matrix = np.zeros((3,3))
    affine_matrix[2,2] = 1
    affine_matrix[:2] = warp_mat
    affined = cv2.warpPerspective(mask, affine_matrix, (h, w), cv2.INTER_LANCZOS4)
    affined_fixed_pts = cv2.perspectiveTransform(fixed_points, affine_matrix)
    return affined, affined_fixed_pts, affine_matrix

def _rotate_x_axis(affined, affined_fixed_pts, rot_rd, h, w):
    tl, tr, br, bl = order_points(affined_fixed_pts[0])
    mid_l = (tl+bl)/2
    mid_r = (tr+br)/2
    mid_pts = np.expand_dims(np.vstack((mid_l, mid_r)), axis=0)
    #The angle between the legs' axis (middle line between legs) vs horizontal axis
    angle_z = np.arctan((mid_r-mid_l)[1]/(mid_r-mid_l)[0])*180/np.pi

    # Rotate image
    rotate_matrix_x = get_tranformation_matrix(h, w, alpha=90+rot_rd, beta=90, gamma=90, dx=0, dy=0, dz=200, f=200)
    rotated_x = cv2.warpPerspective(affined, rotate_matrix_x, (h, w), cv2.INTER_LANCZOS4)
    return rotated_x, rotate_matrix_x, mid_pts, angle_z

def _rotate_z_axis(mid_pts, rotate_matrix_x, h, w, angle_z):
    #Rotate z-axis to match the angle after transformed with before
    rotated_mid_pts = cv2.perspectiveTransform(mid_pts, rotate_matrix_x)
    rotated_mid_l = rotated_mid_pts[0,0]
    rotated_mid_r = rotated_mid_pts[0,1]

    #After x-axis rotation, the angle between the legs' axis (middle line between legs) vs horizontal axis is:
    rotated_angle_z = np.arctan((rotated_mid_r-rotated_mid_l)[1]/(rotated_mid_r-rotated_mid_l)[0])*180/np.pi
    #Correct this angle with the original angle_z
    rotate_matrix_z = get_tranformation_matrix(h, w, alpha=90, beta=90, gamma=90+angle_z-rotated_angle_z, dx=0, dy=0, dz=200, f=200)
    return rotate_matrix_z

def _flip_and_rotate_z_axis(h, w, angle_z):
    # Rotate the legs' axis to align with the horizontal axis, before vertical flipping
    rotate_matrix_z = get_tranformation_matrix(h, w, alpha=90, beta=90, gamma=90-angle_z, dx=0, dy=0, dz=200, f=200)
    # rotated_z = cv2.warpPerspective(rotated_x, rotate_matrix_z, (h, w), cv2.INTER_LANCZOS4)

    x_flip_matrix = np.eye(3)
    x_flip_matrix[1,1]=-1
    x_flip_matrix[1,2]=h-1
    # flipped = cv2.warpPerspective(rotated_z, x_flip_matrix, (h, w), cv2.INTER_LANCZOS4)

    return_rotate_matrix_z = get_tranformation_matrix(h, w, alpha=90, beta=90, gamma=90+angle_z, dx=0, dy=0, dz=200, f=200)
    # rotated = cv2.warpPerspective(flipped, return_rotate_matrix_z, (h, w), cv2.INTER_LANCZOS4)
    return rotate_matrix_z, x_flip_matrix, return_rotate_matrix_z

# TODO: bug if there is more than 1 object in the mask
def mask_to_shadow(mask, fixed_points, rot_rd_low, rot_rd_high, is_vertical_flip = False):
    r, b = np.max(fixed_points[0], axis=0)
    l, t = np.min(fixed_points[0], axis=0)

    # Affine transform to make the shadow cast in the left or right direction compared to the original mask
    h, w = mask.shape[0], mask.shape[1]
    rd = np.random.randint(low = -h//8, high = h//8, size=2)
    affined, affined_fixed_pts, affine_matrix = _affine_transform(mask, fixed_points, rd)

    rot_rd = np.random.randint(low = rot_rd_low, high = rot_rd_high)
    rotated_x, rotate_matrix_x, mid_pts, angle_z = _rotate_x_axis(affined, affined_fixed_pts, rot_rd, h, w)

    # OpenCV applied transformation weirdly (A then B => B.dot(A))
    all_matrix = np.matmul(matrix_padding(rotate_matrix_x), matrix_padding(affine_matrix))

    # Flip shadow
    if not is_vertical_flip:
        rotate_matrix_z = _rotate_z_axis(mid_pts, rotate_matrix_x, h, w, angle_z)
        all_matrix = np.matmul(matrix_padding(rotate_matrix_z), all_matrix)
        rotated = cv2.warpPerspective(mask, all_matrix[:3,:3], (h, w), cv2.INTER_LANCZOS4)

        # Matching shadow: matching the bottom of 2 bboxes
        final_fixed_points = cv2.perspectiveTransform(fixed_points, all_matrix[:3,:3])
        new_r, new_b = np.max(final_fixed_points[0], axis=0)
        new_l, new_t = np.min(final_fixed_points[0], axis=0)
        dy=int(b-new_b)
    else:
        rotate_matrix_z, x_flip_matrix, return_rotate_matrix_z = _flip_and_rotate_z_axis(h, w, angle_z)
        all_matrix = np.matmul(matrix_padding(rotate_matrix_z), all_matrix)
        all_matrix = np.matmul(matrix_padding(x_flip_matrix), all_matrix)
        all_matrix = np.matmul(matrix_padding(return_rotate_matrix_z), all_matrix)
        rotated = cv2.warpPerspective(mask, all_matrix[:3,:3], (h, w), cv2.INTER_LANCZOS4)

        # Matching shadow
        rotated = cv2.warpPerspective(mask, all_matrix[:3,:3], (h, w), cv2.INTER_LANCZOS4)
        center_x = (min(fixed_points[0,:,0]) + max(fixed_points[0,:,0]))/2
        center_y = (min(fixed_points[0,:,1]) + max(fixed_points[0,:,1]))/2
        center_coordinates = np.array([center_x, center_y]).astype('int')
        center_coordinates = np.expand_dims(center_coordinates, axis=(0,1)).astype('float64')
        final_centers = cv2.perspectiveTransform(center_coordinates, all_matrix[:3,:3])
        dy=int(center_y-final_centers[0,0,1])

        final_fixed_points = cv2.perspectiveTransform(fixed_points, all_matrix[:3,:3])
        new_r, new_b = np.max(final_fixed_points[0], axis=0)
        new_l, new_t = np.min(final_fixed_points[0], axis=0)

    # matching the x-axis center of 2 bboxes
    dx=int((l+(r-l)/2)-(new_l+(new_r-new_l)/2))
    trans_matrix = get_tranformation_matrix(h, w, alpha=90, beta=90, gamma=90,
                                     dx=dx, dy=dy, dz=200, f=200)
    all_matrix = np.matmul(matrix_padding(trans_matrix), all_matrix)

    final = cv2.warpPerspective(mask, all_matrix[:3,:3], (h, w), cv2.INTER_LANCZOS4)
    return final, all_matrix[:3,:3]

def create_grad_img(org_img, center, radius, speed, overflow, offset, thickness = 2):
    """
    Create an image that gradually fading from a center of a circle. Stop fading after reaching full radius
    """
    grad_img = np.zeros_like(org_img, dtype='uint8')
    it = int(overflow*radius)
    for i in range(1,radius):
        color_weight = np.exp(-speed*i/it)
        color = 255-int(i/it*255)
        color*=color_weight
        color-=offset

        grad_img = cv2.circle(grad_img, center, i, (color, color, color), thickness)
    return grad_img

def create_gradient_shadow_alpha_mask(shadow_mask, rd_low, rd_high, is_vertical_flip,
                                      speed, overflow, offset, thickness=2):
    """
    The rd_low and high randomize the location of the shadow.
    speed: higher means fade faster
    overflow: 1.2 = 120% original shadow size
    offset: reduce shadow's intensity
    """
    final_l, final_r, final_t, final_b = get_mask_bbox(shadow_mask)
    rd = np.random.randint(low = rd_low, high = rd_high)
    if not is_vertical_flip:
        y_center = final_b+rd
    else:
        y_center = final_t-rd
    radius = max(shadow_mask.shape[0], shadow_mask.shape[1])
    grad_img = create_grad_img(shadow_mask, ((final_l+final_r)//2,y_center), radius,
                               speed, overflow, offset, thickness) # Hard-code here

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = grad_img.astype(float)/255
    return alpha

def cut_mask(org_mask):
    """
    Find the obj insize the mask and cut a square patch around it
    """
    h, w = org_mask.shape[0], org_mask.shape[1]
    l,r,t,b = get_mask_bbox(org_mask)
    obj_center_x = int(l+(r-l)/2)
    obj_center_y = int(t+(b-t)/2)
    new_l = 0
    new_t = 0
    if h<w:
        if obj_center_x>=h//2: # obj is NOT too far left
            if obj_center_x<=w-h//2: # obj is NOT too far right
                new_l = obj_center_x-h//2
            else: # obj is too far right
                new_l = w-h
        else: # obj is too far left, new_l=0
            pass
        mask = org_mask[:, new_l:new_l+h]
    elif h>w:
        if obj_center_y>=w//2: # obj is NOT too far top
            if obj_center_y<=h-w//2: # obj is NOT too far bottom
                new_t = obj_center_y-w//2
            else: # obj is too far bottom
                new_t = h-w
        else: # obj is too far top
            pass
        mask = org_mask[new_t:new_t+w]
    else:
        mask = org_mask
    return mask, (new_l, new_t)

def restore_mask(org_mask, mask, cut_info):
    """
    padding the mask back to org_mask size
    """
    h, w = org_mask.shape[0], org_mask.shape[1]
    new_l, new_t = cut_info
    if h<w:
        pad_mask = np.zeros_like(org_mask)
        pad_mask[:, new_l:new_l+h] = mask
    elif h>w:
        pad_mask = np.zeros_like(org_mask)
        pad_mask[new_t:new_t+w] = mask
    else:
        pad_mask = mask
    return pad_mask

# def show_warped(img, warped, c_src, c_dst):
#     fig, axs = plt.subplots(1, 2, figsize=(16,8))
#     axs[0].axis('off')
#     axs[1].axis('off')
#     axs[0].imshow(img[...,::-1], origin='upper')
#     axs[0].scatter(c_src[:, 0]*img.shape[1], c_src[:, 1]*img.shape[0], marker='+', color='red')
#     axs[1].imshow(warped[...,::-1], origin='upper')
#     axs[1].scatter(c_dst[:, 0]*warped.shape[1], c_dst[:, 1]*warped.shape[0], marker='+', color='red')
#     plt.show()

def convert_pts(src_shape, src):
    src = np.squeeze(src, axis=0)
    c_src = np.zeros_like(src)
    c_src[:,0] = src[:,0]/src_shape[1]
    c_src[:,1] = src[:,1]/src_shape[0]
    return c_src

def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

def match_legs_TPS(img, fixed_points, rotate_matrix, rd_low, rd_high):
    rotated_points = cv2.perspectiveTransform(fixed_points, rotate_matrix)
    conv_fixed_pts = convert_pts(img.shape[:2], fixed_points)
    conv_rotated_pts = convert_pts(img.shape[:2], rotated_points)

    center_x = (min(fixed_points[0,:,0]) + max(fixed_points[0,:,0]))/2
    center_y = (min(fixed_points[0,:,1]) + max(fixed_points[0,:,1]))/2
    rd = np.random.uniform(low=rd_low, high=rd_high)
    src_pts = np.vstack((conv_rotated_pts,np.float32([[center_x,center_y]])))
    dst_pts = np.vstack((conv_fixed_pts,np.float32([[center_x+rd,center_y]])))
    return warp_image_cv(img, src_pts, dst_pts)

def pointwise_distance(pts1, pts2):
    """Calculates the distance between pairs of points

    Args:
        pts1 (np.ndarray): array of form [[x1, y1], [x2, y2], ...]
        pts2 (np.ndarray): array of form [[x1, y1], [x2, y2], ...]

    Returns:
        np.array: distances between corresponding points
    """
    dist = np.sqrt(np.sum((pts1 - pts2)**2, axis=1))
    return dist

def order_points(pts):
    """Orders points in form [top left, top right, bottom right, bottom left].
    Source: https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/

    Args:
        pts (np.ndarray): list of points of form [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    Returns:
        [type]: [description]
    """
    # sort the points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point. Note: this is a valid assumption because
    # we are dealing with rectangles only.
    # We need to use this instead of just using min/max to handle the case where
    # there are points that have the same x or y value.
    D = pointwise_distance(np.vstack([tl, tl]), right_most)
    
    br, tr = right_most[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def create_grad_img_ellipse(img, center, axesLength, angle, startAngle, endAngle,
                            thickness=2, speed=1, overflow=1.1, offset = 40):
    """
    speed: how fast the shadow fades
    offset: lessen the "blackness" of the shadow
    overflow: how much bigger the shadow should be compared to the original (in %)
    """
    grad_img = np.zeros_like(img)
    it = int(overflow*(axesLength[0]-axesLength[1]))
    for i in range(1,it):
        color_weight = np.exp(-speed*i/it)
        color = 255-int(i/it*255)
        color*=color_weight
        color-=offset
        axesLength_i = np.array([axesLength[0]-axesLength[1]+i, i],dtype='int')
        grad_img = cv2.ellipse(grad_img, center, axesLength_i,
           angle, startAngle, endAngle, (color,color,color), thickness)
    return grad_img

def create_ellipse_alpha_mask(org_mask, fixed_points, rd_low, rd_high,
                              speed=0.1, overflow=1.2, offset = 90):
    shadow_img = np.copy(org_mask)
    # shadow_img = np.zeros_like(org_mask)
    rd = np.random.randint(low = rd_low, high = rd_high)
    center_x = (min(fixed_points[0,:,0]) + max(fixed_points[0,:,0]))/2 + rd
    rd = np.random.randint(low = rd_low, high = rd_high)
    center_y = (min(fixed_points[0,:,1]) + max(fixed_points[0,:,1]))/2 + rd
    center_coordinates = np.array([center_x, center_y]).astype('int')

    tl, tr, br, bl = order_points(fixed_points[0])
    mid_l = (tl+bl)/2
    mid_r = (tr+br)/2
    d = np.max(fixed_points[0]-center_coordinates, axis=0)*1.5
    d[0]+=np.random.randint(low = rd_low, high = rd_high)
    d[1]+=np.random.randint(low = rd_low, high = rd_high)
    axesLength = np.sort(d)[::-1].astype('int')

    angle = np.arctan((mid_r-mid_l)[1]/(mid_r-mid_l)[0])*180/np.pi
    startAngle = 0  
    endAngle = 360

    color = (255,255,255)
    thickness = 5
    grad_img = create_grad_img_ellipse(org_mask, center_coordinates, axesLength, 
                                       angle, startAngle, endAngle, speed=speed, 
                                       overflow=overflow, offset=offset)
    
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = grad_img.astype(float)/255
    return alpha

def match_legs_ARAP(image_rgb, mask_gray, src_pts, target_pts, DEFORM_MESH_PATH=f'./deformed_mesh.obj', TEMP_IMG_PATH=f'./temp.png'):
    """
    poses2d_raw are the source points
    constraint_v_coords are the target points (for deformation)
    """
    h, w = image_rgb.shape[:2]
    cv2.imwrite(TEMP_IMG_PATH, image_rgb)

    poses2d_raw = src_pts
    constraint_v_coords = target_pts[0].copy()

    #
    # tri_mc = TriangleMeshCreator(interval=20, angle_constraint=20, area_constraint=200, dilated_pixel=5)
    tri_mc = TriangleMeshCreator(interval=5, angle_constraint=5, area_constraint=50, dilated_pixel=1)
    mesh = tri_mc.create(image_rgb, mask_gray)

    #
    vertices = 0.5 * (mesh.vertices + 1) * np.array([w, h]).reshape((1, 2)).astype(np.float32)
    distance = cdist(poses2d_raw, vertices)
    constraint_v_ids = np.argmin(distance, axis=1)
    poses2d = vertices[constraint_v_ids]

    #
    vis_image = mesh.get_image()
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

    for x, y in poses2d.astype(int):
        cv2.circle(vis_image, (x, y), radius=3, color=(255, 0, 0), thickness=2)

    for x, y in constraint_v_coords.astype(int):
        cv2.circle(vis_image, (x, y), radius=3, color=(0, 255, 0), thickness=2)

    # im_utils.imshow(vis_image)

    #
    constraint_v_coords_normed = Mesh.normalize_vertices(constraint_v_coords, size=(w, h))

    # build vertices texture
    vts = 0.5 * (mesh.vertices + 1)
    vts[:, 1] = 1. - vts[:, 1]

    # deform
    arap_deform = ARAPDeformation()
    arap_deform.load_from_mesh(mesh)
    arap_deform.setup()

    deformed_mesh = arap_deform.deform(constraint_v_ids, constraint_v_coords_normed, w=1000.)
    save_obj_format(file_path=DEFORM_MESH_PATH, vertices=deformed_mesh.vertices, faces=deformed_mesh.faces,
                    texture_vertices=vts)

    # # Visualize
    # vis_image = deformed_mesh.get_image(size=(w, h))
    # im_utils.imshow(vis_image)

    #
    pt_renderer = render_utils.PytorchRenderer(use_gpu=False)
    deformed_image = pt_renderer.render_w_texture(DEFORM_MESH_PATH, TEMP_IMG_PATH)
    deformed_image = deformed_image[::-1, :, :]
    deformed_image = cv2.cvtColor(deformed_image, cv2.COLOR_BGR2RGB)
    if os.path.exists(TEMP_IMG_PATH):
        os.remove(TEMP_IMG_PATH)
    # im_utils.imshow(deformed_image)
    return deformed_image
