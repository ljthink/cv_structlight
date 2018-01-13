# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import sys
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_avg   = (ref_white + ref_black) / 2.0
    ref_on    = ref_avg + 0.05 # a threshold for ON pixels
    ref_off   = ref_avg - 0.05 # add a small buffer region
    h,w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h,w), dtype=np.uint16)
    data = np.zeros((h,w,3), np.float64)
    datadest = np.zeros((h,w,3), np.uint8)

    # analyze the binary patterns from the camera
    for i in range(0,15):
        # read the file
        #patt_gray = cv2.resize(cv2.imread("images/aligned%03d.jpg"%(i+2)), (0,0), fx=scale_factor,fy=scale_factor)
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg" % (i + 2), cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0),
                               fx=scale_factor, fy=scale_factor)
        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)

        # TODO: populate scan_bits by putting the bit_code according to on_mask
        for x in range(w):
            for y in range(h):
                if on_mask[y,x] == True:
                    scan_bits[y,x] = scan_bits[y,x] + bit_code

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    camera_points = []
    projector_points = []
    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code

            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points
            x_p, y_p = binary_codes_ids_codebook[scan_bits[y, x]]
            if x_p >= 1279 or y_p >= 799:  # filter
                continue
            camera_points.append([x / 2.0, y / 2.0])
            projector_points.append([x_p,y_p])
            data[y,x] =  (0,y_p,x_p)
            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2
    for i in range(3):
        minval = data[..., i].min()
        maxval = data[..., i].max()
        if minval != maxval:
            data[..., i] -= minval
            data[..., i] *= (255.0 / (maxval - minval))
    cv2.imwrite("correspondence.jpg",data)
    # now that we have 2D-2D correspondances, we can triangulate 3D points!

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d
    pts_uv = np.array(camera_points, dtype=np.float32)
    num_pts = pts_uv.size / 2
    pts_uv.shape = (num_pts, 1, 2)
    camera__undist_pnts = cv2.undistortPoints(pts_uv,np.array(camera_K,dtype=np.float32) ,np.array(camera_d,dtype=np.float32))
    print("cam norm pts shape is "+str(camera__undist_pnts.shape))
    pts_proj = np.array(projector_points, dtype=np.float32)
    num_pts = pts_proj.size / 2
    pts_proj.shape = (num_pts, 1, 2)
    print("numpoints is "+str(num_pts))
    projector_undist_pnts = cv2.undistortPoints(pts_proj, np.array(projector_K ,dtype=np.float32), np.array(projector_d, dtype=np.float32))
    print("proj norm pts shape is " + str(projector_undist_pnts.shape))
    P1 = np.hstack((projector_R, projector_t))
    #P0 = np.eye(4)
    print (P1.shape)
    P0 = np.array([[1.0, 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, 1.0, 0]])
    print (P0.shape)
    res = cv2.triangulatePoints(P0, P1, camera__undist_pnts.reshape(2,num_pts), projector_undist_pnts.reshape(2,num_pts))
    print(res.shape)
    points_3d = cv2.convertPointsFromHomogeneous(res.T);
    return points_3d
    # TODO: use cv2.triangulatePoints to triangulate the normalized points
    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
	# TODO: name the resulted 3D points as "points_3d"


def write_3d_points(points_3d):
	
	# ===== DO NOT CHANGE THIS FUNCTION =====
    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
 #          if (p[0,2] <1400) & (p[0,2]>200):
                f.write("%d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2]))

   # return points_3d, camera_points, projector_points


def write_3d_color(points_3d):
    scale_factor = 1.0
    ref_color = cv2.resize(cv2.imread("images/aligned001.jpg", cv2.IMREAD_COLOR) / 255.0, (0, 0), fx=scale_factor,
                           fy=scale_factor)
    print ref_color.shape
    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "outputcolor.xyz"
    minx = points_3d[..., 0].min()
    maxx = points_3d[..., 0].max()
    with open(output_name, "w") as f:
        for p in points_3d:
            #           print("%d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2]))
            x = p[0, 0]
            y = p[0, 1]
            z = p[0, 2]
            if (z < 1400) & (z > 200):
                print("%d %d %d %d %d %d\n" % (x, y, z, ref_color[x][y][2], ref_color[x][y][1], ref_color[x][y][0]))
                # f.write("%d %d %d %d %d %d\n" % (x,y,z,ref_color[x][y][2],ref_color[x][y][1],ref_color[x][y][0]))

                # return points_3d, camera_points, projector_points

if __name__ == '__main__':

	# ===== DO NOT CHANGE THIS FUNCTION =====
	
	# validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
    #write_3d_color(points_3d)
	
