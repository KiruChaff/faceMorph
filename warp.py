
import numpy as np
import os
import sys
import cv2
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from skimage.draw import polygon
from numpy.linalg import solve, LinAlgError
from faceDetection import *
from glob import glob
import subprocess
import argparse
from livePhoto import *


steps = 20
reverse = False


def triangulate(p1, p2):
    """corresponding traingles for both shapes"""
    points_mean = np.round((p1 + p2)/2.)
    return Delaunay(points_mean)

def findM(p, p_prime):
    """find Matrix for affine warp
        p = M*p'
    """
    p = np.vstack([p.T, [1, 1, 1]])
    p_prime = np.vstack([p_prime.T, [1, 1, 1]])
    try:
        AT = solve(p.T, p_prime.T)
    except LinAlgError:
        return np.eye(3, 3)
    return AT.T

def findMs(src_crds, tgt_crds):
    """find Matrices for affine warp"""
    Ms = []
    for p, p_prime in zip(tgt_crds, src_crds):
        Ms.append(findM(p, p_prime))
    return np.array(Ms)

def mask(shape, tgt):
    """The mask of the triangle"""
    xs, ys = tgt.T[0], tgt.T[1]
    rs, cs = polygon(ys, xs)
    _mask = np.zeros(shape, dtype=bool)
    _mask[rs, cs] = 1
    return _mask


def warp(src, tgt, M, image):
    """return warped section of image"""
    ys, xs = np.where(mask(image.shape, tgt)[:, :, 0])
    tgt_pts = np.vstack([xs, ys, np.ones(xs.shape)])

    src_pts = M.dot(tgt_pts)


    src_pts = np.round(src_pts).astype(int)
    color = image[src_pts[1], src_pts[0], :]
    return color, tgt_pts

def affine_warp(img, pts, inter_pts, tris):
    """affine warp the triangels of the source image into the intermediate shape"""
    src_crds = np.round(pts[tris.simplices].copy()).astype(float)
    tgt_crds = inter_pts[tris.simplices].copy()
    tgt_image = np.zeros(img.shape, dtype = img.dtype)

    Ms = findMs(src_crds, tgt_crds)

    for source, target, M in zip(src_crds, tgt_crds, Ms):
        color, target_points = warp(source, target, M, img)
        target_points = np.round(target_points).astype(int)
        tgt_image[target_points[1], target_points[0], :] = color
    return tgt_image


def morph(img, tgt_image, pts1, pts2, tris, warp, dissolve):
    """Set an intermediate shape and warp towards it"""
    interm_pts = np.round((warp * pts2) + ((1. - warp) * pts1))
    # warp both images into an intermediate shape
    img_warp = affine_warp(img, pts1, interm_pts, tris)
    tgt_img_warp = affine_warp(tgt_image, pts2, interm_pts, tris)
    return dissolve * tgt_img_warp + (1. - dissolve) * img_warp #cross dissolve



def morp_images(img1, pts1, img2, pts2, n1, n2, append=False):
    """morph two faces and safe as gif"""
    global reverse
    tri = triangulate(pts1, pts2) ## associate triangles in both images
    #empty output folder
    out_path = './output'
    junk = glob(out_path+'/*')
    for file in junk:
        os.remove(file)
    for frac in np.linspace(0., 1., steps): # for every frame
        morph_img = morph(im1, im2, points1, points2, tri, frac, frac) # morphed frame
        img = morph_img/256. # convert to float image
        plt.imsave(out_path+'/{}_{}_{}.jpg'.format(n1, n2, int(frac*100)), img)
    # convert to gif
    print("save output..")
    arg1s = sorted(glob(out_path + '/*.jpg'), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # append prev output if set
    if append:
        prepend = []
        for folder in sorted(next(os.walk('./append'))[1]):
            # print(folder)
            prepend += sorted(glob('./append/'+folder + '/*.jpg'), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        arg1s = prepend + arg1s
    arg1s = arg1s + arg1s[::-1] if reverse else arg1s # append reverse order if set
    arg2 = './warped.gif'
    subprocess.check_output('convert -delay 5 -loop 0'.split() + arg1s + [arg2])
    print("done..")

def get_shape(path1, path2, force=False):
    """evaluate the warp-points for both images"""
    data1 = path1[:-4]+"_shape.txt"
    data2 = path2[:-4]+"_shape.txt"
    if len(glob(data1)) == 0 or len(glob(data2))==0 or force:
        im1 = cv2.imread(path1, cv2.IMREAD_COLOR)
        im2 = cv2.imread(path2, cv2.IMREAD_COLOR)
        d1 = eval_points(im1, im2, data1)
        d2 = eval_points(im2, im1, data2)
    else :
        d1 = open(data1, "r")
        d2 = open(data2, "r")
    points1 = np.array(np.mat(";".join(d1)))
    points2 = np.array(np.mat(";".join(d2)))
    return points1, points2



## ------------------------------------------------------------------ ##
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", action="store", dest="images", help="The Path to two images to warp")
    parser.add_argument("-l", "--live", action='store_true', default=False, help="Take live photos to warp")
    parser.add_argument("-f", "--frames",  help="Set framecount", type=int)
    parser.add_argument("--force", action="store_true", default=False, help="Force to add warp points")
    parser.add_argument("-r", "--reverse", action='store_true', default=False, help="Make the gif seemless (twice the framecount)")
    parser.add_argument("--append", action='store_true', default=False, help="Append the output of this folder to a previous result")
    args = parser.parse_args()

    if args.images:
        path1,path2 = args.images.split(" ")

    elif args.live:
        livePhoto()
        path1,path2 = "Image_1.jpg", "Image_2.jpg"
    else:
        parser.print_help()
        exit()
    if args.frames:
        steps = args.frames
    if args.reverse:
        reverse = args.reverse
    im1 = plt.imread(path1)
    im2 = plt.imread(path2)
    name1 = path1.split("/")[-1].split(".")[0]
    name2 = path2.split("/")[-1].split(".")[0]
    points1, points2 = get_shape(path1, path2, args.force)
    print("starting to morph..")
    morp_images(im1, points1, im2, points2, name1, name2, args.append)
