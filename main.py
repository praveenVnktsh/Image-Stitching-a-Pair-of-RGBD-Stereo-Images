from blending import Blender
import glob
import cv2
import numpy as np
import os
from tqdm import tqdm
import shutil
from itertools import combinations
def detectFeaturesAndMatch(img1, img2, nFeaturesReturn = -1):
    '''
    takes in two images, and returns a set of correspondences between the two images matched using ORB features, sorted from best to worst match using an L2 norm distance.
    '''
    kpts1 = orb.detect(img1, None)
    kpts2 = orb.detect(img2, None)
    kp1, des1 = descriptor.compute(img1,kpts1)
    kp2, des2 = descriptor.compute(img2,kpts2)
    # kp1, des1 = orb.detectAndCompute(img1,None)
    # kp2, des2 = orb.detectAndCompute(img2,None)
    print(len(des1), len(des2))
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance) 
    correspondences = []
    for match in matches:
        correspondences.append((kp1[match.queryIdx].pt, kp2[match.trainIdx].pt))
    print('Totally', len(correspondences), 'matches')
    src = np.float32([ m[0] for m in correspondences[:nFeaturesReturn] ]).reshape(-1,1,2)
    dst = np.float32([ m[1] for m in correspondences[:nFeaturesReturn] ]).reshape(-1,1,2)
    # tempImg = cv2.drawMatches(img2, kp1, img2, kp2, matches, None, flags = 2)
    # cv2.imwrite('matches.png', tempImg)
    # exit()
    
    return np.array(correspondences[:nFeaturesReturn]), src, dst

def getHomography(matches, depthmap):
    '''
    Takes in the points of correspondences and returns the homography matrix by 
    solving for the best fit transform using SVD.
    '''
    A = np.zeros((2*len(matches), 9))
    for i, match in enumerate(matches):
        src = match[0]
        dst = match[1]
        x1 = src[0]
        x2 = dst[0]
        y1 = src[1]
        y2 = dst[1]

        A[2*i] = np.array([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2])
        A[2*i+1] = np.array([0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2])
    
    U, S, V = np.linalg.svd(A)
    H = np.reshape(V[-1], (3, 3) )
    return H

def getBestHomographyRANSAC(matches, imageIndex, trials = 10000, threshold = 10, toChooseSamples = 4):
    '''
    Applies the RANSAC algorithm and tries out different homography matrices to compute the best matrix.
    '''
    matches = np.array(matches)
    if len(matches) == 4:
        H = getHomography(matches, depthQuantizer(cv2.imread(depthPaths[imageIndex] , 0)))
        return H, matches, -1

    finalH = None
    nMaxInliers = 0
    randomSample = None
    bestSample = None
    # randomchoices = np.random.randint(0, len(matches))
    

    for trialIndex in tqdm(range(trials)):
        inliers = []

        # randomly sample from the correspondences, and then compute homography matrix
        # after finding homography, see if the number of inliers is the best so far. If yes, we take that homography.
        # the number of correspondences for which we can compute the homography is a parameter.

        randomchoice = np.random.choice(len(matches), size=toChooseSamples, replace=False).tolist()
        randomSample = matches[randomchoice] 
        H = getHomography(randomSample, depthQuantizer(cv2.imread(depthPaths[imageIndex] , 0)))

        matches = np.array(matches)

        # vectorized implementation

        src = np.hstack((matches[:, 0], np.ones((len(matches[:, 0]), 1)))).T
        dst = np.hstack((matches[:, 1], np.ones((len(matches[:, 0]), 1)))).T
        transformed = np.dot(H, src)
        transformed = np.divide(transformed, transformed[2].T).T
        dst = dst.T


        distances = np.linalg.norm(transformed - dst, axis = 1)
        inliersIndices = distances < threshold


        inliers = matches[inliersIndices]

        #  # non vectorized implementation

        # for match in matches:
        #     src = np.append(match[0], 1).T
        #     dst = np.append(match[1], 1).T
        #     transformed = np.dot(H, src)
        #     transformed /= transformed[2]
        #     if np.linalg.norm(transformed - dst) < threshold:
        #         inliers.append(match)


        # best match => store 
        if len(inliers) > nMaxInliers:
            nMaxInliers = len(inliers)
            finalH = H
            bestSample = randomSample
    print('nInliers = ', nMaxInliers)
    return finalH, bestSample, nMaxInliers

def transformPoint(i, j, H):
    '''
    Helper function that simply transforms the point according to a given homography matrix
    '''
    transformed = np.dot(H, np.array([i, j, 1]))
    transformed /= transformed[2]
    transformed = transformed.astype(np.int)[:2]
    return np.array(transformed)

def transformImage(img, H, dst, forward = False, offset = [0, 0]):
    '''
    Helper function that computes the transformed image after applying homography.
    '''
    h, w, _ = img.shape
    if forward:
        # direct conversion from image to warped image without gap filling.
        coords = np.indices((w, h)).reshape(2, -1)
        coords = np.vstack((coords, np.ones(coords.shape[1]))).astype(np.int)    
        transformedPoints = np.dot(H, coords)
        yo, xo = coords[1, :], coords[0, :]
        # projective transform. Output's 3rd index should be one to convert to cartesian coords.
        yt = np.divide(np.array(transformedPoints[1, :]),np.array(transformedPoints[2, :])).astype(np.int)
        xt = np.divide(np.array(transformedPoints[0, :]),np.array(transformedPoints[2, :])).astype(np.int)
        dst[yt + offset[1], xt + offset[0]] = img[yo, xo]
    else:
        # applies inverse sampling to prevent any aliasing and hole effects in output image.
        Hinv = np.linalg.inv(H)
        topLeft = transformPoint(0, 0, H) 
        topRight = transformPoint(w-1, 0, H) 
        bottomLeft = transformPoint(0, h-1, H) 
        bottomRight = transformPoint(w-1, h-1, H)
        box = np.array([topLeft, topRight, bottomLeft, bottomRight])
        minX = np.min(box[:, 0])
        maxX = np.max(box[:, 0])
        minY = np.min(box[:, 1])
        maxY = np.max(box[:, 1])

        # instead of iterating through the pixels, we take indices and do
        # H.C, where C = coordinates to get the transformed pixels.
        # print(minX, maxX, minY, maxY)

        coords = np.indices((maxX - minX, maxY - minY)).reshape(2, -1)
        coords = np.vstack((coords, np.ones(coords.shape[1]))).astype(np.int)   

        coords[0, :] += minX
        coords[1, :] += minY
        # here, we use the inverse transformation from the transformed bounding box to compute the pixel value of the transformed image.
        transformedPoints = np.dot(Hinv, coords)
        yo, xo = coords[1, :], coords[0, :]

        # projective transform. Output's 3rd index should be one to convert to cartesian coords.
        yt = np.divide(np.array(transformedPoints[1, :]),np.array(transformedPoints[2, :])).astype(np.int)
        xt = np.divide(np.array(transformedPoints[0, :]),np.array(transformedPoints[2, :])).astype(np.int)


        # to prevent out of range errors
        indices = np.where((yt >= 0) & (yt < h) & (xt >= 0) & (xt < w))

        xt = xt[indices]
        yt = yt[indices]

        xo = np.clip(xo[indices] + offset[0], 0, 8191)
        yo = np.clip(yo[indices] + offset[1], 0, 4191)

        # assign pixel values!
        dst[yo , xo ] = img[yt, xt]


def execute(index1, index2, prevH):
    '''
    Function that, for a given pair of indices, computes the best homography and saves the warped images to disk.
    '''
    global m, depthPaths, imagePaths
    shutil.rmtree('outputs/' + str(imageSet) + '/warped/', ignore_errors=True)
    os.makedirs('outputs/' + str(imageSet) + '/warped/', exist_ok = True)
    os.makedirs('outputs/' + str(imageSet) + '/ransac/', exist_ok = True)

    depthPaths = sorted(glob.glob('dataset/' + str(imageSet) + '/depth_*'))
    imagePaths = sorted(glob.glob('dataset/' + str(imageSet) + '/im_*'))

    print('Images = ', len(imagePaths), 'in', 'dataset/' + str(imageSet) + '/depth_*')
    print('Depths = ', len(depthPaths))

    print('------------------------------------------')
    print('Warping images = ', (index1, index2))
    img1 = cv2.imread(imagePaths[index1])
    dimg1 = cv2.imread(depthPaths[index1], 0)
    img2 = cv2.imread(imagePaths[index2])
    img1 = cv2.resize(img1, shape)
    img2 = cv2.resize(img2,shape)
    dimg1 = cv2.resize(dimg1, shape)

    matches, src, dst = detectFeaturesAndMatch(img2, img1, -1)
    print('Total matches for computation= ', len(matches))
    
    imout = np.zeros((4192, 8192, 3))
    transformImage(img1, np.eye(3), dst = imout, offset = offset)

    cv2.imwrite('outputs/' + str(imageSet) + '/warped/a.jpg', imout)

    info = []
    maxInliersMatches = []
    for depthLevel in range(1, m + 1 ):
        levelinfo = depthQuantizer(dimg1, depthLevel, m)
        indices = np.stack((levelinfo, levelinfo, levelinfo), axis = -1)
        inpimg1 = img1.copy()
        inpimg1[~indices] = 0
        subsetMatches  = []
        # cv2.imshow('im1', inpimg1)
        # cv2.waitKey(0)
        # print(np.sum(level.astype(int)))
        for match in matches:
            if levelinfo[int(match[0][1]), int(match[0][0])] == True:
                subsetMatches.append(match)
        
        if len(subsetMatches) < 4:
            info.append({'matches' : maxInliersMatches, 'indices' : indices})
        else:
            info.append({'matches' : subsetMatches, 'indices' : indices})
            if len(maxInliersMatches) < len(subsetMatches):
                maxInliersMatches = subsetMatches

        levelinfo = info[-1]

        H, subsetMatches, nInliers = getBestHomographyRANSAC(levelinfo['matches'], index1, trials = trials, threshold = threshold)

        # print('Level = ', depthLevel, 'Match Ratio = ', nInliers*100//len(levelinfo['matches']), '|| # Points =', np.sum(levelinfo['indices'].astype(int)))

        inpimg = img2.copy()
        inpimg[~levelinfo['indices']] = 0

        warpedImage = np.zeros((4192, 8192, 3))
        transformImage(inpimg, np.dot(prevH, H), dst = warpedImage, offset = offset)

        cv2.imwrite('outputs/' + str(imageSet) + '/warped/warped_' + str(index2) +  str(depthLevel) + '.png', warpedImage)
    
    blendedImage = np.zeros((4192, 8192, 3))
    for path in glob.glob('outputs/' + str(imageSet) + '/warped/*'):
        print('blending', path)
        img = cv2.imread(path)
        indices = img[:, :, 0] == 0
        indices1 = img[:, :, 1] == 0
        indices2 = img[:, :, 2] == 0
        indices = ~(indices & indices1 & indices2)

        indices = np.stack((indices, indices, indices), axis = -1)
        blendedImage[indices] = img[indices]
        
    cv2.imwrite(f'outputs/{imageSet}/ransac/i{index1}_i{index2}_depth_{m}.png', blendedImage)

    
    return prevH

def depthQuantizer(image, depthLevel = 0, m = 12):

    assert depthLevel < m + 1
    # print((depthLevel-1)*255//m, (depthLevel*255//m))
    indices1 = image >= ((depthLevel-1)*255//m)
    indices2 = image < (depthLevel*255//m)
    # print(np.sum(indices1.astype(int)), np.sum(indices2.astype(int)))
    indices = np.logical_and(indices2, indices1)
    # image[indices] = 255*depthLevel//m

    return indices








if __name__ == "__main__":
    threshold = 2
    trials = 5000
    offset = [2300, 800]
    shape = (640, 360) # resize in order to improve speed and relax size constraints.
    orb = cv2.ORB_create(5000)
    descriptor = cv2.xfeatures2d.BEBLID_create(0.75)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for imageSet in range(1, 7):
        for m in [1, 5, 13]:
            try:

                execute(2, 3, np.eye(3))

                execute(1, 2, np.eye(3))

                execute(0, 1, np.eye(3))

            except Exception as e:
                print(e)
        