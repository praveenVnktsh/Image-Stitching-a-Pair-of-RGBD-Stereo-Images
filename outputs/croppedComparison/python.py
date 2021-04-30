import glob
import cv2

for path in glob.glob('*.jpg'):
    im = cv2.imread(path)
    im = cv2.resize(im, (700, 400))
    cv2.imwrite(path, im)