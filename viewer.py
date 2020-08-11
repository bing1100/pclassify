import matplotlib.pyplot as plt
import matplotlib.image as mpimg

f = open("./keypoints/1.txt", "r")
a = (f.readline()).split(",")
a = (f.readline()).split(",")
a = (f.readline()).split(",")
xs = [int(float(i)) for i in a[::2]]
ys = [int(float(i)) for i in a[1::2]]

img=mpimg.imread('./images/1.tif')
imgplot = plt.imshow(img)
# plt.scatter(xs[0], ys[0])
plt.scatter(xs[1], ys[1])
plt.scatter(xs[2], ys[2])
#plt.scatter(xs[3], ys[3])
# plt.scatter(xs[4], ys[4])
plt.show()