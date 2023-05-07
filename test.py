#### Experimentation only
import cv2
import matplotlib.pyplot as plt
import numpy as np

test_img = cv2.imread("./Dataset/images_prepped_train/0001TP_006690.png")
print(test_img.shape)
plt.imshow(np.hstack((test_img[:, :, 0], test_img[:, :, 1], test_img[:, :, 2],cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY))))
plt.show()
test_anotation = cv2.imread("./Dataset/annotations_prepped_train/0001TP_006690.png")
print(test_anotation.shape)
plt.imshow(np.hstack((test_anotation[:, :, 0], test_anotation[:, :, 1], test_anotation[:, :, 2],cv2.cvtColor(test_anotation,cv2.COLOR_BGR2GRAY))))
plt.show()
cv2.waitKey(0)
