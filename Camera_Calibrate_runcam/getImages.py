import cv2

w, h = 2560,1440
# h, w = 720, 1280
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS,60)
# cap = cv2.VideoCapture(2)
frW = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps,frW,frH)
num = 0

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == ord('s'):
        break
    elif k == 27: # wait for 'esc' key to save and exit
        cv2.imwrite('runcam_images_2k/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()