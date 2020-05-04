import cv2
import numpy
import imutils


def count_bacteria(image: numpy.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 3, 5, 5)
    edged = cv2.Canny(blurred, 50, 100)

    a = 3
    b = 2
    kernel_ab = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a, b))
    kernel_ba = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (b, a))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel_ab)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_ba)

    cnts = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cv2.drawContours(image, cnts, -1, (0, 0, 255), 1)

    return len(cnts)


def process_image(image_name: str):
    image = cv2.imread(image_name)
    
    number_of_bacteria = count_bacteria(image)

    print("{} bacteria found".format(number_of_bacteria))

    cv2.imshow(image_name, image)
    cv2.waitKey(0)


def main():
    image_path = "./src/images/"
    for i in range(1, 2):
        image_name = "1 ({}).png".format(i)
        process_image(image_path + image_name)


if __name__ == "__main__":
    main()
