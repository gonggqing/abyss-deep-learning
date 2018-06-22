import PIL.Image
import cv2

def cv2_to_Pil(image):
    image = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(image)
