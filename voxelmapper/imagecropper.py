import cv2
import os

def crop_image(image_path):

    # Check if the file exists and is readable
    if not os.path.isfile(image_path) or not os.access(image_path, os.R_OK):
        print(f"Error: File {image_path} does not exist or is not readable")
        return
    # Load the image
    image = cv2.imread(image_path)
    cv2.namedWindow("Image")
    cv2.imshow("Image", image)

    # Define the callback function for mouse events
    def mouse_callback(event, x, y, flags, param):
        global image_cropped, x2, y2

        # If left mouse button is pressed, start drawing a rectangle
        if event == cv2.EVENT_LBUTTONDOWN:
            image_cropped = image.copy()
            cv2.rectangle(image_cropped, (x, y), (x, y), (0, 255, 0), 2)
            x2, y2 = x, y  # Initialize x2 and y2

        # If left mouse button is released, finish drawing the rectangle
        elif event == cv2.EVENT_LBUTTONUP:
            if image_cropped is not None and 'x2' in globals() and 'y2' in globals():
                cv2.rectangle(image_cropped, (x, y), (x, y), (0, 255, 0), 2)
                cv2.imshow("Image", image_cropped)
                x1, y1 = min(x, x2), min(y, y2)
                x2, y2 = max(x, x2), max(y, y2)
                image_cropped = image[y1:y2, x1:x2]
                fn = image_path + "_cropped2.png"
                cv2.imwrite(fn, image_cropped)
                print("Cropped image saved as", fn)

    cv2.setMouseCallback("Image", mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with the path to your image
image_folder = "./voxelmapper/images/"
image_name = "240529_135724_37.png"
crop_image(image_folder + image_name)
