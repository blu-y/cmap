import rclpy
from rclpy.node import Node
# Import the message type, adjust the import according to your package structure and message type
from theora_image_transport.msg import Packet
# You might need additional imports for Theora decoding and image display, such as OpenCV
import cv2
import numpy as np
import pytheora

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber_node')
        self.subscription = self.create_subscription(
            Packet,
            '/oakd/rgb/preview/image_raw/theora',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('Received image')
        # Here, you would decode the Theora packet contained in msg.data
        # Decoding to an OpenCV image might look something like this:
        decoded_image = decode_theora_to_cv2(msg.data)
        # For demonstration, let's assume it's already decoded to a format cv2 can display
        cv2.imshow('Decoded Image', decoded_image)
        cv2.waitKey(1)

def decode_theora_to_cv2(theora_packet):
    # Decode the Theora packet to a YUV420p image
    yuv_image = pytheora.decode(theora_packet)

    # Convert the YUV420p image to RGB
    rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB_I420)


    return rgb_image
def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
