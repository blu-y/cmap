import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
import os
import cv2
from cv_bridge import CvBridge

# This node publishes an image message from files
# image filenames are timestamp
# publish image message with appropriate timestamp
class ImagePublisher(Node):
    def __init__(self, src='/home/iram/images/rgb_04082250/'):
        super().__init__('image_publisher')
        self.src = src
        self.bridge = CvBridge()
        self.image_list = [float(fn[:-4]) for fn in os.listdir(src)]
        self.image_list.sort()
        self.get_logger().info('Image list sorted')
        self.publisher_ = self.create_publisher(Image, '/oakd/rgb/preview/image_raw', 100)
        self.clock_subscriber_ = self.create_subscription(Clock, 'clock', self.clock_callback, qos_profile_sensor_data)
    
    def publish_image(self, fn):
        msg = self.bridge.cv2_to_imgmsg(
            cv2.imread(os.path.join(self.src, str(fn)+'.png')), 'bgr8')
        msg.header.stamp.sec = int(fn)
        msg.header.stamp.nanosec = int((fn - int(fn)) * 1e9)
        msg.header.frame_id = 'oakd_rgb_camera_optical_frame'
        self.publisher_.publish(msg)
        self.get_logger().info('Published image with timestamp: {}'.format(msg.header.stamp))
        

    def clock_callback(self, msg):
        time = msg.clock.sec + msg.clock.nanosec * 1e-9
        if time >= self.image_list[0]:
            fn = self.image_list.pop(0)
            self.publish_image(fn)

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)
    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()