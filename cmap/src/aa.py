import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class MarkerPublisher(Node):
    def __init__(self):
        super().__init__('marker_publisher')
        self.publisher_ = self.create_publisher(MarkerArray, 'pose_markers', 10)
        self.marker_id = 0
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.id = self.marker_id
        self.marker_id += 1
        
        # Set your pose data here
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2  # Length of the arrow
        marker.scale.y = 0.05 # Width of the arrow
        marker.color.a = 1.0  # Don't forget to set the alpha!
        marker.color.r = 1.0
        
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        
        self.publisher_.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    marker_publisher = MarkerPublisher()
    rclpy.spin(marker_publisher)
    marker_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()