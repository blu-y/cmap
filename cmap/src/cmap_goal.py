import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading

class GoalPublisher(Node):
    def __init__(self):
        super().__init__('cmap_goal_publisher')
        self.publisher_ = self.create_publisher(String, '/cmap/goal', 1)
        # 콘솔 입력 처리를 위한 별도의 스레드 시작
        threading.Thread(target=self.read_input, daemon=True).start()

    def read_input(self):
        while True:
            input_str = input("Enter CMAP goal: ")
            self.publish_string(input_str)

    def publish_string(self, input_str):
        msg = String()
        msg.data = input_str
        self.publisher_.publish(msg)
        
        self.get_logger().info('Publishing: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    goal_publisher = GoalPublisher()
    try:
        rclpy.spin(goal_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        goal_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
