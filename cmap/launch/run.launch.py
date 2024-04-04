from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_tb4_nav = get_package_share_directory('turtlebot4_navigation')
    pkg_tb4_viz = get_package_share_directory('turtlebot4_viz')
    launch_slam = PathJoinSubstitution(
        [pkg_tb4_nav, 'launch', 'slam.launch.py'])
    launch_viz = PathJoinSubstitution(
        [pkg_tb4_viz, 'launch', 'view_robot_cmap.launch.py'])
    launch_nav2 = PathJoinSubstitution(
        [pkg_tb4_nav, 'launch', 'nav2.launch.py'])

    ld = LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_slam)
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_viz)
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_nav2)
        ),
    ])

    return ld
