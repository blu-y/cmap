from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node, PushRosNamespace

import os
# ros2 launch turtlebot4_viz view_robot.launch.py


ARGUMENTS = [
    DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'),
    DeclareLaunchArgument(
        'description',
        default_value='false',
        description='Launch turtlebot4 description'
    ),
    DeclareLaunchArgument(
        'model',
        default_value='standard',
        choices=['standard', 'lite'],
        description='Turtlebot4 Model'
    ),
    DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace'
    ),
]


def generate_launch_description():

    pkg_turtlebot4_viz = get_package_share_directory('turtlebot4_viz')
    pkg_cmap = PathJoinSubstitution([os.path.expanduser("~"), 'cmap', 'cmap'])
    pkg_turtlebot4_description = get_package_share_directory('turtlebot4_description')

    rviz2_config = PathJoinSubstitution(
        [pkg_cmap, 'rviz', 'robot.rviz'])
    description_launch = PathJoinSubstitution(
        [pkg_turtlebot4_description, 'launch', 'robot_description.launch.py']
    )

    namespace = LaunchConfiguration('namespace')

    rviz = GroupAction([
        PushRosNamespace(namespace),

        Node(package='rviz2',
             executable='rviz2',
             name='rviz2',
             arguments=['-d', rviz2_config],
             parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
             remappings=[
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static')
             ],
             output='screen'),

        # Delay launch of robot description to allow Rviz2 to load first.
        # Prevents visual bugs in the model.
        TimerAction(
            period=3.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([description_launch]),
                    launch_arguments=[('model', LaunchConfiguration('model'))],
                    condition=IfCondition(LaunchConfiguration('description'))
                )])
    ])

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(rviz)
    return ld