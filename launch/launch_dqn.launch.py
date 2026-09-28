from ament_index_python.packages import get_package_share_directory
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    # RSP
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('my_bot'),
                'launch',
                'rsp.launch.py'
            )
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ),
        launch_arguments={
            'world': os.path.join(
                get_package_share_directory('my_bot'),
                'worlds',
                'obstacles.world'
            )
        }.items()
    )

    # URDF
    robot_urdf = os.path.join(
        get_package_share_directory('my_bot'),
        'urdf',
        'my_robot.urdf'
    )

    # 🔥 FIX: position initiale
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', robot_urdf,
            '-entity', 'my_bot',
            '-x', '0',
            '-y', '0',
            '-z', '0.1'
        ],
        output='screen'
    )

    # DQN
    train_dqn_node = Node(
        package='my_bot',
        executable='train_dqn',
        name='train_dqn_node',
        output='screen'
    )

    return LaunchDescription([
        rsp,
        gazebo,
        spawn_entity,
        train_dqn_node
    ])
