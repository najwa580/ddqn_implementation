import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    # =========================
    # RSP (robot description)
    # =========================
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

    # =========================
    # GAZEBO
    # =========================
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
            ),
            'verbose': 'true'
        }.items()
    )

    # =========================
    # SPAWN ROBOT (avec délai)
    # =========================
    spawn_entity = TimerAction(
        period=3.0,   # 🔥 attendre Gazebo
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-topic', 'robot_description',
                    '-entity', 'my_bot',
                    '-x', '0',
                    '-y', '0',
                    '-z', '0.1'
                ],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        rsp,
        gazebo,
        spawn_entity
    ])
