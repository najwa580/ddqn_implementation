from setuptools import setup
import os
from glob import glob

package_name = 'my_bot'

setup(
    name=package_name,
    version='0.0.0',

    packages=[package_name],
    package_dir={'my_bot': 'my_bot'},

    data_files=[
        # ROS index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        # package.xml
        ('share/' + package_name,
            ['package.xml']),

        # launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),

        # description (URDF/XACRO)
        (os.path.join('share', package_name, 'description'),
            glob('description/*.xacro')),

        # worlds Gazebo
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*')),

        # config
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='ahlem',
    maintainer_email='ahlemloued1@gmail.com',
    description='DQN robot navigation ROS2',
    license='MIT',

    entry_points={
        'console_scripts': [
            # 🔥 adapte selon ton fichier réel
            'train_dqn = my_bot.train_dqn:main',
            'test_dqn = my_bot.test_dqn:main',
            # ou si tu utilises step.py :
            # 'train_dqn = my_bot.step:main',
        ],
    },
)
