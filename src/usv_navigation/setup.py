from setuptools import find_packages, setup

package_name = 'usv_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sanjeev Kumar',
    maintainer_email='sanjeevrs2000@gmail.com',
    description='USV perception and velocity control nodes.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stereo_detect = usv_navigation.object_detect_stereo:main',
            'pd_velocity_controller = usv_navigation.velocity_controller:main',
        ],
    },
)
