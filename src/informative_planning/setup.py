from setuptools import find_packages, setup

package_name = 'informative_planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/hybrid_plan_log.launch.py']),
        ('share/' + package_name + '/launch', ['launch/vis_rviz.launch.py']),
        ('share/' + package_name + '/config', ['config/def_config.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sanjeev Kumar',
    maintainer_email='sanjeevrs2000@gmail.com',
    description='MPPI-based search and capture planner for autonomous surface vehicles.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'time_varying_og = informative_planning.occ_grid:main',
            'hybrid_planner = informative_planning.hybrid_plan:main',
            'data_logger = informative_planning.logger:main',
            'mppi_inf_planner = informative_planning.mppi_planner:main',
        ],
    },
)
