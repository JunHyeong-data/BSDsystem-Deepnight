from setuptools import setup, find_packages
import os
from glob import glob

package_name = "bsd_deepnight"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
            glob("launch/*.py") + glob("launch/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="park june",
    maintainer_email="parkjune0310@gmail.com",
    description="SGLDet-based BSD system for nighttime driving",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # ros2 run bsd_deepnight detector_node
            "detector_node = bsd_deepnight.detector_node:main",
        ],
    },
)
