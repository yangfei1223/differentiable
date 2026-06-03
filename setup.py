from setuptools import setup, find_packages

setup(
    name="differentiable-bake",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0", "numpy>=1.24.0", "opencv-python>=4.8.0",
        "trimesh>=4.0.0", "PyYAML>=6.0", "Pillow>=10.0.0",
        "tqdm>=4.65.0", "imageio>=2.31.0",
    ],
)
