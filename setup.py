from setuptools import setup, find_packages

setup(
    name="zerotrace-ai",
    version="1.0.0",
    description="Real-Time Hardware Energy Telemetry & Green AI Inference Gateway",
    author="ZeroTrace AI Team",
    packages=find_packages(),
    py_modules=["cli", "server"],
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "psutil>=5.9.0",
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
    ],
    include_package_data=True,
    package_data={
        "zerotrace": ["*.html", "index.html"],
    },
    entry_points={
        "console_scripts": [
            "zerotrace=cli:main",
        ],
    },
    python_requires=">=3.9",
)
