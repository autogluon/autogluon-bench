from setuptools import find_packages, setup

install_requires = [
    "pandas>=1.2.5,<2.0",
    "boto3<=1.17.74",
]


setup(
    install_requires=install_requires,
    name = 'autogluon.bench',
    packages = find_packages("src"),
    package_dir ={"": "src"},
)