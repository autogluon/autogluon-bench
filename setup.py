from setuptools import find_packages, setup

install_requires = [
    "pandas>=1.2.5,<2.0",
    "boto3>=1.26.0,<1.26.99",
    "aws-cdk-lib>=2.0.0,<2.70.0",
    "aws-cdk.aws-batch-alpha>=2.0.0a1,<2.70.0a0",
    "constructs>=10.0.0,<10.1.289",
    "pyyaml>=5.4,<=6.0"
]

extras_require = {}
test_requirements = ["pytest"]
extras_require["tests"] = test_requirements

setup(
    install_requires=install_requires,
    extras_require=extras_require,
    name = 'autogluon.bench',
    packages = find_packages("src"),
    package_dir ={"": "src"},
)