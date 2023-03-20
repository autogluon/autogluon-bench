from setuptools import find_packages, setup

install_requires = [
    "pandas>=1.2.5,<2.0",
    "boto3<=1.26.55",
    "aws-cdk-lib>=2.0.0",
    "aws-cdk.aws-codestar-alpha>=2.0.0alpha1",
    "aws-cdk.aws-batch-alpha>=2.0.0alpha1",
    "constructs>=10.0.0"
]


setup(
    install_requires=install_requires,
    name = 'autogluon.bench',
    packages = find_packages("src"),
    package_dir ={"": "src"},
)