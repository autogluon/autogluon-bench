import os

from setuptools import setup, find_packages

AUTOGLUON = "autogluon"
BENCH = "bench"

PYTHON_REQUIRES = ">=3.8, <3.10"


def create_version_file(*, version):
    print("-- Building version " + version)
    version_path = os.path.join("src", AUTOGLUON, BENCH, "version.py")
    with open(version_path, "w") as f:
        f.write(f'"""This is the {AUTOGLUON}.{BENCH} version file."""\n')
        f.write("__version__ = '{}'\n".format(version))


def update_version(version, use_file_if_exists=True, create_file=False):
    """
    To release a new stable version on PyPi, simply tag the release on github, and the Github CI will automatically publish
    a new stable version to PyPi using the configurations in .github/workflows/pypi_release.yml .
    You need to increase the version number after stable release, so that the nightly pypi can work properly.
    """
    try:
        if not os.getenv("RELEASE"):
            from datetime import date

            minor_version_file_path = "VERSION.minor"
            if use_file_if_exists and os.path.isfile(minor_version_file_path):
                with open(minor_version_file_path) as f:
                    day = f.read().strip()
            else:
                today = date.today()
                day = today.strftime("b%Y%m%d")
            version += day
    except Exception:
        pass
    if create_file and not os.getenv("RELEASE"):
        with open("VERSION.minor", "w") as f:
            f.write(day)
    return version


def default_setup_args(*, version):
    long_description = open("README.md").read()
    name = f"{AUTOGLUON}.{BENCH}"
    setup_args = dict(
        name=name,
        version=version,
        author="AutoGluon Community",
        url="https://github.com/autogluon/autogluon-bench",
        description="",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="Apache-2.0",
        license_files=("LICENSE", "NOTICE"),
        # Package info
        packages=find_packages("src"),
        package_dir={"": "src"},
        namespace_packages=[AUTOGLUON],
        zip_safe=True,
        include_package_data=True,
        python_requires=PYTHON_REQUIRES,
        package_data={
            "": ["Dockerfile", "*.sh", "*.txt", "*.yaml"],
            AUTOGLUON: [
                "LICENSE",
            ]
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Education",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Customer Service",
            "Intended Audience :: Financial and Insurance Industry",
            "Intended Audience :: Healthcare Industry",
            "Intended Audience :: Telecommunications Industry",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Image Recognition",
        ],
        project_urls={
            "Bug Reports": "https://github.com/autogluon/autogluon-bench/issues",
            "Source": "https://github.com/autogluon/autogluon-bench/",
        },
        entry_points="""
            [console_scripts]
            agbench=autogluon.bench.main:app
        """,
    )
    return setup_args


version = "0.0.3"
version = update_version(version, use_file_if_exists=False, create_file=True)

install_requires = [
    "autogluon.common>=0.7,<1.0",
    "awscliv2>=2.2.0,<2.3.0",
    "pandas>=1.2.5,<2.0",
    "boto3>=1.26.0,<1.26.99",
    "aws-cdk-lib>=2.0.0,<2.70.0",
    "aws-cdk.aws-batch-alpha>=2.0.0a1,<2.70.0a0",
    "constructs>=10.0.0,<10.1.289",
    "pyyaml>=5.4,<=6.0",
    "tqdm>4.60.0,<=4.65.0",
    "twine>=4.0.0,<=4.0.2",
    "typer>=0.9.0,<1.0.0",
    "requests>2.20.0,<=2.30.0",
    "pyarrow>11.0.0,<=12.0.0",
    "wheel>0.38.0,<=0.40.0",
]

test_requirements = ["pytest", "pytest-mock", "tox"]
extras_require = {}
extras_require["tests"] = test_requirements

if __name__ == "__main__":
    create_version_file(version=version)
    setup_args = default_setup_args(version=version)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
