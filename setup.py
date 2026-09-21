#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2022 Huawei Technologies Co., Ltd.All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup."""

import sys
import os
import shutil
import stat
import platform
import subprocess
from importlib import import_module
from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py
from setuptools.command.install import install


def get_readme_content():
    pwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(pwd, 'README.md'), encoding='UTF-8') as f:
        return f.read()


def get_platform():
    """
    Get platform name.

    Returns:
        str, platform name in lowercase.
    """
    return platform.system().strip().lower()


def get_description():
    """
    Get description.

    Returns:
        str, wheel package description.
    """
    os_info = get_platform()
    cpu_info = platform.machine().strip()

    return f'mindformers platform: {os_info}, cpu: {cpu_info}'


def get_install_requires():
    """
    Get install requirements.

    Returns:
        list, list of dependent packages.
    """
    with open('requirements.txt', encoding='UTF-8') as file:
        return file.read().strip().splitlines()


def update_permissions(path):
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IEXEC | stat.S_IWRITE)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD | stat.S_IWRITE)


def _git_output(*args):
    """
    Run git with an argument list, without a shell.

    Returns:
        bytes, the command's stdout, or None if git is not installed or the command fails.
    """
    git = shutil.which('git')
    if git is None:
        return None
    try:
        return subprocess.run([git, *args], stdout=subprocess.PIPE, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def write_commit_id():
    """Record the source branch and the last commit in mindformers/.commit_id."""
    branch = _git_output('rev-parse', '--abbrev-ref', 'HEAD')
    last_commit = _git_output('log', '--abbrev-commit', '-1') if branch is not None else None
    if last_commit is None:
        sys.stdout.write("Warning: Can not get commit id information. Please make sure git is available.")
        commit_info = b'git is not available while building.\n'
    else:
        commit_info = branch + last_commit
    with open(os.path.join('mindformers', '.commit_id'), 'wb') as f:
        f.write(commit_info)


class EggInfo(egg_info):
    """Egg info."""

    def run(self):
        egg_info_dir = os.path.join(os.path.dirname(__file__), 'mindformers.egg-info')
        shutil.rmtree(egg_info_dir, ignore_errors=True)
        super().run()
        update_permissions(egg_info_dir)


class BuildPy(build_py):
    """Build py files."""

    def run(self):
        mindspore_transformer_lib_dir = os.path.join(os.path.dirname(__file__), 'build', 'lib', 'mindformers')
        shutil.rmtree(mindspore_transformer_lib_dir, ignore_errors=True)
        super().run()
        update_permissions(mindspore_transformer_lib_dir)


class Install(install):
    """Install."""

    def run(self):
        super().run()
        if sys.argv[-1] == 'install':
            pip = import_module('pip')
            mindspore_transformer_dir = os.path.join(os.path.dirname(pip.__path__[0]), 'mindformers')
            update_permissions(mindspore_transformer_dir)


if __name__ == '__main__':
    version_info = sys.version_info
    if (version_info.major, version_info.minor) < (3, 10):
        sys.stderr.write('Python version should be at least 3.10\r\n')
        sys.exit(1)

    write_commit_id()

    setup(
        name='mindformers',
        version='1.9.0',
        author='The MindSpore Authors',
        author_email='contact@mindspore.cn',
        url='https://www.mindspore.cn',
        download_url='https://atomgit.com/mindspore/mindformers/tags',
        project_urls={
            'Sources': 'https://atomgit.com/mindspore/mindformers',
            'Issue Tracker': 'https://atomgit.com/mindspore/mindformers/issues',
        },
        description=get_description(),
        long_description=get_readme_content(),
        long_description_content_type="text/markdown",
        test_suite="tests",
        packages=find_packages(exclude=["*tests*"]),
        platforms=[get_platform()],
        include_package_data=True,
        package_data={'mindformers': ['../configs/**/*.yaml',
                                      '../configs/**/*.yml',
                                      '../configs/**/*.md',
                                      './*.json',
                                      './dataset/blended_datasets/*',
                                      '.commit_id']},
        cmdclass={
            'egg_info': EggInfo,
            'build_py': BuildPy,
            'install': Install,
        },
        python_requires='>=3.10',
        install_requires=get_install_requires(),
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Environment :: Web Environment',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        license='Apache 2.0',
        keywords='mindformers',
    )
