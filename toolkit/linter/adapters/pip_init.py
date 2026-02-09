"""
Initializer script that installs stuff to pip.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time


def run_command(
    args: list[str],
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute external system commands with enhanced observability and strict validation.

        A wrapper around subprocess.run that adds command logging, execution time tracking,
        consistent UTF-8 encoding, and mandatory exit code checking. This function ensures
        command execution is traceable, secure, and follows consistent encoding standards.

        Args:
            args: List of command arguments (e.g., ["git", "clone", "https://repo.git"]).
                Using a list avoids shell injection risks and ensures cross-platform compatibility.
            env: Optional dictionary of environment variables to override for the command.
                If None, the current process environment is used (default: None).

        Returns:
            subprocess.CompletedProcess[str]: Object containing command execution results,
                including returncode, stdout (UTF-8 encoded string), and stderr (UTF-8 encoded string).

        Raises:
            subprocess.CalledProcessError: If the command exits with a non-zero status (enforced by check=True).
            FileNotFoundError: If the command executable is not found in the system PATH.
            PermissionError: If the user lacks permission to execute the command.

        Notes:
            - Uses time.monotonic() for accurate execution time measurement (unaffected by system time changes).
            - Logs the full command and execution duration at DEBUG level (always, regardless of success/failure).
            - Enforces text=True and encoding="utf-8" to return string (not bytes) output consistently.
        """
    logger = logging.getLogger(__name__)
    logger.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(args, env=env, text=True, encoding="utf-8", check=True)
    finally:
        end_time = time.monotonic()
        logger.debug("took %dms", (end_time - start_time) * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description="pip initializer")
    parser.add_argument(
        "packages",
        nargs="+",
        help="pip packages to install",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "--dry-run", help="do not install anything, just print what would be done."
    )
    parser.add_argument(
        "--no-black-binary",
        help="do not use pre-compiled binaries from pip for black.",
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET if args.verbose else logging.DEBUG,
        stream=sys.stderr,
    )

    env: dict[str, str] = {
        **os.environ,
        "UV_PYTHON": sys.executable,
        "UV_PYTHON_DOWNLOADS": "never",
        "FORCE_COLOR": "1",
        "CLICOLOR_FORCE": "1",
    }
    uv_index = env.get("UV_INDEX", env.get("PIP_EXTRA_INDEX_URL"))
    if uv_index:
        env["UV_INDEX"] = uv_index

    # If we are in a global install, use `--user` to install so that you do not
    # need root access in order to initialize linters.
    #
    # However, `pip install --user` interacts poorly with virtualenvs (see:
    # https://bit.ly/3vD4kvl) and conda (see: https://bit.ly/3KG7ZfU). So in
    # these cases perform a regular installation.
    in_conda = env.get("CONDA_PREFIX") is not None
    in_virtualenv = env.get("VIRTUAL_ENV") is not None
    need_user_flag = not in_conda and not in_virtualenv

    uv: str | None = shutil.which("uv")
    is_uv_managed_python = "uv/python" in sys.base_prefix.replace("\\", "/")
    if uv and (is_uv_managed_python or not need_user_flag):
        pip_args = [uv, "pip", "install"]
    elif sys.executable:
        pip_args = [sys.executable, "-mpip", "install"]
    else:
        pip_args = ["pip3", "install"]

    if need_user_flag:
        pip_args.append("--user")

    pip_args.extend(args.packages)

    for package in args.packages:
        package_name, _, version = package.partition("=")
        if version == "":
            raise RuntimeError(
                "Package {package_name} did not have a version specified. "
                "Please specify a version to produce a consistent linting experience."
            )
        if args.no_black_binary and "black" in package_name:
            pip_args.append(f"--no-binary={package_name}")

    dry_run = args.dry_run == "1"
    if dry_run:
        print(f"Would have run: {pip_args}")
        sys.exit(0)

    run_command(pip_args, env=env)


if __name__ == "__main__":
    main()
