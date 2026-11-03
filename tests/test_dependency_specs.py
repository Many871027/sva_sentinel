"""Test edge runtime dependencies and lockfile resolution.

Validates:
- R14: Declared edge runtime dependencies (fastapi, uvicorn, opencv-python-headless, openvino, llama-cpp-python, mlflow, supervision, pydantic, httpx, psutil, pyyaml)
- R16: Dependency resolution via package lock tooling without conflicts
"""

import os
import shutil
import subprocess
import tomllib
import unittest
from pathlib import Path


class TestDependencySpecs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.pyproject_path = cls.project_root / "pyproject.toml"
        cls.lockfile_path = cls.project_root / "uv.lock"

        with open(cls.pyproject_path, "rb") as f:
            cls.pyproject_data = tomllib.load(f)

    def test_r14_mandatory_edge_runtime_dependencies_declared(self):
        """Verify all mandatory edge dependencies are declared in pyproject.toml (R14)."""
        project = self.pyproject_data.get("project", {})
        dependencies = project.get("dependencies", [])
        self.assertTrue(len(dependencies) > 0, "project.dependencies must not be empty")

        # Map base package names
        dep_names = set()
        for dep in dependencies:
            # e.g., "fastapi>=0.115.0", "uvicorn[standard]>=0.32.0"
            clean_name = dep.split(">=")[0].split("==")[0].split("<=")[0].split(">")[0].split("<")[0].split("[")[0].strip()
            dep_names.add(clean_name.lower())

        required_edge_deps = [
            "fastapi",
            "uvicorn",
            "opencv-python-headless",
            "openvino",
            "llama-cpp-python",
            "mlflow",
            "supervision",
            "pydantic",
            "httpx",
            "psutil",
            "pyyaml",
        ]

        for req in required_edge_deps:
            self.assertIn(
                req.lower(),
                dep_names,
                f"Mandatory edge dependency missing from pyproject.toml: {req}",
            )

    def test_r14_dev_test_dependencies_declared(self):
        """Verify testing tools are declared in dev-dependencies or dependency-groups."""
        dev_deps = []
        if "dependency-groups" in self.pyproject_data:
            dev_deps.extend(self.pyproject_data["dependency-groups"].get("dev", []))
        if "tool" in self.pyproject_data and "uv" in self.pyproject_data["tool"]:
            dev_deps.extend(self.pyproject_data["tool"]["uv"].get("dev-dependencies", []))

        clean_devs = {
            d.split(">=")[0].split("==")[0].split("[")[0].strip().lower()
            for d in dev_deps
        }
        self.assertIn("pytest", clean_devs)
        self.assertIn("pytest-asyncio", clean_devs)

    def test_r16_dependency_resolution_lockfile(self):
        """Verify dependencies resolve cleanly without version conflicts via uv.lock (R16)."""
        self.assertTrue(
            self.lockfile_path.is_file(),
            "uv.lock lockfile must exist to guarantee deterministic conflict-free resolution",
        )
        lock_size = self.lockfile_path.stat().st_size
        self.assertGreater(lock_size, 1000, "uv.lock file must contain resolved package manifests")

        # If uv CLI is present in environment, verify lockfile check passes
        uv_bin = shutil.which("uv") or shutil.which("uv.exe")
        if uv_bin:
            res = subprocess.run(
                [uv_bin, "lock", "--check"],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
            )
            self.assertEqual(
                res.returncode,
                0,
                f"uv lock --check failed with conflicts or uncommitted changes:\n{res.stderr}\n{res.stdout}",
            )


if __name__ == "__main__":
    unittest.main()
