"""Test workspace layout and modular package scaffolding.

Validates:
- R1: Discrete root-level directories (src/, frontend/, colab_notebooks/, configs/, data/, models/, docker/)
- R2: Modular Python packages inside src/ (api/, core/, engine/, pipeline/, storage/, utils/)
"""

import os
import unittest
from pathlib import Path


class TestWorkspaceLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate project root based on test file location
        cls.project_root = Path(__file__).resolve().parent.parent

    def test_r1_root_level_directories_exist(self):
        """Verify presence of all mandatory root-level directories (R1)."""
        required_root_dirs = [
            "src",
            "frontend",
            "colab_notebooks",
            "configs",
            "data",
            "models",
            "docker",
        ]
        for dir_name in required_root_dirs:
            target_path = self.project_root / dir_name
            self.assertTrue(
                target_path.is_dir(),
                f"Required root-level directory missing: {dir_name} (path: {target_path})",
            )

    def test_r2_src_modular_packages_exist(self):
        """Verify modular subpackages and __init__.py files in src/ (R2)."""
        required_subpackages = [
            "api",
            "core",
            "engine",
            "pipeline",
            "storage",
            "utils",
        ]
        src_root = self.project_root / "src"
        self.assertTrue(src_root.is_dir(), "src directory must exist")

        # Root src package init
        src_init = src_root / "__init__.py"
        self.assertTrue(src_init.is_file(), "src/__init__.py must exist")

        for pkg_name in required_subpackages:
            pkg_path = src_root / pkg_name
            self.assertTrue(
                pkg_path.is_dir(),
                f"Required modular subpackage missing: src/{pkg_name}",
            )
            init_file = pkg_path / "__init__.py"
            self.assertTrue(
                init_file.is_file(),
                f"Package init missing: src/{pkg_name}/__init__.py",
            )


if __name__ == "__main__":
    unittest.main()
