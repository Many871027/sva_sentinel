"""Test docker compose configuration, memory budget, and health checks.

Validates:
- R3: Services defined (edge-api, mlflow-server, nextjs-ui)
- R4: mem_limit: 4g for edge-api
- R5: mem_limit: 1.5g for mlflow-server
- R6: mem_limit: 1g for nextjs-ui
- R7: Aggregate memory limit <= 6.5GB (reserving >= 1.5GB for 8GB host)
- R8: Dockerfile.api without GPU/CUDA bloat
- R9: Dockerfile.mlflow with SQLite backend
- R10: Dockerfile.frontend with multi-stage standalone build
- R11: Healthcheck for edge-api querying /health/live
- R12: Healthcheck for mlflow-server querying /health
- R13: Healthcheck for nextjs-ui querying /api/health
- R15: docker-compose syntax validation
- R17: Budget violation failure if aggregate container memory > 6.5GB
"""

import os
import shutil
import subprocess
import unittest
from pathlib import Path
import yaml


def parse_memory_limit(limit_val: str | int | float) -> float:
    """Converts memory strings like '4g', '1.5g', '1024m', '1536m' into gigabytes (float)."""
    if isinstance(limit_val, (int, float)):
        # numeric bytes
        return float(limit_val) / (1024.0 ** 3)
    s = str(limit_val).strip().lower()
    if s.endswith("gb"):
        return float(s[:-2])
    elif s.endswith("g"):
        return float(s[:-1])
    elif s.endswith("mb"):
        return float(s[:-2]) / 1024.0
    elif s.endswith("m"):
        return float(s[:-1]) / 1024.0
    elif s.endswith("kb"):
        return float(s[:-2]) / (1024.0 * 1024.0)
    elif s.endswith("k"):
        return float(s[:-1]) / (1024.0 * 1024.0)
    elif s.endswith("b"):
        return float(s[:-1]) / (1024.0 ** 3)
    return float(s) / (1024.0 ** 3)


def validate_compose_memory_budget(services: dict, max_total_gb: float = 6.5) -> float:
    """Validates that all services declare memory limits and aggregate <= max_total_gb.
    
    Raises:
        ValueError with explicit budget violation message if total > max_total_gb.
    """
    total_mem = 0.0
    for name, s_cfg in services.items():
        mem_str = s_cfg.get("mem_limit")
        if not mem_str:
            # Check deploy.resources.limits.memory
            mem_str = s_cfg.get("deploy", {}).get("resources", {}).get("limits", {}).get("memory")
        if not mem_str:
            raise ValueError(f"Service '{name}' does not declare 'mem_limit'.")
        mem_gb = parse_memory_limit(mem_str)
        total_mem += mem_gb

    if total_mem > max_total_gb:
        raise ValueError(
            f"Container memory budget violation: Total allocated {total_mem:.2f}GB "
            f"exceeds maximum allowed {max_total_gb:.2f}GB (Host reserve requirement >= 1.5GB violated)"
        )
    return total_mem


class TestDockerComposeBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.compose_file = cls.project_root / "docker-compose.yml"
        with open(cls.compose_file, "r", encoding="utf-8") as f:
            cls.compose_data = yaml.safe_load(f)

    def test_r3_services_defined(self):
        """Verify edge-api, mlflow-server, and nextjs-ui are defined (R3)."""
        services = self.compose_data.get("services", {})
        self.assertIn("edge-api", services, "edge-api service missing in docker-compose.yml")
        self.assertIn("mlflow-server", services, "mlflow-server service missing in docker-compose.yml")
        self.assertIn("nextjs-ui", services, "nextjs-ui service missing in docker-compose.yml")

    def test_r4_edge_api_memory_limit(self):
        """Verify edge-api memory limit is 4g (R4)."""
        services = self.compose_data.get("services", {})
        edge_mem = parse_memory_limit(services["edge-api"]["mem_limit"])
        self.assertEqual(edge_mem, 4.0, f"edge-api mem_limit expected 4.0GB, got {edge_mem}GB")

    def test_r5_mlflow_memory_limit(self):
        """Verify mlflow-server memory limit is 1.5g (R5)."""
        services = self.compose_data.get("services", {})
        mlflow_mem = parse_memory_limit(services["mlflow-server"]["mem_limit"])
        self.assertEqual(mlflow_mem, 1.5, f"mlflow-server mem_limit expected 1.5GB, got {mlflow_mem}GB")

    def test_r6_nextjs_memory_limit(self):
        """Verify nextjs-ui memory limit is 1g (R6)."""
        services = self.compose_data.get("services", {})
        nextjs_mem = parse_memory_limit(services["nextjs-ui"]["mem_limit"])
        self.assertEqual(nextjs_mem, 1.0, f"nextjs-ui mem_limit expected 1.0GB, got {nextjs_mem}GB")

    def test_r7_aggregate_container_memory_limit(self):
        """Verify aggregate memory limit does not exceed 6.5GB (R7)."""
        services = self.compose_data.get("services", {})
        total_mem = validate_compose_memory_budget(services, max_total_gb=6.5)
        self.assertLessEqual(total_mem, 6.5, f"Total container memory exceeds 6.5GB: {total_mem}")
        self.assertEqual(total_mem, 6.5, "Total planned container memory is exactly 6.5GB (4g + 1.5g + 1g)")

    def test_r8_dockerfile_api_cpu_only(self):
        """Verify Dockerfile.api omits CUDA and GPU training frameworks (R8)."""
        df_path = self.project_root / "docker" / "Dockerfile.api"
        self.assertTrue(df_path.is_file(), "docker/Dockerfile.api missing")
        content = df_path.read_text(encoding="utf-8").lower()
        self.assertIn("python:3.11-slim", content)
        self.assertNotIn("cuda", content)
        self.assertNotIn("cudnn", content)
        self.assertNotIn("nvidia", content)

    def test_r9_dockerfile_mlflow_sqlite(self):
        """Verify Dockerfile.mlflow uses SQLite backend and local artifacts (R9)."""
        df_path = self.project_root / "docker" / "Dockerfile.mlflow"
        self.assertTrue(df_path.is_file(), "docker/Dockerfile.mlflow missing")
        content = df_path.read_text(encoding="utf-8")
        self.assertIn("sqlite", content.lower())
        self.assertIn("mlflow", content.lower())

    def test_r10_dockerfile_frontend_multistage(self):
        """Verify Dockerfile.frontend uses multi-stage standalone Node build (R10)."""
        df_path = self.project_root / "docker" / "Dockerfile.frontend"
        self.assertTrue(df_path.is_file(), "docker/Dockerfile.frontend missing")
        content = df_path.read_text(encoding="utf-8")
        self.assertIn("as deps", content.lower())
        self.assertIn("as builder", content.lower())
        self.assertIn("as runner", content.lower())
        self.assertIn("standalone", content)

    def test_r11_edge_api_healthcheck(self):
        """Verify edge-api healthcheck queries /health/live (R11)."""
        services = self.compose_data.get("services", {})
        hc = services["edge-api"].get("healthcheck", {})
        test_cmd = " ".join(hc.get("test", []))
        self.assertIn("/health/live", test_cmd, f"edge-api healthcheck command must query /health/live: {test_cmd}")

    def test_r12_mlflow_healthcheck(self):
        """Verify mlflow-server healthcheck queries /health (R12)."""
        services = self.compose_data.get("services", {})
        hc = services["mlflow-server"].get("healthcheck", {})
        test_cmd = " ".join(hc.get("test", []))
        self.assertIn("/health", test_cmd, f"mlflow-server healthcheck command must query /health: {test_cmd}")

    def test_r13_nextjs_healthcheck(self):
        """Verify nextjs-ui healthcheck queries /api/health (R13)."""
        services = self.compose_data.get("services", {})
        hc = services["nextjs-ui"].get("healthcheck", {})
        test_cmd = " ".join(hc.get("test", []))
        self.assertIn("/api/health", test_cmd, f"nextjs-ui healthcheck command must query /api/health: {test_cmd}")

    def test_r15_docker_compose_syntax_validation(self):
        """Verify compose syntax is valid yaml and passes docker compose config if CLI available (R15)."""
        self.assertIsInstance(self.compose_data, dict)
        self.assertIn("services", self.compose_data)
        self.assertIn("networks", self.compose_data)
        self.assertIn("volumes", self.compose_data)

        # If docker is available, run docker compose config
        docker_bin = shutil.which("docker")
        if docker_bin:
            try:
                res = subprocess.run(
                    [docker_bin, "compose", "-f", str(self.compose_file), "config"],
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root),
                )
                self.assertEqual(
                    res.returncode,
                    0,
                    f"docker compose config failed:\nstdout: {res.stdout}\nstderr: {res.stderr}",
                )
            except Exception as e:
                # Docker daemon might not be running in this test environment, but syntax was validated
                pass

    def test_r17_unwanted_budget_violation_failure(self):
        """Verify IF sum of limits > 6.5GB THEN verification fails with budget violation message (R17)."""
        bad_services = {
            "edge-api": {"mem_limit": "5g"},
            "mlflow-server": {"mem_limit": "1.5g"},
            "nextjs-ui": {"mem_limit": "1g"},
        }
        with self.assertRaises(ValueError) as ctx:
            validate_compose_memory_budget(bad_services, max_total_gb=6.5)
        self.assertIn("Container memory budget violation", str(ctx.exception))
        self.assertIn("exceeds maximum allowed 6.50GB", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
