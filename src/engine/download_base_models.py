"""Hugging Face Teacher & Student Base Model Checkpoint Acquisition Engine.

Covers:
- R1: Download PyTorch and SafeTensors model checkpoints from Hugging Face Hub using huggingface_hub.
- R2: Store acquired checkpoints into structured subdirectories models/base/<model_name>/.
- R3: Compute cryptographic SHA256 checksum and verify against catalog profile expected hash.
- R4: Remove invalid file, raise ModelChecksumError, and abort registration on hash mismatch.
- R5: Support HTTP resume capability and local cache snapshot re-use.
- R6: Offline execution mode resolving checkpoints strictly from local files.
- R7: Predefined catalog profiles (yolo11x, yolo11n, qwen2.5-0.5b-instruct).
- R14: Raise ModelNotFoundError with list of valid models if model is unregistered.
- R18: Expose CLI entrypoint supporting download, verify, and inspect commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download

from src.core.exceptions import ModelChecksumError, ModelNotFoundError, ModelRegistryError
from src.pipeline.schemas import TARGET_MODEL_PROFILES, BaseModelProfile

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Compute the hexadecimal SHA256 digest of a local file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot compute checksum: file not found at {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class BaseModelDownloader:
    """Orchestrates checkpoint acquisition, caching, and cryptographic verification."""

    def __init__(
        self,
        base_dir: Path | str = Path("models/base"),
        profiles: Optional[Dict[str, BaseModelProfile]] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.profiles: Dict[str, BaseModelProfile] = (
            profiles if profiles is not None else dict(TARGET_MODEL_PROFILES)
        )

    def get_profile(self, model_id: str) -> BaseModelProfile:
        """Retrieve a catalog profile by identifier, or raise ModelNotFoundError (R14)."""
        if model_id not in self.profiles:
            valid_models = ", ".join(sorted(self.profiles.keys()))
            raise ModelNotFoundError(
                f"Model profile '{model_id}' not found. Available valid models: [{valid_models}]",
                details={"model_id": model_id, "available_models": list(self.profiles.keys())},
            )
        return self.profiles[model_id]

    def list_available_models(self) -> List[str]:
        """List all available model identifiers in the catalog."""
        return sorted(list(self.profiles.keys()))

    def get_model_target_path(self, model_id: str) -> Path:
        """Resolve the persistent destination path for a model's weights file (R2)."""
        profile = self.get_profile(model_id)
        return self.base_dir / model_id / profile.file_name

    def write_metadata(self, model_id: str, verified_sha256: str) -> Path:
        """Write architecture configuration and versioning metadata to models/base/<model_name>/ (R2)."""
        profile = self.get_profile(model_id)
        dest_dir = self.base_dir / model_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / profile.file_name

        file_size = dest_file.stat().st_size if dest_file.exists() else profile.file_size_bytes

        metadata = {
            "model_id": profile.model_id,
            "model_name": profile.model_name,
            "model_role": profile.model_role,
            "model_family": profile.model_family,
            "version": profile.version,
            "format": profile.format,
            "source_repo": profile.source_repo,
            "file_name": profile.file_name,
            "sha256": verified_sha256,
            "file_size_bytes": file_size,
            "input_resolution": profile.input_resolution,
            "parameter_count": profile.parameter_count,
            "description": profile.description,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        }

        meta_path = dest_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return meta_path

    def download(
        self,
        model_id: str,
        offline: bool = False,
        force: bool = False,
    ) -> Path:
        """Download checkpoint with resume and verify SHA256 integrity (R1, R2, R3, R4, R5, R6).

        Args:
            model_id: Unique model profile identifier.
            offline: If True, do not attempt network download; resolve strictly from local storage.
            force: If True, bypass existing valid files and re-download.

        Returns:
            Path to verified model checkpoint weights file.

        Raises:
            ModelNotFoundError: If model_id not in catalog, or offline file missing.
            ModelChecksumError: If SHA256 checksum does not match expected hash (file is removed).
        """
        profile = self.get_profile(model_id)
        dest_dir = self.base_dir / model_id
        dest_file = dest_dir / profile.file_name

        # Offline execution mode (R6)
        if offline:
            if not dest_file.is_file():
                raise ModelNotFoundError(
                    f"Offline checkpoint for '{model_id}' does not exist at '{dest_file}'. "
                    f"External network egress is disabled.",
                    details={"model_id": model_id, "expected_path": str(dest_file)},
                )
            computed_sha256 = compute_sha256(dest_file)
            if computed_sha256 != profile.expected_sha256:
                dest_file.unlink(missing_ok=True)
                raise ModelChecksumError(
                    f"SHA256 checksum mismatch for offline model '{model_id}'. "
                    f"Expected: {profile.expected_sha256}, got: {computed_sha256}. File removed.",
                    details={
                        "model_id": model_id,
                        "expected_sha256": profile.expected_sha256,
                        "computed_sha256": computed_sha256,
                    },
                )
            self.write_metadata(model_id, computed_sha256)
            return dest_file

        # Check existing cached file
        if dest_file.is_file() and not force:
            computed_sha256 = compute_sha256(dest_file)
            if computed_sha256 == profile.expected_sha256:
                logger.info(f"Using verified cached weights for '{model_id}' at {dest_file}")
                self.write_metadata(model_id, computed_sha256)
                return dest_file
            else:
                logger.warning(
                    f"Cached file for '{model_id}' failed checksum. Removing invalid file and re-downloading."
                )
                dest_file.unlink(missing_ok=True)

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Download from Hugging Face Hub with resume_download=True (R1, R5)
        try:
            downloaded_path_str = hf_hub_download(
                repo_id=profile.source_repo,
                filename=profile.file_name,
                local_dir=str(dest_dir),
                resume_download=True,
            )
            downloaded_path = Path(downloaded_path_str)

            # Ensure downloaded file resides at expected dest_file
            if downloaded_path.resolve() != dest_file.resolve() and downloaded_path.exists():
                shutil.copy2(downloaded_path, dest_file)
        except Exception as exc:
            if not isinstance(exc, (ModelChecksumError, ModelNotFoundError)):
                raise ModelRegistryError(
                    f"Failed to download checkpoint '{model_id}' from Hugging Face Hub: {exc}",
                    details={"model_id": model_id, "repo": profile.source_repo},
                ) from exc
            raise

        # SHA256 checksum verification (R3, R4)
        if not dest_file.is_file():
            raise ModelRegistryError(
                f"Downloaded checkpoint file was not found at expected location '{dest_file}'"
            )

        computed_sha256 = compute_sha256(dest_file)
        if computed_sha256 != profile.expected_sha256:
            dest_file.unlink(missing_ok=True)
            raise ModelChecksumError(
                f"SHA256 checksum mismatch for downloaded checkpoint '{model_id}' ({profile.file_name}). "
                f"Expected: {profile.expected_sha256}, got: {computed_sha256}. Invalid file removed.",
                details={
                    "model_id": model_id,
                    "expected_sha256": profile.expected_sha256,
                    "computed_sha256": computed_sha256,
                },
            )

        self.write_metadata(model_id, computed_sha256)
        logger.info(f"Successfully acquired and verified '{model_id}' -> {dest_file}")
        return dest_file

    def verify_checkpoint(self, model_id: str) -> bool:
        """Verify checksum of existing local checkpoint. Returns True if valid, False otherwise."""
        profile = self.get_profile(model_id)
        dest_file = self.base_dir / model_id / profile.file_name
        if not dest_file.is_file():
            return False
        return compute_sha256(dest_file) == profile.expected_sha256

    def download_all(self, offline: bool = False, force: bool = False) -> Dict[str, Path]:
        """Download and verify all target models in the catalog."""
        results: Dict[str, Path] = {}
        for model_id in self.profiles:
            results[model_id] = self.download(model_id=model_id, offline=offline, force=force)
        return results


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for sva-download-models (R18)."""
    parser = argparse.ArgumentParser(
        prog="sva-download-models",
        description="SVA Sentinel Base Model Checkpoint Acquisition & Verification CLI",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model profile ID to download (e.g. yolo11n, yolo11x, qwen2.5-0.5b-instruct)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Acquire all predefined base models",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Operate strictly from existing local files without network egress",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify SHA256 checksums of local files without downloading",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all predefined base model profiles in catalog",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="models/base",
        help="Root directory for base model weights (default: models/base)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    downloader = BaseModelDownloader(base_dir=args.base_dir)

    if args.list:
        print("Available base model profiles:")
        for model_id in downloader.list_available_models():
            prof = downloader.get_profile(model_id)
            print(f"  - {prof.model_id}: {prof.model_name} ({prof.model_role}, {prof.format})")
        return 0

    if args.verify:
        models_to_check = [args.model] if args.model else downloader.list_available_models()
        all_ok = True
        for mid in models_to_check:
            is_valid = downloader.verify_checkpoint(mid)
            status = "VALID" if is_valid else "INVALID or MISSING"
            print(f"[{status}] {mid}")
            if not is_valid:
                all_ok = False
        return 0 if all_ok else 1

    if args.all:
        try:
            results = downloader.download_all(offline=args.offline)
            print(f"Successfully processed {len(results)} base models.")
            return 0
        except Exception as exc:
            print(f"Error acquiring models: {exc}", file=sys.stderr)
            return 1

    if args.model:
        try:
            path = downloader.download(args.model, offline=args.offline)
            print(f"Model '{args.model}' ready at: {path}")
            return 0
        except Exception as exc:
            print(f"Error acquiring model '{args.model}': {exc}", file=sys.stderr)
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
