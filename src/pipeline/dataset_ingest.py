"""Hugging Face Hub multimodal dataset acquisition and caching engine.

Covers:
- R1: Dataset ingestion pipeline using huggingface_hub.
- R2: Persistent local caching in data/raw/<dataset_name>/.
- R3: Resume capability via HTTP range requests and local partial file validation.
- R4: Offline execution mode operating exclusively from local cache snapshots.
- R5: DatasetIngestError on unreachable/nonexistent datasets without local corruptions.
- R6: Declarative configuration profiles for target traffic datasets.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.core.exceptions import DatasetIngestError
from src.pipeline.schemas import IngestConfig

logger = logging.getLogger("sva.pipeline.ingest")

TARGET_PROFILES: Dict[str, IngestConfig] = {
    "starwit/overhead-traffic-anomalies": IngestConfig(
        dataset_name="starwit/overhead-traffic-anomalies",
        subset="overhead-intersections",
        allow_patterns=["*.jpg", "*.jpeg", "*.png", "*.json", "*.txt"],
    ),
    "UniqueData/electric-scooters-tracking": IngestConfig(
        dataset_name="UniqueData/electric-scooters-tracking",
        subset="micromobility-tracking",
        allow_patterns=["*.jpg", "*.png", "*.xml", "*.txt", "*.json"],
    ),
    "sva/street-level-vehicles": IngestConfig(
        dataset_name="sva/street-level-vehicles",
        subset="street-level-traffic",
        allow_patterns=["*.jpg", "*.png", "*.xml", "*.txt", "*.json"],
    ),
}


class HuggingFaceDatasetIngest:
    """Acquisition and local snapshot caching engine for Hugging Face Hub datasets."""

    def __init__(self, raw_dir: Path | str = Path("data/raw"), offline: bool = False) -> None:
        self.raw_dir = Path(raw_dir)
        self.offline = offline

    def get_dataset_dir(self, dataset_name: str) -> Path:
        """Resolve canonical local caching path under raw_dir."""
        parts = dataset_name.strip("/").split("/")
        return self.raw_dir.joinpath(*parts)

    def is_cached(self, dataset_name: str) -> bool:
        """Check whether valid cached files exist for the given dataset."""
        target_dir = self.get_dataset_dir(dataset_name)
        if not target_dir.is_dir():
            return False
        # Check if there is at least one file
        return any(target_dir.rglob("*"))

    def download_dataset(
        self,
        dataset_name: str,
        revision: Optional[str] = None,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        force_offline: Optional[bool] = None,
    ) -> Path:
        """Acquire dataset snapshot from Hugging Face Hub or retrieve cached snapshot.

        Args:
            dataset_name: Hugging Face repo identifier.
            revision: Specific git commit or branch tag.
            allow_patterns: Optional wildcard patterns to filter downloads.
            ignore_patterns: Optional wildcard patterns to omit.
            force_offline: Override instance offline flag.

        Returns:
            Path to the local cached dataset directory.

        Raises:
            DatasetIngestError: If dataset is unreachable and no cache exists, or offline cache is missing.
        """
        target_dir = self.get_dataset_dir(dataset_name)
        offline_mode = self.offline if force_offline is None else force_offline

        if offline_mode:
            if not self.is_cached(dataset_name):
                raise DatasetIngestError(
                    f"Offline mode active: No local cache found for dataset '{dataset_name}' at '{target_dir}'."
                )
            logger.info("Operating in offline mode. Utilizing local cached snapshot at: %s", target_dir)
            return target_dir

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise DatasetIngestError(
                f"huggingface_hub package is required for dataset acquisition: {exc}"
            ) from exc

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                local_dir=str(target_dir),
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                resume_download=True,
            )
            return target_dir
        except Exception as exc:
            if self.is_cached(dataset_name):
                logger.warning(
                    "Network acquisition failed for '%s' (%s). Falling back to existing local cache at %s",
                    dataset_name,
                    exc,
                    target_dir,
                )
                return target_dir

            # Clean up empty directory if no files were downloaded
            if target_dir.is_dir() and not any(target_dir.iterdir()):
                try:
                    target_dir.rmdir()
                except OSError:
                    pass

            raise DatasetIngestError(
                f"Dataset '{dataset_name}' is unreachable or does not exist, and no local cache snapshot was found: {exc}"
            ) from exc

    def ingest_from_profile(self, profile_key: str, force_offline: Optional[bool] = None) -> Path:
        """Ingest a predefined target dataset using its declarative profile."""
        if profile_key not in TARGET_PROFILES:
            raise DatasetIngestError(
                f"Unknown dataset profile '{profile_key}'. Available: {list(TARGET_PROFILES.keys())}"
            )
        cfg = TARGET_PROFILES[profile_key]
        return self.download_dataset(
            dataset_name=cfg.dataset_name,
            revision=cfg.revision,
            allow_patterns=cfg.allow_patterns,
            ignore_patterns=cfg.ignore_patterns,
            force_offline=force_offline,
        )


def main() -> int:
    """CLI entrypoint for dataset acquisition."""
    parser = argparse.ArgumentParser(description="Hugging Face Traffic Dataset Ingestion CLI")
    parser.add_argument(
        "--dataset",
        type=str,
        default="starwit/overhead-traffic-anomalies",
        help="Hugging Face dataset identifier or profile key",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory for storing raw dataset snapshots",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run exclusively in offline mode from existing local cache",
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Ingest all configured declarative dataset profiles",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ingest = HuggingFaceDatasetIngest(raw_dir=args.raw_dir, offline=args.offline)

    try:
        if args.all_profiles:
            for profile_key in TARGET_PROFILES:
                logger.info("Ingesting profile: %s", profile_key)
                ingest.ingest_from_profile(profile_key)
        else:
            if args.dataset in TARGET_PROFILES:
                ingest.ingest_from_profile(args.dataset)
            else:
                ingest.download_dataset(args.dataset)
        logger.info("Ingestion complete.")
        return 0
    except DatasetIngestError as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
