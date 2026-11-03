"""Deterministic synthetic traffic telemetry generator for forensic auditor fine-tuning.

Covers:
- R9: Deterministic dataset generator synthesizing 1,000 balanced samples (seed=42).
- R10: Balanced infraction distributions (wrong-way vehicular, micromobility, speeding, benign controls).
- R11: Serialization into structured JSONL pairing system/user prompts with IncidentAuditVerdict.
- R12: 100% schema validation and class/violation distribution reporting.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from src.core.exceptions import TelemetryGenerationError
from src.pipeline.schemas import (
    CANONICAL_CLASSES,
    AuditorTrainingSample,
    IncidentAuditVerdict,
    RiskLevel,
    TrafficTelemetryRecord,
)

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous residential traffic forensic auditor. "
    "Analyze the vehicle telemetry and lane constraints to determine if a traffic infraction occurred. "
    "Output strictly a valid JSON object matching the IncidentAuditVerdict schema with no conversational markdown or preamble."
)

RESIDENTIAL_LANES = [
    {"lane_id": "Lane-1-EB", "legal_heading": 90.0, "speed_limit": 25.0, "pedestrian_zone": False, "description": "Eastbound One-Way Residential Street"},
    {"lane_id": "Lane-2-WB", "legal_heading": 270.0, "speed_limit": 25.0, "pedestrian_zone": False, "description": "Westbound One-Way Residential Street"},
    {"lane_id": "Lane-3-NB", "legal_heading": 0.0, "speed_limit": 30.0, "pedestrian_zone": False, "description": "Northbound Two-Way Collector Avenue"},
    {"lane_id": "Lane-4-SB", "legal_heading": 180.0, "speed_limit": 30.0, "pedestrian_zone": False, "description": "Southbound Two-Way Collector Avenue"},
    {"lane_id": "Walkway-North", "legal_heading": 90.0, "speed_limit": 10.0, "pedestrian_zone": True, "description": "North Sidewalk and Pedestrian Crossing"},
    {"lane_id": "Walkway-South", "legal_heading": 270.0, "speed_limit": 10.0, "pedestrian_zone": True, "description": "South Sidewalk and School Crosswalk"},
]


class TelemetryGenerator:
    """Synthesizes high-fidelity residential traffic telemetry for LLM auditor training."""

    def __init__(self, seed: int = 42, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.seed = seed
        self.system_prompt = system_prompt
        self.rng = random.Random(seed)

    def _generate_trajectory(
        self, start_x: float, start_y: float, heading: float, speed_kmh: float, duration_s: float
    ) -> List[Tuple[float, float]]:
        """Simulate discrete (x, y) coordinates along heading angle at 10Hz."""
        points: List[Tuple[float, float]] = []
        import math

        rad = math.radians(heading)
        speed_ms = speed_kmh / 3.6
        dt = 0.1  # 100ms per point
        num_points = max(5, int(duration_s / dt))

        curr_x, curr_y = start_x, start_y
        for _ in range(num_points):
            points.append((round(curr_x, 2), round(curr_y, 2)))
            # Slight random jitter
            jitter_x = self.rng.gauss(0, 0.05)
            jitter_y = self.rng.gauss(0, 0.05)
            curr_x += speed_ms * dt * math.sin(rad) + jitter_x
            curr_y += speed_ms * dt * math.cos(rad) + jitter_y

        return points

    def generate_sample(self, sample_idx: int, scenario_type: str) -> AuditorTrainingSample:
        """Synthesize a single training sample for the specified scenario."""
        incident_id = f"INC-{sample_idx + 1:05d}"
        iso_timestamp = datetime(2026, 11, 2, 10, 0, 0, tzinfo=timezone.utc).isoformat()

        if scenario_type == "vehicular_wrong_way":
            # 30% Wrong-way vehicular ingress (cars, motorcycles)
            entity = self.rng.choice(["car", "motorcycle"])
            lane = self.rng.choice([l for l in RESIDENTIAL_LANES if not l["pedestrian_zone"] and "One-Way" in l["description"]])
            legal_heading = lane["legal_heading"]
            # Opposite direction with slight angular variance
            observed_heading = (legal_heading + 180.0 + self.rng.uniform(-20.0, 20.0)) % 360.0
            speed = round(self.rng.uniform(22.0, 48.0), 1)
            speed_limit = lane["speed_limit"]
            pedestrian_zone = False
            is_wrong_way = True
            violation_detected = True

            is_speeding = speed > speed_limit
            if is_speeding and speed > 40.0:
                risk_level = RiskLevel.CRITICAL
                summary = f"{entity.capitalize()} traveling wrong-way at excessive speed ({speed} km/h) on one-way {lane['lane_id']}."
            else:
                risk_level = RiskLevel.HIGH
                summary = f"{entity.capitalize()} entered one-way {lane['lane_id']} in opposite direction ({observed_heading:.1f}° vs legal {legal_heading:.1f}°)."

            reasoning = (
                f"The observed movement heading of {observed_heading:.1f}° directly opposes the designated legal heading "
                f"of {legal_heading:.1f}° for one-way lane {lane['lane_id']} (angular deviation > 160°). "
                f"The vehicle was clocked at {speed} km/h against a posted limit of {speed_limit} km/h, "
                f"generating a severe head-on collision risk rated as {risk_level.value.upper()}."
            )

        elif scenario_type == "micromobility_infraction":
            # 25% Sidewalk / wrong-way micromobility (bicycles, e-scooters)
            entity = self.rng.choice(["bicycle", "e-scooter"])
            # Either riding on restricted sidewalk or counter-flow on roadway
            on_sidewalk = self.rng.choice([True, False])
            if on_sidewalk:
                lane = self.rng.choice([l for l in RESIDENTIAL_LANES if l["pedestrian_zone"]])
                legal_heading = lane["legal_heading"]
                observed_heading = (legal_heading + self.rng.uniform(-30.0, 30.0)) % 360.0
                speed = round(self.rng.uniform(16.0, 28.0), 1)
                speed_limit = lane["speed_limit"]
                pedestrian_zone = True
                is_wrong_way = False
                violation_detected = True
                risk_level = RiskLevel.HIGH if speed > 20.0 else RiskLevel.MEDIUM
                summary = f"Unauthorized {entity} operating on pedestrian {lane['lane_id']} at {speed} km/h."
                reasoning = (
                    f"Operating a motorized or fast micromobility device ({entity}) on designated pedestrian infrastructure "
                    f"({lane['lane_id']}) violates municipal safety ordinances. Recorded velocity of {speed} km/h exceeds "
                    f"pedestrian safe flow threshold of {speed_limit} km/h, endangering vulnerable pedestrians."
                )
            else:
                lane = self.rng.choice([l for l in RESIDENTIAL_LANES if not l["pedestrian_zone"]])
                legal_heading = lane["legal_heading"]
                observed_heading = (legal_heading + 180.0 + self.rng.uniform(-25.0, 25.0)) % 360.0
                speed = round(self.rng.uniform(18.0, 32.0), 1)
                speed_limit = lane["speed_limit"]
                pedestrian_zone = False
                is_wrong_way = True
                violation_detected = True
                risk_level = RiskLevel.HIGH
                summary = f"{entity.capitalize()} operating counter-flow on {lane['lane_id']} at {speed} km/h."
                reasoning = (
                    f"The {entity} traveled at heading {observed_heading:.1f}°, directly opposing lane legal vector "
                    f"{legal_heading:.1f}°. Counter-flow micromobility severely limits motorist reaction margins at intersections."
                )

        elif scenario_type == "speeding_violation":
            # 20% Excessive speeding in residential zone (> 35 km/h in 25-30 km/h zones)
            entity = self.rng.choice(["car", "motorcycle"])
            lane = self.rng.choice([l for l in RESIDENTIAL_LANES if not l["pedestrian_zone"]])
            legal_heading = lane["legal_heading"]
            # Correct heading (not wrong-way)
            observed_heading = (legal_heading + self.rng.uniform(-10.0, 10.0)) % 360.0
            speed = round(self.rng.uniform(38.0, 65.0), 1)
            speed_limit = lane["speed_limit"]
            pedestrian_zone = False
            is_wrong_way = False
            violation_detected = True

            if speed >= 50.0:
                risk_level = RiskLevel.CRITICAL
            elif speed >= 40.0:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.MEDIUM

            summary = f"{entity.capitalize()} exceeding residential limit on {lane['lane_id']} ({speed} km/h in {speed_limit} km/h zone)."
            reasoning = (
                f"The {entity} adhered to legal direction of travel ({observed_heading:.1f}° vs {legal_heading:.1f}°), "
                f"but recorded ground speed was {speed} km/h, substantially surpassing the posted residential limit of "
                f"{speed_limit} km/h (speed delta +{speed - speed_limit:.1f} km/h). Classified as {risk_level.value.upper()} hazard."
            )

        else:
            # 25% Benign compliant traffic baseline
            entity = self.rng.choice(["car", "motorcycle", "bicycle", "e-scooter", "pedestrian"])
            if entity == "pedestrian":
                lane = self.rng.choice([l for l in RESIDENTIAL_LANES if l["pedestrian_zone"]])
                speed = round(self.rng.uniform(3.0, 6.0), 1)
                pedestrian_zone = True
            elif entity in ["bicycle", "e-scooter"]:
                lane = self.rng.choice([l for l in RESIDENTIAL_LANES if not l["pedestrian_zone"]])
                speed = round(self.rng.uniform(10.0, 22.0), 1)
                pedestrian_zone = False
            else:
                lane = self.rng.choice([l for l in RESIDENTIAL_LANES if not l["pedestrian_zone"]])
                speed = round(self.rng.uniform(18.0, 25.0), 1)
                pedestrian_zone = False

            legal_heading = lane["legal_heading"]
            observed_heading = (legal_heading + self.rng.uniform(-8.0, 8.0)) % 360.0
            speed_limit = lane["speed_limit"]
            is_wrong_way = False
            violation_detected = False
            risk_level = RiskLevel.LOW
            summary = f"Normal compliant traffic movement for {entity} in {lane['lane_id']}."
            reasoning = (
                f"Trajectory conforms with designated flow direction ({observed_heading:.1f}° within tolerance of legal {legal_heading:.1f}°). "
                f"Operating speed of {speed} km/h remains safely below posted limit of {speed_limit} km/h. No infraction detected."
            )

        # Generate trajectory coordinates
        duration = round(self.rng.uniform(1.5, 4.0), 1)
        trajectory = self._generate_trajectory(
            start_x=0.0, start_y=0.0, heading=observed_heading, speed_kmh=speed, duration_s=duration
        )

        try:
            telemetry = TrafficTelemetryRecord(
                incident_id=incident_id,
                timestamp=iso_timestamp,
                entity_type=entity,
                lane_id=lane["lane_id"],
                lane_legal_heading=legal_heading,
                observed_heading=observed_heading,
                estimated_speed_kmh=speed,
                speed_limit_kmh=speed_limit,
                trajectory_points=trajectory,
                dwell_time_seconds=duration,
                pedestrian_zone=pedestrian_zone,
            )

            verdict = IncidentAuditVerdict(
                incident_id=incident_id,
                timestamp=iso_timestamp,
                entity_type=entity,
                lane_id=lane["lane_id"],
                is_wrong_way=is_wrong_way,
                heading_degrees=observed_heading,
                estimated_speed_kmh=speed,
                speed_limit_kmh=speed_limit,
                risk_level=risk_level,
                violation_detected=violation_detected,
                summary=summary[:300],
                reasoning=reasoning,
            )
        except ValidationError as e:
            raise TelemetryGenerationError(f"Validation failed for generated sample {incident_id}: {e}") from e

        user_prompt = telemetry.format_telemetry_prompt()

        return AuditorTrainingSample(
            sample_id=incident_id,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            ground_truth_verdict=verdict,
            telemetry=telemetry,
        )

    def generate_dataset(self, num_samples: int = 1000) -> List[AuditorTrainingSample]:
        """Generate balanced dataset across the 4 scenario types (R9, R10)."""
        if num_samples <= 0:
            raise TelemetryGenerationError(f"num_samples must be positive, got {num_samples}")

        # Distribution quotas: 30% wrong-way vehicle, 25% micromobility, 20% speeding, 25% benign
        n_wrong_way = int(num_samples * 0.30)
        n_micromobility = int(num_samples * 0.25)
        n_speeding = int(num_samples * 0.20)
        n_benign = num_samples - (n_wrong_way + n_micromobility + n_speeding)

        scenario_plan = (
            ["vehicular_wrong_way"] * n_wrong_way
            + ["micromobility_infraction"] * n_micromobility
            + ["speeding_violation"] * n_speeding
            + ["benign_control"] * n_benign
        )
        self.rng.shuffle(scenario_plan)

        samples: List[AuditorTrainingSample] = []
        for idx, scenario in enumerate(scenario_plan):
            sample = self.generate_sample(idx, scenario)
            samples.append(sample)

        return samples

    def calculate_distribution_stats(self, samples: List[AuditorTrainingSample]) -> Dict[str, Any]:
        """Compute statistical breakdown of generated dataset (R12)."""
        total = len(samples)
        if total == 0:
            return {"total": 0}

        entity_counts: Dict[str, int] = {}
        risk_counts: Dict[str, int] = {}
        violation_count = 0
        wrong_way_count = 0

        for s in samples:
            v = s.ground_truth_verdict
            entity_counts[v.entity_type] = entity_counts.get(v.entity_type, 0) + 1
            risk_counts[v.risk_level.value] = risk_counts.get(v.risk_level.value, 0) + 1
            if v.violation_detected:
                violation_count += 1
            if v.is_wrong_way:
                wrong_way_count += 1

        return {
            "total_samples": total,
            "violations_count": violation_count,
            "violations_percentage": round((violation_count / total) * 100, 2),
            "wrong_way_count": wrong_way_count,
            "wrong_way_percentage": round((wrong_way_count / total) * 100, 2),
            "entity_distribution": entity_counts,
            "risk_distribution": risk_counts,
        }

    def save_jsonl(self, samples: List[AuditorTrainingSample], output_path: str | Path) -> Path:
        """Serialize dataset into JSONL format with ChatML formatted training records (R11)."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                record = {
                    "sample_id": s.sample_id,
                    "messages": s.to_qwen_chat_format()["messages"],
                    "telemetry": s.telemetry.model_dump(),
                    "ground_truth": s.ground_truth_verdict.model_dump(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Persisted {len(samples)} training samples to {out_path}")
        return out_path


def generate_synthetic_telemetry_dataset(
    num_samples: int = 1000,
    seed: int = 42,
    output_path: Optional[str | Path] = None,
) -> Tuple[List[AuditorTrainingSample], Dict[str, Any]]:
    """Convenience functional interface to generate and optionally persist telemetry."""
    generator = TelemetryGenerator(seed=seed)
    samples = generator.generate_dataset(num_samples=num_samples)
    stats = generator.calculate_distribution_stats(samples)

    if output_path is not None:
        generator.save_jsonl(samples, output_path)

    return samples, stats


def main() -> None:
    """CLI entrypoint for sva-generate-telemetry."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic traffic telemetry dataset for LLM auditor fine-tuning.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of telemetry samples to generate (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument("--output", type=str, default="data/telemetry/synthetic_auditor_train.jsonl", help="Output JSONL path")

    args = parser.parse_args()
    samples, stats = generate_synthetic_telemetry_dataset(num_samples=args.samples, seed=args.seed, output_path=args.output)
    print(f"Generated {len(samples)} samples. Distribution stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()

