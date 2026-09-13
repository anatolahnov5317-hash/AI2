import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest import mock

import numpy as np

from text_factors import ModelConfig, TextFactorModel


def persistence_config(**overrides: Any) -> ModelConfig:
    values: dict[str, Any] = {
        "input_bits": 48,
        "active_bits_per_symbol": 4,
        "positions": 5,
        "frame_size": 3,
        "context_count": 5,
        "receptive_bits": 12,
        "point_count": 128,
        "output_bits": 48,
        "create_threshold": 3,
        "activation_threshold": 2,
        "min_active_points": 1,
        "probation_after": 2,
        "stable_after": 3,
        "max_clusters_per_point": 8,
        "prediction_vote_threshold": 1,
        "seed": 17,
    }
    values.update(overrides)
    return ModelConfig(**values)


def archive_arrays(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    with np.load(path, allow_pickle=False) as state:
        return {name: state[name].copy() for name in state.files}


def rewrite_archive(
    path: Path,
    *,
    changes: dict[str, np.ndarray[Any, Any]] | None = None,
    omit: str | None = None,
) -> None:
    arrays = archive_arrays(path)
    if changes:
        arrays.update(changes)
    if omit is not None:
        arrays.pop(omit)
    np.savez_compressed(path, **cast(dict[str, Any], arrays))


def rewrite_metadata(path: Path, update: Any) -> None:
    arrays = archive_arrays(path)
    metadata = json.loads(str(arrays["metadata"].item()))
    update(metadata)
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **cast(dict[str, Any], arrays))


def state_snapshot(model: TextFactorModel) -> tuple[Any, ...]:
    arrays = (
        model.encoder.codebook,
        model.memory.receptors,
        model.memory.output_map,
    )
    array_state = tuple(
        (value.dtype.str, value.shape, value.tobytes()) for value in arrays
    )
    clusters = tuple(
        (
            point,
            cluster.bits.tobytes(),
            cluster.bit_hits.tobytes(),
            cluster.created_at,
            cluster.last_seen,
            int(cluster.status),
            cluster.partial_hits,
            cluster.exact_hits,
            cluster.partial_errors,
            cluster.complete_errors,
        )
        for point, cluster in model.memory.iter_clusters()
    )
    return (
        model.training_mode,
        model.context_pair,
        model.memory.step,
        array_state,
        clusters,
        tuple(sorted(model.memory._nonempty_points)),
        tuple(tuple(sorted(values)) for values in model.memory._signatures),
    )


class PersistenceTests(unittest.TestCase):
    def test_v2_round_trip_preserves_trajectory_and_training_operator(self) -> None:
        first = TextFactorModel(persistence_config(), alphabet="abc")
        resumed_source = TextFactorModel(persistence_config(), alphabet="abc")
        for model in (first, resumed_source):
            for _ in range(3):
                model.learn_context_transform("abc", source_context=1, target_context=3)

        expected_before = resumed_source.predict_context_transform(
            "abc", source_context=1
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            resumed_source.save(path)
            resumed = TextFactorModel.load(path)

        self.assertEqual(resumed.training_mode, "supervised")
        self.assertEqual(resumed.context_pair, (1, 3))
        with self.assertRaisesRegex(ValueError, "operators"):
            resumed.learn_context_transform("abc", source_context=0, target_context=3)
        np.testing.assert_array_equal(
            resumed.predict_context_transform("abc", source_context=1).output,
            expected_before.output,
        )
        self.assertEqual(state_snapshot(resumed), state_snapshot(resumed_source))

        for model in (first, resumed):
            for window in ("bca", "cab", "abc"):
                model.learn_context_transform(
                    window, source_context=1, target_context=3
                )
        self.assertEqual(state_snapshot(resumed), state_snapshot(first))

    def test_transform_does_not_change_any_internal_state(self) -> None:
        model = TextFactorModel(persistence_config(), alphabet="abc")
        model.fit_text("abcabcabc", epochs=3)
        before = state_snapshot(model)
        model.transform_window("abc")
        self.assertEqual(state_snapshot(model), before)

    def test_v1_populated_load_requires_explicit_resume_mode(self) -> None:
        model = TextFactorModel(persistence_config(), alphabet="abc")
        model.fit_text("abcabc", epochs=2)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            model.save(path)

            def make_v1(metadata: dict[str, Any]) -> None:
                metadata["format_version"] = 1
                metadata.pop("training")

            rewrite_metadata(path, make_v1)
            loaded = TextFactorModel.load(path)

        self.assertEqual(loaded.training_mode, "unknown")
        loaded.predict_context_transform("abc")
        with self.assertRaisesRegex(ValueError, "assume_training_mode"):
            loaded.partial_fit_window("abc")
        loaded.assume_training_mode("unsupervised")
        loaded.partial_fit_window("abc")

    def test_empty_v1_can_learn_without_guessing_a_prior_mode(self) -> None:
        model = TextFactorModel(persistence_config(), alphabet="abc")
        model.memory.step = 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty-legacy.npz"
            model.save(path)

            def make_v1(metadata: dict[str, Any]) -> None:
                metadata["format_version"] = 1
                metadata.pop("training")

            rewrite_metadata(path, make_v1)
            loaded = TextFactorModel.load(path)

            resaved = Path(directory) / "resaved-v2.npz"
            loaded.save(resaved)
            reloaded = TextFactorModel.load(resaved)

        self.assertEqual(loaded.training_mode, "untrained")
        self.assertEqual(reloaded.training_mode, "untrained")
        reloaded.partial_fit_window("abc")
        self.assertEqual(reloaded.training_mode, "unsupervised")

    def test_corrupt_missing_and_malformed_archives_raise_value_error(self) -> None:
        model = TextFactorModel(persistence_config(), alphabet="abc")
        model.fit_text("abc", epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrupt = root / "corrupt.npz"
            corrupt.write_bytes(b"not a zip archive")
            with self.assertRaises(ValueError):
                TextFactorModel.load(corrupt)

            missing = root / "missing.npz"
            model.save(missing)
            rewrite_archive(missing, omit="cluster_bits")
            with self.assertRaisesRegex(ValueError, "missing data"):
                TextFactorModel.load(missing)

            malformed = root / "malformed.npz"
            model.save(malformed)
            arrays = archive_arrays(malformed)
            arrays["metadata"] = np.asarray("[]")
            np.savez_compressed(malformed, **cast(dict[str, Any], arrays))
            with self.assertRaisesRegex(ValueError, "metadata"):
                TextFactorModel.load(malformed)

            with self.assertRaises(ValueError):
                TextFactorModel.load(root / "does-not-exist.npz")

    def test_invalid_array_layout_dtype_offsets_status_and_counters_rejected(
        self,
    ) -> None:
        model = TextFactorModel(persistence_config(), alphabet="abc")
        model.fit_text("abc", epochs=1)

        def mutation_cases(
            arrays: dict[str, np.ndarray[Any, Any]],
        ) -> list[tuple[str, dict[str, np.ndarray[Any, Any]]]]:
            metadata = json.loads(str(arrays["metadata"].item()))
            malformed_metadata = dict(metadata)
            malformed_metadata["training"] = {"mode": "supervised"}
            return [
                (
                    "dtype",
                    {"cluster_offsets": arrays["cluster_offsets"].astype(np.float64)},
                ),
                (
                    "dimension",
                    {
                        "cluster_point_indices": arrays[
                            "cluster_point_indices"
                        ].reshape(1, -1)
                    },
                ),
                (
                    "offset",
                    {
                        "cluster_offsets": np.concatenate(
                            (
                                np.asarray([1], dtype=np.int64),
                                arrays["cluster_offsets"][1:],
                            )
                        )
                    },
                ),
                (
                    "status",
                    {"cluster_statuses": np.full_like(arrays["cluster_statuses"], 9)},
                ),
                (
                    "hits",
                    {"cluster_exact_hits": arrays["cluster_partial_hits"] + 1},
                ),
                (
                    "bit_hits",
                    {"cluster_bit_hits": np.zeros_like(arrays["cluster_bit_hits"])},
                ),
                (
                    "timestamp",
                    {
                        "cluster_last_seen": np.full_like(
                            arrays["cluster_last_seen"], metadata["step"] + 1
                        )
                    },
                ),
                (
                    "training metadata",
                    {
                        "metadata": np.asarray(
                            json.dumps(malformed_metadata, sort_keys=True)
                        )
                    },
                ),
            ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.npz"
            model.save(original)
            arrays = archive_arrays(original)
            self.assertGreater(len(arrays["cluster_point_indices"]), 0)
            for name, changes in mutation_cases(arrays):
                with self.subTest(name=name):
                    path = root / f"{name.replace(' ', '-')}.npz"
                    model.save(path)
                    rewrite_archive(path, changes=changes)
                    with self.assertRaises(ValueError):
                        TextFactorModel.load(path)

    def test_resource_limits_are_enforced_and_configurable(self) -> None:
        model = TextFactorModel(persistence_config(), alphabet="abc")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            with self.assertRaisesRegex(ValueError, "model file"):
                TextFactorModel.load(path, max_file_bytes=path.stat().st_size - 1)
            with self.assertRaisesRegex(ValueError, "uncompressed payload"):
                TextFactorModel.load(path, max_uncompressed_bytes=1)
            with self.assertRaisesRegex(ValueError, "model member"):
                TextFactorModel.load(path, max_array_bytes=1)
            loaded = TextFactorModel.load(
                path,
                max_file_bytes=path.stat().st_size,
                max_uncompressed_bytes=10 * 1024**2,
                max_array_bytes=10 * 1024**2,
            )
        self.assertEqual(loaded.summary(), model.summary())

    def test_failed_save_does_not_replace_existing_file(self) -> None:
        model = TextFactorModel(persistence_config(), alphabet="abc")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model.npz"
            path.write_bytes(b"existing")
            with (
                mock.patch(
                    "text_factors.model.np.savez_compressed",
                    side_effect=RuntimeError("write failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "write failed"),
            ):
                model.save(path)
            self.assertEqual(path.read_bytes(), b"existing")
            self.assertEqual(list(root.glob(".model.npz.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
