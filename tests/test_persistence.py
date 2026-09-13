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


def rewrite_as_legacy(path: Path, version: int) -> None:
    """Rewrite a v3 frequency archive using the actual v1/v2 member schema."""

    arrays = archive_arrays(path)
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata["format_version"] = version
    for name in (
        "consolidation_method",
        "coactivation_history_size",
        "coactivation_passes",
    ):
        metadata["config"].pop(name, None)
    if version == 1:
        metadata.pop("training")
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    for name in (
        "cluster_history_counts",
        "cluster_history_offsets",
        "cluster_history_values",
    ):
        arrays.pop(name)
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
            tuple(
                (row.dtype.str, row.shape, row.tobytes())
                for row in cluster.activation_history
            ),
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
    def test_v3_round_trip_preserves_trajectory_and_training_operator(self) -> None:
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

    def test_v3_coactivation_round_trip_preserves_continued_training(self) -> None:
        config = persistence_config(
            consolidation_method="coactivation",
            coactivation_history_size=4,
            coactivation_passes=2,
        )
        uninterrupted = TextFactorModel(config, alphabet="abc")
        saved_source = TextFactorModel(config, alphabet="abc")
        initial = ("abc", "bca", "abc", "cab")
        for model in (uninterrupted, saved_source):
            for window in initial:
                model.partial_fit_window(window)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coactivation.npz"
            saved_source.save(path)
            arrays = archive_arrays(path)
            self.assertEqual(arrays["cluster_history_counts"].dtype, np.int32)
            self.assertEqual(arrays["cluster_history_offsets"].dtype, np.int64)
            self.assertEqual(arrays["cluster_history_values"].dtype, np.bool_)
            resumed = TextFactorModel.load(path)

        self.assertEqual(state_snapshot(resumed), state_snapshot(saved_source))
        self.assertTrue(
            all(
                cluster.activation_history
                for _, cluster in resumed.memory.iter_clusters()
            )
        )
        for model in (uninterrupted, resumed):
            for window in ("bca", "cab", "abc", "abc"):
                model.partial_fit_window(window)
        self.assertEqual(state_snapshot(resumed), state_snapshot(uninterrupted))

    def test_frequency_v3_persists_empty_histories(self) -> None:
        model = TextFactorModel(persistence_config(), alphabet="abc")
        model.fit_text("abcabc", epochs=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frequency.npz"
            model.save(path)
            arrays = archive_arrays(path)
            loaded = TextFactorModel.load(path)

        self.assertTrue(np.all(arrays["cluster_history_counts"] == 0))
        self.assertTrue(np.all(arrays["cluster_history_offsets"] == 0))
        self.assertEqual(arrays["cluster_history_values"].shape, (0,))
        self.assertTrue(
            all(
                cluster.activation_history == []
                for _, cluster in loaded.memory.iter_clusters()
            )
        )

    def test_actual_v2_schema_loads_with_new_frequency_defaults(self) -> None:
        model = TextFactorModel(persistence_config(), alphabet="abc")
        model.fit_text("abcabc", epochs=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-v2.npz"
            model.save(path)
            rewrite_as_legacy(path, 2)
            self.assertFalse(
                any(
                    name.startswith("cluster_history_") for name in archive_arrays(path)
                )
            )
            loaded = TextFactorModel.load(path)

        self.assertEqual(loaded.config.consolidation_method, "frequency")
        self.assertEqual(state_snapshot(loaded), state_snapshot(model))

    def test_legacy_schema_rejects_coactivation_config_without_history(self) -> None:
        model = TextFactorModel(persistence_config(), alphabet="abc")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-v2.npz"
            model.save(path)
            rewrite_as_legacy(path, 2)
            rewrite_metadata(
                path,
                lambda metadata: metadata["config"].update(
                    {"consolidation_method": "coactivation"}
                ),
            )
            with self.assertRaisesRegex(ValueError, "cannot restore coactivation"):
                TextFactorModel.load(path)

    def test_malformed_v3_history_payloads_are_rejected(self) -> None:
        config = persistence_config(
            consolidation_method="coactivation",
            coactivation_history_size=4,
            coactivation_passes=2,
        )
        model = TextFactorModel(config, alphabet="abc")
        model.fit_text("abcabc", epochs=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "coactivation.npz"
            model.save(original)
            arrays = archive_arrays(original)
            self.assertGreater(len(arrays["cluster_history_values"]), 0)

            bad_count = arrays["cluster_history_counts"].copy()
            bad_count[0] = arrays["cluster_partial_hits"][0] + 1
            multirow_indices = np.flatnonzero(arrays["cluster_history_counts"] > 1)
            self.assertGreater(len(multirow_indices), 0)
            dropped_index = int(multirow_indices[0])
            dropped_width = int(
                arrays["cluster_offsets"][dropped_index + 1]
                - arrays["cluster_offsets"][dropped_index]
            )
            dropped_start = int(arrays["cluster_history_offsets"][dropped_index])
            dropped_counts = arrays["cluster_history_counts"].copy()
            dropped_counts[dropped_index] -= 1
            dropped_offsets = arrays["cluster_history_offsets"].copy()
            dropped_offsets[dropped_index + 1 :] -= dropped_width
            dropped_values = np.concatenate(
                (
                    arrays["cluster_history_values"][:dropped_start],
                    arrays["cluster_history_values"][dropped_start + dropped_width :],
                )
            )
            cases = {
                "dtype": {
                    "cluster_history_values": arrays["cluster_history_values"].astype(
                        np.int8
                    )
                },
                "count": {"cluster_history_counts": bad_count},
                "offset": {
                    "cluster_history_offsets": np.concatenate(
                        (
                            arrays["cluster_history_offsets"][:-1],
                            arrays["cluster_history_offsets"][-1:] + 1,
                        )
                    )
                },
                "logical-bound": {
                    "cluster_history_values": np.zeros(
                        len(arrays["cluster_bits"]) * config.coactivation_history_size
                        + 1,
                        dtype=np.bool_,
                    )
                },
                "missing-row": {
                    "cluster_history_counts": dropped_counts,
                    "cluster_history_offsets": dropped_offsets,
                    "cluster_history_values": dropped_values,
                },
            }
            for name, changes in cases.items():
                with self.subTest(name=name):
                    path = root / f"bad-{name}.npz"
                    rewrite_archive(original, changes=changes)
                    original.replace(path)
                    with self.assertRaises(ValueError):
                        TextFactorModel.load(path)
                    model.save(original)

            frequency_metadata = root / "frequency-metadata.npz"
            model.save(frequency_metadata)
            rewrite_metadata(
                frequency_metadata,
                lambda metadata: metadata["config"].update(
                    {"consolidation_method": "frequency"}
                ),
            )
            with self.assertRaisesRegex(ValueError, "frequency clusters"):
                TextFactorModel.load(frequency_metadata)

            missing_history = root / "missing-history.npz"
            frequency = TextFactorModel(persistence_config(), alphabet="abc")
            frequency.fit_text("abc", epochs=1)
            frequency.save(missing_history)
            rewrite_metadata(
                missing_history,
                lambda metadata: metadata["config"].update(
                    {"consolidation_method": "coactivation"}
                ),
            )
            with self.assertRaisesRegex(ValueError, "history count must equal"):
                TextFactorModel.load(missing_history)

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

            rewrite_as_legacy(path, 1)
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

            rewrite_as_legacy(path, 1)
            loaded = TextFactorModel.load(path)

            resaved = Path(directory) / "resaved-v3.npz"
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
