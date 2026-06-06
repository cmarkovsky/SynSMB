"""
test_experiment.py
==================
Tests for the Experiment dataclass.

Run directly for step-by-step printed output:
    python test_experiment.py

Run via pytest for pass/fail:
    pytest test_experiment.py -v
"""

import numpy as np
import pytest
from syn_smb import Experiment


# ======================================================================
# 1. Default construction
# ======================================================================

class TestDefaultConstruction:

    def test_default_n_years(self):
        exp = Experiment()
        assert exp.n_years == 1000

    def test_default_n_members(self):
        exp = Experiment()
        assert exp.n_members == 30

    def test_default_seed(self):
        exp = Experiment()
        assert exp.seed == 0

    def test_default_band_scales_is_none(self):
        exp = Experiment()
        assert exp.band_scales is None

    def test_default_name(self):
        exp = Experiment()
        assert exp.name == "baseline"

    def test_custom_fields(self):
        exp = Experiment(
            n_years=500,
            n_members=10,
            seed=42,
            band_scales=[(0.8, 1.5, 2.0)],
            name="my_experiment",
            description="test",
        )
        assert exp.n_years == 500
        assert exp.n_members == 10
        assert exp.seed == 42
        assert exp.band_scales == [(0.8, 1.5, 2.0)]
        assert exp.name == "my_experiment"


# ======================================================================
# 2. Validation
# ======================================================================

class TestValidation:

    def test_raises_on_zero_n_years(self):
        with pytest.raises(ValueError, match="n_years must be positive"):
            Experiment(n_years=0)

    def test_raises_on_negative_n_years(self):
        with pytest.raises(ValueError, match="n_years must be positive"):
            Experiment(n_years=-100)

    def test_raises_on_zero_n_members(self):
        with pytest.raises(ValueError, match="n_members must be positive"):
            Experiment(n_members=0)

    def test_raises_on_negative_seed(self):
        with pytest.raises(ValueError, match="seed must be non-negative"):
            Experiment(seed=-1)

    def test_raises_on_non_list_band_scales(self):
        with pytest.raises(TypeError, match="band_scales must be a list"):
            Experiment(band_scales={(0.8, 1.5): 2.0})

    def test_raises_on_wrong_tuple_length(self):
        with pytest.raises(ValueError, match="3-tuple"):
            Experiment(band_scales=[(0.8, 1.5)])

    def test_raises_on_negative_pmin(self):
        with pytest.raises(ValueError, match="period bounds must be positive"):
            Experiment(band_scales=[(-1.0, 1.5, 2.0)])

    def test_raises_on_pmin_equal_pmax(self):
        with pytest.raises(ValueError, match="pmin must be strictly less than pmax"):
            Experiment(band_scales=[(1.5, 1.5, 2.0)])

    def test_raises_on_pmin_greater_than_pmax(self):
        with pytest.raises(ValueError, match="pmin must be strictly less than pmax"):
            Experiment(band_scales=[(2.0, 1.5, 2.0)])

    def test_raises_on_negative_factor(self):
        with pytest.raises(ValueError, match="non-negative"):
            Experiment(band_scales=[(0.8, 1.5, -1.0)])

    def test_zero_factor_is_valid(self):
        """Factor of 0 fully suppresses a band — should be allowed."""
        exp = Experiment(band_scales=[(0.8, 1.5, 0.0)])
        assert exp.band_scales[0][2] == 0.0

    def test_multiple_bands_validated(self):
        """Each entry in band_scales is validated independently."""
        with pytest.raises(ValueError):
            Experiment(band_scales=[
                (0.8, 1.5, 2.0),   # valid
                (8.0, 5.0, 1.0),   # invalid: pmin > pmax
            ])


# ======================================================================
# 3. Properties
# ======================================================================

class TestProperties:

    def test_rng_returns_generator(self):
        exp = Experiment(seed=42)
        assert isinstance(exp.rng, np.random.Generator)

    def test_rng_same_seed_same_sequence(self):
        """Same seed should produce identical random sequences."""
        exp = Experiment(seed=42)
        r1 = exp.rng.standard_normal(10)
        r2 = exp.rng.standard_normal(10)
        np.testing.assert_array_equal(r1, r2)

    def test_rng_different_seeds_different_sequences(self):
        exp1 = Experiment(seed=0)
        exp2 = Experiment(seed=1)
        r1 = exp1.rng.standard_normal(10)
        r2 = exp2.rng.standard_normal(10)
        assert not np.allclose(r1, r2)

    def test_n_months_correct(self):
        exp = Experiment(n_years=100)
        assert exp.n_months == 1200

    def test_has_band_scaling_false_for_baseline(self):
        exp = Experiment.baseline()
        assert not exp.has_band_scaling

    def test_has_band_scaling_true_when_set(self):
        exp = Experiment.annual_enhanced()
        assert exp.has_band_scaling


# ======================================================================
# 4. Preset constructors
# ======================================================================

class TestPresets:

    def test_baseline_has_no_band_scales(self):
        exp = Experiment.baseline()
        assert exp.band_scales is None

    def test_baseline_name(self):
        assert Experiment.baseline().name == "baseline"

    def test_annual_enhanced_band_covers_annual(self):
        exp = Experiment.annual_enhanced(factor=10.0)
        assert exp.band_scales is not None
        pmin, pmax, factor = exp.band_scales[0]
        # Annual band should include period=1.0 yr
        assert pmin < 1.0 < pmax
        assert factor == 10.0

    def test_annual_suppressed_factor_less_than_one(self):
        exp = Experiment.annual_suppressed(factor=0.1)
        _, _, factor = exp.band_scales[0]
        assert factor < 1.0

    def test_decadal_enhanced_band_covers_decadal(self):
        exp = Experiment.decadal_enhanced(factor=10.0)
        pmin, pmax, factor = exp.band_scales[0]
        # Decadal band should include period=10.0 yr
        assert pmin < 10.0 < pmax
        assert factor == 10.0

    def test_decadal_suppressed_factor_less_than_one(self):
        exp = Experiment.decadal_suppressed(factor=0.1)
        _, _, factor = exp.band_scales[0]
        assert factor < 1.0

    def test_preset_respects_n_years(self):
        exp = Experiment.annual_enhanced(n_years=500)
        assert exp.n_years == 500

    def test_preset_respects_n_members(self):
        exp = Experiment.annual_enhanced(n_members=10)
        assert exp.n_members == 10

    def test_preset_respects_seed(self):
        exp = Experiment.baseline(seed=99)
        assert exp.seed == 99


# ======================================================================
# 5. standard_suite
# ======================================================================

class TestStandardSuite:

    def test_suite_returns_five_experiments(self):
        suite = Experiment.standard_suite()
        assert len(suite) == 5

    def test_suite_first_is_baseline(self):
        suite = Experiment.standard_suite()
        assert suite[0].band_scales is None

    def test_suite_all_same_n_years(self):
        suite = Experiment.standard_suite(n_years=500)
        assert all(e.n_years == 500 for e in suite)

    def test_suite_all_same_n_members(self):
        suite = Experiment.standard_suite(n_members=10)
        assert all(e.n_members == 10 for e in suite)

    def test_suite_all_same_seed(self):
        suite = Experiment.standard_suite(seed=7)
        assert all(e.seed == 7 for e in suite)

    def test_suite_all_have_distinct_names(self):
        suite = Experiment.standard_suite()
        names = [e.name for e in suite]
        assert len(names) == len(set(names)), "Suite experiments should have unique names"

    def test_suite_contains_annual_enhanced(self):
        suite = Experiment.standard_suite()
        names = [e.name for e in suite]
        assert any("annual_enhanced" in n for n in names)

    def test_suite_contains_decadal_enhanced(self):
        suite = Experiment.standard_suite()
        names = [e.name for e in suite]
        assert any("decadal_enhanced" in n for n in names)


# ======================================================================
# 6. Serialisation
# ======================================================================

class TestSerialisation:

    def test_to_dict_returns_dict(self):
        exp = Experiment.annual_enhanced()
        d = exp.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_expected_keys(self):
        exp = Experiment.baseline()
        d = exp.to_dict()
        expected = {"name", "description", "n_years", "n_members", "seed", "band_scales"}
        assert set(d.keys()) == expected

    def test_roundtrip_baseline(self):
        exp = Experiment.baseline()
        reconstructed = Experiment.from_dict(exp.to_dict())
        assert reconstructed.name       == exp.name
        assert reconstructed.n_years    == exp.n_years
        assert reconstructed.n_members  == exp.n_members
        assert reconstructed.seed       == exp.seed
        assert reconstructed.band_scales == exp.band_scales

    def test_roundtrip_with_band_scales(self):
        exp = Experiment.annual_enhanced(factor=5.0)
        reconstructed = Experiment.from_dict(exp.to_dict())
        assert reconstructed.band_scales == exp.band_scales

    def test_roundtrip_preserves_validation(self):
        """from_dict should still trigger validation."""
        with pytest.raises(ValueError):
            Experiment.from_dict({
                "name": "bad", "description": "",
                "n_years": -1, "n_members": 10, "seed": 0,
                "band_scales": None,
            })


# ======================================================================
# 7. repr and display
# ======================================================================

class TestDisplay:

    def test_repr_contains_name(self):
        exp = Experiment.baseline()
        assert "baseline" in repr(exp)

    def test_repr_contains_n_years(self):
        exp = Experiment(n_years=500)
        assert "500" in repr(exp)

    def test_repr_contains_band_scales_when_set(self):
        exp = Experiment.annual_enhanced()
        assert "band_scales" in repr(exp)

    def test_repr_no_band_scales_for_baseline(self):
        exp = Experiment.baseline()
        assert "band_scales" not in repr(exp)

    def test_summary_runs_without_error(self, capsys):
        exp = Experiment.annual_enhanced()
        exp.summary()
        captured = capsys.readouterr()
        assert "annual" in captured.out.lower()


# ======================================================================
# Standalone debug runner
# ======================================================================

def run_debug():
    print("=" * 60)
    print("Experiment — step-by-step debug run")
    print("=" * 60)

    # --- Default construction ---
    print("\n[1] Default construction...")
    exp = Experiment()
    print(f"    {exp}")
    print(f"    n_months:       {exp.n_months}")
    print(f"    has_band_scaling: {exp.has_band_scaling}")

    # --- Preset constructors ---
    print("\n[2] Preset constructors...")
    presets = [
        Experiment.baseline(),
        Experiment.annual_enhanced(factor=10.0),
        Experiment.annual_suppressed(factor=0.1),
        Experiment.decadal_enhanced(factor=10.0),
        Experiment.decadal_suppressed(factor=0.1),
    ]
    for p in presets:
        bands = p.band_scales[0] if p.band_scales else "None"
        print(f"    {p.name:<35}  band_scales={bands}")

    # --- standard_suite() ---
    print("\n[3] standard_suite()...")
    suite = Experiment.standard_suite(n_years=1000, n_members=30, seed=0)
    print(f"    {len(suite)} experiments in suite:")
    for e in suite:
        e.summary()
        print()

    # --- rng reproducibility ---
    print("[4] RNG reproducibility check...")
    exp = Experiment(seed=42)
    r1 = exp.rng.standard_normal(5)
    r2 = exp.rng.standard_normal(5)
    print(f"    Same seed, two calls:")
    print(f"    r1: {r1.round(4)}")
    print(f"    r2: {r2.round(4)}")
    print(f"    Identical: {np.allclose(r1, r2)}")

    # --- Serialisation round-trip ---
    print("\n[5] Serialisation round-trip...")
    original = Experiment.annual_enhanced(factor=5.0, n_years=500, seed=7)
    d = original.to_dict()
    print(f"    to_dict(): {d}")
    restored = Experiment.from_dict(d)
    print(f"    from_dict(): {restored}")
    print(f"    Round-trip identical: {original == restored}")

    # --- Validation catches bad inputs ---
    print("\n[6] Validation — bad inputs are caught...")
    bad_inputs = [
        dict(n_years=-1),
        dict(n_members=0),
        dict(seed=-5),
        dict(band_scales=[(2.0, 1.0, 1.0)]),  # pmin > pmax
        dict(band_scales=[(0.8, 1.5, -2.0)]), # negative factor
    ]
    for kwargs in bad_inputs:
        try:
            Experiment(**kwargs)
            print(f"    {kwargs} — NOT caught (unexpected)")
        except (ValueError, TypeError) as e:
            print(f"    {kwargs} — caught: {e}")

    print("\n" + "=" * 60)
    print("Debug run complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_debug()