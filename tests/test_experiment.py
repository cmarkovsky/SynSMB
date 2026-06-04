import numpy as np
import pytest

from syn_smb.core.experiment import (
    ANNUAL_ENHANCED,
    ANNUAL_SUPPRESSED,
    BASELINE,
    DECADAL_ENHANCED,
    DECADAL_SUPPRESSED,
    MEAN_ONLY,
    Experiment,
    ExperimentSuite,
)


# ---------------------------------------------------------------------------
# Experiment — instantiation
# ---------------------------------------------------------------------------

class TestExperimentInstantiation:

    def test_default_values(self):
        exp = Experiment()
        assert exp.name == "baseline"
        assert exp.band_scales == {}
        assert exp.include_noise is True

    def test_custom_name(self):
        exp = Experiment(name="my_experiment")
        assert exp.name == "my_experiment"

    def test_custom_band_scales(self):
        scales = {"annual": (0.8, 1.2, 5.0)}
        exp = Experiment(band_scales=scales)
        assert exp.band_scales == scales

    def test_include_noise_false(self):
        exp = Experiment(include_noise=False)
        assert exp.include_noise is False

    def test_band_scales_are_independent(self):
        """Two Experiment instances must not share the same dict object."""
        a = Experiment()
        b = Experiment()
        assert a.band_scales is not b.band_scales


# ---------------------------------------------------------------------------
# Experiment.build_scale_profile — correctness
# ---------------------------------------------------------------------------

class TestBuildScaleProfile:

    def test_output_shape_matches_freqs(self, short_freqs):
        exp = Experiment()
        profile = exp.build_scale_profile(short_freqs)
        assert profile.shape == short_freqs.shape

    def test_dc_always_zero_baseline(self, short_freqs):
        profile = BASELINE.build_scale_profile(short_freqs)
        assert profile[0] == 0.0

    def test_dc_always_zero_with_bands(self, short_freqs):
        """DC must be zero even if a band definition starts at 0."""
        exp = Experiment(band_scales={"low": (0.0, 0.5, 5.0)})
        profile = exp.build_scale_profile(short_freqs)
        assert profile[0] == 0.0

    def test_baseline_all_ones_except_dc(self, short_freqs):
        profile = BASELINE.build_scale_profile(short_freqs)
        assert np.all(profile[1:] == 1.0)

    def test_mean_only_all_zeros(self, short_freqs):
        profile = MEAN_ONLY.build_scale_profile(short_freqs)
        assert np.all(profile == 0.0)

    def test_annual_enhanced_in_band(self, monthly_freqs):
        profile = ANNUAL_ENHANCED.build_scale_profile(monthly_freqs)
        annual_mask = (monthly_freqs >= 0.8) & (monthly_freqs <= 1.2)
        assert np.all(profile[annual_mask] == 10.0)

    def test_annual_enhanced_outside_band(self, monthly_freqs):
        profile = ANNUAL_ENHANCED.build_scale_profile(monthly_freqs)
        annual_mask = (monthly_freqs >= 0.8) & (monthly_freqs <= 1.2)
        outside = ~annual_mask
        outside[0] = False  # exclude DC
        assert np.all(profile[outside] == 1.0)

    def test_annual_suppressed_in_band(self, monthly_freqs):
        profile = ANNUAL_SUPPRESSED.build_scale_profile(monthly_freqs)
        annual_mask = (monthly_freqs >= 0.8) & (monthly_freqs <= 1.2)
        assert np.all(profile[annual_mask] == pytest.approx(0.1))

    def test_decadal_enhanced_in_band(self, monthly_freqs):
        profile = DECADAL_ENHANCED.build_scale_profile(monthly_freqs)
        decadal_mask = (monthly_freqs >= 0.05) & (monthly_freqs <= 0.15)
        assert np.all(profile[decadal_mask] == 10.0)

    def test_decadal_suppressed_in_band(self, monthly_freqs):
        profile = DECADAL_SUPPRESSED.build_scale_profile(monthly_freqs)
        decadal_mask = (monthly_freqs >= 0.05) & (monthly_freqs <= 0.15)
        assert np.all(profile[decadal_mask] == pytest.approx(0.1))

    def test_multiple_bands(self, monthly_freqs):
        """Both bands should be scaled independently."""
        exp = Experiment(
            band_scales={
                "annual":  (0.8, 1.2,  10.0),
                "decadal": (0.05, 0.15, 0.1),
            }
        )
        profile = exp.build_scale_profile(monthly_freqs)

        annual_mask  = (monthly_freqs >= 0.8)  & (monthly_freqs <= 1.2)
        decadal_mask = (monthly_freqs >= 0.05) & (monthly_freqs <= 0.15)

        assert np.all(profile[annual_mask]  == 10.0)
        assert np.all(profile[decadal_mask] == pytest.approx(0.1))

    def test_output_dtype_float(self, short_freqs):
        profile = BASELINE.build_scale_profile(short_freqs)
        assert profile.dtype == float


# ---------------------------------------------------------------------------
# Built-in presets — names and types
# ---------------------------------------------------------------------------

class TestBuiltInPresets:

    @pytest.mark.parametrize("preset, expected_name", [
        (BASELINE,           "baseline"),
        (MEAN_ONLY,          "mean_only"),
        (ANNUAL_ENHANCED,    "annual_enhanced_10x"),
        (ANNUAL_SUPPRESSED,  "annual_suppressed_0.1x"),
        (DECADAL_ENHANCED,   "decadal_enhanced_10x"),
        (DECADAL_SUPPRESSED, "decadal_suppressed_0.1x"),
    ])
    def test_preset_name(self, preset, expected_name):
        assert preset.name == expected_name

    def test_all_presets_are_experiment_instances(self):
        for preset in [BASELINE, MEAN_ONLY, ANNUAL_ENHANCED,
                       ANNUAL_SUPPRESSED, DECADAL_ENHANCED, DECADAL_SUPPRESSED]:
            assert isinstance(preset, Experiment)


# ---------------------------------------------------------------------------
# ExperimentSuite
# ---------------------------------------------------------------------------

class TestExperimentSuite:

    def test_instantiation(self):
        suite = ExperimentSuite([BASELINE, MEAN_ONLY])
        assert len(suite) == 2

    def test_iteration_order(self):
        suite = ExperimentSuite([BASELINE, ANNUAL_ENHANCED, DECADAL_ENHANCED])
        names = [exp.name for exp in suite]
        assert names == ["baseline", "annual_enhanced_10x", "decadal_enhanced_10x"]

    def test_len(self):
        suite = ExperimentSuite([BASELINE, MEAN_ONLY, ANNUAL_ENHANCED])
        assert len(suite) == 3

    def test_add(self):
        suite = ExperimentSuite([BASELINE])
        suite.add(MEAN_ONLY)
        assert len(suite) == 2
        names = [e.name for e in suite]
        assert "mean_only" in names

    def test_repr(self):
        suite = ExperimentSuite([BASELINE])
        assert "baseline" in repr(suite)

    def test_default_length(self):
        assert len(ExperimentSuite.DEFAULT) == 6

    def test_default_contains_all_presets(self):
        default_names = {e.name for e in ExperimentSuite.DEFAULT}
        expected = {
            "baseline",
            "mean_only",
            "annual_enhanced_10x",
            "annual_suppressed_0.1x",
            "decadal_enhanced_10x",
            "decadal_suppressed_0.1x",
        }
        assert default_names == expected

    def test_default_is_experiment_suite(self):
        assert isinstance(ExperimentSuite.DEFAULT, ExperimentSuite)

    def test_empty_suite(self):
        suite = ExperimentSuite([])
        assert len(suite) == 0
        assert list(suite) == []