"""
tests/test_geo_utils.py
───────────────────────
Unit tests for the pure geometric functions in geo_utils.py.

Coverage
────────
1. metres_to_deg / deg_to_metres  — round-trip and known values
2. haversine_m                    — scalar accuracy on Moon radius
3. haversine_vec                  — vectorised vs scalar consistency
4. pixel_offset_deg               — symmetry, centre, cardinal directions
5. pixel_to_latlon                — corner/centre mapping to metres
6. Projection consistency         — pixel_offset_deg ↔ pixel_to_latlon
"""

import math

import numpy as np
import pytest

import geo_utils as gu

# ── Shared fixtures / constants ───────────────────────────────────────────────
R  = gu.MOON_RADIUS_M        # 1 737 400 m
PX = gu.METERS_PER_PIXEL     # 100.0 m/px
SZ = gu.IMAGE_SIZE           # 416 px

# A standard square tile centred at projected origin, matching one full tile
TILE_HALF_M  = SZ / 2.0 * PX          # 20 800 m
TILE_SIZE_M  = SZ * PX                # 41 600 m

LAT_MIN_M    = 0.0
LAT_MAX_M    = TILE_SIZE_M
LON_MIN_M    = 0.0
LON_MAX_M    = TILE_SIZE_M

# Tile-centre in degrees under the flat equirectangular projection
TILE_CX_DEG  = gu.metres_to_deg(TILE_HALF_M)
TILE_CY_DEG  = gu.metres_to_deg(TILE_HALF_M)

# One projected degree in metres (flat, not geodetic)
DEG_IN_M     = R * math.pi / 180.0    # ≈ 30 324.8 m


# ════════════════════════════════════════════════════════════════════════════════
# 1. metres_to_deg and deg_to_metres
# ════════════════════════════════════════════════════════════════════════════════

class TestMetresDegConversion:
    def test_zero(self):
        assert gu.metres_to_deg(0.0) == 0.0
        assert gu.deg_to_metres(0.0) == 0.0

    def test_one_radian_equivalent(self):
        # R metres → exactly (180/π) degrees
        assert gu.metres_to_deg(R) == pytest.approx(180.0 / math.pi, rel=1e-9)

    def test_180_degrees(self):
        # (π × R) metres → 180 degrees
        assert gu.metres_to_deg(math.pi * R) == pytest.approx(180.0, rel=1e-9)

    def test_round_trip_metres(self):
        for m in [0.0, 100.0, 1_737_400.0, 41_600.0, 5_000_000.0]:
            assert gu.deg_to_metres(gu.metres_to_deg(m)) == pytest.approx(m, rel=1e-9)

    def test_round_trip_degrees(self):
        for d in [0.0, 1.0, 90.0, -45.0, 180.0]:
            assert gu.metres_to_deg(gu.deg_to_metres(d)) == pytest.approx(d, rel=1e-9)

    def test_linearity(self):
        # Scale factor must be identical for both axes (flat projection guarantee)
        scale = 180.0 / (math.pi * R)
        assert gu.metres_to_deg(10_000.0) == pytest.approx(10_000.0 * scale, rel=1e-12)
        assert gu.metres_to_deg(20_000.0) == pytest.approx(20_000.0 * scale, rel=1e-12)

    def test_negative_values(self):
        assert gu.metres_to_deg(-PX) == pytest.approx(-gu.metres_to_deg(PX), rel=1e-12)
        assert gu.deg_to_metres(-1.0) == pytest.approx(-gu.deg_to_metres(1.0), rel=1e-12)


# ════════════════════════════════════════════════════════════════════════════════
# 2. haversine_m  (scalar)
# ════════════════════════════════════════════════════════════════════════════════

class TestHaversineScalar:
    def test_same_point_is_zero(self):
        assert gu.haversine_m(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-6)

    def test_same_point_arbitrary(self):
        assert gu.haversine_m(34.5, -120.3, 34.5, -120.3) == pytest.approx(0.0, abs=1e-6)

    def test_one_degree_latitude_on_equator(self):
        # Along a meridian 1° of arc = R × π/180
        expected = R * math.pi / 180.0
        assert gu.haversine_m(0.0, 0.0, 1.0, 0.0) == pytest.approx(expected, rel=1e-6)

    def test_one_degree_longitude_on_equator(self):
        # On the equator, 1° of longitude = same as 1° of latitude (sphere)
        expected = R * math.pi / 180.0
        assert gu.haversine_m(0.0, 0.0, 0.0, 1.0) == pytest.approx(expected, rel=1e-6)

    def test_antipodal_points(self):
        # Antipodal distance = π × R (half great circle)
        expected = math.pi * R
        assert gu.haversine_m(0.0, 0.0, 0.0, 180.0) == pytest.approx(expected, rel=1e-6)

    def test_north_south_poles(self):
        # Pole-to-pole = π × R
        expected = math.pi * R
        assert gu.haversine_m(90.0, 0.0, -90.0, 0.0) == pytest.approx(expected, rel=1e-6)

    def test_symmetry(self):
        d_ab = gu.haversine_m(10.0, 20.0, 30.0, 40.0)
        d_ba = gu.haversine_m(30.0, 40.0, 10.0, 20.0)
        assert d_ab == pytest.approx(d_ba, rel=1e-12)

    def test_known_moon_radius(self):
        # At exactly R metres, metres_to_deg gives 180/π degrees;
        # Haversine of that arc should be exactly R metres.
        arc_deg = gu.metres_to_deg(R)           # = 180/π ≈ 57.296°
        assert gu.haversine_m(0.0, 0.0, arc_deg, 0.0) == pytest.approx(R, rel=1e-6)

    def test_small_distance_linearity(self):
        # For small angles haversine ≈ straight-line; both should match
        tiny = 0.001   # degrees
        dist = gu.haversine_m(0.0, 0.0, tiny, 0.0)
        linear = R * math.radians(tiny)
        assert dist == pytest.approx(linear, rel=1e-5)


# ════════════════════════════════════════════════════════════════════════════════
# 3. haversine_vec  (vectorised)
# ════════════════════════════════════════════════════════════════════════════════

class TestHaversineVec:
    def test_output_shape(self):
        lats = np.array([0.0, 1.0, 2.0])
        lons = np.array([0.0, 0.0, 0.0])
        result = gu.haversine_vec(0.0, 0.0, lats, lons)
        assert result.shape == (3,)

    def test_matches_scalar(self):
        lat0, lon0 = 15.0, 30.0
        lats = np.array([16.0, 14.0, 15.0, 0.0])
        lons = np.array([30.0, 30.0, 31.0, 0.0])
        vec_result = gu.haversine_vec(lat0, lon0, lats, lons)
        for i, (la, lo) in enumerate(zip(lats, lons)):
            scalar = gu.haversine_m(lat0, lon0, la, lo)
            assert vec_result[i] == pytest.approx(scalar, rel=1e-10), \
                f"Mismatch at index {i}: vec={vec_result[i]:.4f}  scalar={scalar:.4f}"

    def test_self_distance_is_zero(self):
        lats = np.full(5, 20.0)
        lons = np.full(5, 45.0)
        result = gu.haversine_vec(20.0, 45.0, lats, lons)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_non_negative(self):
        lats = np.random.uniform(-90, 90, 100)
        lons = np.random.uniform(-180, 180, 100)
        result = gu.haversine_vec(0.0, 0.0, lats, lons)
        assert np.all(result >= 0.0)


# ════════════════════════════════════════════════════════════════════════════════
# 4. pixel_offset_deg
# ════════════════════════════════════════════════════════════════════════════════

class TestPixelOffsetDeg:
    def test_centre_gives_zero_offset(self):
        dlat, dlon = gu.pixel_offset_deg(0.5, 0.5)
        assert dlat == pytest.approx(0.0, abs=1e-12)
        assert dlon == pytest.approx(0.0, abs=1e-12)

    def test_right_edge_is_positive_longitude(self):
        dlat, dlon = gu.pixel_offset_deg(1.0, 0.5)
        assert dlon > 0.0
        assert dlat == pytest.approx(0.0, abs=1e-12)

    def test_left_edge_is_negative_longitude(self):
        dlat, dlon = gu.pixel_offset_deg(0.0, 0.5)
        assert dlon < 0.0
        assert dlat == pytest.approx(0.0, abs=1e-12)

    def test_top_edge_is_positive_latitude(self):
        """cy = 0 → top → north → positive Δlat."""
        dlat, dlon = gu.pixel_offset_deg(0.5, 0.0)
        assert dlat > 0.0
        assert dlon == pytest.approx(0.0, abs=1e-12)

    def test_bottom_edge_is_negative_latitude(self):
        dlat, dlon = gu.pixel_offset_deg(0.5, 1.0)
        assert dlat < 0.0
        assert dlon == pytest.approx(0.0, abs=1e-12)

    def test_magnitude_at_edge(self):
        # Half-tile offset in metres = TILE_HALF_M; converted to degrees flat
        expected = gu.metres_to_deg(TILE_HALF_M)
        _, dlon = gu.pixel_offset_deg(1.0, 0.5)
        assert dlon == pytest.approx(expected, rel=1e-9)

        dlat, _ = gu.pixel_offset_deg(0.5, 0.0)
        assert dlat == pytest.approx(expected, rel=1e-9)

    def test_symmetry_horizontal(self):
        _, dlon_r = gu.pixel_offset_deg(0.75, 0.5)
        _, dlon_l = gu.pixel_offset_deg(0.25, 0.5)
        assert dlon_r == pytest.approx(-dlon_l, rel=1e-12)

    def test_symmetry_vertical(self):
        dlat_t, _ = gu.pixel_offset_deg(0.5, 0.25)
        dlat_b, _ = gu.pixel_offset_deg(0.5, 0.75)
        assert dlat_t == pytest.approx(-dlat_b, rel=1e-12)

    def test_no_cos_lat_applied(self):
        """
        Verify that dlon uses the FLAT scale, not geodetic (cos-lat) scale.
        For any cx, the ratio dlat/dlon (both from same TILE_HALF_M offset)
        must be exactly 1.0 on the diagonal since both axes use identical scale.
        """
        dlat, dlon = gu.pixel_offset_deg(1.0, 0.0)   # top-right corner
        assert dlat == pytest.approx(dlon, rel=1e-9), (
            "Flat projection requires dlat == dlon for equal-pixel diagonal offset. "
            "A cos(lat) factor would break this."
        )

    def test_unit_pixel_step(self):
        # Moving one pixel right should change lon by exactly PX/R*(180/π)
        step_deg = gu.metres_to_deg(PX)
        _, dlon_a = gu.pixel_offset_deg(0.5, 0.5)
        _, dlon_b = gu.pixel_offset_deg(0.5 + 1.0 / SZ, 0.5)
        assert (dlon_b - dlon_a) == pytest.approx(step_deg, rel=1e-9)


# ════════════════════════════════════════════════════════════════════════════════
# 5. pixel_to_latlon
# ════════════════════════════════════════════════════════════════════════════════

class TestPixelToLatlon:
    def test_top_left_corner(self):
        """cx=0, cy=0 → lon_min, lat_max."""
        lat, lon = gu.pixel_to_latlon(0.0, 0.0, LAT_MIN_M, LAT_MAX_M, LON_MIN_M, LON_MAX_M)
        assert lat == pytest.approx(gu.metres_to_deg(LAT_MAX_M), rel=1e-9)
        assert lon == pytest.approx(gu.metres_to_deg(LON_MIN_M), rel=1e-9)

    def test_bottom_right_corner(self):
        """cx=1, cy=1 → lon_max, lat_min."""
        lat, lon = gu.pixel_to_latlon(1.0, 1.0, LAT_MIN_M, LAT_MAX_M, LON_MIN_M, LON_MAX_M)
        assert lat == pytest.approx(gu.metres_to_deg(LAT_MIN_M), rel=1e-9)
        assert lon == pytest.approx(gu.metres_to_deg(LON_MAX_M), rel=1e-9)

    def test_centre_equals_tile_centre_degrees(self):
        """cx=0.5, cy=0.5 → tile centre in degrees."""
        lat, lon = gu.pixel_to_latlon(0.5, 0.5, LAT_MIN_M, LAT_MAX_M, LON_MIN_M, LON_MAX_M)
        assert lat == pytest.approx(TILE_CY_DEG, rel=1e-9)
        assert lon == pytest.approx(TILE_CX_DEG, rel=1e-9)

    def test_top_right_corner(self):
        lat, lon = gu.pixel_to_latlon(1.0, 0.0, LAT_MIN_M, LAT_MAX_M, LON_MIN_M, LON_MAX_M)
        assert lat == pytest.approx(gu.metres_to_deg(LAT_MAX_M), rel=1e-9)
        assert lon == pytest.approx(gu.metres_to_deg(LON_MAX_M), rel=1e-9)

    def test_bottom_left_corner(self):
        lat, lon = gu.pixel_to_latlon(0.0, 1.0, LAT_MIN_M, LAT_MAX_M, LON_MIN_M, LON_MAX_M)
        assert lat == pytest.approx(gu.metres_to_deg(LAT_MIN_M), rel=1e-9)
        assert lon == pytest.approx(gu.metres_to_deg(LON_MIN_M), rel=1e-9)

    def test_lat_decreases_with_cy(self):
        """Increasing cy (moving down) must decrease latitude (north-up)."""
        lat1, _ = gu.pixel_to_latlon(0.5, 0.2, LAT_MIN_M, LAT_MAX_M, LON_MIN_M, LON_MAX_M)
        lat2, _ = gu.pixel_to_latlon(0.5, 0.8, LAT_MIN_M, LAT_MAX_M, LON_MIN_M, LON_MAX_M)
        assert lat1 > lat2

    def test_lon_increases_with_cx(self):
        """Increasing cx (moving right) must increase longitude (east)."""
        _, lon1 = gu.pixel_to_latlon(0.2, 0.5, LAT_MIN_M, LAT_MAX_M, LON_MIN_M, LON_MAX_M)
        _, lon2 = gu.pixel_to_latlon(0.8, 0.5, LAT_MIN_M, LAT_MAX_M, LON_MIN_M, LON_MAX_M)
        assert lon2 > lon1

    def test_negative_tile_bounds(self):
        """Tiles with negative metre bounds (e.g. south/west of projected origin)."""
        lat, lon = gu.pixel_to_latlon(
            0.5, 0.5,
            -TILE_SIZE_M, 0.0,
            -TILE_SIZE_M, 0.0,
        )
        assert lat == pytest.approx(gu.metres_to_deg(-TILE_HALF_M), rel=1e-9)
        assert lon == pytest.approx(gu.metres_to_deg(-TILE_HALF_M), rel=1e-9)


# ════════════════════════════════════════════════════════════════════════════════
# 6. Projection consistency — the key correctness invariant
# ════════════════════════════════════════════════════════════════════════════════

class TestProjectionConsistency:
    """
    Core invariant:

        pixel_to_latlon(cx, cy, ...) == tile_centre_deg + pixel_offset_deg(cx, cy)

    This ensures that database building (pixel_to_latlon) and query localisation
    (tile_centre + pixel_offset_deg) use the exact same coordinate arithmetic.
    A mismatch here means the voting stage would be comparing apples to oranges.
    """

    @pytest.mark.parametrize("cx,cy", [
        (0.5, 0.5),   # centre
        (0.0, 0.0),   # top-left
        (1.0, 1.0),   # bottom-right
        (1.0, 0.0),   # top-right
        (0.0, 1.0),   # bottom-left
        (0.25, 0.75), # arbitrary interior point
        (0.1, 0.9),
        (0.9, 0.1),
    ])
    def test_offset_plus_centre_equals_absolute(self, cx: float, cy: float):
        dlat, dlon = gu.pixel_offset_deg(cx, cy)
        abs_lat, abs_lon = gu.pixel_to_latlon(
            cx, cy, LAT_MIN_M, LAT_MAX_M, LON_MIN_M, LON_MAX_M
        )
        assert abs_lat == pytest.approx(TILE_CY_DEG + dlat, rel=1e-9), (
            f"Lat mismatch at cx={cx}, cy={cy}: "
            f"absolute={abs_lat:.8f}  centre+offset={TILE_CY_DEG + dlat:.8f}"
        )
        assert abs_lon == pytest.approx(TILE_CX_DEG + dlon, rel=1e-9), (
            f"Lon mismatch at cx={cx}, cy={cy}: "
            f"absolute={abs_lon:.8f}  centre+offset={TILE_CX_DEG + dlon:.8f}"
        )

    def test_hypothesis_round_trip(self):
        """
        Simulate the voting loop: project a crater to global coords under a
        known tile-centre hypothesis, then verify it maps back to the right
        pixel position.

        True tile centre: (lat_c, lon_c)
        Query crater at (cx, cy) → DB lat/lon = (lat_c + dlat, lon_c + dlon)
        Implied tile centre = DB lat/lon − offset = (lat_c, lon_c) ✓
        """
        lat_c = TILE_CY_DEG
        lon_c = TILE_CX_DEG

        for cx, cy in [(0.3, 0.7), (0.8, 0.2), (0.1, 0.1)]:
            dlat, dlon = gu.pixel_offset_deg(cx, cy)
            db_lat = lat_c + dlat
            db_lon = lon_c + dlon

            # Reconstruct hypothesis (as localize.py does)
            lat_hyp = db_lat - dlat
            lon_hyp = db_lon - dlon

            assert lat_hyp == pytest.approx(lat_c, rel=1e-12)
            assert lon_hyp == pytest.approx(lon_c, rel=1e-12)

    def test_no_bias_at_high_latitude(self):
        """
        The flat projection must produce the same offset at any latitude.
        A cos(lat) bug would cause the longitude offset to shrink as latitude
        increases, producing a systematic bias (the observed ~57° error).
        """
        cx, cy = 0.75, 0.5

        dlat_eq, dlon_eq   = gu.pixel_offset_deg(cx, cy)   # at equator
        # pixel_offset_deg is location-independent (no lat argument)
        # so dlon must be identical everywhere — no hidden cos(lat)
        dlat_hi, dlon_hi   = gu.pixel_offset_deg(cx, cy)

        assert dlon_eq == pytest.approx(dlon_hi, rel=1e-12), (
            "pixel_offset_deg should not depend on latitude in flat projection."
        )
