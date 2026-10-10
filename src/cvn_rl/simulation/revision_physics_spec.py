"""Physical-domain validation and artifact helpers, independent of any review contract."""
from pathlib import Path
import hashlib
import json
import numpy as np

STANDARD_URL = 'https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/18.01.00_60/tr_138901v180100p.pdf'
PATHLOSS_MODELS = {'UMi_LOS_Table7.4.1-1', 'UMi_NLOS_Table7.4.1-1'}
REQUIRED = {
    'spec_version', 'standard_url', 'pathloss', 'carrier_hz', 'bs_position_m', 'vt_height_m',
    'initial_x_range_m', 'road_y_m', 'speed_kmh', 'reference_max_speed_kmh', 'direction_mode',
    'pu_receiver_x_range_m', 'pu_receiver_y_m', 'pu_receiver_height_m', 'pu_transmitter_offset_m',
    'bandwidth_mhz', 'p_max_mw', 'p_ref_mw', 'thermal_noise_dbm_hz', 'receiver_noise_figure_db',
    'rician_k_db', 'shadow_sigma_db', 'shadow_correlation_distance_m', 'fading_temporal_model',
    'pu_tx_power_mw', 'pu_desired_rx_power_dbm', 'pu_tolerance_dbm', 'cam_update_slots',
    'cap_update_slots', 'protection_margin', 'underlay_ceiling_fraction', 'report_probability',
    'position_error_std_m', 'estimated_gain_bias_db', 'additional_report_delay_slots',
}
OPTIONAL = {'link_overrides', 'speed_schedule', 'estimator_pathloss', 'position_error_distribution', 'estimated_gain_error_std_db'}
LINK_FIELDS = {'pathloss', 'shadow_sigma_db', 'shadow_correlation_distance_m', 'rician_k_db', 'small_scale_fading'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_hash(arrays):
    h = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        a = np.asarray(value)
        h.update(name.encode()); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
        h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _number(value, name, minimum=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if minimum is not None and (value <= minimum if strict else value < minimum):
        raise ValueError(f'{name} outside physical domain')


def validate_physics(s):
    """Check domains; experiment contracts separately restrict their reference values."""
    if not REQUIRED <= set(s) or set(s) - REQUIRED - OPTIONAL:
        raise ValueError(f'physical fields missing/unknown: {REQUIRED - set(s)}, {set(s) - REQUIRED - OPTIONAL}')
    if s['pathloss'] not in PATHLOSS_MODELS or '18.1.0' not in s['spec_version'] or s['standard_url'] != STANDARD_URL:
        raise ValueError('supported TR38.901 v18.1.0 pathloss source required')
    if len(s['bs_position_m']) != 3 or s['bs_position_m'][2] != 10 or s['pu_receiver_height_m'] != 10:
        raise ValueError('UMi elevated endpoints require 10m height')
    if not 1.5 <= s['vt_height_m'] <= 22.5 or not .5e9 < s['carrier_hz'] < 100e9:
        raise ValueError('UMi frequency/UT-height domain')
    for key in ['p_max_mw', 'p_ref_mw', 'reference_max_speed_kmh', 'shadow_correlation_distance_m', 'pu_tx_power_mw']:
        _number(s[key], key, 0, True)
    for key in ['speed_kmh', 'shadow_sigma_db', 'position_error_std_m', 'receiver_noise_figure_db']:
        _number(s[key], key, 0)
    for key in ['rician_k_db', 'thermal_noise_dbm_hz', 'pu_desired_rx_power_dbm', 'pu_tolerance_dbm', 'estimated_gain_bias_db', 'road_y_m', 'pu_receiver_y_m']:
        _number(s[key], key)
    for key in ['protection_margin', 'underlay_ceiling_fraction', 'report_probability', 'vt_height_m', 'carrier_hz']:
        _number(s[key], key)
    _number(s.get('estimated_gain_error_std_db', 0.), 'estimated gain error sigma', 0)
    if s.get('position_error_distribution', 'gaussian') not in ['gaussian', 'uniform']:
        raise ValueError('unknown horizontal position error distribution')
    for key in ['bs_position_m', 'pu_transmitter_offset_m', 'initial_x_range_m', 'pu_receiver_x_range_m', 'bandwidth_mhz']:
        if not np.all(np.isfinite(s[key])):
            raise ValueError(f'nonfinite {key}')
    if len(s['pu_transmitter_offset_m']) != 3 or s['pu_transmitter_offset_m'][2] != 0:
        raise ValueError('PU transmitter must retain the supported 10m endpoint height')
    for key in ['initial_x_range_m', 'pu_receiver_x_range_m']:
        if len(s[key]) != 2 or s[key][0] > s[key][1]:
            raise ValueError(f'invalid {key}')
    if not len(s['bandwidth_mhz']) or np.any(np.asarray(s['bandwidth_mhz']) <= 0):
        raise ValueError('positive bandwidths required')
    if not 0 <= s['speed_kmh'] <= s['reference_max_speed_kmh'] or not 0 <= s['report_probability'] <= 1:
        raise ValueError('speed/report domain')
    if s['protection_margin'] < 1 or not 0 < s['underlay_ceiling_fraction'] <= 1:
        raise ValueError('margin/power ceiling domain')
    for key in ['cam_update_slots', 'cap_update_slots', 'additional_report_delay_slots']:
        if isinstance(s[key], bool) or not isinstance(s[key], int) or s[key] < (0 if key == 'additional_report_delay_slots' else 1):
            raise ValueError(f'{key} must have integer slot units')
    if s['direction_mode'] not in ['alternating', 'same_direction'] or s['fading_temporal_model'] != 'complex_gaussian_ar1_J0_one_step':
        raise ValueError('unsupported mobility/fading model')
    if s.get('estimator_pathloss', s['pathloss']) not in PATHLOSS_MODELS:
        raise ValueError('unsupported estimated pathloss model')
    for link, overrides in s.get('link_overrides', {}).items():
        if link not in ['vt_bs', 'vt_pu', 'pu_vt'] or set(overrides) - LINK_FIELDS:
            raise ValueError('unknown link/override fields')
        if overrides.get('pathloss', s['pathloss']) not in PATHLOSS_MODELS:
            raise ValueError('unsupported link pathloss')
        if overrides.get('small_scale_fading', 'rician') not in ['rician', 'rayleigh', 'none']:
            raise ValueError('unsupported small-scale fading family')
        for key in ['shadow_sigma_db', 'shadow_correlation_distance_m', 'rician_k_db']:
            if key in overrides:
                _number(overrides[key], f'{link}.{key}', 0 if key != 'rician_k_db' else None,
                        key == 'shadow_correlation_distance_m')
    previous = -1
    for row in s.get('speed_schedule', []):
        if set(row) != {'start_slot', 'speed_kmh'} or isinstance(row['start_slot'], bool) or not isinstance(row['start_slot'], int) or row['start_slot'] <= previous:
            raise ValueError('speed schedule requires increasing nonnegative integer start slots')
        _number(row['speed_kmh'], 'scheduled speed', 0)
        if row['speed_kmh'] > s['reference_max_speed_kmh']:
            raise ValueError('scheduled speed exceeds configured physical maximum')
        previous = row['start_slot']
    return {
        'values': s,
        'standard': {'version': s['spec_version'], 'url': s['standard_url'], 'table': '7.4.1-1',
                     'domain': 'UMi, hBS=10m, hUT=1.5..22.5m, d2D=10..5000m, hE=1m'},
        'standard_derived': ['LOS/NLOS pathloss equations, max(LOS,NLOS) rule, breakpoint, applicability domain'],
        'independent_assumptions': ['geometry/mobility', 'Rician K and simplified temporal correlation',
             'selected shadow sigma/correlation distance', 'power/noise/PU tolerance',
             'PU occupancy/sensing/packet arrivals', 'reporting/cap clocks and estimation errors'],
        'notes': ['LOS-only is the default reference; NLOS must be explicitly selected.',
                  'A shadow sigma matching a standard table value does not make the whole simulator 3GPP-compliant.',
                  'Rician AR1 only matches one-step J0 correlation, not full Jakes covariance or 3GPP clusters.',
                  'Registered initial position is available at slot0; extra report delay affects later CAM reports.'],
    }
