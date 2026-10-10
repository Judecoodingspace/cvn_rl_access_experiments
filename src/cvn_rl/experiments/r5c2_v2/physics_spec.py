"""Explicit physical constants; supplemental choices are not attributed to 3GPP."""
import numpy as np

REQUIRED={'spec_version','standard_url','pathloss','carrier_hz','bs_position_m','vt_height_m','initial_x_range_m','road_y_m',
 'speed_kmh','reference_max_speed_kmh','direction_mode','pu_receiver_x_range_m','pu_receiver_y_m','pu_receiver_height_m',
 'pu_transmitter_offset_m','bandwidth_mhz','p_max_mw','p_ref_mw','thermal_noise_dbm_hz','receiver_noise_figure_db','rician_k_db',
 'shadow_sigma_db','shadow_correlation_distance_m','fading_temporal_model','pu_tx_power_mw','pu_desired_rx_power_dbm',
 'pu_tolerance_dbm','cam_update_slots','cap_update_slots','protection_margin','underlay_ceiling_fraction','report_probability',
 'position_error_std_m','estimated_gain_bias_db','additional_report_delay_slots'}

def validate(s):
    if set(s)!=REQUIRED: raise ValueError(f'physical fields missing/unknown: {REQUIRED-set(s)}, {set(s)-REQUIRED}')
    if s['pathloss']!='UMi_LOS_Table7.4.1-1' or '18.1.0' not in s['spec_version']: raise ValueError('adopted UMi LOS Release18 source required')
    if s['bs_position_m'][2]!=10 or s['pu_receiver_height_m']!=10 or not 1.5<=s['vt_height_m']<=22.5: raise ValueError('UMi antenna heights')
    if s['carrier_hz']!=5.9e9 or s['rician_k_db']!=3 or s['shadow_sigma_db']!=4: raise ValueError('paper fc/K/shadow required')
    for k in ['p_max_mw','p_ref_mw','reference_max_speed_kmh','shadow_correlation_distance_m','pu_tx_power_mw']:
        if not np.isfinite(s[k]) or s[k]<=0: raise ValueError(k)
    if not 0<=s['speed_kmh']<=s['reference_max_speed_kmh'] or not 0<=s['report_probability']<=1: raise ValueError('speed/report probability')
    if s['protection_margin']<1 or not 0<s['underlay_ceiling_fraction']<=1: raise ValueError('protection/power ceiling')
    if s['cam_update_slots']!=100 or s['cap_update_slots']!=10: raise ValueError('separate CAM100/cap10 clocks')
    if s['direction_mode']!='alternating' or s['fading_temporal_model']!='complex_gaussian_ar1_J0_one_step': raise ValueError('explicit mobility/fading mode')
    if s['position_error_std_m']<0 or s['additional_report_delay_slots']<0: raise ValueError('error/delay nonnegative')
    paper=['carrier_hz','rician_k_db','shadow_sigma_db','cam_update_slots']
    standard=['spec_version','standard_url','pathloss','bs_position_m','vt_height_m','pu_receiver_height_m']
    return {'values':s,'provenance':{k:('paper' if k in paper else '3GPP pathloss/domain with supplemental geometry' if k in standard else 'explicit supplemental v2 assumption') for k in s},
            'standard_table':'7.4.1-1 / UMi LOS, environment height1m',
            'fading_note':'simplified Rician marginal; AR1 coefficient J0 at one-slot radial Doppler lag, not full Jakes covariance or 3GPP cluster simulation',
            'shadow_note':'Gaussian dB process with exponential distance correlation; 10m correlation is supplemental',
            'measurement_note':'reference link quality conditional on channel availability; no PU activity in DQI labels'}
