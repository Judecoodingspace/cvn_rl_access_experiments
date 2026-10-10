"""Reusable scenario/view construction; experiment runners own training and statistics."""
from dataclasses import dataclass
from copy import deepcopy
from types import MappingProxyType
import re
import numpy as np
from .revision_physics import PhysicalBundle, generate, pathloss
from .revision_physics_spec import array_hash, digest, validate_physics

OBSERVATION_FIELDS = {
    'position_error_std_m', 'estimated_gain_bias_db', 'additional_report_delay_slots',
    'protection_margin', 'cam_update_slots', 'cap_update_slots', 'estimator_pathloss',
    'position_error_distribution', 'estimated_gain_error_std_db',
}
OBSERVATION_ARRAYS = {
    'reported_position_m', 'gain_pu_estimated', 'cam_source_slot', 'cap_source_slot',
    'cap_mw', 'power_cap',
}
EXOGENOUS_ARRAYS = {'occupancy', 'sensing_idle', 'arrivals', 'contention_priority', 'report_mask'}
FROZEN_FIELDS = {'policy_sha256', 'dqi_model_sha256', 'occupancy_model_sha256', 'preprocessing_sha256', 'dqi_reference_mbps'}


def component_hashes(bundle):
    arrays = bundle.arrays
    return {
        'truth_hash': array_hash({k: v for k, v in arrays.items() if k not in OBSERVATION_ARRAYS | EXOGENOUS_ARRAYS}),
        'observation_hash': array_hash({k: v for k, v in arrays.items() if k in OBSERVATION_ARRAYS}),
        'exogenous_hash': array_hash({k: v for k, v in arrays.items() if k in EXOGENOUS_ARRAYS}),
        'full_physical_hash': bundle.physical_hash(),
    }


def spec_hash(entry, vehicles):
    return digest({'physics': entry['physics'], 'traffic': entry['traffic'], 'packet': entry['packet_traffic'],
                   'vehicles': vehicles, 'channels': entry['num_channels'], 'slot': entry['slot_duration_ms']})


def observation_variant(base, entry, overrides):
    """Change received PU-protection information while preserving every true-world array."""
    if set(overrides) - OBSERVATION_FIELDS:
        raise ValueError('observation changes cannot modify truth or service-quality parameters')
    vehicles = base.arrays['position_m'].shape[1]
    if base.spec_hash != spec_hash(entry, vehicles):
        raise ValueError('base trace and supplied physical entry differ')
    updated = deepcopy(entry)
    updated['physics'].update(overrides)
    validate_physics(updated['physics'])
    s = updated['physics']; total = base.slots + base.future_slots; m = entry['num_channels']
    arrays = {k: v.copy() for k, v in base.arrays.items()}
    cam = np.maximum((np.arange(total) - s['additional_report_delay_slots']) // s['cam_update_slots'] * s['cam_update_slots'], 0)
    error_rng = np.random.default_rng(np.random.SeedSequence(base.seed).spawn(12)[8])
    if s.get('position_error_distribution', 'gaussian') == 'uniform':
        bound = np.sqrt(3) * s['position_error_std_m']
        error = error_rng.uniform(-bound, bound, arrays['position_m'].shape)
    else:
        error = error_rng.normal(0, s['position_error_std_m'], arrays['position_m'].shape)
    reported = arrays['position_m'][cam] + error[cam]
    reported[:, :, 2] = s['vt_height_m']
    pu = np.stack([np.linspace(*s['pu_receiver_x_range_m'], m), np.full(m, s['pu_receiver_y_m']),
                   np.full(m, s['pu_receiver_height_m'])], axis=-1)
    delta = reported[:, :, None, :] - pu[None, None, :, :]
    estimator = {**s, 'pathloss': s.get('estimator_pathloss', s['pathloss'])}
    estimated = 10 ** (-pathloss(np.linalg.norm(delta[..., :2], axis=-1), np.linalg.norm(delta, axis=-1), estimator) / 10)
    estimated *= 10 ** (s['estimated_gain_bias_db'] / 10)
    if s.get('estimated_gain_error_std_db', 0) > 0:
        gain_error_rng = np.random.default_rng(np.random.SeedSequence([base.seed, 103]))
        gain_error = gain_error_rng.normal(0, s['estimated_gain_error_std_db'], estimated.shape)[cam]
        estimated *= 10 ** (gain_error / 10)
    cap_source = np.arange(total) // s['cap_update_slots'] * s['cap_update_slots']
    cap = np.minimum(s['p_max_mw'], arrays['pu_tolerance_mw'][:, None, :] / (s['protection_margin'] * estimated))[cap_source]
    arrays.update(reported_position_m=reported, gain_pu_estimated=estimated[cap_source], cam_source_slot=cam,
                  cap_source_slot=cap_source, cap_mw=cap, power_cap=cap / s['p_max_mw'])
    result = PhysicalBundle(arrays, base.slots, base.future_slots, base.seed, base.split, spec_hash(updated, vehicles))
    old = component_hashes(base); new = component_hashes(result)
    if old['truth_hash'] != new['truth_hash'] or old['exogenous_hash'] != new['exogenous_hash']:
        raise AssertionError('observation-only scenario altered the physical world')
    return result, updated


def scenario_entry(training_entry, overrides):
    """Construct an evaluation distribution without mutating training configuration."""
    if set(overrides) - {'physics', 'traffic', 'packet_traffic'}:
        raise ValueError('only explicit distribution sections may change; controller/model budgets are separate')
    result = deepcopy(training_entry)
    for section, fields in overrides.items():
        if not isinstance(fields, dict):
            raise ValueError('scenario sections must be mappings')
        result[section].update(deepcopy(fields))
    validate_physics(result['physics'])
    traffic = result['traffic']
    if not 0 <= traffic['sensing_pd'] <= 1 or not 0 <= traffic['sensing_pfa'] <= 1:
        raise ValueError('sensing probabilities outside [0,1]')
    for key in ['p01', 'p10']:
        if key in traffic and not 0 <= traffic[key] <= 1:
            raise ValueError('transition probabilities outside [0,1]')
    if not 0 <= result['packet_traffic']['arrival_probability'] <= 1:
        raise ValueError('arrival probability outside [0,1]')
    return result


def distribution_hash(entry):
    return digest({k: entry[k] for k in ['physics', 'traffic', 'packet_traffic', 'num_channels', 'slot_duration_ms']})


@dataclass(frozen=True)
class FrozenEvaluationCase:
    """Evaluation trace and provenance; no policy training or scaler fitting occurs here."""
    bundle: PhysicalBundle
    entry: dict
    manifest: dict
    references: dict

    def assert_frozen_references(self, actual):
        if actual != self.references:
            raise ValueError('OOD evaluation changed policy, predictors, preprocessing, or training Rref')


def build_frozen_evaluation_case(training_entry, overrides, references, seed, slots, vehicles,
                                 case_id, future_slots=0):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', case_id) or set(references) != FROZEN_FIELDS:
        raise ValueError('named case and complete frozen-artifact references required')
    for key in FROZEN_FIELDS - {'dqi_reference_mbps'}:
        if not re.fullmatch(r'[0-9a-fA-F]{64}', references[key]):
            raise ValueError(f'{key} must be SHA256')
    if not np.isfinite(references['dqi_reference_mbps']) or references['dqi_reference_mbps'] <= 0:
        raise ValueError('positive training Rref required')
    entry = scenario_entry(training_entry, overrides)
    bundle = generate(entry, seed, slots, vehicles, f'evaluation_{case_id}', future_slots)
    manifest = {
        'case_id': case_id, 'purpose': 'frozen held-out distribution evaluation', 'seed': seed,
        'training_distribution_hash': distribution_hash(training_entry),
        'evaluation_distribution_hash': distribution_hash(entry), 'overrides': deepcopy(overrides),
        'frozen_references': deepcopy(references), 'controller_training_performed': False,
        'preprocessing_fit_performed': False, 'dimensions': {'channels': entry['num_channels'], 'vehicles': vehicles},
        'paired_seed_note': 'same seed preserves latent streams where the corresponding process is unchanged; changed distributions need not have identical arrays',
        **component_hashes(bundle),
    }
    return FrozenEvaluationCase(bundle, entry, manifest, MappingProxyType(deepcopy(references)))
