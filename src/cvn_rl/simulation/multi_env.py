from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from cvn_rl.simulation.actions import build_action_mask, decode_action, encode_action
from cvn_rl.simulation.multi_traces import MultiVehicleAccessTrace
from cvn_rl.simulation.rate_model import (
    potential_access_rate_mbps,
    underlay_rate_mbps,
    underlay_transmit_power_mw,
)
from cvn_rl.simulation.state_builder import feature_slices


DEFAULT_MULTI_FEATURES = [
    'delta', 'fresh_dqi', 'theta', 'queue_occupancy', 'hol_age',
    'previous_channel_load', 'vehicle_context', 'dqi_available',
    'prediction_available',
    'sensing_available',
]


class MultiVehicleAccessEnv:
    '''Concurrent access environment with causal descriptors and MAC contention.'''

    def __init__(
        self,
        trace: MultiVehicleAccessTrace,
        config: dict[str, Any],
        state_features: list[str] | None = None,
        reward_layers: list[str] | None = None,
    ) -> None:
        self.trace = trace
        self.config = config
        self.protection_variant = str(config.get('protection_variant', 'gate_only')).lower()
        self.state_features = list(
            state_features or config.get('state_features') or DEFAULT_MULTI_FEATURES
        )
        if 'occupancy_type' in self.state_features:
            raise ValueError(
                'occupancy_type is privileged simulator state and cannot be an agent feature'
            )
        if self.protection_variant in {'gate_plus_state', 'gate_state_mask'}:
            for feature in ('power_cap', 'dqi_age'):
                if feature not in self.state_features:
                    self.state_features.append(feature)
        if reward_layers is not None:
            self.reward_layers = list(reward_layers)
        elif self.protection_variant == 'gate_state_mask':
            self.reward_layers = ['reward_gate', 'action_mask']
        else:
            self.reward_layers = ['reward_gate']
        self.module_variant = str(config.get('module_variant', 'full')).lower()
        self.sensing_info_source = str(
            config.get('sensing_info_source', 'real')
        ).lower()
        self.prediction_info_source = str(
            config.get(
                'prediction_info_source',
                config.get('ablation_info_source', 'real'),
            )
        ).lower()
        self.sensing_mask_source = str(
            config.get('sensing_mask_source', 'real')
        ).lower()
        self.prediction_mask_source = str(
            config.get(
                'prediction_mask_source',
                config.get('ablation_mask_source', 'real'),
            )
        ).lower()
        for name, source in (
            ('sensing_info_source', self.sensing_info_source),
            ('prediction_info_source', self.prediction_info_source),
            ('sensing_mask_source', self.sensing_mask_source),
            ('prediction_mask_source', self.prediction_mask_source),
        ):
            if source not in {'real', 'neutral'}:
                raise ValueError(f'unsupported {name}: {source}')
        self.use_dqi = self.module_variant not in {
            'no_dqi', 'sensing_only', 'state_sensing_only',
        }
        default_prediction = self.module_variant not in {
            'no_prediction', 'no_prediction_info', 'sensing_only',
            'state_sensing_only',
        }
        self.ablation_info_source = str(
            config.get(
                'ablation_info_source',
                'real' if default_prediction else 'neutral',
            )
        ).lower()
        self.ablation_mask_source = str(
            config.get(
                'ablation_mask_source',
                'real' if default_prediction else 'neutral',
            )
        ).lower()
        if 'prediction_info_source' not in config:
            self.prediction_info_source = self.ablation_info_source
        if 'prediction_mask_source' not in config:
            self.prediction_mask_source = self.ablation_mask_source
        valid_sources = {'real', 'neutral'}
        if self.ablation_info_source not in valid_sources:
            raise ValueError(
                f'unsupported ablation_info_source: {self.ablation_info_source}'
            )
        if self.ablation_mask_source not in valid_sources:
            raise ValueError(
                f'unsupported ablation_mask_source: {self.ablation_mask_source}'
            )
        self.use_sensing = self.sensing_info_source == 'real'
        self.use_prediction = self.prediction_info_source == 'real'
        self.action_mask_thresholds = dict(config.get('action_mask_thresholds', {}))
        self.enabled_modes = list(
            config.get('action', {}).get('enabled_modes', ['overlay', 'underlay'])
        )
        self.mask_policy_version = str(config.get('mask_policy_version', 'legacy_v1'))
        switching_cfg = config.get('switching', {})
        self.switch_blackout_slots = max(
            int(switching_cfg.get('blackout_slots', 0)),
            0,
        )
        self.minimum_dwell_slots = max(
            int(switching_cfg.get('min_dwell_slots', 0)),
            0,
        )
        self.switching_enabled = bool(
            self.switch_blackout_slots or self.minimum_dwell_slots
        )
        self.dwell_semantics = str(
            switching_cfg.get('dwell_semantics', 'legacy_after_blackout')
        )
        if self.dwell_semantics not in {
            'legacy_after_blackout',
            'total_from_switch',
        }:
            raise ValueError(
                f'unknown dwell_semantics {self.dwell_semantics!r}; '
                f"expected 'legacy_after_blackout' or 'total_from_switch'"
            )
        packet_cfg = config.get('packet_traffic', {})
        self.queue_capacity = int(packet_cfg.get('queue_capacity_packets', 10))
        self.packet_size_bits = float(packet_cfg.get('packet_size_bits', 32_000))
        self.slot_duration_ms = float(config.get('slot_duration_ms', 1.0))
        self.hol_age_scale_slots = max(float(packet_cfg.get('hol_age_scale_slots', 1_000)), 1.0)
        self.queues: list[deque[list[float]]] = []
        self.generated_packets = np.zeros(trace.num_vehicles, dtype=int)
        self.dropped_packets = np.zeros(trace.num_vehicles, dtype=int)
        self.completed_packets = np.zeros(trace.num_vehicles, dtype=int)
        self.previous_channel_load = np.zeros(trace.num_channels, dtype=float)
        self.current_channels = np.full(trace.num_vehicles, -1, dtype=int)
        self.dwell_remaining = np.zeros(trace.num_vehicles, dtype=int)
        self.blackout_remaining = np.zeros(trace.num_vehicles, dtype=int)
        self._arrivals_this_slot = np.zeros(trace.num_vehicles, dtype=int)
        self._drops_this_slot = np.zeros(trace.num_vehicles, dtype=int)
        self.t = 0

    def reset(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        self.t = 0
        self.queues = [deque() for _ in range(self.trace.num_vehicles)]
        self.generated_packets.fill(0)
        self.dropped_packets.fill(0)
        self.completed_packets.fill(0)
        self.previous_channel_load.fill(0.0)
        self.current_channels.fill(-1)
        self.dwell_remaining.fill(0)
        self.blackout_remaining.fill(0)
        self._apply_arrivals(0)
        return self._observations(), self.agent_infos()

    def step(
        self,
        action_indices: np.ndarray | list[int],
    ) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        actions_raw = np.asarray(action_indices, dtype=int)
        if actions_raw.shape != (self.trace.num_vehicles,):
            raise ValueError(f'expected {self.trace.num_vehicles} actions, got {actions_raw.shape}')

        t = self.t
        n = self.trace.num_vehicles
        m = self.trace.num_channels
        decision_backlog = np.asarray([bool(queue) for queue in self.queues], dtype=bool)
        decision_infos = self.agent_infos()
        decision_masks = [
            np.asarray(info['action_mask'], dtype=bool) for info in decision_infos
        ]
        raw_physical_masks = [
            np.asarray(info['raw_action_mask'], dtype=bool) for info in decision_infos
        ]
        requested = [decode_action(int(index), m) for index in actions_raw]
        decoded, switch_started, dwell_violation, blackout_active = (
            self._resolve_switching_actions(requested)
        )
        active = np.asarray(
            [
                action.mode != 'no_access' and action.channel >= 0 and bool(self.queues[i])
                for i, action in enumerate(decoded)
            ],
            dtype=bool,
        )
        channels = np.asarray(
            [action.channel if active[i] else -1 for i, action in enumerate(decoded)],
            dtype=int,
        )
        modes = np.asarray(
            [action.mode if active[i] else 'no_access' for i, action in enumerate(decoded)],
            dtype=object,
        )
        contention_lost = np.zeros(n, dtype=bool)
        contention_winner = np.zeros(n, dtype=bool)
        channel_contenders = np.zeros(m, dtype=int)
        for channel in range(m):
            users = np.flatnonzero(active & (channels == channel))
            channel_contenders[channel] = len(users)
            if len(users) == 0:
                continue
            priorities = self.trace.contention_priority[t, users, channel]
            winner = int(users[int(np.argmin(priorities))])
            contention_winner[winner] = True
            contention_lost[users] = True
            contention_lost[winner] = False
        transmitted = active & contention_winner

        p_max = float(self.config.get('underlay_power', {}).get('p_max_mw', 1.0))
        transmit_power = np.zeros(n, dtype=float)
        for vehicle in np.flatnonzero(transmitted):
            channel = channels[vehicle]
            transmit_power[vehicle] = (
                p_max if modes[vehicle] == 'overlay'
                else underlay_transmit_power_mw(
                    float(self.trace.transmit_power_cap_mw[t, vehicle, channel]),
                    self.config,
                )
            )
        actual_interference = np.zeros(m, dtype=float)
        for vehicle in np.flatnonzero(transmitted):
            channel = channels[vehicle]
            actual_interference[channel] += (
                float(self.trace.su_pu_gain_true[t, vehicle, channel])
                * transmit_power[vehicle]
            )
        busy = self.trace.occupancy[t].astype(bool)
        threshold = self.trace.pu_threshold_mw[t]
        channel_violation = busy & (actual_interference > threshold)
        interference_ratio = np.divide(
            actual_interference,
            np.maximum(threshold, 1e-12),
            out=np.zeros_like(actual_interference),
        )
        pu_throughput = np.zeros(m, dtype=float)
        if np.any(busy):
            sinr = self.trace.pu_signal_mw[t] / np.maximum(
                self.trace.noise_mw[t] + actual_interference,
                1e-12,
            )
            pu_throughput[busy] = self.trace.bandwidth_mhz[t, busy] * np.log2(1.0 + sinr[busy])

        vt_throughput = np.zeros(n, dtype=float)
        served_bits = np.zeros(n, dtype=float)
        selected_dqi = np.full(n, np.nan, dtype=float)
        selected_theta = np.full(n, np.nan, dtype=float)
        pu_interruption = np.zeros(n, dtype=bool)
        low_rate_backlog = np.zeros(n, dtype=bool)
        completed_this_slot = np.zeros(n, dtype=int)
        for vehicle in range(n):
            if not active[vehicle]:
                continue
            channel = channels[vehicle]
            selected_dqi[vehicle] = float(self.trace.fresh_dqi[t, channel])
            selected_theta[vehicle] = float(self.trace.theta[t, channel])
            if not transmitted[vehicle]:
                low_rate_backlog[vehicle] = bool(self.queues[vehicle])
                continue
            is_busy = bool(busy[channel])
            if modes[vehicle] == 'overlay':
                pu_interruption[vehicle] = is_busy
                rate = 0.0 if is_busy else float(self.trace.quality_mbps[t, vehicle, channel])
            else:
                rate = underlay_rate_mbps(
                    float(self.trace.quality_mbps[t, vehicle, channel]),
                    float(self.trace.bandwidth_mhz[t, channel]),
                    float(transmit_power[vehicle]),
                    is_busy,
                    self.config,
                )
            vt_throughput[vehicle] = rate
            available_service = rate * self.slot_duration_ms * 1_000.0
            served_bits[vehicle] = self._serve_queue(vehicle, available_service, completed_this_slot)
            low_rate_backlog[vehicle] = (
                bool(self.queues[vehicle]) and served_bits[vehicle] < self.packet_size_bits
            )

        overlay_selected = active & (modes == 'overlay')
        underlay_selected = active & (modes == 'underlay')
        overlay_transmitted = transmitted & overlay_selected
        underlay_transmitted = transmitted & underlay_selected
        overlay_collision = overlay_transmitted & pu_interruption
        underlay_violation = np.zeros(n, dtype=bool)
        for vehicle in np.flatnonzero(underlay_transmitted):
            underlay_violation[vehicle] = bool(channel_violation[channels[vehicle]])

        safe_action_regret = np.full(n, np.nan, dtype=float)
        selected_future_dqi = np.full(n, np.nan, dtype=float)
        channel_selection_regret = np.full(n, np.nan, dtype=float)
        oracle_top1_channel = np.full(n, -1, dtype=int)
        chosen_overlay_busy = 0
        chosen_overlay_idle = 0
        legal_overlay_busy_opportunities = 0
        legal_overlay_idle_opportunities = 0
        blackout_loss_bits = np.zeros(n, dtype=float)
        for vehicle in range(n):
            if not decision_backlog[vehicle]:
                continue
            legal_rates: list[float] = []
            legal_channels: set[int] = set()
            for channel in range(m):
                is_busy = bool(busy[channel])
                overlay_index = encode_action(channel, 'overlay', m)
                underlay_index = encode_action(channel, 'underlay', m)
                if decision_masks[vehicle][overlay_index]:
                    legal_channels.add(channel)
                    legal_rates.append(
                        potential_access_rate_mbps(
                            mode='overlay',
                            full_power_quality_mbps=float(
                                self.trace.quality_mbps[t, vehicle, channel]
                            ),
                            bandwidth_mhz=float(self.trace.bandwidth_mhz[t, channel]),
                            transmit_power_cap_mw=float(
                                self.trace.transmit_power_cap_mw[t, vehicle, channel]
                            ),
                            busy=is_busy,
                            config=self.config,
                        )
                    )
                    if is_busy:
                        legal_overlay_busy_opportunities += 1
                    else:
                        legal_overlay_idle_opportunities += 1
                if decision_masks[vehicle][underlay_index]:
                    legal_channels.add(channel)
                    legal_rates.append(
                        potential_access_rate_mbps(
                            mode='underlay',
                            full_power_quality_mbps=float(
                                self.trace.quality_mbps[t, vehicle, channel]
                            ),
                            bandwidth_mhz=float(self.trace.bandwidth_mhz[t, channel]),
                            transmit_power_cap_mw=float(
                                self.trace.transmit_power_cap_mw[t, vehicle, channel]
                            ),
                            busy=is_busy,
                            config=self.config,
                        )
                    )
            chosen_rate = 0.0
            chosen = requested[vehicle]
            if chosen.channel >= 0 and chosen.mode != 'no_access':
                chosen_rate = potential_access_rate_mbps(
                    mode=chosen.mode,
                    full_power_quality_mbps=float(
                        self.trace.quality_mbps[t, vehicle, chosen.channel]
                    ),
                    bandwidth_mhz=float(self.trace.bandwidth_mhz[t, chosen.channel]),
                    transmit_power_cap_mw=float(
                        self.trace.transmit_power_cap_mw[t, vehicle, chosen.channel]
                    ),
                    busy=bool(busy[chosen.channel]),
                    config=self.config,
                )
                if blackout_active[vehicle] or switch_started[vehicle]:
                    chosen_rate = 0.0
                if chosen.mode == 'overlay':
                    if bool(busy[chosen.channel]):
                        chosen_overlay_busy += 1
                    else:
                        chosen_overlay_idle += 1
            safe_action_regret[vehicle] = max(max(legal_rates, default=0.0) - chosen_rate, 0.0)

            oracle_scores = self.trace.future_dqi_oracle[t]
            if np.any(np.isfinite(oracle_scores)):
                oracle_top1_channel[vehicle] = int(np.nanargmax(oracle_scores))
                if chosen.channel >= 0:
                    selected_future_dqi[vehicle] = float(oracle_scores[chosen.channel])
                legal_scores = [
                    float(oracle_scores[channel])
                    for channel in legal_channels
                    if np.isfinite(oracle_scores[channel])
                ]
                if legal_scores:
                    selected_score = (
                        selected_future_dqi[vehicle]
                        if np.isfinite(selected_future_dqi[vehicle])
                        else 0.0
                    )
                    channel_selection_regret[vehicle] = max(
                        max(legal_scores) - selected_score,
                        0.0,
                    )

            if blackout_active[vehicle] or switch_started[vehicle]:
                target = int(self.current_channels[vehicle])
                if target >= 0:
                    target_rate = max(
                        potential_access_rate_mbps(
                            mode=mode,
                            full_power_quality_mbps=float(
                                self.trace.quality_mbps[t, vehicle, target]
                            ),
                            bandwidth_mhz=float(self.trace.bandwidth_mhz[t, target]),
                            transmit_power_cap_mw=float(
                                self.trace.transmit_power_cap_mw[t, vehicle, target]
                            ),
                            busy=bool(busy[target]),
                            config=self.config,
                        )
                        for mode in ('overlay', 'underlay')
                    )
                    blackout_loss_bits[vehicle] = (
                        target_rate * self.slot_duration_ms * 1_000.0
                    )

        reward = self._physical_reward(
            served_bits,
            contention_lost,
            overlay_collision | underlay_violation,
            selected_dqi,
        )
        queue_lengths = np.asarray([len(queue) for queue in self.queues], dtype=int)
        hol_age = self._hol_ages()
        self.previous_channel_load = channel_contenders.astype(float) / max(n, 1)
        step_info: dict[str, Any] = {
            't': t,
            'action_indices': actions_raw.copy(),
            'channels': channels,
            'modes': modes,
            'rewards': reward.copy(),
            'mean_reward': float(np.mean(reward)),
            'vt_throughput_per_vehicle': vt_throughput,
            'system_vt_throughput': float(np.sum(vt_throughput)),
            'mean_vt_throughput': float(np.mean(vt_throughput)),
            'served_bits': served_bits,
            'pu_throughput_per_channel': pu_throughput,
            'system_pu_throughput': float(np.sum(pu_throughput)),
            'pu_busy_channels': busy,
            'pu_violation_per_channel': channel_violation,
            'actual_interference_mw': actual_interference,
            'pu_threshold_mw': threshold.copy(),
            'interference_ratio': interference_ratio,
            'selected_dqi': selected_dqi,
            'selected_theta': selected_theta,
            'overlay_selected': overlay_selected,
            'underlay_selected': underlay_selected,
            'overlay_transmitted': overlay_transmitted,
            'underlay_transmitted': underlay_transmitted,
            'overlay_collision': overlay_collision,
            'underlay_violation': underlay_violation,
            'pu_access_violation': overlay_collision | underlay_violation,
            'attempted': active,
            'transmitted': transmitted,
            'contention_lost': contention_lost,
            'contention_winner': contention_winner,
            'channel_contenders': channel_contenders,
            'vt_collision': contention_lost,
            'physical_collision': np.zeros(n, dtype=bool),
            'pu_interruption': pu_interruption,
            'low_rate_backlog': low_rate_backlog,
            'packet_generated': self._arrivals_this_slot.copy(),
            'packet_dropped': self._drops_this_slot.copy(),
            'packet_completed': completed_this_slot,
            'queue_lengths': queue_lengths,
            'hol_age_slots': hol_age,
            'decision_backlog': decision_backlog,
            'safe_action_regret_mbps': safe_action_regret,
            'chosen_overlay_busy': int(chosen_overlay_busy),
            'chosen_overlay_idle': int(chosen_overlay_idle),
            'legal_overlay_busy_opportunities': int(
                legal_overlay_busy_opportunities
            ),
            'legal_overlay_idle_opportunities': int(
                legal_overlay_idle_opportunities
            ),
            'selected_future_dqi': selected_future_dqi,
            'channel_selection_regret': channel_selection_regret,
            'oracle_top1_channel': oracle_top1_channel,
            'switch_started': switch_started,
            'dwell_violation': dwell_violation,
            'blackout_active': blackout_active | switch_started,
            'blackout_backlog': (
                blackout_active | switch_started
            ) & decision_backlog,
            'blackout_loss_bits': blackout_loss_bits,
            'current_channels': self.current_channels.copy(),
            'dwell_remaining': self.dwell_remaining.copy(),
            'blackout_remaining': self.blackout_remaining.copy(),
        }
        self._advance_switch_timers()
        self.t += 1
        done = self.t >= self.trace.slots
        if not done:
            self._apply_arrivals(self.t)
        else:
            self._arrivals_this_slot.fill(0)
            self._drops_this_slot.fill(0)
        return self._observations(), reward, done, step_info

    def agent_infos(self) -> list[dict[str, Any]]:
        t = min(self.t, self.trace.slots - 1)
        infos: list[dict[str, Any]] = []
        mask_theta = (
            self.trace.theta[t]
            if self.prediction_mask_source == 'real'
            else np.full(self.trace.num_channels, 0.5)
        )
        for vehicle in range(self.trace.num_vehicles):
            mask_sensing = (
                self.trace.sensing_idle[t, vehicle]
                if self.sensing_mask_source == 'real'
                else np.full(self.trace.num_channels, 0.5)
            )
            raw_mask = build_action_mask(
                mask_sensing,
                mask_theta,
                self.trace.power_cap[t, vehicle],
                self.reward_layers,
                self.action_mask_thresholds,
                self.enabled_modes,
            )
            switch_mask = self._apply_switch_mask(raw_mask, vehicle)
            mask = switch_mask.copy()
            if not self.queues[vehicle]:
                mask[:] = False
                mask[0] = True
            infos.append(
                {
                    't': t,
                    'vehicle_id': vehicle,
                    'num_channels': self.trace.num_channels,
                    'state_features': list(self.state_features),
                    'feature_slices': feature_slices(
                        self.trace.num_channels,
                        self.state_features,
                    ),
                    'action_mask': mask,
                    'raw_action_mask': raw_mask,
                    'switch_action_mask': switch_mask,
                    'queue_length': len(self.queues[vehicle]),
                    'module_variant': self.module_variant,
                    'protection_variant': self.protection_variant,
                    'ablation_info_source': self.ablation_info_source,
                    'ablation_mask_source': self.ablation_mask_source,
                    'sensing_info_source': self.sensing_info_source,
                    'prediction_info_source': self.prediction_info_source,
                    'sensing_mask_source': self.sensing_mask_source,
                    'prediction_mask_source': self.prediction_mask_source,
                    'mask_policy_version': self.mask_policy_version,
                    'current_channel': int(self.current_channels[vehicle]),
                    'dwell_remaining': int(self.dwell_remaining[vehicle]),
                    'blackout_remaining': int(self.blackout_remaining[vehicle]),
                }
            )
        return infos

    def _observations(self) -> np.ndarray:
        t = min(self.t, self.trace.slots - 1)
        n = self.trace.num_vehicles
        m = self.trace.num_channels
        queue_occupancy = np.asarray(
            [len(queue) / max(self.queue_capacity, 1) for queue in self.queues],
            dtype=float,
        )[:, None]
        hol_age = (self._hol_ages() / self.hol_age_scale_slots)[:, None]
        parts: list[np.ndarray] = []
        for feature in self.state_features:
            if feature == 'delta':
                value = (
                    self.trace.sensing_idle[t]
                    if self.use_sensing
                    else np.full((n, m), 0.5, dtype=float)
                )
            elif feature == 'fresh_dqi':
                shared = self.trace.fresh_dqi[t]
                value = np.repeat(shared[None, :], n, axis=0)
                if not self.use_dqi:
                    value = np.full((n, m), 0.5, dtype=float)
            elif feature == 'theta':
                theta = self.trace.theta[t] if self.use_prediction else np.full(m, 0.5)
                value = np.repeat(theta[None, :], n, axis=0)
            elif feature == 'current_cqi':
                if t == 0:
                    value = np.full((n, m), 0.5, dtype=float)
                else:
                    r_ref = max(
                        float(self.config.get('current_cqi_reference_mbps', self.config.get('dqi', {}).get('r_ref_mbps', 80.0))),
                        1e-9,
                    )
                    # When _shuffled_quality_mbps is present (Diagnostic 3:
                    # shuffle_current_cqi), use the permuted copy so that
                    # current_cqi becomes unreliable — testing whether the
                    # agent was depending on it instead of DQI.
                    quality_source = getattr(
                        self.trace, '_shuffled_quality_mbps', self.trace.quality_mbps
                    )
                    value = np.clip(quality_source[t - 1] / r_ref, 0.0, 1.0)
            elif feature == 'current_channel':
                value = np.zeros((n, m), dtype=float)
                selected = np.flatnonzero(self.current_channels >= 0)
                if selected.size:
                    value[selected, self.current_channels[selected]] = 1.0
            elif feature == 'dwell_remaining':
                scale = max(self.minimum_dwell_slots, 1)
                value = (self.dwell_remaining.astype(float) / scale)[:, None]
            elif feature == 'blackout_remaining':
                scale = max(self.switch_blackout_slots, 1)
                value = (self.blackout_remaining.astype(float) / scale)[:, None]
            elif feature == 'power_cap':
                value = self.trace.power_cap[t]
            elif feature == 'dqi_age':
                maximum = max(float(np.max(self.trace.dqi_validity_slots[t])), 1.0)
                shared_age = self.trace.dqi_age[t] / maximum
                if self.config.get('r5c2_protocol_version'):
                    shared_age = (self.trace.dqi_age[t] * self.slot_duration_ms /
                                  float(self.config['r5c2_age_feature_scale_ms']))
                    if not self.use_dqi:
                        shared_age = np.zeros_like(shared_age)
                value = np.repeat(shared_age[None, :], n, axis=0)
            elif feature == 'queue_occupancy':
                value = queue_occupancy
            elif feature == 'hol_age':
                value = np.clip(hol_age, 0.0, 1.0)
            elif feature == 'previous_channel_load':
                value = np.repeat(self.previous_channel_load[None, :], n, axis=0)
            elif feature == 'vehicle_context':
                value = ((np.arange(n, dtype=float) + 0.5) / max(n, 1))[:, None]
            elif feature == 'dqi_available':
                value = np.full((n, 1), float(self.use_dqi))
                if self.config.get('r5c2_protocol_version'):
                    value[:] = float(self.use_dqi and self.trace.dqi_source_slot[t] >= 0)
            elif feature == 'prediction_available':
                value = np.full((n, 1), float(self.use_prediction))
            elif feature == 'sensing_available':
                value = np.full((n, 1), float(self.use_sensing))
            else:
                raise ValueError(f'unknown multi-vehicle state feature: {feature}')
            parts.append(np.asarray(value, dtype=float))
        return np.concatenate(parts, axis=1)

    def _apply_switch_mask(self, raw_mask: np.ndarray, vehicle: int) -> np.ndarray:
        mask = np.asarray(raw_mask, dtype=bool).copy()
        if not self.switching_enabled:
            return mask
        if self.blackout_remaining[vehicle] > 0:
            mask[:] = False
            mask[0] = True
            return mask
        current = int(self.current_channels[vehicle])
        if current >= 0 and self.dwell_remaining[vehicle] > 0:
            restricted = np.zeros_like(mask)
            restricted[0] = True
            for mode in ('overlay', 'underlay'):
                index = encode_action(current, mode, self.trace.num_channels)
                restricted[index] = mask[index]
            return restricted
        return mask

    def _resolve_switching_actions(self, requested: list[Any]) -> tuple[
        list[Any], np.ndarray, np.ndarray, np.ndarray
    ]:
        n = self.trace.num_vehicles
        no_access = decode_action(0, self.trace.num_channels)
        effective = list(requested)
        switch_started = np.zeros(n, dtype=bool)
        dwell_violation = np.zeros(n, dtype=bool)
        blackout_active = self.blackout_remaining > 0
        if not self.switching_enabled:
            return effective, switch_started, dwell_violation, blackout_active

        for vehicle, action in enumerate(requested):
            if blackout_active[vehicle]:
                effective[vehicle] = no_access
                continue
            if action.mode == 'no_access' or action.channel < 0:
                continue
            current = int(self.current_channels[vehicle])
            if current < 0:
                self.current_channels[vehicle] = int(action.channel)
                self.dwell_remaining[vehicle] = self.minimum_dwell_slots
                continue
            if action.channel == current:
                continue
            if self.dwell_remaining[vehicle] > 0:
                dwell_violation[vehicle] = True
                effective[vehicle] = no_access
                continue
            self.current_channels[vehicle] = int(action.channel)
            switch_started[vehicle] = True
            if self.dwell_semantics == 'total_from_switch':
                self.dwell_remaining[vehicle] = self.minimum_dwell_slots
            if self.switch_blackout_slots > 0:
                self.blackout_remaining[vehicle] = self.switch_blackout_slots
                effective[vehicle] = no_access
            elif self.dwell_semantics != 'total_from_switch':
                self.dwell_remaining[vehicle] = self.minimum_dwell_slots
        return effective, switch_started, dwell_violation, blackout_active

    def _advance_switch_timers(self) -> None:
        if not self.switching_enabled:
            return
        for vehicle in range(self.trace.num_vehicles):
            if self.dwell_semantics == 'total_from_switch':
                if self.blackout_remaining[vehicle] > 0:
                    self.blackout_remaining[vehicle] -= 1
                if self.dwell_remaining[vehicle] > 0:
                    self.dwell_remaining[vehicle] -= 1
            else:
                if self.blackout_remaining[vehicle] > 0:
                    self.blackout_remaining[vehicle] -= 1
                    if self.blackout_remaining[vehicle] == 0:
                        self.dwell_remaining[vehicle] = self.minimum_dwell_slots
                elif self.dwell_remaining[vehicle] > 0:
                    self.dwell_remaining[vehicle] -= 1

    def _physical_reward(
        self,
        served_bits: np.ndarray,
        contention_lost: np.ndarray,
        access_violation: np.ndarray,
        selected_dqi: np.ndarray,
    ) -> np.ndarray:
        reward_cfg = self.config.get('reward', {})
        service_weight = float(reward_cfg.get('service_weight', 1.0))
        drop_weight = float(reward_cfg.get('drop_weight', 1.0))
        contention_weight = float(reward_cfg.get('contention_weight', 0.25))
        violation_weight = float(reward_cfg.get('violation_weight', 2.0))
        queue_weight = float(reward_cfg.get('queue_weight', 0.05))
        service_fraction = served_bits / max(self.packet_size_bits, 1.0)
        if reward_cfg.get('type', 'physical') == 'physical_dqi_shaping':
            gamma = float(reward_cfg.get('dqi_gamma', 2.0))
            dqi = np.where(np.isfinite(selected_dqi), selected_dqi, 0.5)
            service_fraction = service_fraction * np.power(dqi, gamma)
        violation = np.asarray(access_violation, dtype=float)
        queue_fraction = np.asarray(
            [len(queue) / max(self.queue_capacity, 1) for queue in self.queues],
            dtype=float,
        )
        reward = (
            service_weight * service_fraction
            - drop_weight * self._drops_this_slot
            - contention_weight * contention_lost.astype(float)
            - violation_weight * violation
            - queue_weight * queue_fraction
        )
        low, high = reward_cfg.get('clip_reward', [-3.0, 2.0])
        return np.clip(reward, float(low), float(high))

    def _serve_queue(
        self,
        vehicle: int,
        service_bits: float,
        completed_this_slot: np.ndarray,
    ) -> float:
        available = max(float(service_bits), 0.0)
        served = 0.0
        while available > 0.0 and self.queues[vehicle]:
            packet = self.queues[vehicle][0]
            amount = min(available, packet[0])
            packet[0] -= amount
            available -= amount
            served += amount
            if packet[0] <= 1e-9:
                self.queues[vehicle].popleft()
                self.completed_packets[vehicle] += 1
                completed_this_slot[vehicle] += 1
        return served

    def _hol_ages(self) -> np.ndarray:
        ages = np.zeros(self.trace.num_vehicles, dtype=float)
        for vehicle, queue in enumerate(self.queues):
            if queue:
                ages[vehicle] = max(self.t - int(queue[0][1]), 0)
        return ages

    def _apply_arrivals(self, t: int) -> None:
        arrivals = self.trace.packet_arrivals[t].astype(bool)
        self._arrivals_this_slot = arrivals.astype(int)
        self._drops_this_slot = np.zeros(self.trace.num_vehicles, dtype=int)
        for vehicle in np.flatnonzero(arrivals):
            self.generated_packets[vehicle] += 1
            if len(self.queues[vehicle]) >= self.queue_capacity:
                self.dropped_packets[vehicle] += 1
                self._drops_this_slot[vehicle] = 1
            else:
                self.queues[vehicle].append([self.packet_size_bits, float(t)])
