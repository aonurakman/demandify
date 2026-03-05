"""
Seeded demand generation for SUMO.
"""
from collections import Counter
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import logging
import math

from demandify.sumo.network import SUMONetwork
from demandify.sumo.departure_schedule import (
    sequential_departure_times,
    format_departure_time,
)

logger = logging.getLogger(__name__)


class DemandGenerator:
    """Generate seeded synthetic demand for SUMO."""

    THROUGH_TRAFFIC_SHARE = 0.7
    PREFERRED_BOUNDARY_ROLE_BOOST = 2.3
    OPPOSITE_BOUNDARY_ROLE_FACTOR = 0.5
    BOUNDARY_SCARCITY_EXPONENT = 0.5
    BOUNDARY_ADAPTIVE_GAIN_MIN = 0.6
    BOUNDARY_ADAPTIVE_GAIN_MAX = 12.0
    BOUNDARY_ADAPTIVE_EPS = 1e-9
    BOUNDARY_MARGIN_RATIO = 0.08
    BOUNDARY_MARGIN_MIN_METERS = 25.0
    BOUNDARY_MARGIN_MAX_METERS = 200.0
    
    def __init__(self, network: SUMONetwork, seed: int = 42):
        """
        Initialize demand generator.
        
        Args:
            network: SUMO network
            seed: Random seed for reproducibility
        """
        self.network = network
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        adjacency_source = {}
        if self.network is not None:
            adjacency_source = getattr(self.network, "adjacency", {}) or {}
        # Freeze adjacency traversal order for cross-process reproducibility.
        self._deterministic_adjacency = {
            edge_id: tuple(sorted(neighbors))
            for edge_id, neighbors in adjacency_source.items()
        }
        self._route_cache: Dict[Tuple[str, str], bool] = {}
    
    def select_od_pairs(
        self,
        max_od_pairs: int = 150,
        max_consecutive_failures: int = 10000,
        min_trip_distance: float = 0.0,
    ) -> List[Tuple[str, str]]:
        """
        Select origin/destination edges by building validated OD pairs.
        Validates EACH pair individually (reachability + min distance) and resamples failures.

        Notes:
            - Validation is based on the network topology and lane permissions for passenger vehicles.
            - Returned pairs are unique.

        Args:
            max_od_pairs: Target number of OD pairs to create
            max_consecutive_failures: Max failures before giving up
            min_trip_distance: Minimum Euclidean distance between origin and destination O/D

        Returns:
            List of (origin_edge, destination_edge) pairs that should be routable in SUMO.
        """
        all_edges = self.network.get_all_edges()
        
        if len(all_edges) < 2:
            raise ValueError(f"Insufficient edges in network: {len(all_edges)}")
        
        sampling_profiles = self._build_od_sampling_profiles(all_edges)
        edge_roles = sampling_profiles["edge_roles"]
        has_boundary_bias = sampling_profiles["has_boundary_bias"]
        
        # Build valid OD pairs one at a time
        valid_pairs: List[Tuple[str, str]] = []
        valid_pairs_set = set()
        consecutive_failures = 0
        total_attempts = 0
        selected_origin_roles: Counter = Counter()
        selected_destination_roles: Counter = Counter()
        
        logger.info(f"Building up to {max_od_pairs} validated OD pairs (min_dist={min_trip_distance}m)...")
        logger.info(
            "OD selection profiles: incoming=%s, outgoing=%s, internal=%s, "
            "through_share=%.2f, boundary_margin=%.1fm",
            sampling_profiles["role_counts"].get("incoming", 0),
            sampling_profiles["role_counts"].get("outgoing", 0),
            sampling_profiles["role_counts"].get("internal", 0),
            self.THROUGH_TRAFFIC_SHARE if has_boundary_bias else 0.0,
            sampling_profiles["boundary_margin_m"],
        )

        current_min_dist = min_trip_distance
        logger.info(f"Generating {max_od_pairs} OD pairs (min_dist={int(min_trip_distance)}m)...")
        
        while len(valid_pairs) < max_od_pairs:
            # Safety break
            if total_attempts > max_od_pairs * 100 and total_attempts > 10000:
                logger.warning(f"Reached maximum attempt limit ({total_attempts}). Stopping with {len(valid_pairs)} pairs.")
                break
                
            if consecutive_failures > max_consecutive_failures:
                logger.warning(f"Stopped after {consecutive_failures} consecutive failures. Created {len(valid_pairs)} pairs.")
                break
                
            # Progress Logging (every 100 pairs)
            if len(valid_pairs) > 0 and len(valid_pairs) % 100 == 0 and consecutive_failures == 0:
                 logger.info(f"  ... {len(valid_pairs)}/{max_od_pairs} OD pairs found")

            # Adaptive Relaxation: If stuck, reduce min distance requirement
            if consecutive_failures > 500 and consecutive_failures % 500 == 0 and current_min_dist > 0:
                 old_dist = current_min_dist
                 current_min_dist *= 0.8
                 logger.info(f"  Relaxing min_dist from {int(old_dist)}m to {int(current_min_dist)}m after failures")

            # Most attempts use a boundary-biased origin/destination profile to mimic
            # traffic entering and leaving a partial-city calibration bbox. The rest
            # use neutral road-importance weights so internal trips still appear.
            use_through_profile = has_boundary_bias and self.rng.rand() < self.THROUGH_TRAFFIC_SHARE
            if use_through_profile:
                origin = self._sample_edge(
                    all_edges,
                    sampling_profiles["origin_probs"],
                    sampling_profiles["edge_index"],
                )
                destination = self._sample_edge(
                    all_edges,
                    sampling_profiles["destination_probs"],
                    sampling_profiles["edge_index"],
                    exclude_edge=origin,
                )
            else:
                origin = self._sample_edge(
                    all_edges,
                    sampling_profiles["base_probs"],
                    sampling_profiles["edge_index"],
                )
                destination = self._sample_edge(
                    all_edges,
                    sampling_profiles["base_probs"],
                    sampling_profiles["edge_index"],
                    exclude_edge=origin,
                )

            if origin is None or destination is None:
                raise ValueError("Could not sample a valid OD pair from the network")
            
            total_attempts += 1
            
            # 1. Filter by minimum Euclidean distance
            valid_dist = True
            if current_min_dist > 0:
                ox, oy = self.network.get_edge_centroid(origin)
                dx, dy = self.network.get_edge_centroid(destination)
                dist = math.hypot(dx - ox, dy - oy)
                if dist < current_min_dist:
                    valid_dist = False
            
            if not valid_dist:
                consecutive_failures += 1
                continue

            pair = (origin, destination)
            if pair in valid_pairs_set:
                consecutive_failures += 1
                continue

            # 2. Validate reachability for this specific pair
            if self._has_route(origin, destination):
                valid_pairs.append(pair)
                valid_pairs_set.add(pair)
                selected_origin_roles[edge_roles.get(origin, "internal")] += 1
                selected_destination_roles[edge_roles.get(destination, "internal")] += 1
                consecutive_failures = 0  # Reset on success
            else:
                consecutive_failures += 1
        
        if len(valid_pairs) == 0:
            raise ValueError("Could not create any valid OD pairs - network may be disconnected or constraints too strict")
        
        if consecutive_failures >= max_consecutive_failures:
            logger.warning(f"Stopped after {consecutive_failures} consecutive failures. "
                          f"Created {len(valid_pairs)} pairs (target was {max_od_pairs})")
        
        origins = {o for o, _ in valid_pairs}
        destinations = {d for _, d in valid_pairs}

        logger.info(
            f"Created {len(valid_pairs)} valid OD pairs from {total_attempts} attempts: "
            f"{len(origins)} unique origins, {len(destinations)} unique destinations"
        )
        logger.info(
            "Selected OD role mix: origins [%s], destinations [%s]",
            self._format_role_counts(selected_origin_roles),
            self._format_role_counts(selected_destination_roles),
        )

        return valid_pairs

    def _build_od_sampling_profiles(self, edges: List[str]) -> Dict[str, object]:
        """Build neutral and boundary-biased sampling profiles for OD generation."""
        base_weights = self._calculate_edge_weights(edges)
        base_probs = self._normalize_weights(base_weights)
        edge_index = {edge_id: idx for idx, edge_id in enumerate(edges)}

        boundary = self.network.get_network_boundary()
        boundary_margin = self._boundary_margin(boundary)
        edge_roles: Dict[str, str] = {}
        incoming_degree_map = self._build_incoming_degree_map(edges)
        if boundary is not None:
            for edge_id in edges:
                edge_roles[edge_id] = self._classify_edge_boundary_role(
                    edge_id,
                    boundary,
                    boundary_margin,
                    incoming_degree_map.get(edge_id, 0),
                )
        else:
            edge_roles = {edge_id: "internal" for edge_id in edges}

        role_counts = Counter(edge_roles.values())
        role_masses = self._compute_role_masses(edges, base_weights, edge_roles)
        has_boundary_bias = role_counts.get("incoming", 0) > 0 and role_counts.get("outgoing", 0) > 0
        if has_boundary_bias:
            origin_bias_meta = self._compute_boundary_bias_meta(role_masses, preferred_role="incoming")
            destination_bias_meta = self._compute_boundary_bias_meta(role_masses, preferred_role="outgoing")
            origin_probs = self._normalize_weights(
                self._apply_boundary_role_bias(
                    edges,
                    base_weights,
                    edge_roles,
                    preferred_role="incoming",
                    bias_meta=origin_bias_meta,
                )
            )
            destination_probs = self._normalize_weights(
                self._apply_boundary_role_bias(
                    edges,
                    base_weights,
                    edge_roles,
                    preferred_role="outgoing",
                    bias_meta=destination_bias_meta,
                )
            )
            total_mass = max(float(role_masses.get("total", 0.0)), self.BOUNDARY_ADAPTIVE_EPS)
            logger.debug(
                "Adaptive OD boundary bias: role_mass_share [incoming=%.4f, outgoing=%.4f, internal=%.4f], "
                "origin [gain=%.3f, preferred_factor=%.3f, opposite_factor=%.3f], "
                "destination [gain=%.3f, preferred_factor=%.3f, opposite_factor=%.3f]",
                float(role_masses.get("incoming", 0.0)) / total_mass,
                float(role_masses.get("outgoing", 0.0)) / total_mass,
                float(role_masses.get("internal", 0.0)) / total_mass,
                float(origin_bias_meta["gain"]),
                float(origin_bias_meta["preferred_factor"]),
                float(origin_bias_meta["opposite_factor"]),
                float(destination_bias_meta["gain"]),
                float(destination_bias_meta["preferred_factor"]),
                float(destination_bias_meta["opposite_factor"]),
            )
        else:
            origin_bias_meta = None
            destination_bias_meta = None
            origin_probs = list(base_probs)
            destination_probs = list(base_probs)

        effective_origin_probs = self._mix_probabilities(origin_probs, base_probs, self.THROUGH_TRAFFIC_SHARE)
        effective_destination_probs = self._mix_probabilities(
            destination_probs, base_probs, self.THROUGH_TRAFFIC_SHARE
        )

        return {
            "edge_roles": edge_roles,
            "role_counts": role_counts,
            "role_masses": role_masses,
            "boundary_margin_m": boundary_margin,
            "has_boundary_bias": has_boundary_bias,
            "origin_bias_meta": origin_bias_meta,
            "destination_bias_meta": destination_bias_meta,
            "edge_index": edge_index,
            "base_probs": base_probs,
            "origin_probs": origin_probs,
            "destination_probs": destination_probs,
            "effective_origin_probs": effective_origin_probs,
            "effective_destination_probs": effective_destination_probs,
        }

    def _apply_boundary_role_bias(
        self,
        edges: List[str],
        base_weights: List[float],
        edge_roles: Dict[str, str],
        preferred_role: str,
        bias_meta: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        """Bias weights toward preferred boundary roles while keeping internal edges alive."""
        if bias_meta is None:
            role_masses = self._compute_role_masses(edges, base_weights, edge_roles)
            bias_meta = self._compute_boundary_bias_meta(role_masses, preferred_role)

        preferred_factor = float(bias_meta["preferred_factor"])
        opposite_factor = float(bias_meta["opposite_factor"])

        biased_weights = []
        for edge_id, base_weight in zip(edges, base_weights):
            role = edge_roles.get(edge_id, "internal")
            factor = 1.0
            if role == preferred_role:
                factor = preferred_factor
            elif role in {"incoming", "outgoing"}:
                factor = opposite_factor
            biased_weights.append(base_weight * factor)
        return biased_weights

    @staticmethod
    def _compute_role_masses(
        edges: List[str],
        base_weights: List[float],
        edge_roles: Dict[str, str],
    ) -> Dict[str, float]:
        """Compute total weighted mass for each role and overall."""
        role_masses: Dict[str, float] = {"incoming": 0.0, "outgoing": 0.0, "internal": 0.0}
        for edge_id, base_weight in zip(edges, base_weights):
            role = edge_roles.get(edge_id, "internal")
            role_masses[role] = role_masses.get(role, 0.0) + float(base_weight)
        role_masses["total"] = float(sum(base_weights))
        return role_masses

    @classmethod
    def _compute_adaptive_boundary_gain(cls, preferred_mass: float, total_mass: float) -> float:
        """Compute clipped adaptive gain based on preferred-role scarcity."""
        eps = float(cls.BOUNDARY_ADAPTIVE_EPS)
        preferred_mass = float(preferred_mass)
        total_mass = float(total_mass)
        if preferred_mass <= eps or total_mass <= eps:
            return 1.0

        non_preferred_mass = max(0.0, total_mass - preferred_mass)
        scarcity = (non_preferred_mass + eps) / (preferred_mass + eps)
        gain = scarcity ** float(cls.BOUNDARY_SCARCITY_EXPONENT)
        return float(min(cls.BOUNDARY_ADAPTIVE_GAIN_MAX, max(cls.BOUNDARY_ADAPTIVE_GAIN_MIN, gain)))

    @classmethod
    def _compute_boundary_bias_meta(
        cls,
        role_masses: Dict[str, float],
        preferred_role: str,
    ) -> Dict[str, float]:
        """Compute adaptive gain and multiplicative factors for a preferred role."""
        total_mass = float(role_masses.get("total", 0.0))
        preferred_mass = float(role_masses.get(preferred_role, 0.0))
        eps = float(cls.BOUNDARY_ADAPTIVE_EPS)

        if preferred_mass <= eps or total_mass <= eps:
            gain = 1.0
            scarcity = 1.0
        else:
            gain = cls._compute_adaptive_boundary_gain(preferred_mass, total_mass)
            non_preferred_mass = max(0.0, total_mass - preferred_mass)
            scarcity = (non_preferred_mass + eps) / (preferred_mass + eps)

        return {
            "gain": float(gain),
            "scarcity": float(scarcity),
            "preferred_factor": float(cls.PREFERRED_BOUNDARY_ROLE_BOOST * gain),
            "opposite_factor": float(cls.OPPOSITE_BOUNDARY_ROLE_FACTOR / gain),
        }

    def _classify_edge_boundary_role(
        self,
        edge_id: str,
        boundary: Tuple[float, float, float, float],
        boundary_margin: float,
        incoming_degree: int,
    ) -> str:
        """Classify an edge as incoming, outgoing, or internal relative to the bbox."""
        geom = self.network.get_edge_geometry(edge_id)
        if geom is None or len(geom.coords) < 2:
            return "internal"

        outgoing_degree = len(self.network.adjacency.get(edge_id, set()))

        start = geom.coords[0]
        end = geom.coords[-1]
        start_dist = self._distance_to_boundary(start[0], start[1], boundary)
        end_dist = self._distance_to_boundary(end[0], end[1], boundary)

        start_near_boundary = start_dist <= boundary_margin
        end_near_boundary = end_dist <= boundary_margin
        distance_delta = max(5.0, boundary_margin * 0.15)

        if start_near_boundary and (incoming_degree == 0 or end_dist > start_dist + distance_delta):
            return "incoming"
        if end_near_boundary and (outgoing_degree == 0 or start_dist > end_dist + distance_delta):
            return "outgoing"
        return "internal"

    @staticmethod
    def _normalize_weights(weights: List[float]) -> List[float]:
        """Normalize a weight list to probabilities."""
        total_weight = float(sum(weights))
        if total_weight <= 0.0:
            if not weights:
                return []
            uniform_prob = 1.0 / float(len(weights))
            return [uniform_prob] * len(weights)
        return [float(weight) / total_weight for weight in weights]

    @staticmethod
    def _mix_probabilities(biased: List[float], neutral: List[float], biased_share: float) -> List[float]:
        """Blend biased and neutral probability profiles."""
        return [
            (float(biased_share) * float(biased_prob)) + ((1.0 - float(biased_share)) * float(neutral_prob))
            for biased_prob, neutral_prob in zip(biased, neutral)
        ]

    def _sample_edge(
        self,
        edges: List[str],
        probabilities: List[float],
        edge_index: Dict[str, int],
        exclude_edge: Optional[str] = None,
    ) -> Optional[str]:
        """Sample one edge, optionally excluding a previously sampled edge."""
        if not edges:
            return None
        if exclude_edge is None:
            return str(self.rng.choice(edges, p=probabilities))

        adjusted_probs = np.array(probabilities, dtype=float)
        exclude_idx = edge_index.get(exclude_edge)
        if exclude_idx is not None and 0 <= exclude_idx < len(adjusted_probs):
            adjusted_probs[exclude_idx] = 0.0

        prob_sum = float(adjusted_probs.sum())
        if prob_sum <= 0.0:
            remaining_edges = [edge_id for edge_id in edges if edge_id != exclude_edge]
            if not remaining_edges:
                return None
            return str(self.rng.choice(remaining_edges))

        adjusted_probs /= prob_sum
        return str(self.rng.choice(edges, p=adjusted_probs))

    def _build_incoming_degree_map(self, edges: List[str]) -> Dict[str, int]:
        """Count incoming passenger-accessible predecessors for each edge."""
        counts = {edge_id: 0 for edge_id in edges}
        for from_edge, neighbors in self.network.adjacency.items():
            if from_edge not in counts:
                continue
            for to_edge in neighbors:
                if to_edge in counts:
                    counts[to_edge] += 1
        return counts

    def _boundary_margin(self, boundary: Optional[Tuple[float, float, float, float]]) -> float:
        """Compute a reasonable boundary band width in network meters."""
        if boundary is None:
            return self.BOUNDARY_MARGIN_MIN_METERS
        min_x, min_y, max_x, max_y = boundary
        width = max(0.0, max_x - min_x)
        height = max(0.0, max_y - min_y)
        shortest_side = min(width, height) if width > 0.0 and height > 0.0 else max(width, height)
        if shortest_side <= 0.0:
            return self.BOUNDARY_MARGIN_MIN_METERS
        return min(
            self.BOUNDARY_MARGIN_MAX_METERS,
            max(self.BOUNDARY_MARGIN_MIN_METERS, shortest_side * self.BOUNDARY_MARGIN_RATIO),
        )

    @staticmethod
    def _distance_to_boundary(
        x: float,
        y: float,
        boundary: Tuple[float, float, float, float],
    ) -> float:
        """Return the shortest distance from a point to the bbox boundary."""
        min_x, min_y, max_x, max_y = boundary
        return min(abs(x - min_x), abs(x - max_x), abs(y - min_y), abs(y - max_y))

    @staticmethod
    def _format_role_counts(counts: Counter) -> str:
        """Format role counts consistently for logs."""
        ordered_roles = ("incoming", "outgoing", "internal")
        return ", ".join(f"{role}={int(counts.get(role, 0))}" for role in ordered_roles)
    
    def _calculate_edge_weights(self, edges: List[str]) -> List[float]:
        """Calculate selection weights for edges based on road importance."""
        weights = []
        for edge in edges:
            attrs = self.network.get_edge_attributes(edge)
            # Default priority 1 (minor), speed 13.89 (50kmh), lanes 1
            p = max(1, attrs.get('priority', 1))
            s = max(5.0, attrs.get('speed', 13.89))
            l = max(1, attrs.get('numLanes', 1))
            
            # Boost highways significantly
            weight = p * s * l
            weights.append(weight)
        return weights
    
    def _has_route(self, from_edge: str, to_edge: str) -> bool:
        """
        Check if there exists a directed route from from_edge to to_edge.
        Uses BFS on the network adjacency graph.
        
        Args:
            from_edge: Origin edge ID
            to_edge: Destination edge ID
            
        Returns:
            True if a route exists, False otherwise
        """
        if from_edge == to_edge:
            return True

        cache_key = (from_edge, to_edge)
        cached = self._route_cache.get(cache_key)
        if cached is not None:
            return cached

        # BFS for reachability (complete traversal; visited guarantees termination)
        visited = {from_edge}
        queue = [from_edge]
        idx = 0

        while idx < len(queue):
            current = queue[idx]
            idx += 1

            # Deterministic outgoing-neighbor order avoids PYTHONHASHSEED drift.
            neighbors = self._deterministic_adjacency.get(current, ())

            for neighbor in neighbors:
                if neighbor == to_edge:
                    self._route_cache[cache_key] = True
                    return True

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        self._route_cache[cache_key] = False
        return False
    
    def genome_to_demand_csv(
        self,
        genome: np.ndarray,
        od_pairs: List[Tuple[str, str]],
        departure_bins: List[Tuple[int, int]],
        output_file: Path
    ) -> pd.DataFrame:
        """
        Convert a genome (vehicle counts per OD pair and time bin) to demand.csv.
        
        Args:
            genome: 1D array of vehicle counts (length = num_od_pairs * num_bins)
            od_pairs: List of (origin_edge, destination_edge) tuples
            departure_bins: List of (start_time, end_time) tuples in seconds
            output_file: Path to save demand.csv
        
        Returns:
            DataFrame with demand
        """
        num_od = len(od_pairs)
        num_bins = len(departure_bins)
        
        assert len(genome) == num_od * num_bins, "Genome size mismatch"
        
        # Reshape genome to (num_od, num_bins)
        counts = genome.reshape(num_od, num_bins)
        
        # Generate individual trips
        trips = []
        trip_id = 0
        
        for od_idx, (origin, dest) in enumerate(od_pairs):
            for bin_idx, (start_time, end_time) in enumerate(departure_bins):
                count = int(max(0, round(counts[od_idx, bin_idx])))
                
                # Generate individual departure times within the bin
                if count > 0:
                    departure_times = sequential_departure_times(start_time, end_time, count)
                    
                    for dep_time in departure_times:
                        trips.append({
                            'ID': f'trip_{trip_id}',
                            'origin link id': origin,
                            'destination link id': dest,
                            'departure timestep': float(dep_time)
                        })
                        trip_id += 1
        
        # Create DataFrame
        demand_df = pd.DataFrame(
            trips,
            columns=["ID", "origin link id", "destination link id", "departure timestep"],
        )
        
        # Sort deterministically for reproducibility.
        # Multiple trips often share the same departure timestep, so we include
        # stable tie-breakers to avoid run-to-run row reordering.
        if not demand_df.empty:
            demand_df = demand_df.sort_values(
                by=["departure timestep", "origin link id", "destination link id", "ID"],
                kind="mergesort",
            ).reset_index(drop=True)
        
        # Save to CSV
        output_file.parent.mkdir(parents=True, exist_ok=True)
        demand_df.to_csv(output_file, index=False)
        
        logger.debug(f"Generated {len(demand_df)} trips in demand.csv: {output_file}")
        
        return demand_df
    
    def demand_csv_to_trips_xml(
        self,
        demand_csv: Path,
        output_trips_file: Path
    ):
        """
        Convert demand.csv to SUMO trips.xml format.
        
        Args:
            demand_csv: Path to demand.csv
            output_trips_file: Path for output trips.xml
        """
        # Read demand
        demand_df = pd.read_csv(demand_csv)
        
        # Create XML
        root = ET.Element('routes')
        
        for _, row in demand_df.iterrows():
            trip = ET.SubElement(root, 'trip')
            trip.set('id', row['ID'])
            trip.set('depart', format_departure_time(row['departure timestep']))
            trip.set('from', row['origin link id'])
            trip.set('to', row['destination link id'])
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space='  ')
        tree.write(output_trips_file, encoding='utf-8', xml_declaration=True)
        
        logger.debug(f"Created trips.xml: {output_trips_file}")
