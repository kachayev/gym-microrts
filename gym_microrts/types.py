from dataclasses import dataclass
from typing import List, Any


@dataclass
class UnitType:
    id: int
    name: str
    cost: int
    hp: int
    min_damage: int
    max_damage: int
    attack_range: int
    produce_time: int
    move_time: int
    attack_time: int
    harvest_time: int
    return_time: int
    harvest_amount: int
    sight_radius: int
    is_resource: bool
    is_stockpile: bool
    can_harvest: bool
    can_move: bool
    can_attack: bool
    produces: List[str]
    produced_by: List[str]


@dataclass
class GameInfo:
    move_conflict_resolution_strategy: int
    unit_types: List[UnitType]


@dataclass
class Player:
    id: int
    resources: int


@dataclass
class Unit:
    type: str
    id: int
    player: int
    x: int
    y: int
    resources: int
    hitpoints: int


@dataclass
class Pgs:
    width: int
    height: int
    terrain: str
    players: List[Player]
    units: List[Unit]


@dataclass
class GameState:
    time: int
    pgs: Pgs
    actions: List[Any]
