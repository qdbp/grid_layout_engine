from __future__ import annotations
from collections import Set, defaultdict
from dataclasses import dataclass
from typing import Union, AbstractSet, Dict, DefaultDict, Tuple, Iterable

from pulp import LpProblem, LpBinary
from pulp_lparray import lparray

from gle.util import pod


@dataclass(frozen=True, order=True)
class Tag:
    name: str


@dataclass(frozen=True, order=True)
class TileXY:
    name: str
    tags: Set[Tag]

    exclude_tags: Set[Tag]
    exclude_tiles: Set[TileXY]

    match_r_tags: Set[Tag] = None
    match_r_tiles: Set[Tag] = None

    match_d_tags: Set[Tag] = None
    match_d_tiles: Set[Tag] = None

    match_l_tags: Set[Tag] = None
    match_l_tiles: Set[Tag] = None

    match_u_tags: Set[Tag] = None
    match_u_tiles: Set[Tag] = None


# toolbox:

# coverage req't:
#   forall xy. conv(coverer, Kcoverer) - covered >= 0

# no overlap req't
#   forall xy. conv(p, Kp) + conv(q, Kq) <= 1


# target problems:

# factorio belt layout with fixed edge conditions and continuity req'ts w/o splitters
#  --> stretch goal w/ splitters

# openttd rail routing w/o {terraforming, signalling}
#  --> stretch goal w/o terraforming w/ signalling

# factorio bots-only factory packing
# factorio solar farm packing


# noinspection PyPep8Naming
def create_problem_xy(
    w: int,
    h: int,
    tile_universe: AbstractSet[TileXY],
    edge_conditions: Iterable[Tuple[TileXY, int, int]],
    allow_null_tiles: bool = True,
):

    tile_map = {tile: ix for ix, tile in enumerate(sorted(tile_universe))}
    N = len(tile_map)

    tag_map = defaultdict(set)
    for tile, ix in tile_map.items():
        for tag in tile.tags:
            if not isinstance(tag, Tag):
                continue
            tag_map[tile].add(tag)

    prob = LpProblem()

    Z = lparray.create_anon("TileField", shape=(h, w, N), cat=LpBinary)

    # set up edge conditions
    for tile, x, y in edge_conditions:
        (Z[x, y, tile_map[tile]] == 1).constrain(prob, f"EdgeCondition[{x},{y}]")

    # disallow empty tiles, if needed
    if not allow_null_tiles:
        (Z.sum(axis=-1) >= 1).constrain(prob, "NoNullTiles")

    # create tile exclusions
    for tile, tile_ix in tile_map.items():
        exclude_ixes = sorted(
            (
                set.union(*[tag_map[exclude_tag] for exclude_tag in tile.exlude_tags])
                | {tile_map[exclude_tile] for exclude_tile in tile.exclude_tiles}
            )
            - {tile_ix}
        )
        if len(exclude_ixes) > 0:
            (Z[:, :, exclude_ixes].sum(axis=-1) <= 1).constrain(
                prob, f"Exclusion_{tile.name}"
            )
