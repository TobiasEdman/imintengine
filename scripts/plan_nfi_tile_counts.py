#!/usr/bin/env python3
"""Read-only aggregate tile sizing. Never freeze, fetch, score or emit plot coordinates."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from imint.eval.fieldtruth import NFI_KEY, NFI_MODELS, observation_keys, resolve_observation_year
from imint.training.tile_config import TileConfig
from imint.training.tile_time import resolve_tile_year

TILE = TileConfig(size_px=512)
CROP_PX = 496
SEED = 20261008


def crop_mask(frame: pd.DataFrame, bbox: dict) -> np.ndarray:
    """Same floor and centre-crop geometry as colocation and all model readers."""
    rows = np.floor((bbox['north'] - frame.Northing.to_numpy()) / TILE.gsd_m)
    cols = np.floor((frame.Easting.to_numpy() - bbox['west']) / TILE.gsd_m)
    pad = (TILE.size_px - CROP_PX) // 2
    return (rows >= pad) & (rows < pad + CROP_PX) & (cols >= pad) & (cols < pad + CROP_PX)


def tile_cover(frame: pd.DataFrame) -> dict:
    """Feasible centred-lattice cover, recursively split for the common crop.

    This extends the versioned 2024 placement; it is not a global minimum.
    No tile coordinates or observation identities leave this function.
    """
    if frame.empty:
        return {'tiles': 0, 'plots': 0, 'original_lattice_cells': 0, 'extra_crop_splits': 0}
    if frame.Year.nunique() != 1 or observation_keys(frame).has_duplicates:
        raise ValueError("tile cover requires unique plot-years from one year")
    groups = frame.assign(_x=np.floor(frame.Easting / TILE.size_m).astype('int64'),
                          _y=np.floor(frame.Northing / TILE.size_m).astype('int64'))
    pending = [g for _, g in groups.groupby(['_x', '_y'], sort=True)]
    cells, tiles, covered = len(pending), 0, 0
    while pending:
        group = pending.pop()
        bbox = TILE.bbox_from_center((group.Easting.min()+group.Easting.max())/2,
                                     (group.Northing.min()+group.Northing.max())/2)
        TILE.assert_bbox_matches(bbox)
        if crop_mask(group, bbox).all():
            tiles += 1; covered += len(group)
            continue
        assert len(group) > 1, 'a centred singleton must lie in every native crop'
        axis = max(('Easting', 'Northing'), key=lambda k: group[k].max()-group[k].min())
        ordered = group.sort_values([axis, *NFI_KEY])
        midpoint = len(group)//2
        pending.extend([ordered.iloc[:midpoint], ordered.iloc[midpoint:]])
    assert covered == len(frame)
    return {'tiles': tiles, 'plots': covered, 'original_lattice_cells': cells,
            'extra_crop_splits': tiles-cells}


def balanced_candidates(frame: pd.DataFrame, years: list[int], count: int) -> pd.DataFrame:
    selected = frame[frame.Year.isin(years)].copy()
    selected['_rank'] = [hashlib.sha256(f'{SEED}:{t}:{p}:{y}'.encode('ascii')).hexdigest()
                         for t,p,y in observation_keys(selected)]
    selected = (selected.sort_values(['_rank', *NFI_KEY]).groupby('Year').head(count)
                .drop(columns='_rank').sort_values(NFI_KEY).reset_index(drop=True))
    assert selected.Year.value_counts().to_dict() == dict.fromkeys(years, count)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--years', type=int, nargs='+', required=True)
    parser.add_argument('--observations-per-year', type=int, nargs='+', default=[],
                        help='Additional planning scenarios; never a frozen quota.')
    args = parser.parse_args()
    if any(n <= 0 for n in args.observations_per_year):
        parser.error('--observations-per-year must be positive')
    years = sorted(set(args.years))
    assert len(years) == len(args.years) and len(years) >= 2
    root = args.data_root
    inputs = {}
    def read(path, parquet=True):
        payload = path.read_bytes()
        inputs[str(path)] = {'bytes':len(payload), 'sha256':hashlib.sha256(payload).hexdigest()}
        return pd.read_parquet(io.BytesIO(payload)) if parquet else json.loads(payload)

    full = read(root/'nfi/nfi_plots.parquet')
    full = full[full.Year.isin(years)].copy()
    assert not observation_keys(full).has_duplicates
    values = full[['VolPine','VolContorta','VolSpruce','VolBirch','VolOtherDec','Easting','Northing']].to_numpy(float)
    valid = np.isfinite(values).all(axis=1) & (values[:,:5] >= 0).all(axis=1)
    invalid = {str(y):int((~valid & (full.Year.to_numpy() == y)).sum()) for y in years}
    full = full.loc[valid].copy()
    source = read(root/'nfi/nfi_index_unified_v2_512.parquet')
    features = []
    for model in NFI_MODELS:
        frame = read(root/f'distill/heads/{model}_r2_plot_features.parquet')
        features.append(resolve_observation_year(frame, source)[NFI_KEY])
    exposure = pd.concat(features).drop_duplicates(NFI_KEY)
    exposed = observation_keys(full).isin(observation_keys(exposure))
    eligible = full.loc[~exposed].copy().reset_index(drop=True)
    assert not observation_keys(eligible).isin(observation_keys(exposure)).any()
    available = eligible.groupby('Year').size().to_dict()
    assert set(available) == set(years)
    ceiling = min(available.values())
    if any(n > ceiling for n in args.observations_per_year):
        parser.error(f'planning scenario exceeds annual candidate ceiling {ceiling}')
    balanced = balanced_candidates(eligible, years, int(ceiling))

    # Existing index includes cohort and the475 evaluation-only campaign.
    index = read(root/'nfi/nfi_index_v4_2026-09-30.parquet')
    pad = (TILE.size_px-CROP_PX)//2
    in_crop = index.row.between(pad,pad+CROP_PX-1) & index.col.between(pad,pad+CROP_PX-1)
    known = observation_keys(index.loc[in_crop]).unique()
    reuse = observation_keys(eligible).isin(known)
    reused_existing_index = {str(y):int((reuse & (eligible.Year.to_numpy()==y)).sum()) for y in years}
    # Legacy validation may need source/modality repair and independence review.
    manifest = read(root/'audits/holdout_val_phaseA.json', parquet=False)
    old_root = root/'holdout_val_512'
    paths = {p.stem:p for p in old_root.glob('*.npz') if not p.name.endswith('.tmp.npz')}
    assert len({r['name'] for r in manifest}) == len(manifest)
    inventory, issues, tessera_sources = Counter(), Counter(), Counter()
    for record in manifest:
        path = paths.get(record['name'])
        if path is None or record['year'] not in years:
            continue
        try:
            with np.load(path, allow_pickle=False) as tile:
                with tile.zip.open('spectral.npy') as spectral:
                    version = np.lib.format.read_magic(spectral)
                    reader = {(1,0):np.lib.format.read_array_header_1_0,
                              (2,0):np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:
                        issues['spectral_header_version'] += 1; continue
                    shape, _, dtype = reader(spectral)
                if shape != (24,512,512) or dtype.kind not in 'iuf':
                    issues['spectral_shape_or_dtype'] += 1; continue
                year = resolve_tile_year(tile)
                if year != record['year']:
                    issues['year_mismatch_or_unknown'] += 1; continue
                if 'bbox_3006' in tile:
                    values = np.asarray(tile['bbox_3006']).reshape(-1)
                    if values.shape != (4,):
                        issues['invalid_bbox'] += 1; continue
                    bbox = dict(zip(('west','south','east','north'),map(float,values)))
                else:
                    bbox = TILE.bbox_from_center(float(tile['easting']),float(tile['northing']))
                expected_bbox = {k:float(record['bbox_3006'][k]) for k in bbox}
                if not np.isfinite([*bbox.values(), *expected_bbox.values()]).all():
                    issues['nonfinite_bbox'] += 1; continue
                if not np.isclose(np.mod(list(bbox.values()),TILE.gsd_m),0).all():
                    issues['off_grid_bbox'] += 1; continue
                TILE.assert_bbox_matches(bbox)
                if any(abs(bbox[k]-expected_bbox[k]) > 1 for k in bbox):
                    issues['manifest_bbox_mismatch'] += 1; continue
                stamp = tile.get('tessera_source')
                tessera_sources[str(stamp) if stamp is not None else 'unstamped'] += 1
            inventory[str(year)] += 1
            reuse |= (eligible.Year.to_numpy()==year) & crop_mask(eligible,bbox)
        except (OSError,ValueError,KeyError,EOFError) as exc:
            issues['metadata_'+type(exc).__name__] += 1
    # Unknown files are not credited as reusable.
    unknown_files = len(set(paths)-{r['name'] for r in manifest})
    reuse_keys = observation_keys(eligible.loc[reuse])
    rows=[]
    for year in years:
        all_year = eligible[eligible.Year==year]
        selected = balanced[balanced.Year==year]
        missing_all = all_year.loc[~observation_keys(all_year).isin(reuse_keys)]
        missing_selected = selected.loc[~observation_keys(selected).isin(reuse_keys)]
        rows.append({'year':year,'valid_plot_years':int((full.Year==year).sum()),
                     'excluded_teacher_plot_years':int((full.loc[exposed].Year==year).sum()),
                     'eligible_plot_years':len(all_year),'balanced_planning_plot_years':len(selected),
                     'existing_v4_index_common_crop':reused_existing_index[str(year)],
                     'geometric_reuse_all_roots':len(all_year)-len(missing_all),
                     'balanced_geometric_reuse':len(selected)-len(missing_selected),
                     'cover_all_eligible_from_scratch':tile_cover(all_year),
                     'cover_balanced_from_scratch':tile_cover(selected),
                     'additional_all_if_legacy_reusable':tile_cover(missing_all),
                     'additional_balanced_if_legacy_reusable':tile_cover(missing_selected)})
    scenarios = []
    for count in sorted(set(args.observations_per_year)):
        sample = balanced_candidates(eligible, years, count)
        annual = []
        for year in years:
            selected = sample[sample.Year == year]
            missing = selected.loc[~observation_keys(selected).isin(reuse_keys)]
            annual.append({'year': year, 'plot_years': len(selected),
                           'geometric_reuse': len(selected) - len(missing),
                           'cover_from_scratch': tile_cover(selected),
                           'additional_if_legacy_reusable': tile_cover(missing)})
        scenarios.append({'observations_per_year': count, 'by_year': annual})
    print(json.dumps({'schema':'nfi-tile-sizing-v1','preliminary':True,'input_freeze':False,
        'fetch_started':False,'gpu_started':False,'nmd_scoring':False,'years':years,
        'identity':NFI_KEY,'inputs':inputs,'teacher_feature_union':len(exposure),
        'common_crop_px':CROP_PX,'tile_size_px':TILE.size_px,'invalid_rows_by_year':invalid,
        'balanced_planning_ceiling_per_year':int(ceiling),'selection_seed':SEED,
        'legacy_existing_files':len(paths),'legacy_metadata_verified_by_year':dict(inventory),
        'legacy_metadata_issues':dict(issues),'legacy_unmatched_files':unknown_files,
        'legacy_tessera_sources':dict(tessera_sources),'by_year':rows,
        'additional_planning_scenarios': scenarios,
        'limitations':['Counts are geometric feasible covers, not global minimum or imagery success forecasts.',
            'Legacy reuse is conditional on modality/source readiness and independence/model-selection audit.',
            'v4 index geometry assumed512px as produced; full tile readiness remains a freeze gate.',
            'Final equal count must be recalculated after all coverage and modality exclusions, before scores.',
            'Read-only sizing does not authorize acquisition, input freeze, inference or joint NMD scoring.']},
        indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
