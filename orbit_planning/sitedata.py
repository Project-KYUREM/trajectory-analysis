"""Read and write one run's site payload in the compact on-disk format.

A run directory ``docs/data/<model>/<sequence>/`` holds three files:

* ``cloud.json``: run metadata (model, sequence, labels, cost axis, blurb,
  settings, frame rotation) and a ``points_file`` header describing the
  binary point table.
* ``points.bin.gz``: every plotted point as gzipped little-endian columns, in
  this order (so each typed array is aligned): ``launch_mjd2000`` float64,
  ``cost`` float32, ``tof_days`` float32, ``index`` uint32 (the point's
  position in the raw ``results/.../pareto.json``, which also names it:
  ``p00042``) and ``flags`` uint8 (1 Pareto, 2 polished, 4 described). About
  21 bytes per point before compression, against ~125 as JSON objects.
* ``detail.json.gz``: ``{point id: trajectory.describe() output}``, gzipped;
  the repeated key names compress ~5-10x.

The previous format was one ``cloud.json`` with ``points`` and ``detail``
inline; ``read_run`` still reads it, ``convert`` rewrites it in place, and
the site's ``loadCloud`` reads both. This module imports neither SPICE nor
pykep::

    python orbit_planning/sitedata.py          # convert every old-format run under docs/data
"""
import argparse
import gzip
import json

import numpy as np

import paths

POINTS_FILE = 'points.bin.gz'
DETAIL_FILE = 'detail.json.gz'
COLUMNS = (('launch_mjd2000', '<f8'), ('cost', '<f4'), ('tof_days', '<f4'),
           ('index', '<u4'), ('flags', 'u1'))
FLAGS = {'pareto': 1, 'polished': 2, 'detail': 4}


def point_id(index):
    """Return the id of the point at ``index`` of the raw result (``p00042``)."""
    return f"p{index:05d}"


def write_run(directory, meta, points, detail):
    """Write one run's payload.

    Args:
        directory (pathlib.Path): The run directory (created if needed).
        meta (dict): Run metadata; must not contain ``points`` or ``detail``.
        points (list[dict]): Records with ``id`` (``p<index>``), ``cost``,
            ``tof_days``, ``launch_mjd2000``, ``pareto``, ``polished`` and
            ``detail``.
        detail (dict): Point id to trajectory description.

    Returns:
        int: Bytes written across the three files.
    """
    directory.mkdir(parents=True, exist_ok=True)
    n = len(points)
    columns = {
        'launch_mjd2000': np.array([p['launch_mjd2000'] for p in points], '<f8'),
        'cost': np.array([p['cost'] for p in points], '<f4'),
        'tof_days': np.array([p['tof_days'] for p in points], '<f4'),
        'index': np.array([int(p['id'][1:]) for p in points], '<u4'),
        'flags': np.array([sum(bit for name, bit in FLAGS.items() if p.get(name))
                           for p in points], 'u1'),
    }
    blob = b''.join(columns[name].astype(dtype).tobytes() for name, dtype in COLUMNS)
    (directory / POINTS_FILE).write_bytes(gzip.compress(blob, 6))
    (directory / DETAIL_FILE).write_bytes(
        gzip.compress(json.dumps(detail, separators=(',', ':')).encode(), 6))
    header = dict(meta, points_file=dict(name=POINTS_FILE, n=n, columns=[list(c) for c in COLUMNS],
                                         flags=FLAGS),
                  detail_file=DETAIL_FILE)
    (directory / 'cloud.json').write_text(json.dumps(header, separators=(',', ':')))
    return sum((directory / f).stat().st_size for f in ('cloud.json', POINTS_FILE, DETAIL_FILE))


def read_run(directory):
    """Read one run's payload in either format.

    Args:
        directory (pathlib.Path): The run directory.

    Returns:
        tuple[dict, list[dict], dict]: ``(meta, points, detail)``, with points
        as records like ``write_run`` takes.
    """
    meta = json.loads((directory / 'cloud.json').read_text())
    if 'points' in meta:   # previous single-file format
        points, detail = meta.pop('points'), meta.pop('detail', {})
        return meta, points, detail
    header = meta['points_file']
    blob = gzip.decompress((directory / header['name']).read_bytes())
    n, offset, cols = header['n'], 0, {}
    for name, dtype in header['columns']:
        width = np.dtype(dtype).itemsize
        cols[name] = np.frombuffer(blob, dtype, n, offset)
        offset += width * n
    flags = header['flags']
    points = [dict(id=point_id(int(cols['index'][i])), cost=float(cols['cost'][i]),
                   tof_days=float(cols['tof_days'][i]),
                   launch_mjd2000=float(cols['launch_mjd2000'][i]),
                   **{name: bool(cols['flags'][i] & bit) for name, bit in flags.items()})
              for i in range(n)]
    detail = json.loads(gzip.decompress((directory / meta['detail_file']).read_bytes()))
    return meta, points, detail


def is_old_format(directory):
    """Return whether a run directory still holds the single-file format."""
    path = directory / 'cloud.json'
    return path.is_file() and not (directory / POINTS_FILE).is_file()


def convert(directory):
    """Rewrite an old-format run in the compact format; return (bytes before, after)."""
    before = (directory / 'cloud.json').stat().st_size
    meta, points, detail = read_run(directory)
    for p in points:
        p.setdefault('detail', p['id'] in detail)
    return before, write_run(directory, meta, points, detail)


def convert_all(log=print):
    """Convert every old-format run under ``docs/data``.

    Returns:
        int: Number of runs converted.
    """
    done = 0
    for cloud in sorted(paths.DOCS_DATA_DIR.glob('*/*/cloud.json')):
        if is_old_format(cloud.parent):
            before, after = convert(cloud.parent)
            log(f"converted {cloud.parent.relative_to(paths.DOCS_DATA_DIR)}: "
                f"{before / 1e6:.1f} -> {after / 1e6:.2f} MB")
            done += 1
    return done


def main():
    """Convert old-format runs in place."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    print(f"{convert_all()} run(s) converted")


if __name__ == '__main__':
    main()
