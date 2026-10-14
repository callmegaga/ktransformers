#!/usr/bin/env python3
"""Aggregate the latest complete cumulative trace per (layer, phase).

Input is a server log or perf.py tee log. Fields are totals, not means. The
last trace need not include the final tail when TRACE_EVERY > 1. Never sum
all repeated cumulative records, or double-count host wait and device time.
"""
import argparse
import json
from pathlib import Path
import statistics


def summarize(text):
    latest, previous = {}, {}
    for line in text.splitlines():
        marker = 'KT_SYCL_PIPELINE '
        if marker not in line:
            continue
        try:
            record = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError as error:
            raise ValueError('Malformed pipeline JSON') from error
        key = (record['layer'], record['phase'])
        if record['totals'].get('calls', 0) <= 0:
            raise ValueError('Trace has no calls')
        if key in latest:
            previous[key] = latest[key]
        latest[key] = record
    if not latest:
        raise ValueError('No KT_SYCL_PIPELINE records found')
    phases = {}
    for phase in sorted({key[1] for key in latest}):
        layers = [r for (_, p), r in sorted(latest.items()) if p == phase]
        totals = {}
        for layer in layers:
            for key, value in layer['totals'].items():
                totals[key] = totals.get(key, 0) + value
        calls = totals['calls']
        phases[phase] = {'layer_count': len(layers), 'totals': totals,
                         'mean_per_layer_call': {k: v / calls for k, v in totals.items()},
                         'median_per_layer_us': {
                             key: statistics.median(r['totals'].get(key, 0) / r['totals']['calls'] for r in layers)
                             for key in totals if key.endswith('_us')},
                         'layers': layers}
    intervals = {}
    for (layer, phase), record in latest.items():
        before = previous.get((layer, phase))
        if before is None or record['totals']['calls'] <= before['totals']['calls']:
            continue
        delta = {key: value - before['totals'].get(key, 0) for key, value in record['totals'].items()}
        stats = intervals.setdefault(phase, {'layer_count': 0, 'totals': {}})
        stats['layer_count'] += 1
        for key, value in delta.items():
            stats['totals'][key] = stats['totals'].get(key, 0) + value
    for stats in intervals.values():
        stats['mean_per_layer_call'] = {k: v / stats['totals']['calls'] for k, v in stats['totals'].items()}
    return {'phases': phases, 'last_intervals': intervals,
            'caveat': 'Cumulative totals include initialization. Last-interval differences exclude earlier startup, but tail may be missing. Diagnostic instrumentation affects latency.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.log.read_text(errors='replace'))
    result['source'] = str(args.log.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print('Final reported intervals (exclude earlier startup):', json.dumps(result['last_intervals'], indent=2))
    for phase, record in result['phases'].items():
        print(phase, 'layers:', record['layer_count'], 'layer calls:', record['totals']['calls'])
        print(json.dumps(record['mean_per_layer_call'], indent=2))


if __name__ == '__main__': main()
