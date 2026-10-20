"""Recompute the manuscript's position metrics from its archived traces."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE / 'evidence/manifest.json').read_text())
    for case in manifest['cases']:
        source = HERE / 'evidence' / (case['id'] + '.json')
        assert hashlib.sha256(source.read_bytes()).hexdigest() == case['source_sha256'], case['id']
        metrics = json.loads(source.read_text())
        with np.load(HERE / 'evidence' / case['archived_trace']) as data:
            reference, actual = data['reference'], data['actual']
            assert reference.shape == actual.shape == (metrics['steps'], 3)
            assert np.isfinite(reference).all() and np.isfinite(actual).all()
            error_mm = np.linalg.norm(actual-reference, axis=1)*1000
            rmse = float(np.sqrt(np.mean(error_mm**2)))
            maximum = float(error_mm.max())
            assert np.isclose(rmse, metrics['rmse_mm'], atol=1e-9, rtol=0)
            assert np.isclose(maximum, metrics['max_error_mm'], atol=1e-9, rtol=0)
            assert np.isclose((len(actual)-1)*float(data['dt']), case['writing_duration_s'])
        print(f"{case['id']}: RMSE={rmse:.4f} mm; max={maximum:.4f} mm; verified")
    print(f"Verified {len(manifest['cases'])} archived trajectories. No simulations or hardware started.")


if __name__ == '__main__':
    main()
