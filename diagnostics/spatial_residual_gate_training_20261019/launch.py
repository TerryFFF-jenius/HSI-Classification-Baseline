"""Cross-platform gate launch. Default: read-only preflight. No implicit approval."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from diagnostics.honghu_single_seed_20261018 import diagnostic_train as trainer
from diagnostics.spatial_residual_gate_training_20261019 import runtime

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--assets-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--report', help='New preflight JSON outside the proposed run output')
    parser.add_argument('--execute', action='store_true', help='Future use only: requires explicit separately reviewed approval')
    parser.add_argument('--approval', help='Separate user-approved authorization; inert template is rejected')
    args = parser.parse_args()
    command = [sys.executable, *sys.argv]
    report = runtime.preflight(trainer, args.project_root, args.assets_dir, args.output_dir, args.expected_commit)
    report['actual_command'] = command
    if args.report:
        target = Path(args.report).resolve()
        output = Path(args.output_dir).resolve()
        if target == output or output in target.parents:
            raise ValueError('Preflight report cannot populate the future run directory')
        runtime.write_new(target, report)
    if not args.execute:
        print(json.dumps({'status': report['status'], 'training_authorized': False,
                          'head': report['runtime_code_identity']['git']['head'],
                          'worktree_clean': report['runtime_code_identity']['git']['worktree_clean'],
                          'asset_count': len(report['assets']), 'report': args.report,
                          'output_created': False, 'training_started': False}))
        return 0
    if not args.approval:
        raise PermissionError('--execute requires separate reviewed approval; preparation is not authorization')
    approval = runtime.validate_approval(args.approval, report)
    # Exclusive, permanent consumed marker: failures also consume this approved attempt.
    lock = Path(args.approval).resolve().with_name(Path(args.approval).name + '.consumed.json')
    launch_id = runtime.sha256(args.approval)
    runtime.write_new(lock, {'launch_id': launch_id, 'actual_command': command,
                            'status': 'approved_attempt_consumed', 'utc': datetime.now(timezone.utc).isoformat()})
    runtime.ACTIVE_LAUNCH = {'id': launch_id}
    trainer_args = [str(Path(trainer.__file__).resolve()), '--project-root', str(Path(args.project_root).resolve()),
                    '--assets-dir', str(Path(args.assets_dir).resolve()), '--model-name', 'gscvit_dssr_spatial',
                    '--output-dir', str(Path(args.output_dir).resolve()), '--spatial-state-strength', '0.5',
                    '--spatial-residual-gate', '--gate-approval', str(Path(args.approval).resolve()),
                    '--gate-launch-id', launch_id]
    original_argv = sys.argv
    code, error = 1, None
    try:
        sys.argv = trainer_args
        with lock.with_suffix('.stdout.log').open('x', encoding='utf-8') as out, lock.with_suffix('.stderr.log').open('x', encoding='utf-8') as err:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    trainer.main()
                    code = 0
                except BaseException:
                    error = traceback.format_exc()
                    print(error, file=sys.stderr)
    except BaseException:
        error = traceback.format_exc()
    finally:
        sys.argv = original_argv
        runtime.ACTIVE_LAUNCH = None
        exit_report = {'actual_command': command, 'trainer_argv': trainer_args, 'exit_code': code,
                       'error': error, 'launch_id': launch_id, 'utc': datetime.now(timezone.utc).isoformat()}
        runtime.write_new(lock.with_suffix('.exit.json'), exit_report)
        identity = Path(args.output_dir) / 'gate_run_identity.json'
        if identity.is_file() and runtime.read_json(identity)['gate_context']['launch_id'] == launch_id:
            runtime.write_new(Path(args.output_dir) / 'launch_exit.json', exit_report)
    print(json.dumps({'exit_code': code, 'output_dir': args.output_dir, 'attempt_record': str(lock)}))
    return code

if __name__ == '__main__':
    raise SystemExit(main())
