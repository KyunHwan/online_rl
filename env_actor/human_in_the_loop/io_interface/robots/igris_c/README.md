# env_actor/human_in_the_loop/io_interface/robots/igris_c

> **Status: scaffolding for a not-yet-wired subsystem inside another not-yet-wired subsystem.** No path through [run_online_rl.py](../../../../../run_online_rl.py) reaches this directory. See [../../../README.md](../../../README.md#status) for the parent-level Status callout.

This was intended as the HIL-side igris_c bridge — equivalent to [`../../../auto/io_interface/...`](../../../) in an earlier project layout — but the directory contains only a stub and outdated references. If you are looking for the live igris_c bridge work, see [`env_actor/robot_io_interface/robots/igris_c/README.md`](../../../../robot_io_interface/robots/igris_c/README.md) instead.

If you intend to bring HIL up for igris_c eventually:

1. First finish the live (auto-mode) igris_c bridge under [`env_actor/robot_io_interface/robots/igris_c/`](../../../../robot_io_interface/robots/igris_c/) — see that directory's README.
2. Then revisit this directory as part of the larger HIL activation effort.

The original content here was an aspirational migration plan from a removed `inference_engine` package; it referenced files and paths that no longer exist in the repo.
