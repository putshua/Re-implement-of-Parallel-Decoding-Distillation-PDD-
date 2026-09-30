# RCM6 restart fix

Observed current run reached step117 and has complete checkpoints25/50/75/100. Subsequent restart logs show FileExistsError at the existing-output guard. Previous launcher unconditionally unset RESUME and RESUME_PATH, incorrectly extending the initial fresh-start request to every restart.

Main launcher now defaults to --resume auto and honors explicit RESUME_PATH/RESUME. Rank0 selects the highest numbered completed checkpoint in the current output directory, requiring model.pt and every expected rank file; broadcasts the result once so all ranks agree. Empty output starts fresh. Existing metrics without a valid checkpoint fail without overwriting results. RESUME=none requires a new output for a new experiment.

Read-only selection against the actual16-rank RCM6 run returned step_000100. Successful restoration archives metrics before discarding records after the restored step; TensorBoard purge behavior already handles rollback. No existing experiment files were modified by validation. The remote cluster job was not restarted from this session.

Validation includes fresh-start selection, incomplete newer checkpoint fallback, missing-rank exclusion, explicit overrides, preserving metrics on missing checkpoints, and archiving/trimming metrics. Shell syntax and Ruff checks pass. See auto_resume_tests.log and auto_resume_launch.txt.
