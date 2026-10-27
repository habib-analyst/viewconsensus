# Contributing

Issues and pull requests are welcome on GitHub.

For a bug report, include the smallest model/input example that reproduces the issue, the full `ConsensusResult` if one was produced, and the Python, PyTorch, NumPy, and Pillow versions involved.

For a code change:

1. Add or update tests for the behavior you changed.
2. Run `python -m pytest -q`.
3. Keep the public API small. A new option should solve a concrete classifier-integration problem rather than expose another internal detail.
4. Do not add benchmark claims without the script, data description, and measured output needed to reproduce them.
