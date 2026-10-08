"""Real contract sources: harvesters that write snapshots, loaders that read them.

Two halves, kept apart on purpose:

  harvest   Network. Run by hand (python -m src.sources.harvest), politely paced,
            writing a dated snapshot under data/raw/ with a sha256 manifest.
  load      No network, ever. A ContractSource reads a snapshot from disk, so a
            scan, a test or an eval is reproducible from files alone and never
            depends on a government website being up or unchanged.
"""
