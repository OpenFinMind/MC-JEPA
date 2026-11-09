"""Feature baselines for ``tools.eval_jepa_probe``.

Each baseline module builds features for every (symbol, decision date) and returns ``{"symbol", "date", <name>}``
arrays, ``<name>`` of shape [N, d]. ``eval_jepa_probe`` saves them as an npz, and ``probe`` scores them with the
same ridge / GBDT heads as the encoder states: ``--embeddings <npz> --features <name>`` alone, or
``--extra <npz>`` joined to an encoder's embeddings (same rows, and ``online+<name>``).
"""
