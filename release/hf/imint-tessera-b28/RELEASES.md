# imint-tessera-b28 releases

Each row is a revision of the private Hugging Face repository
`RISE-Sweden/imint-tessera-b28`. Pass the `MANIFEST.json` sha256 to
`load_release(..., expected_manifest_sha256=...)`: it is the trust anchor,
and it lives here, on GitHub, not next to the files it vouches for. A row is
added through a reviewed PR in a commit after the exporter commit (the
manifest names that commit, so it cannot carry its own row), and the anchor
is only as trustworthy as write access to `main`.

| Published | HF revision | Exporter commit | MANIFEST.json sha256 | Source checkpoint sha256 |
|---|---|---|---|---|
