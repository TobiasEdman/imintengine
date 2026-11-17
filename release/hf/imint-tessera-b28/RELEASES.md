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
| 2026-11-14 | `a40411aed116faaaeea95b1ec271f7dfab63b858` | `09e135c67f0714f8aa049ba20db5d9fe9416369d` | `89e4f6d3b558e1bc96da3a23950d003e08bcb24c174c196c6d9c09eb96686bf7` | `4fc82ff5a967c54550278e74b79a10cb1f6808a71d988b768fd61238453889e0` |
| 2026-11-17 | `65a60e950c62760ed73782009a96999209a05a7f` | `7557298d813dfb5a33aee5bbe12dd34b59508563` | `e047991ab417a7d373d75d25d040653b93a3760657ba38541b2306efac2eb7ec` | `4fc82ff5a967c54550278e74b79a10cb1f6808a71d988b768fd61238453889e0` |

`65a60e9` supersedes `a40411a`: the same tensors and `config.json`, only the
model card changed (how class 28 is labelled, #167). Use the newest row.
