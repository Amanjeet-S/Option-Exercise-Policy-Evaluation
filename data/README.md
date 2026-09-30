# Data organisation

`public/` contains attributed source snapshots that can be distributed with the repository. Each snapshot preserves original source bytes, observation coverage and hashes. These are retrieval vintages, not reconstructions of information available to historical decision-makers.

- [US Treasury par yields](public/treasury/README.md)
- [Germany production, prices and ECB shocks](public/germany_monetary_policy/README.md)

`private/` is ignored by Git. Restricted provider exports, demonstration archives with unconfirmed redistribution rights and account-specific files belong there and are not required by continuous integration.

Research findings derived from LSEG observations may be published under the confirmation recorded on 27 September 2026. Raw exports, cleaned subsets, quote-level transformations and other recoverable source observations remain private. The [publication policy](../docs/publication_policy.md) also covers values embedded in figures, notebooks, PDF files and supplementary tables. Existing open datasets retain their documented reuse terms.

Future cleaned analysis datasets must be generated from an identified snapshot by documented code. Record units, seasonal adjustments, transformations, date alignment, exclusions and missing values. A missing observation must never silently become a zero. Preserve source files when creating a new vintage.

The [paper reproduction route](../docs/reproducing_paper.md) uses published mathematical fixtures, released research summaries and original computational timing measurements. It does not read or export restricted financial observations. Its local outputs are explicitly distinct from the paper's original financial simulation results.

See [third-party notices](../THIRD_PARTY_NOTICES.md) and [provenance](../PROVENANCE.md).
