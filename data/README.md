# Data organisation

`public/` contains attributed source snapshots that can be distributed with the repository. Each snapshot preserves original source bytes, observation coverage and hashes. These are retrieval vintages, not reconstructions of information available to historical decision-makers.

- [US Treasury par yields](public/treasury/README.md)
- [Germany production, prices and ECB shocks](public/germany_monetary_policy/README.md)

`private/` is ignored by Git. Restricted provider exports, demonstration archives with unconfirmed redistribution rights and account-specific files belong there and are not required by continuous integration.

Future cleaned analysis datasets must be generated from an identified snapshot by documented code. Record units, seasonal adjustments, transformations, date alignment, exclusions and missing values. A missing observation must never silently become a zero. Preserve source files when creating a new vintage.

See [third-party notices](../THIRD_PARTY_NOTICES.md) and [provenance](../PROVENANCE.md).
