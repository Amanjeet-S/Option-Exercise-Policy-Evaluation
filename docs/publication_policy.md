# Publication policy

## 1. Basis and scope

I apply the University of St Andrews guidance supplied on 9 October 2026 to data obtained through its LSEG subscription. It permits individual study and research, identifies non-commercial academic papers and university applications as acceptable uses of calculated findings, and prohibits redistribution of the underlying observations. Distribution of derived data is limited to insubstantial portions, released non-systematically and without independent commercial value. These conditions qualify the earlier confirmation of 27 September; transformation alone does not establish permission to publish.

| Material | Treatment |
| --- | --- |
| Raw provider exports, quote histories, saved responses and copies | Excluded from GitHub and third-party distribution; private retention only for the authorised research use, followed by completion-stage deletion |
| Cleaned subsets, rounded quotes, midpoints, spreads, rescaled series, calibrated input datasets and reversible encodings | Transformation does not clear redistribution; assess source restrictions and the retention requirement |
| Statistical summaries, calculated findings, selected tables and figures based on LSEG inputs | Review for non-commercial academic purpose, insubstantial quantity, non-systematic distribution, absence of independent commercial value and disclosure or reconstruction of restricted observations |
| Original methods and extraction code | Review attribution, credentials, embedded observations and copied vendor material before release; code does not convey data rights |
| Independently stipulated mathematical models and their simulations | Assess their own provenance and permissions; do not attribute them to LSEG when no LSEG input was used |
| Open datasets with documented reuse permissions | Apply their existing source terms and attribution requirements |

## 2. Review of findings, figures and supplements

I review each proposed paper and repository release together, including tables, figure coordinates, notebook outputs, numerical supplements, attachments and the history made available. Small releases can collectively reconstruct a dataset or become systematic distribution. An aggregate or chart can also have independent commercial value without reconstructing its inputs. Removing a raw CSV or changing its numbers is therefore insufficient.

A figure based on subscription inputs is a proposed derived research output, not an automatically cleared image. I select only the quantities needed to explain the research finding and check companion tables and downloadable chart data. Quote histories, complete fitted input curves and observation-level panels remain excluded unless separately permitted. Publication of an academic finding does not clear an internship or external-company use described as commercial in the university guidance.

The current model-only revision uses stated mathematical parameters and generated paths. These outputs have different provenance from the historical financial specifications informed by LSEG. Simulating from a market-calibrated input would still require a source assessment; simulation alone does not remove that dependency. This policy does not itself approve any pending release.

## 3. Acquisition and attribution

I do not use the subscription for systematic bulk downloading or to assemble a redistributed market-data service. Extraction code must support the bounded research question and the account's permitted access. Publishing that code does not authorise a recipient's acquisition or retention practice.

For each LSEG-dependent result, I identify LSEG Workspace and the University of St Andrews access, distinguish observation dates from retrieval dates, and identify my subsequent calculation. I verify dates, fields, units and transformations from the acquisition record rather than inventing a retrieval vintage. A concise source acknowledgement is: 'Source: LSEG Workspace, accessed through the University of St Andrews subscription; my calculations.' Relevant captions and the data section also identify the verified period. Independently simulated figures cite their mathematical parameter source instead. The separate Cboe and Treasury sources retain their own attribution and terms.

## 4. Retention and reproduction

I retain underlying subscription data only for its authorised study and research use and delete it when the project for which it was obtained is completed, as the university guidance requires. The review must cover working exports, saved responses, restricted transformed input datasets, copied archives, private history compilations, caches and backups under my control. A Git ignore rule and a private folder do not replace this requirement. A later model-only revision does not by itself justify indefinite retention of an earlier completed project's inputs.

Before deletion, I identify the restricted copies and distinguish them from original code, independent simulations, permitted findings and source-free provenance. Retention decisions and deletion receipts can preserve research chronology without retaining the underlying observations. I preserve original historical records only to the extent lawful retention permits, and record removed dependencies separately rather than rewrite old hashes or findings.

Exact historical replay depends on lawful possession and permitted retention of the matching inputs. I do not distribute the original subscription archive, including to another licence holder. Once required inputs are deleted, their absence is an explicit limit on exact historical reproduction. Public analytical examples and verification of permitted summaries can remain available. A new acquisition or substituted vintage is a new experiment, not an exact replay.

## 5. Original licences and third-party material

Original released software uses [MIT](../LICENSE); original research expression uses [CC BY 4.0](../LICENSES/CC-BY-4.0.txt) within [LICENSING.md](../LICENSING.md). These grants do not relicense LSEG source content or any restricted derived content embedded in a table, figure or paper. A release-specific source notice identifies excluded material where applicable. Academic publication permission alone does not establish permission to grant unrestricted commercial reuse of every component. The original licence terms remain unchanged for independently original material.

The university guidance does not clear another provider's data, vendor documentation, publisher text or third-party figures. I apply their separate source permissions and [notices](../THIRD_PARTY_NOTICES.md).
