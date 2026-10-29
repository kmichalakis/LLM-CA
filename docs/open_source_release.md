# Public Code and Paper Citation

The code release uses the existing MIT license. The repository destination is
`https://github.com/kmichalakis/LLM-CA`. Publication is complete only when that
repository is public and contains the reviewed source.

## Code Versus Research Evidence

The public source includes Python modules, tests, configurations, schemas,
prompts, documentation, citation metadata and the offline test workflow.
Datasets, annotations, generated runs, validation logs, frozen paper packs,
Word drafts and publisher templates are excluded by default. This avoids
publishing unrelated or unreviewed material and does not license that material
as part of the code release. The data generator remains included.

The full paper demonstration additionally needs the frozen annotation evidence
and input snapshot described in `release_reproduction.md`. A source-only
download must not be described as a deposit of those research artifacts.

## Source Package

```sh
python scripts/build_open_source_release.py
```

This creates a source ZIP and a manifest of SHA256 checksums in `releases/`.
The allowlist and common-credential-pattern checks are useful safeguards, not
a guarantee that no sensitive information exists. Review the included files
before publication. Generated files and old frozen archives are left intact.

## Version and DOI

Record the public release tag and commit used for the paper. Version 0.1.0 is
the current package version; do not claim a release tag exists before creating
it. `CITATION.cff` identifies the software contributors as in the existing
package metadata; confirm named software authors before changing that credit.

For a persistent paper citation, optionally archive the public GitHub release
with Zenodo and add the actual issued DOI to the citation file and manuscript.
Do not invent a DOI or imply that the unpublished paper already has one.

GitHub documents [software citation metadata](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)
and [archiving releases with Zenodo](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content).

## Paper Wording After Publication

Once the source is publicly accessible, the code-availability statement can
identify its repository URL and MIT license, while explicitly distinguishing
code from the separately archived synthetic inputs and evaluation evidence.
Use a verified version URL or DOI for the exact version supporting the paper.
Do not describe a source ZIP prepared locally as already publicly available.
