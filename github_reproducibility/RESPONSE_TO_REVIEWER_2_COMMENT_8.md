# Response to Reviewer 2, Comment 8

We thank the reviewer for emphasizing the importance of computational
reproducibility. We agree that the source code for molecular standardization,
descriptor calculation, conformer generation, similarity search and candidate
prioritization should be available.

We have therefore prepared a versioned GitHub reproducibility package and will
release it with the revised manuscript. The package contains portable Python
implementations and command-line entry points for all five workflow components,
the monomer/sequence representation rules used for standardization, explicit
RDKit fingerprint parameters, example inputs, and smoke tests. The default
structure workflow uses RDKit ETKDGv3 embedding followed by MMFF94 optimization,
with UFF as a documented fallback. The similarity workflow records RDKit
MorganGenerator settings of radius 2, 2,048-bit binary fingerprints, default
connectivity and bond-type invariants, `useFeatures=False`, and an explicit
`includeChirality` switch. Candidate prioritization is implemented as a stable
descending similarity ranking followed by structure-confidence score,
evidence score and identifier.

The repository is intended to reproduce the transformations and parameter
choices, rather than to redistribute third-party database contents. Raw records
from dbAMP, DRAMP, DBAASP, SATPdb and APD remain governed by their original
licences and are therefore not bundled. Amber/tleap and xTB routes used for
selected downstream structures are optional external dependencies; the default
open workflow does not require them.

The package will be deposited in a public GitHub repository with a release tag
matching the manuscript version. The exact repository URL will be added to the
revised Code Availability statement before submission. That statement will
also document the RDKit version used for the reported analyses (2024.03.2).
