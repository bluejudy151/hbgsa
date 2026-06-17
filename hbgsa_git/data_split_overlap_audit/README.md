# Data Split Overlap Audit

This folder contains PDBID lists, overlap audit tables, and an overlap-controlled CSAR-HiQ subset evaluation.

## Key Counts

- CSAR protocol source: MBP csar_2016: C:\smiles\zhenshi\学习\MBP-main\MBP-main\MBP\data\csar_2016
- Train PDBIDs: 11906
- Validation PDBIDs: 1000
- PDBbind Core test PDBIDs: 290
- CSAR-HiQ protocol PDBIDs: 135
- CSAR-HiQ after removing train/validation/Core PDBID overlaps: 51
- Overlap-controlled CSAR-HiQ with saved predictions: 47

## Main PDBID Overlaps

- Train vs validation: 0
- Train vs Core test: 0
- Validation vs Core test: 0
- Train vs CSAR-HiQ protocol: 53
- Validation vs CSAR-HiQ protocol: 0
- Core test vs CSAR-HiQ protocol: 31

## Notes

- The 135-complex CSAR-HiQ list is retained as a protocol-matched benchmark for literature comparability.
- The overlap-controlled CSAR-HiQ subset removes all PDBIDs appearing in train, validation, or PDBbind Core test.
- Four overlap-controlled CSAR complexes are not present in the saved prediction CSV because their ligand files were unavailable during preprocessing.
