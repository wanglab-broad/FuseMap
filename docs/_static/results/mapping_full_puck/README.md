# Reference mapping UMAPs: full Slide-seq Puck60

These figures show all 88,295 MERFISH + STARmap reference cells and all 54,787
Slide-seq query beads. They are not the earlier 512-bead validation subset.

The existing complete query mapping was matched to its actual pretrained reference
by comparing every frozen discriminator parameter and all shared pretrained gene
embeddings. FuseMap 1.2.0 Stage-B was then applied to the full query using fixed
reference signatures: 40 archetypes and 1,471 shared genes. All 20 reference files
retained their original hashes.

UMAP is fitted separately for cell-type and tissue-region embeddings on the joint
reference/query matrix, using all 64 latent dimensions, Euclidean distance,
50 neighbors, min_dist=0.5, and random_state=0. All three coloring panels of an
embedding use the same coordinates and point order. Point order is shuffled with
a fixed seed to avoid always overlaying one source on top of the other.

Reference identities are observed annotations where present and inferred otherwise;
query identities are inferred. Cell identities are learned from labeled reference
cell-type embeddings, and tissue-region identities from labeled reference tissue
embeddings. Missing labels such as `NA` and `nan` are excluded, and query labels
never enter classifier training. The same identities and palette are used in both
embedding spaces. These inferred identities summarize beads; they are not mixture
fractions or independent biological validation of mapping.

Reproduce plots from corresponding saved inputs with:

```bash
python examples/plot_mapping_umaps.py \
  --reference ./output_tutorial1 \
  --query ./output_tutorial4/slideseq_Puck60.h5ad \
  --output ./mapping_umaps
```

The script writes full-resolution PNG/PDF figures, both joint AnnData objects with
UMAP coordinates, palettes, input hashes and classifier diagnostics. The PNG files
in this documentation folder are previews from that run. Figure coordinates and
predicted identities depend on the specific pretrained checkpoint; a newly trained
reference need not reproduce these coordinates exactly.
