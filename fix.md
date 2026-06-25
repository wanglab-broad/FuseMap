Check https://github.com/wanglab-broad/FuseMap/blob/main/fusemap/loss.py#L975 (fusemap/loss.py#L975), right now disease dataset will gets pushed toward the normal atlas distribution by the adversarial loss indiscriminately, including niches that don't exist in the normal brain at all. We need the warm-start fine-tuning pipeline preserve disease-specific structure and enable the cross-disease comparisons.

---



Based on my analysis of the FuseMap code at line 975 and surrounding functions, I can confirm your concern about the adversarial loss implementation. Here's a breakdown of the issue and proposed solutions:

## Current Issue Analysis

### The Problem (Lines 975-1085: `compute_dis_loss_map`)

The adversarial loss in `compute_dis_loss_map` and `compute_ae_loss_map` applies **indiscriminate alignment** of all disease data toward the normal atlas distribution:

```python
# Key problematic pattern (lines 1041-1059):
loss_dis_single = F.cross_entropy(
    F.softmax(
        torch.hstack([
            adapt_model.discriminator_single(z_mean_cat_single),  # All disease samples
            adapt_model.discriminator_single_pretrain(z_mean_cat_single),
        ]),
        dim=1,
    ),
    torch.hstack([
        flag_source_cat_single[mask_batch_single_all],  # All treated as "source" to align
        flag_source_cat_single_pretrain,
    ]),
    reduction="none",
)
```

**Critical flaws:**

1. **No disease-specific masking**: All disease samples (including novel pathological niches) receive the same adversarial pressure to match the normal atlas
2. **Binary discriminator objective**: The discriminator only learns to distinguish "atlas vs. query", not "normal-compatible vs. disease-specific"
3. **No gradient gating**: Disease-unique features are actively suppressed during fine-tuning, destroying cross-disease comparability

## Proposed Solutions for Warm-Start Fine-Tuning

### Option 1: Disease-Aware Adversarial Masking

Add a mechanism to identify and protect disease-specific niches:

```python
# Pseudocode for modified compute_ae_loss_map
def compute_ae_loss_map_disease_aware(...):
    # ... existing code ...
  
    # Step 1: Identify disease-specific cells via density estimation
    # Cells with low density in normal atlas = potential disease niches
    normal_density = estimate_density_in_atlas(z_mean_cat_single, atlas_embeddings)
    disease_specific_mask = normal_density < disease_niche_threshold
  
    # Step 2: Apply adversarial loss ONLY to atlas-compatible cells
    loss_dis_single = F.cross_entropy(
        F.softmax(adapt_model.discriminator_single(z_mean_cat_single[~disease_specific_mask]), dim=1),
        flag_source_cat_single[mask_batch_single_all][~disease_specific_mask],
        reduction="none",
    )
  
    # Step 3: Preserve disease-specific structure via reconstruction-only loss
    if disease_specific_mask.any():
        loss_disease_preserve = F.mse_loss(
            z_mean_cat_single[disease_specific_mask],
            z_mean_cat_single[disease_specific_mask].detach()  # Stop-gradient: preserve original structure
        )
        loss_dis = flagconfig.lambda_disc_single * loss_dis_single + \
                   flagconfig.lambda_disease_preserve * loss_disease_preserve
```

### Option 2: Multi-Task Discriminator with Disease-Type Awareness

Modify the discriminator to output three classes instead of two:

- Class 0: Normal atlas cells
- Class 1: Disease cells with normal counterparts (align these)
- Class 2: Disease-specific niches (preserve these)

```python
# Modified discriminator forward pass
def discriminator_forward_with_niche_detection(z, disease_metadata=None):
    # Existing domain discrimination
    domain_logits = discriminator_head(z)
  
    # Optional: niche detection head (trained with weak supervision)
    if disease_metadata is not None:
        niche_logits = niche_detection_head(z, disease_metadata)
        return torch.cat([domain_logits, niche_logits], dim=1)
    return domain_logits
```

### Option 3: Gradient Reversal with Adaptive Weighting

Use a dynamic λ that reduces adversarial pressure for cells dissimilar to the atlas:

```python
# Adaptive lambda computation (replace static flagconfig.lambda_disc_single)
def compute_adaptive_lambda(z_query, z_atlas, base_lambda, similarity_threshold=0.3):
    # Compute cosine similarity to nearest atlas neighbors
    similarities = F.cosine_similarity(z_query.unsqueeze(1), z_atlas.unsqueeze(0), dim=-1).max(dim=1)[0]
  
    # Reduce adversarial weight for low-similarity (disease-specific) cells
    adaptive_weights = torch.where(
        similarities < similarity_threshold,
        torch.tensor(0.1),  # Minimal alignment pressure
        torch.tensor(1.0)   # Full alignment pressure
    )
    return base_lambda * adaptive_weights.mean()
```

## Implementation Recommendations for Cross-Disease Comparison

1. **Add disease-type metadata tracking**: Ensure `flag_source_cat_single` encodes disease type, not just "query vs atlas"
2. **Implement niche preservation loss**: Add a contrastive term that maintains distances between disease-specific clusters:

   ```python
   # Preserve disease-specific structure
   if enable_disease_preservation:
       disease_clusters = cluster_disease_cells(z_mean_cat_single, disease_labels)
       loss_preserve = contrastive_preservation_loss(disease_clusters, margin=1.0)
       total_loss = -loss_dis + sum(loss_AE_all) + flagconfig.lambda_preserve * loss_preserve
   ```
3. **Two-stage fine-tuning protocol**:

   - **Stage 1** (warm-start): Freeze discriminator, train encoder/decoder with reconstruction loss only → preserves disease structure
   - **Stage 2** (selective alignment): Unfreeze discriminator but apply adversarial loss only to cells with atlas similarity > threshold
4. **Evaluation metric**: Add a "disease structure preservation score" to monitor during training:

   ```python
   def compute_structure_preservation_score(z_before, z_after, disease_labels):
       # Measure intra-disease cluster cohesion before/after fine-tuning
       cohesion_before = silhouette_score(z_before, disease_labels)
       cohesion_after = silhouette_score(z_after, disease_labels)
       return cohesion_after / (cohesion_before + 1e-8)  # >1.0 = improved preservation
   ```

## Quick Fix for Immediate Testing

If you need a minimal change to test the hypothesis:

```python
# In compute_ae_loss_map, around line 1182-1223:
# Add this before computing loss_dis_single:

# Compute similarity to atlas for adaptive masking
atlas_z = torch.cat([z_all[i][3] for i in range(ModelType.n_atlas)], dim=0)  # Reference atlas embeddings
query_z = torch.cat([z_all[i][3][mask_batch_single[i]] for i in range(ModelType.n_atlas)], dim=0)

# Simple cosine similarity thresholding
similarities = F.cosine_similarity(query_z.unsqueeze(1), atlas_z.unsqueeze(0), dim=-1).max(dim=1)[0]
align_mask = similarities > 0.4  # Only align cells with reasonable atlas similarity

# Apply mask to adversarial loss computation
if align_mask.any():
    loss_dis_single = F.cross_entropy(
        F.softmax(adapt_model.discriminator_single(query_z[align_mask]), dim=1),
        flag_source_cat_single[mask_batch_single_all][align_mask],
        reduction="mean",  # Changed from "none" for stability
    )
else:
    loss_dis_single = torch.tensor(0.0, device=query_z.device)  # No alignment pressure if all disease-specific
```

This preserves disease-specific niches while still enabling atlas alignment for comparable cell states. Would you like me to help draft a PR with one of these approaches, or would you prefer to explore the similarity thresholding approach first?
