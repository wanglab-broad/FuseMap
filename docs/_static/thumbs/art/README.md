# Tutorial gallery artwork

The 11 gallery covers use a small number of recognizable biological cells, clear conceptual relationships, and a consistent teal/coral/lilac palette. Featured cells retain membrane contours, cytoplasm, nuclei with nucleoli, and sparse organelles. Each cover communicates one tutorial topic. They are conceptual illustrations, not measured scientific results.

Generated on 2026-09-13 through the Harvard HUIT Responses API with `gpt-6-astra` (medium reasoning), calling the `gpt-image-2` image-generation tool at medium quality. The illustrations emphasize basic cellular structures and uncluttered compositions.

Each master was generated at 1536 × 1024 pixels, visually reviewed, then resized with Pillow Lanczos to a 900 × 600 WebP (quality 90, method 6). `manifest.json` records the tutorial mapping, concepts, submitted and tool-revised prompts, and web asset SHA-256 hashes.

`docs/conf.py` explicitly assigns each cover to its notebook. Card titles provide accessible link names; the accompanying images are decorative. Scientific result figures remain in the notebooks. The `-cellular` filenames distinguish this revision from the previous covers in browser caches.

The previous unversioned filenames are retained as byte-identical copies of the current illustrations. Cached gallery HTML can continue to request these paths after deployment. Keep previously published image paths available when replacing gallery artwork.
