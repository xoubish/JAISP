## What’s Happening (Simple but Technical)

### JEPA: Joint-Embedding Predictive Architecture

We use a **Joint-Embedding Predictive Architecture (JEPA)**, a self-supervised learning framework introduced by Yann LeCun. Instead of predicting raw pixels (which is computationally expensive and noisy), the model learns to predict **representations in an abstract embedding space**.

In our setup, JEPA operates on **two views of the same sky region**:

- **Rubin**: ground-based, 6 bands, 0.2"/pixel  
- **Euclid**: space-based, 4 bands, 0.1"/pixel  

Because both instruments image the same underlying sky, their representations should be similar. The model learns this **without labels**, relying purely on the statistical structure of the data.

The predictive aspect is implicit: given a Rubin patch, the model learns what the corresponding Euclid embedding should look like (and vice versa). Using an **NT-Xent contrastive loss**, the training:

- pulls matched Rubin–Euclid pairs together in embedding space
- pushes unmatched pairs apart

This encourages the network to learn **instrument-agnostic features of the sky**, such as:

- galaxy morphologies  
- spatial clustering patterns  
- stellar populations  

These features are robust to PSF differences, pixel scale mismatches, and bandpass variations.

---

### Foundation Model: Transferable Representations

This stage acts as a **foundation model**, learning general-purpose representations that are reused across multiple downstream tasks.

The embedding can be thought of as a **universal language for astronomical scenes**. It captures *what is in a patch* in a way that is useful for:

- **Stage 2**: astrometric alignment  
- **Stage 3**: source deblending  
- **Stage 4**: photometry  

Rather than training separate models for each task, we:

1. pretrain a single representation model (expensive)
2. freeze the learned embeddings
3. attach small task-specific heads on top

This follows the core philosophy of foundation models: **learn once, apply broadly**. While pretraining is costly (≈100 epochs × 144 tiles), downstream tasks train in **minutes instead of days**.

---

### Vision Transformer (ViT): Why Not CNNs?

We use **Vision Transformers (ViTs)** instead of traditional CNNs for several key reasons.

ViTs divide images into fixed-size patches (16×16 pixels) and treat them as a sequence of tokens processed via self-attention. This naturally provides:

- **Global and local context**  
  Self-attention captures both fine-scale structure (individual galaxies) and large-scale patterns (clustering, background gradients). CNNs remain local until very deep layers.

- **Variable resolution handling**  
  Rubin patches (192×192) and Euclid patches (384×384) cover the same physical area. ViTs handle this cleanly:
  - 192×192 → 144 tokens  
  - 384×384 → 576 tokens  
  The architecture remains unchanged.

- **Explicit positional awareness**  
  Position embeddings encode spatial information directly, which is critical in astronomy where *where* a source appears affects systematics.

- **Scalability**  
  Transformers scale better with data and compute. As the dataset grows toward survey scale, performance improves without architectural changes.

---

### Model Capacity and Representation

We use a **6-layer, 6-head ViT** with **384-dimensional embeddings**, balancing expressive power and overfitting risk on ~144 tiles.

Different attention heads naturally specialize in different structures, such as:

- bright galaxy cores  
- extended diffuse halos  
- spatial clustering  

The **CLS token** aggregates this information into a single 384-D vector summarizing *what this patch looks like*. A projection head then maps this to a **256-D embedding space**, where Rubin and Euclid representations can be directly compared.

---

### Why This Approach?

Traditional pipelines process Rubin and Euclid images independently, then struggle to reconcile them due to fundamental differences in PSF, depth, and resolution.

Here, the mapping between surveys is learned **inside the representation itself**. As a result:

- **Astrometric alignment** operates on features that already correspond across instruments  
- **Source deblending** can use Euclid’s high-resolution structure as a prior while understanding how it manifests in Rubin’s lower-resolution data  

The foundation model effectively serves as a **Rosetta Stone** for cross-survey analysis, making downstream processing coherent rather than brittle.
