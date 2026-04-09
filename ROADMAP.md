# EEG Foundation Model - Development Roadmap

This roadmap outlines the incremental development plan for the EEG foundation model, organized by commit units following the principle of **one logical feature per commit**.

---

## ✅ Phase 1: Core Data & Training Infrastructure (COMPLETED)

| # | Feature | Lines | Status | Branch |
|---|---------|-------|--------|--------|
| 1 | Initial skeleton | - | ✅ | main |
| 2 | Data normalization (min-max, z-score) | 42 | ✅ | main |
| 3 | Data augmentation (noise, dropout, scaling, jitter) | 95 | ✅ | main |
| 4 | Multi-format support (.edf, .fif) | 54 | ✅ | main |
| 5 | LR scheduler (cosine annealing + warmup) | 35 | ✅ | main |
| 6 | Gradient clipping | 25 | ✅ | main |
| 7 | Enhanced checkpointing (best model tracking) | 111 | ✅ | main |

**Total**: 362 lines

---

## 🚀 Phase 2: Advanced Architecture (8-10 commits)

### Encoder Improvements

**Commit 8: Multi-scale Temporal Convolutions**
- **Size**: ~80 lines
- **Files**: `src/encoder/simpleEncoder.py`, `config.py`
- **What**:
  - Add parallel conv branches with different kernel sizes
  - Concatenate multi-scale features
  - Improves temporal pattern capture
- **Test**: Compare reconstruction loss with/without multi-scale

**Commit 9: Attention Pooling for Encoder**
- **Size**: ~60 lines
- **Files**: `src/encoder/simpleEncoder.py`
- **What**:
  - Replace average pooling with learned attention pooling
  - Weight important temporal positions
  - Add attention weights visualization
- **Test**: Verify attention weights sum to 1

**Commit 10: Channel-wise Attention**
- **Size**: ~70 lines
- **Files**: `src/encoder/simpleEncoder.py`, `config.py`
- **What**:
  - Self-attention across EEG channels
  - Learn channel dependencies
  - Configurable number of attention heads
- **Test**: Attention between related channels (e.g., FP1-FP2)

### Embedder Enhancements

**Commit 11: Learnable Positional Embeddings**
- **Size**: ~40 lines
- **Files**: `src/embedding/embedder.py`, `config.py`
- **What**:
  - Replace sinusoidal with learnable embeddings
  - Option to switch between learned/sinusoidal/none
  - Add to config
- **Test**: Compare pretraining loss

**Commit 12: Advanced Masking Strategies**
- **Size**: ~90 lines
- **Files**: `src/embedding/embedder.py`, `config.py`
- **What**:
  - Random span masking (mask consecutive chunks)
  - Block masking (mask patterns)
  - Configurable mask ratio
- **Test**: Verify different masking patterns

**Commit 13: Contrastive Loss Option**
- **Size**: ~100 lines
- **Files**: `src/embedding/embedder.py`, `config.py`
- **What**:
  - Add contrastive learning objective
  - Positive/negative pair generation
  - Temperature-scaled cosine similarity
  - Combine with reconstruction loss
- **Test**: Contrastive + reconstruction loss working

### Decoder Improvements

**Commit 14: Add Relative Positional Encoding**
- **Size**: ~70 lines
- **Files**: `src/decoder/transformer.py`, `config.py`
- **What**:
  - Relative position bias in attention
  - Better for variable-length sequences
  - Configurable max relative distance
- **Test**: Compare with absolute positional encoding

**Commit 15: Sparse Attention Patterns**
- **Size**: ~85 lines
- **Files**: `src/decoder/transformer.py`, `config.py`
- **What**:
  - Local + global attention pattern
  - Reduce memory for long sequences
  - Configurable window size
- **Test**: Memory usage vs standard attention

---

## 📊 Phase 3: Training Optimizations (6-8 commits)

**Commit 16: Mixed Precision Training (AMP)**
- **Size**: ~50 lines
- **Files**: `trainer/trainer.py`, `config.py`
- **What**:
  - Use torch.cuda.amp.autocast
  - GradScaler for stable training
  - Faster training on modern GPUs
- **Test**: Speed comparison fp32 vs fp16

**Commit 17: Gradient Accumulation**
- **Size**: ~45 lines
- **Files**: `trainer/trainer.py`, `config.py`
- **What**:
  - Accumulate gradients over N steps
  - Simulate larger batch sizes
  - Configurable accumulation steps
- **Test**: Effective batch size = batch_size × accum_steps

**Commit 18: Dynamic Batch Sizing**
- **Size**: ~60 lines
- **Files**: `data/dataset.py`, `trainer/trainer.py`, `config.py`
- **What**:
  - Adjust batch size based on sequence length
  - Maintain consistent GPU memory usage
  - Handle variable-length sequences efficiently
- **Test**: Memory usage stays constant

**Commit 19: Warmup Restarts (SGDR)**
- **Size**: ~55 lines
- **Files**: `trainer/trainer.py`, `config.py`
- **What**:
  - Cosine annealing with warm restarts
  - Multiple cycles during training
  - Configurable restart period
- **Test**: LR plot shows restart cycles

**Commit 20: Exponential Moving Average (EMA)**
- **Size**: ~70 lines
- **Files**: `trainer/trainer.py`, `model.py`, `config.py`
- **What**:
  - Maintain EMA of model weights
  - Use EMA for validation/testing
  - Configurable decay rate
- **Test**: EMA model performs better than raw

**Commit 21: Label Smoothing for Classification**
- **Size**: ~35 lines
- **Files**: `src/embedding/embedder.py`, `config.py`
- **What**:
  - Add label smoothing to cross-entropy
  - Prevents overconfidence
  - Configurable smoothing parameter
- **Test**: Calibration curve improvement

---

## 📈 Phase 4: Evaluation & Metrics (5-7 commits)

**Commit 22: Comprehensive Metrics Logging**
- **Size**: ~80 lines
- **Files**: `trainer/trainer.py`, `config.py`
- **What**:
  - Per-epoch metrics (loss, acc, F1, AUC)
  - Save metrics to CSV
  - Plot learning curves
- **Test**: Metrics CSV readable

**Commit 23: TensorBoard Integration**
- **Size**: ~90 lines
- **Files**: `trainer/trainer.py`, `config.py`
- **What**:
  - Log scalars (loss, LR, metrics)
  - Log histograms (weights, gradients)
  - Log sample reconstructions
- **Test**: TensorBoard visualization works

**Commit 24: Validation Visualization**
- **Size**: ~100 lines
- **Files**: `trainer/trainer.py`, `utils/visualization.py` (new)
- **What**:
  - Plot original vs reconstructed EEG
  - Save sample outputs every N epochs
  - Attention map visualization
- **Test**: Generated plots are meaningful

**Commit 25: Model Profiling Tools**
- **Size**: ~60 lines
- **Files**: `utils/profiling.py` (new)
- **What**:
  - FLOPs calculation
  - Memory usage tracking
  - Inference speed benchmarking
- **Test**: Accurate FLOPs count

**Commit 26: Cross-Validation Framework**
- **Size**: ~120 lines
- **Files**: `train.py`, `trainer/trainer.py`, `config.py`
- **What**:
  - K-fold cross-validation
  - Leave-one-subject-out (LOSO)
  - Aggregate metrics across folds
- **Test**: CV results consistent

**Commit 27: Early Stopping with Patience**
- **Size**: ~65 lines
- **Files**: `trainer/trainer.py`, `config.py`
- **What**:
  - Stop training when validation plateaus
  - Configurable patience and delta
  - Restore best weights
- **Test**: Training stops at right time

---

## 🔍 Phase 5: Interpretability & Analysis (6-8 commits)

**Commit 28: Attention Weight Extraction**
- **Size**: ~75 lines
- **Files**: `src/decoder/transformer.py`, `utils/interpretability.py` (new)
- **What**:
  - Save attention weights during forward pass
  - Extract specific layer attentions
  - Return attention maps
- **Test**: Attention shape correctness

**Commit 29: Attention Visualization**
- **Size**: ~90 lines
- **Files**: `utils/interpretability.py`, `utils/visualization.py`
- **What**:
  - Heatmap of attention weights
  - Chunk-to-chunk attention
  - Channel-to-channel attention
- **Test**: Visualizations make sense

**Commit 30: Integrated Gradients**
- **Size**: ~110 lines
- **Files**: `utils/interpretability.py`
- **What**:
  - Implement Integrated Gradients
  - Attribute predictions to input features
  - Channel importance scores
- **Test**: Attributions sum to prediction

**Commit 31: SHAP Value Integration**
- **Size**: ~95 lines
- **Files**: `utils/interpretability.py`, `requirements.txt`
- **What**:
  - Use SHAP library for explanations
  - Feature importance plots
  - Per-sample explanations
- **Test**: SHAP values consistent

**Commit 32: Embedding Space Analysis**
- **Size**: ~85 lines
- **Files**: `utils/analysis.py` (new)
- **What**:
  - Extract embeddings for visualization
  - t-SNE/UMAP projections
  - Cluster analysis
- **Test**: Clusters visible in 2D

**Commit 33: Reconstruction Error Analysis**
- **Size**: ~70 lines
- **Files**: `utils/analysis.py`
- **What**:
  - Per-channel reconstruction error
  - Per-time-window error
  - Anomaly detection based on error
- **Test**: Error patterns meaningful

---

## 🎯 Phase 6: Fine-tuning & Transfer Learning (7-9 commits)

**Commit 34: Decoding Mode Implementation**
- **Size**: ~130 lines
- **Files**: `model.py`, `src/decoder/transformer.py`, `config.py`
- **What**:
  - Add classification head
  - CLS token pooling
  - Switch between pretrain/finetune modes
- **Test**: Classification forward pass works

**Commit 35: Downstream Dataset Classes**
- **Size**: ~100 lines
- **Files**: `data/downstream_dataset.py` (new)
- **What**:
  - Classification dataset
  - Load labels from files
  - Support multiple downstream tasks
- **Test**: Labels loaded correctly

**Commit 36: Fine-tuning Strategies**
- **Size**: ~80 lines
- **Files**: `trainer/finetuning.py` (new), `config.py`
- **What**:
  - Full fine-tuning
  - Linear probing (freeze encoder)
  - Gradual unfreezing
- **Test**: Different strategies work

**Commit 37: Layer-wise Learning Rates**
- **Size**: ~70 lines
- **Files**: `trainer/finetuning.py`, `config.py`
- **What**:
  - Different LR for encoder/decoder
  - Discriminative fine-tuning
  - Configurable LR multipliers
- **Test**: Optimizer parameter groups correct

**Commit 38: Few-shot Learning Support**
- **Size**: ~110 lines
- **Files**: `data/fewshot_dataset.py` (new), `trainer/finetuning.py`
- **What**:
  - N-way K-shot sampling
  - Prototypical networks
  - Meta-learning support
- **Test**: Episode generation works

**Commit 39: Multi-task Learning**
- **Size**: ~120 lines
- **Files**: `model.py`, `trainer/multitask.py` (new), `config.py`
- **What**:
  - Multiple task heads
  - Task-specific loss weighting
  - Shared encoder, separate heads
- **Test**: All tasks train together

**Commit 40: Domain Adaptation**
- **Size**: ~100 lines
- **Files**: `trainer/domain_adaptation.py` (new)
- **What**:
  - Domain adversarial training
  - Gradient reversal layer
  - Adapt to new datasets
- **Test**: Domain classifier confused

---

## ⚡ Phase 7: Efficiency & Scalability (5-7 commits)

**Commit 41: Model Quantization**
- **Size**: ~75 lines
- **Files**: `utils/quantization.py` (new), `model.py`
- **What**:
  - Post-training quantization (INT8)
  - Dynamic quantization
  - Export quantized model
- **Test**: Accuracy vs size tradeoff

**Commit 42: Knowledge Distillation**
- **Size**: ~110 lines
- **Files**: `trainer/distillation.py` (new), `config.py`
- **What**:
  - Teacher-student framework
  - Distillation loss
  - Train smaller student model
- **Test**: Student learns from teacher

**Commit 43: Distributed Training (DDP)**
- **Size**: ~90 lines
- **Files**: `train.py`, `trainer/trainer.py`, `config.py`
- **What**:
  - PyTorch DistributedDataParallel
  - Multi-GPU training
  - Gradient synchronization
- **Test**: Speed up on multiple GPUs

**Commit 44: ONNX Export**
- **Size**: ~60 lines
- **Files**: `utils/export.py` (new)
- **What**:
  - Export model to ONNX format
  - Verify exported model
  - Benchmark inference speed
- **Test**: ONNX model produces same output

**Commit 45: Streaming Inference**
- **Size**: ~85 lines
- **Files**: `utils/streaming.py` (new)
- **What**:
  - Process EEG in real-time chunks
  - Maintain state across chunks
  - Online inference API
- **Test**: Real-time processing works

**Commit 46: Model Pruning**
- **Size**: ~95 lines
- **Files**: `utils/pruning.py` (new)
- **What**:
  - Structured pruning (remove channels)
  - Magnitude-based pruning
  - Fine-tune after pruning
- **Test**: Pruned model maintains accuracy

---

## 🏗️ Phase 8: Production & Deployment (6-8 commits)

**Commit 47: Config Management System**
- **Size**: ~100 lines
- **Files**: `config/manager.py` (new), `configs/` (dir)
- **What**:
  - YAML config files
  - Config inheritance
  - Command-line overrides
- **Test**: Load/save configs correctly

**Commit 48: Logging & Monitoring**
- **Size**: ~80 lines
- **Files**: `utils/logging.py` (new)
- **What**:
  - Structured logging (JSON)
  - Log to file + console
  - Integration with logging frameworks
- **Test**: Logs are readable

**Commit 49: Model Registry**
- **Size**: ~90 lines
- **Files**: `utils/registry.py` (new)
- **What**:
  - Track model versions
  - Metadata storage
  - Model comparison tools
- **Test**: Registry CRUD operations

**Commit 50: REST API Endpoint**
- **Size**: ~120 lines
- **Files**: `api/server.py` (new), `requirements.txt`
- **What**:
  - FastAPI server
  - Prediction endpoint
  - Model loading on startup
- **Test**: API responds correctly

**Commit 51: Docker Containerization**
- **Size**: ~60 lines (Dockerfile + docker-compose)
- **Files**: `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- **What**:
  - Production Docker image
  - GPU support
  - Environment configuration
- **Test**: Container builds and runs

**Commit 52: CI/CD Pipeline**
- **Size**: ~100 lines
- **Files**: `.github/workflows/` (dir)
- **What**:
  - GitHub Actions workflows
  - Automated testing
  - Model training on push
- **Test**: Pipeline runs successfully

**Commit 53: Unit & Integration Tests**
- **Size**: ~200 lines
- **Files**: `tests/` (dir)
- **What**:
  - pytest test suite
  - Test all components
  - Mock data generation
- **Test**: All tests pass

---

## 📚 Phase 9: Documentation & Examples (5-6 commits)

**Commit 54: Comprehensive README**
- **Size**: ~150 lines
- **Files**: `README.md`
- **What**:
  - Installation instructions
  - Quick start guide
  - Usage examples
- **Test**: Follow instructions work

**Commit 55: API Documentation**
- **Size**: ~80 lines
- **Files**: `docs/api.md`, docstrings
- **What**:
  - Sphinx documentation
  - API reference
  - Auto-generated docs
- **Test**: Docs build correctly

**Commit 56: Tutorial Notebooks**
- **Size**: ~300 lines (Jupyter)
- **Files**: `examples/` (dir)
- **What**:
  - Pretraining tutorial
  - Fine-tuning tutorial
  - Interpretability examples
- **Test**: Notebooks run end-to-end

**Commit 57: Pretrained Model Zoo**
- **Size**: ~60 lines + models
- **Files**: `models/README.md`, `utils/download.py`
- **What**:
  - Host pretrained weights
  - Download script
  - Model cards
- **Test**: Models download correctly

**Commit 58: Benchmarking Suite**
- **Size**: ~130 lines
- **Files**: `benchmarks/` (dir)
- **What**:
  - Benchmark on standard datasets
  - Performance comparison
  - Leaderboard results
- **Test**: Benchmarks reproduce

---

## 🔬 Phase 10: Advanced Research Features (8-10 commits)

**Commit 59: Self-supervised Pretraining Variants**
- **Size**: ~140 lines
- **Files**: `src/embedding/` (new files)
- **What**:
  - BYOL (Bootstrap Your Own Latent)
  - SimCLR for EEG
  - MoCo (Momentum Contrast)
- **Test**: Different objectives trainable

**Commit 60: Transformer Architecture Variants**
- **Size**: ~180 lines
- **Files**: `src/decoder/` (new files)
- **What**:
  - Performer (linear attention)
  - Reformer (LSH attention)
  - Linformer
- **Test**: All variants work

**Commit 61: Adversarial Robustness**
- **Size**: ~100 lines
- **Files**: `utils/adversarial.py` (new)
- **What**:
  - FGSM/PGD attacks
  - Adversarial training
  - Certified defenses
- **Test**: Model robust to attacks

**Commit 62: Uncertainty Quantification**
- **Size**: ~110 lines
- **Files**: `utils/uncertainty.py` (new)
- **What**:
  - MC Dropout
  - Deep ensembles
  - Calibration metrics
- **Test**: Uncertainty correlates with errors

**Commit 63: Neural Architecture Search**
- **Size**: ~150 lines
- **Files**: `nas/` (dir)
- **What**:
  - DARTS-style NAS
  - Search encoder architecture
  - Efficient search
- **Test**: Search finds good architectures

**Commit 64: Continual Learning**
- **Size**: ~120 lines
- **Files**: `trainer/continual.py` (new)
- **What**:
  - Elastic Weight Consolidation (EWC)
  - Progressive Neural Networks
  - Avoid catastrophic forgetting
- **Test**: Old tasks retained

**Commit 65: Federated Learning**
- **Size**: ~130 lines
- **Files**: `federated/` (dir)
- **What**:
  - FedAvg implementation
  - Privacy-preserving training
  - Client-server architecture
- **Test**: Multiple clients aggregate

**Commit 66: Graph Neural Network Integration**
- **Size**: ~140 lines
- **Files**: `src/encoder/gnn_encoder.py` (new)
- **What**:
  - Model channel connectivity as graph
  - Graph convolutions on EEG
  - Spatial+temporal modeling
- **Test**: GNN encoder works

---

## 📊 Commit Summary by Phase

| Phase | Commits | Estimated Lines | Focus |
|-------|---------|-----------------|-------|
| 1. Core (✅) | 7 | ~362 | Data & basic training |
| 2. Architecture | 8 | ~595 | Better encoder/decoder |
| 3. Training Opt | 6 | ~315 | Faster, stable training |
| 4. Evaluation | 6 | ~515 | Metrics & monitoring |
| 5. Interpretability | 6 | ~525 | Explainability |
| 6. Fine-tuning | 7 | ~710 | Transfer learning |
| 7. Efficiency | 6 | ~515 | Speed & deployment |
| 8. Production | 7 | ~730 | Real-world ready |
| 9. Documentation | 5 | ~720 | User-friendly |
| 10. Research | 8 | ~1070 | Cutting-edge |
| **TOTAL** | **66** | **~6057** | Complete system |

---

## 🎯 Recommended Order

### Immediate Next (Phase 2)
1. Commit 8-10: Better encoder
2. Commit 11-13: Better embedder
3. Commit 14-15: Better decoder

### Short-term (Phases 3-4)
4. Commit 16-21: Training optimizations
5. Commit 22-27: Metrics & evaluation

### Medium-term (Phases 5-6)
6. Commit 28-33: Interpretability
7. Commit 34-40: Fine-tuning capabilities

### Long-term (Phases 7-9)
8. Commit 41-46: Efficiency
9. Commit 47-53: Production
10. Commit 54-58: Documentation

### Research (Phase 10)
11. Commit 59-66: Advanced features

---

## 📏 Commit Size Guidelines

- **Tiny** (15-50 lines): Config additions, small utils
- **Small** (50-100 lines): Single feature, focused change
- **Medium** (100-150 lines): Multiple related changes
- **Large** (150+ lines): Major new capability

**Target**: Most commits should be **small to medium** (50-150 lines).

---

## 🔄 Development Workflow

For each commit:

1. **Create feature branch**: `git checkout -b feat/feature-name`
2. **Implement feature**: Follow spec above
3. **Write tests**: Verify functionality
4. **Test manually**: Run quick validation
5. **Commit**: Clear message with description
6. **Merge to main**: Fast-forward merge
7. **Push**: `git push origin main`
8. **Clean up**: Delete feature branch

---

## 📝 Notes

- This roadmap is **flexible** - adjust based on needs
- Some commits may be **combined** or **split** as needed
- **Prioritize** based on your specific use case
- Each phase can be developed **independently**
- Encourage **contributions** for advanced features

---

**Last Updated**: 2026-04-09
**Status**: Phase 1 Complete (7/66 commits)
**Next Milestone**: Phase 2 - Advanced Architecture
