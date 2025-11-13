# CIFAR-100 WideResNet Training Comparison

**Comparing Two Training Approaches on CIFAR-100**

---

## 📊 Executive Summary

| Metric | Session 02 (Stable) | Session 04 (Aggressive) |
|--------|---------------------|------------------------|
| **Notebook** | `08_ERA_V4_Session-02.ipynb` | `08_ERA_V4_Session-04.ipynb` |
| **Final Test Accuracy** | **71.20%** | **72.98%** |
| **Best Epoch** | Epoch 100 | Epoch 99 |
| **Final Train Accuracy** | 69.26% | 97.85% |
| **Train-Test Gap** | -1.94% ✅ | +24.87% ⚠️ |
| **Generalization** | ✅ Excellent | ⚠️ Overfitting Risk |
| **Training Strategy** | Stable, Consistent | Aggressive, Progressive |
| **Recommendation** | **RECOMMENDED** | Higher Accuracy |
| **HuggingFace Model** | [session8](https://huggingface.co/pandurangpatil/cifar100-wideresnet-session8) | [session04](https://huggingface.co/pandurangpatil/cifar100-wideresnet-session04) |

### ✅ Recommendation: **Session 02**
Despite Session 04 achieving 1.78% higher test accuracy, **Session 02 is recommended for production** due to:
- Excellent generalization (negative train-test gap)
- Lower overfitting risk
- More stable and reliable performance
- Better real-world deployment characteristics

---

## 🔧 Training Configuration Comparison

| Configuration | Session 02 (Stable) | Session 04 (Aggressive) |
|--------------|---------------------|------------------------|
| **Model** | WideResNet-28-10 | WideResNet-28-10 |
| **Parameters** | 36.5M | 36.5M |
| **Batch Size** | 256 | 256 |
| **Total Epochs** | 100 | 100 |
| **Dropout** | 0.3 (constant) | 0.3 → 0.2 → 0.1 (progressive) |
| **MixUp Alpha** | 0.2 (constant) | 0.2 → 0.15 → OFF (progressive) |
| **Label Smoothing** | 0.1 (constant) | 0.1 → 0.05 (progressive) |
| **Augmentation** | Full (constant) | Full → Reduced → Minimal (progressive) |
| **Max LR** | 0.1 | 0.15 (increased) |
| **LR Scheduler** | CosineAnnealingWarmRestarts (T_0=25) | Phase 1: CosineAnnealingWarmRestarts (T_0=20)<br>Phase 2: CosineAnnealingLR |
| **Warmup Epochs** | 5 (0.01 → 0.1) | 5 (0.01 → 0.15) |
| **Weight Decay** | 1e-3 | 1e-3 |
| **Gradient Clipping** | 1.0 | 1.0 |
| **Mixed Precision** | Enabled | Enabled |

---

## 📈 Epoch-by-Epoch Accuracy Comparison

| Epoch | Session 02<br>Test Acc | Session 04<br>Test Acc | Difference | Session 02<br>Events | Session 04<br>Events |
|-------|--------------|--------------|------------|-----------|-----------|
| **1** | 6.64% | 4.15% | +2.49% | - | - |
| **5** | 29.37% | 28.71% | +0.66% | - | - |
| **10** | 47.60% | 51.11% | -3.51% | Checkpoint | Checkpoint |
| **20** | 53.06% | 57.15% | -4.09% | Checkpoint | Checkpoint |
| **25** | 55.27% | 58.95% | -3.68% | Checkpoint (Cycle 1 end) | - |
| **30** | 57.44% | 61.43% | -3.99% | Checkpoint | Checkpoint |
| **40** | 59.54% | 63.88% | -4.34% | Checkpoint | Checkpoint, **Aug: Full → Reduced** |
| **41** | 59.50% | 65.03% | -5.53% | - | **MixUp: 0.2 → 0.15** |
| **50** | 62.94% | 66.27% | -3.33% | Checkpoint | Checkpoint, **Dropout: 0.3 → 0.2, Label Smooth: 0.1 → 0.05** |
| **60** | 64.38% | 69.31% | -4.93% | Checkpoint | Checkpoint, **LR: Warm Restarts → Cosine Decay** |
| **70** | 66.21% | 70.37% | -4.16% | - | Checkpoint |
| **71** | 65.92% | 70.72% | -4.80% | - | **Aug: Reduced → Minimal, MixUp: OFF** |
| **75** | 67.37% | 71.74% | -4.37% | Checkpoint | - |
| **80** | 68.20% | 71.89% | -3.69% | - | Checkpoint |
| **81** | 68.40% | 72.15% | -3.75% | - | **Dropout: 0.2 → 0.1** |
| **90** | 69.53% | 70.00% | -0.47% | Checkpoint | Checkpoint |
| **93** | 70.95% | 71.41% | -0.46% | - | - |
| **99** | 70.69% | **72.98%** | -2.29% | - | **BEST (Session 04)** |
| **100** | **71.20%** | 69.64% | +1.56% | **BEST (Session 02)** | - |

---

## 🎯 Progressive Strategy Transitions (Session 04)

| Epoch | Transition | Details |
|-------|-----------|---------|
| **41** | Augmentation Phase 1 → 2 | Full → Reduced augmentation |
| **41** | MixUp Reduction | Alpha: 0.2 → 0.15 |
| **51** | Dropout Reduction 1 | 0.3 → 0.2 |
| **51** | Label Smoothing Reduction | 0.1 → 0.05 |
| **61** | LR Scheduler Change | CosineAnnealingWarmRestarts → CosineAnnealingLR |
| **71** | Augmentation Phase 2 → 3 | Reduced → Minimal augmentation |
| **71** | MixUp Disabled | Alpha: 0.15 → OFF |
| **81** | Dropout Reduction 2 | 0.2 → 0.1 |

---

## 📊 Final Performance Analysis

### Session 02 (Stable Approach)
```
✅ Final Test Accuracy:    71.20% (epoch 100)
   Final Train Accuracy:   69.26%
   Train-Test Gap:         -1.94% (EXCELLENT - model generalizes well)

   Key Characteristics:
   - Steady, consistent improvement throughout training
   - Negative train-test gap indicates no overfitting
   - Achieved best performance at final epoch
   - Stable configuration without phase transitions
   - Lower variance in performance
```

### Session 04 (Aggressive Approach)
```
⚠️  Final Test Accuracy:    72.98% (epoch 99)
   Final Train Accuracy:   97.85%
   Train-Test Gap:         +24.87% (CRITICAL - severe overfitting)

   Key Characteristics:
   - Higher peak test accuracy (+1.78%)
   - Severe overfitting (train acc 97.85% vs test acc 72.98%)
   - Multiple progressive transitions during training
   - Higher variance due to strategy changes
   - Performance dropped at epoch 100 (69.64%)
```

---

## 📉 Training Curves Summary

### Session 02 Milestones
- **Epoch 10:** 47.60% → First checkpoint
- **Epoch 25:** 55.27% → End of first cosine cycle
- **Epoch 50:** 62.94% → Mid-training
- **Epoch 75:** 67.37% → Late training
- **Epoch 100:** 71.20% → **BEST** (final epoch)

### Session 04 Milestones
- **Epoch 10:** 51.11% → Early lead (+3.51% vs Session 02)
- **Epoch 40:** 63.88% → Before phase transitions
- **Epoch 60:** 69.31% → After scheduler change
- **Epoch 81:** 72.15% → After final dropout reduction
- **Epoch 99:** 72.98% → **BEST** (peaked before final epoch)
- **Epoch 100:** 69.64% → Dropped 3.34%

---

## 🔍 Key Insights

### Why Session 02 is Recommended Despite Lower Accuracy

1. **Generalization Quality**
   - Session 02: Train-Test Gap = -1.94% (✅ Excellent)
   - Session 04: Train-Test Gap = +24.87% (⚠️ Critical)

2. **Stability**
   - Session 02: Consistent configuration throughout
   - Session 04: 8 major configuration changes during training

3. **Real-World Performance**
   - Session 02: More reliable on unseen data
   - Session 04: Likely to perform worse in production despite higher test accuracy

4. **Overfitting Risk**
   - Session 02: Minimal overfitting (negative gap)
   - Session 04: Severe overfitting (97.85% train vs 72.98% test)

5. **Training Reliability**
   - Session 02: Achieved best at epoch 100 (stable)
   - Session 04: Peaked at epoch 99, then dropped

### When to Use Each Approach

**Use Session 02 (Recommended):**
- Production deployments
- When generalization is critical
- Limited validation data available
- Prefer stability over peak performance
- Real-world applications

**Use Session 04:**
- Research purposes
- Understanding progressive training strategies
- When you can afford extensive validation
- Experimental benchmarking
- Can accept higher variance

---

## 📦 Model Artifacts

### Session 02 Checkpoints
Available at: [pandurangpatil/cifar100-wideresnet-session8](https://huggingface.co/pandurangpatil/cifar100-wideresnet-session8)
- `best_model.pth` (Epoch 100, 71.20%)
- `checkpoint_epoch10.pth` (47.60%)
- `checkpoint_epoch20.pth` (53.06%)
- `checkpoint_epoch25.pth` (55.27%)
- `checkpoint_epoch30.pth` (57.44%)
- `checkpoint_epoch40.pth` (59.54%)
- `checkpoint_epoch50.pth` (62.94%)
- `checkpoint_epoch60.pth` (64.38%)
- `checkpoint_epoch75.pth` (67.37%)
- `checkpoint_epoch90.pth` (69.53%)
- `final_model.pth` (71.20%)

### Session 04 Checkpoints
Available at: [pandurangpatil/cifar100-wideresnet-session04](https://huggingface.co/pandurangpatil/cifar100-wideresnet-session04)
- `best_model.pth` (Epoch 99, 72.98%)
- `checkpoint_epoch10.pth` (51.11%)
- `checkpoint_epoch20.pth` (57.15%)
- `checkpoint_epoch30.pth` (61.43%)
- `checkpoint_epoch40.pth` (63.88%)
- `checkpoint_epoch50.pth` (66.27%)
- `checkpoint_epoch60.pth` (69.31%)
- `checkpoint_epoch70.pth` (70.37%)
- `checkpoint_epoch80.pth` (71.89%)
- `checkpoint_epoch90.pth` (70.00%)
- `checkpoint_epoch100.pth` (69.64%)

---

## 🎓 Lessons Learned

### From Session 02 (Stable)
1. **Consistency Works:** Stable configurations can achieve excellent results
2. **Generalization Focus:** Negative train-test gap is a strong indicator of model quality
3. **Patient Training:** Steady improvement over 100 epochs yields reliable models
4. **Cosine Scheduling:** Warm restarts with T_0=25 provided effective learning rate management

### From Session 04 (Aggressive)
1. **Progressive Strategies:** Can achieve higher peak accuracy but with risks
2. **Overfitting Trade-off:** Reducing regularization late in training risks severe overfitting
3. **Scheduler Transitions:** Switching schedulers mid-training can be beneficial but requires careful tuning
4. **Augmentation Phases:** Reducing augmentation progressively increased train accuracy but hurt generalization
5. **Performance Volatility:** Peak performance at epoch 99 with drop at epoch 100 suggests instability

### General Insights
1. **Higher accuracy ≠ Better model:** Session 04's +1.78% comes with +26.81% overfitting gap
2. **Regularization is crucial:** Session 02's consistent regularization prevented overfitting
3. **Simplicity has merit:** Fewer hyperparameter changes = more predictable training
4. **Validation matters:** Train-test gap is as important as test accuracy

---

## 🚀 Reproduction Instructions

### Session 02 (Stable)
```python
# Configuration
dropout = 0.3  # constant
mixup_alpha = 0.2  # constant
label_smoothing = 0.1  # constant
max_lr = 0.1
scheduler = CosineAnnealingWarmRestarts(T_0=25)
augmentation = 'full'  # constant

# Run for 100 epochs
```

### Session 04 (Aggressive)
```python
# Progressive Configuration
# Epochs 1-40:
dropout = 0.3, mixup = 0.2, aug = 'full'

# Epochs 41-50:
mixup = 0.15, aug = 'reduced'

# Epochs 51-60:
dropout = 0.2, label_smoothing = 0.05

# Epochs 61-70:
scheduler = CosineAnnealingLR(T_max=40)

# Epochs 71-80:
mixup = OFF, aug = 'minimal'

# Epochs 81-100:
dropout = 0.1
```

---

## 📚 References

- **Assignment Target:** 73% accuracy from scratch (not achieved by either session)
- **Dataset:** CIFAR-100 (50,000 train, 10,000 test images, 100 classes)
- **Architecture:** WideResNet-28-10 (36.5M parameters)
- **Training Platform:** Google Colab with GPU
- **Framework:** PyTorch with Mixed Precision Training

---

## 🏆 Final Verdict

| Aspect | Winner | Margin |
|--------|--------|--------|
| **Test Accuracy** | Session 04 | +1.78% |
| **Generalization** | Session 02 | +26.81% better gap |
| **Stability** | Session 02 | Fewer config changes |
| **Production Readiness** | Session 02 | ✅ Recommended |
| **Research Value** | Session 04 | Progressive insights |

**Overall Winner for Deployment: Session 02** 🏆

Despite achieving slightly lower test accuracy, Session 02 is the clear winner for production use due to superior generalization, stability, and lower overfitting risk. Session 04 provides valuable insights into progressive training strategies but is not recommended for deployment without addressing the severe overfitting issue.
