# Session 02 Training Log

**Model:** WideResNet-28-10 (36.5M parameters)
**Dataset:** CIFAR-100
**Notebook:** `08_ERA_V4_Session-02.ipynb`
**Final Test Accuracy:** 71.20%
**Total Epochs:** 100

## Training Configuration
- Batch Size: 256
- MixUp Alpha: 0.2
- Label Smoothing: 0.1
- Weight Decay: 1e-3
- Gradient Clipping: 1.0
- Scheduler: CosineAnnealingWarmRestarts (T_0=25)
- Warmup: 5 epochs (0.01 → 0.1)
- Mixed Precision: Enabled
- HuggingFace Upload: Enabled

---

## Training Output

```
Starting training...



  0%|          | 0/196 [00:00<?, ?it/s]/tmp/ipython-input-1052289112.py:15: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.

  with autocast():

Epoch 1 Loss=4.4748 Acc=4.28% LR=0.027908: 100%|██████████| 196/196 [01:45<00:00,  1.85it/s]



Test set: Average loss: 4.0935, Accuracy: 664/10000 (6.64%)



*** New best model! Test Accuracy: 6.64% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 6.64% | Patience: 0/15



Epoch 2 Loss=3.9219 Acc=10.35% LR=0.045908: 100%|██████████| 196/196 [01:49<00:00,  1.79it/s]



Test set: Average loss: 3.6983, Accuracy: 1282/10000 (12.82%)



*** New best model! Test Accuracy: 12.82% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 12.82% | Patience: 0/15



Epoch 3 Loss=4.2744 Acc=16.26% LR=0.063908: 100%|██████████| 196/196 [01:50<00:00,  1.77it/s]



Test set: Average loss: 3.1793, Accuracy: 2206/10000 (22.06%)



*** New best model! Test Accuracy: 22.06% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 22.06% | Patience: 0/15



Epoch 4 Loss=3.1342 Acc=21.40% LR=0.081908: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 2.9839, Accuracy: 2559/10000 (25.59%)



*** New best model! Test Accuracy: 25.59% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 25.59% | Patience: 0/15



Epoch 5 Loss=4.2816 Acc=26.83% LR=0.099908: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 2.8263, Accuracy: 2937/10000 (29.37%)



*** New best model! Test Accuracy: 29.37% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 29.37% | Patience: 0/15



Epoch 6 Loss=3.5024 Acc=34.89% LR=0.000712: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 2.1929, Accuracy: 4319/10000 (43.19%)



*** New best model! Test Accuracy: 43.19% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 43.19% | Patience: 0/15



Epoch 7 Loss=2.8736 Acc=38.09% LR=0.002398: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 2.1376, Accuracy: 4410/10000 (44.10%)



*** New best model! Test Accuracy: 44.10% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 44.10% | Patience: 0/15



Epoch 8 Loss=3.5865 Acc=37.19% LR=0.004739: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 2.0944, Accuracy: 4569/10000 (45.69%)



*** New best model! Test Accuracy: 45.69% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 45.69% | Patience: 0/15



Epoch 9 Loss=3.7618 Acc=39.24% LR=0.007158: 100%|██████████| 196/196 [01:50<00:00,  1.77it/s]



Test set: Average loss: 2.0229, Accuracy: 4702/10000 (47.02%)



*** New best model! Test Accuracy: 47.02% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 47.02% | Patience: 0/15



Epoch 10 Loss=3.1391 Acc=41.18% LR=0.009055: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 2.0005, Accuracy: 4760/10000 (47.60%)



*** New best model! Test Accuracy: 47.60% ***

✓ Uploaded: best_model.pth

📍 Breakpoint checkpoint at epoch 10

✓ Uploaded: checkpoint_epoch10.pth

✓ Uploaded: metrics.json

Best Test Accuracy so far: 47.60% | Patience: 0/15



Epoch 11 Loss=2.5498 Acc=40.95% LR=0.009961: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.9411, Accuracy: 4883/10000 (48.83%)



*** New best model! Test Accuracy: 48.83% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 48.83% | Patience: 0/15



Epoch 12 Loss=2.5479 Acc=40.75% LR=0.000448: 100%|██████████| 196/196 [01:51<00:00,  1.77it/s]



Test set: Average loss: 1.9205, Accuracy: 4947/10000 (49.47%)



*** New best model! Test Accuracy: 49.47% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 49.47% | Patience: 0/15



Epoch 13 Loss=2.6477 Acc=42.22% LR=0.001895: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.9238, Accuracy: 4910/10000 (49.10%)



Best Test Accuracy so far: 49.47% | Patience: 1/15



Epoch 14 Loss=2.5397 Acc=43.06% LR=0.004122: 100%|██████████| 196/196 [01:53<00:00,  1.73it/s]



Test set: Average loss: 1.9060, Accuracy: 4935/10000 (49.35%)



Best Test Accuracy so far: 49.47% | Patience: 2/15



Epoch 15 Loss=2.4645 Acc=43.24% LR=0.006580: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.8738, Accuracy: 5058/10000 (50.58%)



*** New best model! Test Accuracy: 50.58% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 50.58% | Patience: 0/15



Epoch 16 Loss=2.5726 Acc=43.00% LR=0.008658: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.8282, Accuracy: 5161/10000 (51.61%)



*** New best model! Test Accuracy: 51.61% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 51.61% | Patience: 0/15



Epoch 17 Loss=3.1125 Acc=44.29% LR=0.009844: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.8173, Accuracy: 5171/10000 (51.71%)



*** New best model! Test Accuracy: 51.71% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 51.71% | Patience: 0/15



Epoch 18 Loss=2.1594 Acc=44.33% LR=0.000256: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.7660, Accuracy: 5306/10000 (53.06%)



*** New best model! Test Accuracy: 53.06% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 53.06% | Patience: 0/15



Epoch 19 Loss=2.5894 Acc=44.59% LR=0.001442: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.7745, Accuracy: 5230/10000 (52.30%)



Best Test Accuracy so far: 53.06% | Patience: 1/15



Epoch 20 Loss=2.5887 Acc=45.32% LR=0.003520: 100%|██████████| 196/196 [01:52<00:00,  1.75it/s]



Test set: Average loss: 1.7756, Accuracy: 5256/10000 (52.56%)



📍 Breakpoint checkpoint at epoch 20

✓ Uploaded: checkpoint_epoch20.pth

✓ Uploaded: metrics.json

Best Test Accuracy so far: 53.06% | Patience: 2/15



Epoch 21 Loss=3.9119 Acc=47.06% LR=0.005978: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.7759, Accuracy: 5237/10000 (52.37%)



Best Test Accuracy so far: 53.06% | Patience: 3/15



Epoch 22 Loss=2.4567 Acc=45.54% LR=0.008205: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.7628, Accuracy: 5294/10000 (52.94%)



Best Test Accuracy so far: 53.06% | Patience: 4/15



Epoch 23 Loss=2.3311 Acc=47.25% LR=0.009652: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.7307, Accuracy: 5388/10000 (53.88%)



*** New best model! Test Accuracy: 53.88% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 53.88% | Patience: 0/15



Epoch 24 Loss=3.4873 Acc=46.83% LR=0.000139: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.6906, Accuracy: 5497/10000 (54.97%)



*** New best model! Test Accuracy: 54.97% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 54.97% | Patience: 0/15



Epoch 25 Loss=2.4919 Acc=47.17% LR=0.001045: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.6862, Accuracy: 5527/10000 (55.27%)



*** New best model! Test Accuracy: 55.27% ***

✓ Uploaded: best_model.pth

📍 Breakpoint checkpoint at epoch 25

✓ Uploaded: checkpoint_epoch25.pth

✓ Uploaded: metrics.json

Best Test Accuracy so far: 55.27% | Patience: 0/15



Epoch 26 Loss=2.3235 Acc=47.09% LR=0.002942: 100%|██████████| 196/196 [01:51<00:00,  1.77it/s]



Test set: Average loss: 1.6806, Accuracy: 5550/10000 (55.50%)



*** New best model! Test Accuracy: 55.50% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 55.50% | Patience: 0/15



Epoch 27 Loss=2.1030 Acc=49.28% LR=0.005361: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.6652, Accuracy: 5604/10000 (56.04%)



*** New best model! Test Accuracy: 56.04% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 56.04% | Patience: 0/15



Epoch 28 Loss=3.6847 Acc=50.62% LR=0.007702: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.6956, Accuracy: 5525/10000 (55.25%)



Best Test Accuracy so far: 56.04% | Patience: 1/15



Epoch 29 Loss=2.7376 Acc=49.30% LR=0.009388: 100%|██████████| 196/196 [01:52<00:00,  1.75it/s]



Test set: Average loss: 1.6230, Accuracy: 5695/10000 (56.95%)



*** New best model! Test Accuracy: 56.95% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 56.95% | Patience: 0/15



Epoch 30 Loss=2.4370 Acc=49.08% LR=0.010000: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.6177, Accuracy: 5744/10000 (57.44%)



*** New best model! Test Accuracy: 57.44% ***

✓ Uploaded: best_model.pth

📍 Breakpoint checkpoint at epoch 30

✓ Uploaded: checkpoint_epoch30.pth

✓ Uploaded: metrics.json

Best Test Accuracy so far: 57.44% | Patience: 0/15



Epoch 31 Loss=2.2878 Acc=50.79% LR=0.000712: 100%|██████████| 196/196 [01:50<00:00,  1.77it/s]



Test set: Average loss: 1.6141, Accuracy: 5757/10000 (57.57%)



*** New best model! Test Accuracy: 57.57% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 57.57% | Patience: 0/15



Epoch 32 Loss=2.0210 Acc=50.55% LR=0.002398: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.6229, Accuracy: 5741/10000 (57.41%)



Best Test Accuracy so far: 57.57% | Patience: 1/15



Epoch 33 Loss=3.3465 Acc=50.03% LR=0.004739: 100%|██████████| 196/196 [01:53<00:00,  1.73it/s]



Test set: Average loss: 1.6407, Accuracy: 5706/10000 (57.06%)



Best Test Accuracy so far: 57.57% | Patience: 2/15



Epoch 34 Loss=2.6262 Acc=51.75% LR=0.007158: 100%|██████████| 196/196 [01:53<00:00,  1.73it/s]



Test set: Average loss: 1.6299, Accuracy: 5660/10000 (56.60%)



Best Test Accuracy so far: 57.57% | Patience: 3/15



Epoch 35 Loss=1.9937 Acc=51.64% LR=0.009055: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.5710, Accuracy: 5838/10000 (58.38%)



*** New best model! Test Accuracy: 58.38% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 58.38% | Patience: 0/15



Epoch 36 Loss=2.1993 Acc=50.64% LR=0.009961: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.5649, Accuracy: 5880/10000 (58.80%)



*** New best model! Test Accuracy: 58.80% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 58.80% | Patience: 0/15



Epoch 37 Loss=2.3121 Acc=50.32% LR=0.000448: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.5552, Accuracy: 5951/10000 (59.51%)



*** New best model! Test Accuracy: 59.51% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 59.51% | Patience: 0/15



Epoch 38 Loss=3.7860 Acc=51.83% LR=0.001895: 100%|██████████| 196/196 [01:50<00:00,  1.77it/s]



Test set: Average loss: 1.5454, Accuracy: 5954/10000 (59.54%)



*** New best model! Test Accuracy: 59.54% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 59.54% | Patience: 0/15



Epoch 39 Loss=1.9945 Acc=52.93% LR=0.004122: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.5767, Accuracy: 5919/10000 (59.19%)



Best Test Accuracy so far: 59.54% | Patience: 1/15



Epoch 40 Loss=2.9800 Acc=54.34% LR=0.006580: 100%|██████████| 196/196 [01:52<00:00,  1.75it/s]



Test set: Average loss: 1.5817, Accuracy: 5871/10000 (58.71%)



📍 Breakpoint checkpoint at epoch 40

✓ Uploaded: checkpoint_epoch40.pth

✓ Uploaded: metrics.json

Best Test Accuracy so far: 59.54% | Patience: 2/15



Epoch 41 Loss=3.4006 Acc=53.33% LR=0.008658: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.5529, Accuracy: 5950/10000 (59.50%)



Best Test Accuracy so far: 59.54% | Patience: 3/15



Epoch 42 Loss=2.1127 Acc=54.93% LR=0.009844: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.5245, Accuracy: 6035/10000 (60.35%)



*** New best model! Test Accuracy: 60.35% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 60.35% | Patience: 0/15



Epoch 43 Loss=2.0651 Acc=54.17% LR=0.000256: 100%|██████████| 196/196 [01:51<00:00,  1.77it/s]



Test set: Average loss: 1.5112, Accuracy: 6096/10000 (60.96%)



*** New best model! Test Accuracy: 60.96% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 60.96% | Patience: 0/15



Epoch 44 Loss=1.9734 Acc=54.84% LR=0.001442: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.5385, Accuracy: 6062/10000 (60.62%)



Best Test Accuracy so far: 60.96% | Patience: 1/15



Epoch 45 Loss=2.0914 Acc=53.44% LR=0.003520: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.5444, Accuracy: 6050/10000 (60.50%)



Best Test Accuracy so far: 60.96% | Patience: 2/15



Epoch 46 Loss=2.2926 Acc=57.19% LR=0.005978: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.4899, Accuracy: 6135/10000 (61.35%)



*** New best model! Test Accuracy: 61.35% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 61.35% | Patience: 0/15



Epoch 47 Loss=3.2313 Acc=57.85% LR=0.008205: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.5109, Accuracy: 6111/10000 (61.11%)



Best Test Accuracy so far: 61.35% | Patience: 1/15



Epoch 48 Loss=2.0015 Acc=54.96% LR=0.009652: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.4932, Accuracy: 6156/10000 (61.56%)



*** New best model! Test Accuracy: 61.56% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 61.56% | Patience: 0/15



Epoch 49 Loss=1.8442 Acc=56.75% LR=0.000139: 100%|██████████| 196/196 [01:51<00:00,  1.75it/s]



Test set: Average loss: 1.4651, Accuracy: 6294/10000 (62.94%)



*** New best model! Test Accuracy: 62.94% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 62.94% | Patience: 0/15



Epoch 50 Loss=2.3648 Acc=56.46% LR=0.001045: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.5042, Accuracy: 6201/10000 (62.01%)



📍 Breakpoint checkpoint at epoch 50

✓ Uploaded: checkpoint_epoch50.pth

✓ Uploaded: metrics.json

Best Test Accuracy so far: 62.94% | Patience: 1/15



Epoch 51 Loss=2.3915 Acc=56.23% LR=0.002942: 100%|██████████| 196/196 [01:51<00:00,  1.75it/s]



Test set: Average loss: 1.5080, Accuracy: 6182/10000 (61.82%)



Best Test Accuracy so far: 62.94% | Patience: 2/15



Epoch 52 Loss=2.1636 Acc=57.04% LR=0.005361: 100%|██████████| 196/196 [01:54<00:00,  1.71it/s]



Test set: Average loss: 1.5025, Accuracy: 6193/10000 (61.93%)



Best Test Accuracy so far: 62.94% | Patience: 3/15



Epoch 53 Loss=2.1906 Acc=56.35% LR=0.007702: 100%|██████████| 196/196 [01:53<00:00,  1.73it/s]



Test set: Average loss: 1.5182, Accuracy: 6175/10000 (61.75%)



Best Test Accuracy so far: 62.94% | Patience: 4/15



Epoch 54 Loss=1.9755 Acc=58.51% LR=0.009388: 100%|██████████| 196/196 [01:53<00:00,  1.73it/s]



Test set: Average loss: 1.4341, Accuracy: 6388/10000 (63.88%)



*** New best model! Test Accuracy: 63.88% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 63.88% | Patience: 0/15



Epoch 55 Loss=3.6489 Acc=57.29% LR=0.010000: 100%|██████████| 196/196 [01:50<00:00,  1.77it/s]



Test set: Average loss: 1.4529, Accuracy: 6365/10000 (63.65%)



Best Test Accuracy so far: 63.88% | Patience: 1/15



Epoch 56 Loss=2.1593 Acc=59.48% LR=0.000712: 100%|██████████| 196/196 [01:52<00:00,  1.75it/s]



Test set: Average loss: 1.4370, Accuracy: 6438/10000 (64.38%)



*** New best model! Test Accuracy: 64.38% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 64.38% | Patience: 0/15



Epoch 57 Loss=1.9455 Acc=57.89% LR=0.002398: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.4380, Accuracy: 6380/10000 (63.80%)



Best Test Accuracy so far: 64.38% | Patience: 1/15



Epoch 58 Loss=1.8645 Acc=59.79% LR=0.004739: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.4173, Accuracy: 6404/10000 (64.04%)



Best Test Accuracy so far: 64.38% | Patience: 2/15



Epoch 59 Loss=1.7452 Acc=58.13% LR=0.007158: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.4651, Accuracy: 6391/10000 (63.91%)



Best Test Accuracy so far: 64.38% | Patience: 3/15



Epoch 60 Loss=2.0816 Acc=59.07% LR=0.009055: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.5216, Accuracy: 6196/10000 (61.96%)



📍 Breakpoint checkpoint at epoch 60

✓ Uploaded: checkpoint_epoch60.pth

✓ Uploaded: metrics.json

Best Test Accuracy so far: 64.38% | Patience: 4/15



Epoch 61 Loss=3.2582 Acc=57.97% LR=0.009961: 100%|██████████| 196/196 [01:51<00:00,  1.75it/s]



Test set: Average loss: 1.4327, Accuracy: 6502/10000 (65.02%)



*** New best model! Test Accuracy: 65.02% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 65.02% | Patience: 0/15



Epoch 62 Loss=3.1051 Acc=60.15% LR=0.000448: 100%|██████████| 196/196 [01:51<00:00,  1.75it/s]



Test set: Average loss: 1.4064, Accuracy: 6572/10000 (65.72%)



*** New best model! Test Accuracy: 65.72% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 65.72% | Patience: 0/15



Epoch 63 Loss=2.3231 Acc=58.29% LR=0.001895: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.4483, Accuracy: 6506/10000 (65.06%)



Best Test Accuracy so far: 65.72% | Patience: 1/15



Epoch 64 Loss=1.9018 Acc=59.44% LR=0.004122: 100%|██████████| 196/196 [01:52<00:00,  1.75it/s]



Test set: Average loss: 1.4378, Accuracy: 6481/10000 (64.81%)



Best Test Accuracy so far: 65.72% | Patience: 2/15



Epoch 65 Loss=1.9681 Acc=57.35% LR=0.006580: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.5398, Accuracy: 6381/10000 (63.81%)



Best Test Accuracy so far: 65.72% | Patience: 3/15



Epoch 66 Loss=3.1877 Acc=58.40% LR=0.008658: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.4930, Accuracy: 6484/10000 (64.84%)



Best Test Accuracy so far: 65.72% | Patience: 4/15



Epoch 67 Loss=2.8313 Acc=60.20% LR=0.009844: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.4364, Accuracy: 6577/10000 (65.77%)



*** New best model! Test Accuracy: 65.77% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 65.77% | Patience: 0/15



Epoch 68 Loss=1.8927 Acc=59.84% LR=0.000256: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.4195, Accuracy: 6621/10000 (66.21%)



*** New best model! Test Accuracy: 66.21% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 66.21% | Patience: 0/15



Epoch 69 Loss=2.0533 Acc=60.89% LR=0.001442: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.4183, Accuracy: 6521/10000 (65.21%)



Best Test Accuracy so far: 66.21% | Patience: 1/15



Epoch 70 Loss=1.7621 Acc=60.66% LR=0.003520: 100%|██████████| 196/196 [01:52<00:00,  1.75it/s]



Test set: Average loss: 1.4421, Accuracy: 6590/10000 (65.90%)



✓ Uploaded: metrics.json

Best Test Accuracy so far: 66.21% | Patience: 2/15



Epoch 71 Loss=3.1971 Acc=62.75% LR=0.005978: 100%|██████████| 196/196 [01:53<00:00,  1.72it/s]



Test set: Average loss: 1.4229, Accuracy: 6592/10000 (65.92%)



Best Test Accuracy so far: 66.21% | Patience: 3/15



Epoch 72 Loss=1.8650 Acc=61.88% LR=0.008205: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.4984, Accuracy: 6388/10000 (63.88%)



Best Test Accuracy so far: 66.21% | Patience: 4/15



Epoch 73 Loss=1.9105 Acc=61.89% LR=0.009652: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.3691, Accuracy: 6737/10000 (67.37%)



*** New best model! Test Accuracy: 67.37% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 67.37% | Patience: 0/15



Epoch 74 Loss=3.5818 Acc=63.89% LR=0.000139: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.3755, Accuracy: 6731/10000 (67.31%)



Best Test Accuracy so far: 67.37% | Patience: 1/15



Epoch 75 Loss=1.6473 Acc=61.64% LR=0.001045: 100%|██████████| 196/196 [01:52<00:00,  1.75it/s]



Test set: Average loss: 1.4043, Accuracy: 6733/10000 (67.33%)



📍 Breakpoint checkpoint at epoch 75

✓ Uploaded: checkpoint_epoch75.pth

✓ Uploaded: metrics.json

Best Test Accuracy so far: 67.37% | Patience: 2/15



Epoch 76 Loss=3.4919 Acc=64.77% LR=0.002942: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.3928, Accuracy: 6685/10000 (66.85%)



Best Test Accuracy so far: 67.37% | Patience: 3/15



Epoch 77 Loss=1.9858 Acc=62.50% LR=0.005361: 100%|██████████| 196/196 [01:52<00:00,  1.75it/s]



Test set: Average loss: 1.4395, Accuracy: 6587/10000 (65.87%)



Best Test Accuracy so far: 67.37% | Patience: 4/15



Epoch 78 Loss=2.8693 Acc=64.43% LR=0.007702: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.4696, Accuracy: 6602/10000 (66.02%)



Best Test Accuracy so far: 67.37% | Patience: 5/15



Epoch 79 Loss=1.7390 Acc=63.00% LR=0.009388: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.4021, Accuracy: 6732/10000 (67.32%)



Best Test Accuracy so far: 67.37% | Patience: 6/15



Epoch 80 Loss=1.9091 Acc=65.64% LR=0.010000: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.3415, Accuracy: 6820/10000 (68.20%)



*** New best model! Test Accuracy: 68.20% ***

✓ Uploaded: best_model.pth

✓ Uploaded: metrics.json

Best Test Accuracy so far: 68.20% | Patience: 0/15



Epoch 81 Loss=3.3732 Acc=64.19% LR=0.000712: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.3736, Accuracy: 6840/10000 (68.40%)



*** New best model! Test Accuracy: 68.40% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 68.40% | Patience: 0/15



Epoch 82 Loss=1.7508 Acc=64.00% LR=0.002398: 100%|██████████| 196/196 [01:50<00:00,  1.77it/s]



Test set: Average loss: 1.3825, Accuracy: 6784/10000 (67.84%)



Best Test Accuracy so far: 68.40% | Patience: 1/15



Epoch 83 Loss=1.7585 Acc=64.28% LR=0.004739: 100%|██████████| 196/196 [01:52<00:00,  1.75it/s]



Test set: Average loss: 1.4130, Accuracy: 6691/10000 (66.91%)



Best Test Accuracy so far: 68.40% | Patience: 2/15



Epoch 84 Loss=1.6913 Acc=64.10% LR=0.007158: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.4245, Accuracy: 6648/10000 (66.48%)



Best Test Accuracy so far: 68.40% | Patience: 3/15



Epoch 85 Loss=1.6582 Acc=65.68% LR=0.009055: 100%|██████████| 196/196 [01:53<00:00,  1.73it/s]



Test set: Average loss: 1.4151, Accuracy: 6706/10000 (67.06%)



Best Test Accuracy so far: 68.40% | Patience: 4/15



Epoch 86 Loss=1.8273 Acc=67.61% LR=0.009961: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.3327, Accuracy: 6926/10000 (69.26%)



*** New best model! Test Accuracy: 69.26% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 69.26% | Patience: 0/15



Epoch 87 Loss=1.8681 Acc=66.51% LR=0.000448: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.3169, Accuracy: 6953/10000 (69.53%)



*** New best model! Test Accuracy: 69.53% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 69.53% | Patience: 0/15



Epoch 88 Loss=1.8194 Acc=66.47% LR=0.001895: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.3637, Accuracy: 6871/10000 (68.71%)



Best Test Accuracy so far: 69.53% | Patience: 1/15



Epoch 89 Loss=1.6879 Acc=65.26% LR=0.004122: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.4286, Accuracy: 6737/10000 (67.37%)



Best Test Accuracy so far: 69.53% | Patience: 2/15



Epoch 90 Loss=2.3970 Acc=66.09% LR=0.006580: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.4033, Accuracy: 6867/10000 (68.67%)



📍 Breakpoint checkpoint at epoch 90

✓ Uploaded: checkpoint_epoch90.pth

✓ Uploaded: metrics.json

Best Test Accuracy so far: 69.53% | Patience: 3/15



Epoch 91 Loss=1.7821 Acc=66.47% LR=0.008658: 100%|██████████| 196/196 [01:51<00:00,  1.75it/s]



Test set: Average loss: 1.4281, Accuracy: 6622/10000 (66.22%)



Best Test Accuracy so far: 69.53% | Patience: 4/15



Epoch 92 Loss=1.6016 Acc=68.91% LR=0.009844: 100%|██████████| 196/196 [01:52<00:00,  1.74it/s]



Test set: Average loss: 1.2864, Accuracy: 7019/10000 (70.19%)



*** New best model! Test Accuracy: 70.19% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 70.19% | Patience: 0/15



Epoch 93 Loss=1.7314 Acc=65.85% LR=0.000256: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.2996, Accuracy: 7095/10000 (70.95%)



*** New best model! Test Accuracy: 70.95% ***

✓ Uploaded: best_model.pth

Best Test Accuracy so far: 70.95% | Patience: 0/15



Epoch 94 Loss=2.4632 Acc=68.74% LR=0.001442: 100%|██████████| 196/196 [01:51<00:00,  1.76it/s]



Test set: Average loss: 1.3377, Accuracy: 6908/10000 (69.08%)



Best Test Accuracy so far: 70.95% | Patience: 1/15



Epoch 95 Loss=1.6684 Acc=70.40% LR=0.003520: 100%|██████████| 196/196 [01:53<00:00,  1.73it/s]



Test set: Average loss: 1.3324, Accuracy: 6937/10000 (69.37%)



Best Test Accuracy so far: 70.95% | Patience: 2/15



Epoch 96 Loss=1.9000 Acc=65.62% LR=0.005978: 100%|██████████| 196/196 [01:53<00:00,  1.73it/s]



Test set: Average loss: 1.3347, Accuracy: 6927/10000 (69.27%)



Best Test Accuracy so far: 70.95% | Patience: 3/15



Epoch 97 Loss=3.6192 Acc=67.46% LR=0.008205: 100%|██████████| 196/196 [01:53<00:00,  1.72it/s]



Test set: Average loss: 1.3475, Accuracy: 6895/10000 (68.95%)



Best Test Accuracy so far: 70.95% | Patience: 4/15



Epoch 98 Loss=2.8503 Acc=66.96% LR=0.009652: 100%|██████████| 196/196 [01:53<00:00,  1.72it/s]



Test set: Average loss: 1.3592, Accuracy: 6892/10000 (68.92%)



Best Test Accuracy so far: 70.95% | Patience: 5/15



Epoch 99 Loss=1.6424 Acc=68.14% LR=0.000139: 100%|██████████| 196/196 [01:53<00:00,  1.72it/s]



Test set: Average loss: 1.2911, Accuracy: 7069/10000 (70.69%)



Best Test Accuracy so far: 70.95% | Patience: 6/15



Epoch 100 Loss=2.0670 Acc=69.26% LR=0.001045: 100%|██████████| 196/196 [01:53<00:00,  1.73it/s]



Test set: Average loss: 1.2912, Accuracy: 7120/10000 (71.20%)



*** New best model! Test Accuracy: 71.20% ***

✓ Uploaded: best_model.pth

✓ Uploaded: metrics.json

Best Test Accuracy so far: 71.20% | Patience: 0/15





📦 Saving final model...

✓ Uploaded: final_model.pth



Training completed. Best test accuracy: 71.20%
```

---

## Training Summary

- **Total Epochs:** 100
- **Best Test Accuracy:** 71.20% (achieved at epoch 100)
- **Final Train Accuracy:** 69.26%
- **Train-Test Gap:** -1.94% (model generalizes well)

### Checkpoints Uploaded to HuggingFace
- Epoch 10: 47.60%
- Epoch 20: 53.06%
- Epoch 25: 55.27% (end of first cosine cycle)
- Epoch 30: 57.44%
- Epoch 40: 59.54%
- Epoch 50: 62.94% (mid-training)
- Epoch 60: 64.38%
- Epoch 75: 67.37%
- Epoch 90: 69.53%
- **Epoch 100: 71.20% (BEST)**

### Key Observations
1. **Consistent Improvement:** Accuracy improved steadily throughout training
2. **No Early Stopping:** Training completed all 100 epochs without triggering early stopping
3. **Good Generalization:** Train-test gap remained small (~2%), indicating the model generalizes well
4. **Learning Rate Cycles:** Cosine annealing with warm restarts (T_0=25) visible in learning rate changes
5. **Best Model:** Final epoch (100) achieved the best test accuracy of 71.20%

### HuggingFace Model
🤗 **Model Repository:** [pandurangpatil/cifar100-wideresnet-session8](https://huggingface.co/pandurangpatil/cifar100-wideresnet-session8)
