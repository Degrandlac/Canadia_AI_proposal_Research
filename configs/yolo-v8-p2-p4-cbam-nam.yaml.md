# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Parameters
nc: 5 # number of classes
scales: # model compound scaling constants
  # [depth, width, max_channels]
  m: [0.67, 0.75, 768] # YOLOv8m

# YOLOv8 backbone with CBAM attention
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]    # 0-P1/2   (320x320)
  - [-1, 1, C2f, [64, True]]     # 1
  
  - [-1, 1, Conv, [128, 3, 2]]   # 2-P2/4   (160x160)
  - [-1, 3, C2f, [128, True]]    # 3
  - [-1, 1, CBAM, [128]]         # 4: CBAM attention for P2
  
  - [-1, 1, Conv, [256, 3, 2]]   # 5-P3/8   (80x80)
  - [-1, 6, C2f, [256, True]]    # 6
  - [-1, 1, CBAM, [256]]         # 7: CBAM attention for P3
  
  - [-1, 1, Conv, [512, 3, 2]]   # 8-P4/16  (40x40)
  - [-1, 6, C2f, [512, True]]    # 9
  - [-1, 1, CBAM, [512]]         # 10: CBAM attention for P4
  
  - [-1, 1, SPPF, [512, 5]]      # 11: SPPF

# YOLOv8 head with P2, P3, P4 detection (removed P1)
head:
  # Top-down pathway (P4 -> P3 -> P2)
  - [-1, 1, Conv, [256, 1, 1]]                  # 12: reduce channels
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 13: 40→80 (upsample from P4)
  - [[-1, 7], 1, Concat, [1]]                   # 14: concat with P3 CBAM (both 80×80)
  - [-1, 3, C2f, [256]]                         # 15: fuse P3 features
  - [-1, 1, CBAM, [256]]                        # 16: CBAM after P3 fusion

  - [-1, 1, Conv, [128, 1, 1]]                  # 17: reduce channels
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 18: 80→160 (upsample to P2)
  - [[-1, 4], 1, Concat, [1]]                   # 19: concat with P2 CBAM (both 160×160)
  - [-1, 3, C2f, [128]]                         # 20: fuse P2 features
  - [-1, 1, CBAM, [128]]                        # 21: CBAM after P2 fusion

  # Bottom-up pathway (P2 -> P3 -> P4)
  - [-1, 1, Conv, [128, 3, 2]]                  # 22: 160→80 (downsample from P2)
  - [[-1, 16], 1, Concat, [1]]                  # 23: concat with P3 (both 80×80)
  - [-1, 3, C2f, [256]]                         # 24: refined P3
  - [-1, 1, CBAM, [256]]                        # 25: CBAM for refined P3

  - [-1, 1, Conv, [256, 3, 2]]                  # 26: 80→40 (downsample from P3)
  - [[-1, 11], 1, Concat, [1]]                  # 27: concat with SPPF (both 40×40)
  - [-1, 3, C2f, [512]]                         # 28: refined P4
  - [-1, 1, CBAM, [512]]                        # 29: CBAM for refined P4

  # NAM Attention before detection heads
  - [21, 1, NAM, [128]]                         # 30: NAM for P2 detection
  - [25, 1, NAM, [256]]                         # 31: NAM for P3 detection
  - [29, 1, NAM, [512]]                         # 32: NAM for P4 detection

  # Detection heads at P2, P3, P4 (removed P1)
  - [[30, 31, 32], 1, Detect, [nc]]             # 33: Detect(P2, P3, P4)
```

