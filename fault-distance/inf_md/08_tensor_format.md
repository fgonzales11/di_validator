# Stage 7: Final tensor format

## Tensor shape
```
X: (NUM_CHANNELS, SEQ_LENGTH)
```

where:
- `NUM_CHANNELS` = 6 (without symseq) or 12 (with symseq)
- `SEQ_LENGTH` = 400 (default)

## Channel layout

### 6-channel tensor (SYMSEQ_ENABLED = False)
```
X[0, :] = IA   // phase A current [normalized]
X[1, :] = IB   // phase B current [normalized]
X[2, :] = IC   // phase C current [normalized]
X[3, :] = UA   // phase A voltage [normalized]
X[4, :] = UB   // phase B voltage [normalized]
X[5, :] = UC   // phase C voltage [normalized]
```

### 12-channel tensor (SYMSEQ_ENABLED = True)
```
X[0, :]  = IA      // phase A current [normalized]
X[1, :]  = IB      // phase B current [normalized]
X[2, :]  = IC      // phase C current [normalized]
X[3, :]  = |I1|    // positive-sequence current magnitude [normalized]
X[4, :]  = |I2|    // negative-sequence current magnitude [normalized]
X[5, :]  = |I0|    // zero-sequence current magnitude [normalized]
X[6, :]  = UA      // phase A voltage [normalized]
X[7, :]  = UB      // phase B voltage [normalized]
X[8, :]  = UC      // phase C voltage [normalized]
X[9, :]  = |U1|    // positive-sequence voltage magnitude [normalized]
X[10, :] = |U2|    // negative-sequence voltage magnitude [normalized]
X[11, :] = |U0|    // zero-sequence voltage magnitude [normalized]
```

## Batch format (for DataLoader)
```
X_batch: (BATCH_SIZE, NUM_CHANNELS, SEQ_LENGTH)
```

## Label (target)
```
y: scalar float32

// For standard normalization:
y_norm = (y_km - d_min) / (d_max - d_min)

// For p.u. normalization:
y_pu = y_km / L_km
```

## Example
For `BATCH_SIZE = 32`, `NUM_CHANNELS = 12`, `SEQ_LENGTH = 400`:
```
X_batch: (32, 12, 400)   // float32
y_batch: (32, 1)         // float32
```

## Model input
The PyTorch model (Conv1d) expects:
```python
# (B, C, L) — batch, channels, length
input_tensor: (BATCH_SIZE, NUM_CHANNELS, SEQ_LENGTH)
```

## Notes
- All channels are normalized independently
- Symmetrical components are computed using a sliding window (time-varying)
- Padding/trimming ensures a fixed length of SEQ_LENGTH
- The data is ready to be fed into a 1D-CNN / ResNet1D
