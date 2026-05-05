# PyTorch Implementation of Onsets and Frames

This repository is forked from a [PyTorch](https://pytorch.org/) implementation of Google's [Onsets and Frames](https://magenta.tensorflow.org/onsets-frames) model. Originally designed for piano transcription, this version has been significantly enhanced to support high-fidelity **guitar transcription** using the GuitarSet dataset and Constant-Q Transform (CQT) features.

---

## GuitarSet Integration & Performance

The model has been extended to support the **GuitarSet** dataset, shifting the focus from piano to guitar transcription. This involved significant changes to the input features, output space, and convolutional architecture.

### Comparative Evaluation (Player 05)

The following table compares the performance of the baseline **Mel Spectrogram** model against the proposed **CQT Transform** model. Both were evaluated on the Player 05 test set of GuitarSet.

| Metric | Mel Spectrogram (Baseline) | CQT Transform (Proposed) | Improvement |
| :--- | :---: | :---: | :---: |
| **Note F1 Score** | 0.868 | **0.880** | +1.2% |
| **Frame F1 Score** | 0.791 | **0.841** | +5.0% |
| **Note w/ Offsets F1** | 0.583 | **0.676** | **+9.3%** |

> [!NOTE]
> The CQT-based model shows a significant improvement in offset detection and overall frame accuracy, likely due to the higher frequency resolution (36 bins per octave) and alignment with the logarithmic nature of musical pitch.

---

## Major Changes (GuitarSet Branch)

The following architectural and pipeline changes were implemented to adapt the model for guitar transcription:

### 1. Dataset & Requirements
- **Native GuitarSet Support**: Integrated `mirdata` for automated downloading and standardized loading of hexaphonic audio and annotations.
- **Player-Stratified Split**: Implemented a 3-way split to ensure unbiased evaluation:
    - **Training**: Players 00–03.
    - **Validation**: Player 04 (used for real-time monitoring).
    - **Testing**: Player 05 (held out for final evaluation).
- **Dependency Updates**: Added `nnAudio` for GPU-accelerated CQT extraction and `mirdata` for dataset management.

### 2. Configurable Audio Features
- **CQT Implementation**: Added a new `CQT` module wrapping `nnAudio.features.cqt.CQT1992v2`. Configured with **36 bins per octave** (3 bins per semitone) for musically-aligned resolution.
- **Feature Toggling**: The `OnsetsAndFrames` model now supports a configurable `audio_features` argument, allowing seamless switching between `'mel'` and `'cqt'`.

### 3. Guitar-Specific Output Space
- **MIDI Range**: Constrained the output to **MIDI 40 to 83** (44 pitches), matching the standard range of a guitar and reducing the model's output complexity.
- **Resampling**: Standardized input audio to **16kHz** to ensure consistency across the pipeline.

### 4. ConvStack & Temporal Redesign
The `ConvStack` module was rebuilt according to the **Wiggins and Kim (2021)** paper to better capture guitar pitch features:
- **Layer Structure**: 2D Convolutions with 12, 12, and 24 filters respectively.
- **Frequency-Axis Pooling**: Implemented `MaxPool2d((1, 2))` to reduce dimensionality exclusively along the frequency axis, preserving temporal resolution.
- **Dynamic Flattening**: Implemented logic to allow the model to switch between Mel and CQT spectral resolutions without manual recalibration.
- **BiLSTM Bottleneck**: Utilizes a bidirectional LSTM with **768 hidden units** to capture the long-term sustain of guitar notes.

---

## Model Capacity & Parameter Analysis

The shift to CQT features increases the model's "representational bandwidth" at the cost of higher parameter counts:

| Model Configuration | Total Parameters | Input Flattened Dim |
| :--- | :---: | :---: |
| **Mel-Baseline** | 17,275,276 | 2,736 |
| **CQT-Experimental** | **19,487,116** | **3,456** |

The **12.8% increase** in parameters is concentrated in the fully-connected transition layers. This extra capacity allows the CQT model to effectively process the higher semitone-aligned resolution, leading to the improved F1 scores observed in testing.

---

## Key Findings & Conclusion

**Offset Precision**: The most significant performance gain (**+9.3% F1 score**) was observed in **Note Offset** detection. The model was better able to distinguish the decay phase of a string vibration from the onset of a new note, even in dense polyphonic passages.

While the CQT model didn't improve the overall F1 score significantly (+1.2%), it substantially improved the F1 score of note offsets, with the tradeoff of slightly more parameters (17M vs 19M).

---

## Original Piano Model (Maestro/MAPS)

This section contains the instructions and details for the original piano transcription model using the Maestro and MAPS datasets.

### Instructions

This project is quite resource-intensive; 32 GB or larger system memory and 8 GB or larger GPU memory is recommended. 

### Downloading Dataset

The `data` subdirectory already contains the MAPS database. To download the Maestro dataset, first make sure that you have `ffmpeg` executable and run `prepare_maestro.sh` script:

```bash
ffmpeg -version
cd data
./prepare_maestro.sh
```

This will download the full Maestro dataset from Google's server and automatically unzip and encode them as FLAC files in order to save storage. However, you'll still need about 200 GB of space for intermediate storage.

### Training

To train the original piano model, run:

```bash
pip install -r requirements.txt
python train.py
```

`train.py` accepts configuration options such as:

```bash
python train.py with logdir=runs/model iterations=1000000
```

### Testing

To evaluate the trained model using the MAPS database:

```bash
python evaluate.py runs/model/model-100000.pt
```

To test on the Maestro dataset's test split:

```bash
python evaluate.py runs/model/model-100000.pt Maestro test
```

---

## Original Implementation Details

This implementation contains a few of the additional improvements on the model that were reported in the Maestro paper, including:

* Offset head
* Increased model capacity (26M parameters by default for the piano model)
* Gradient stopping of inter-stack connections
* L2 Gradient clipping of each parameter at 3
* Using the HTK mel frequencies

Note: This implementation does not include variable-length input sequences or harmonically decaying weights on the frame loss.
