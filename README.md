# TowardsRobustImageProcessing

**TowardsRobustImageProcessing** is a systematic, multi-tier evaluation of computer vision robustness across ORB, K-Means segmentation, and YOLOv8 on KITTI. Evaluates perception decay across SNR parameter sweeps for noise, compression, and low light, comparing image pre-processing restoration against model fine-tuning for safety-critical automated systems.

> Built as part of the Digital Image Processing course (BIU).

**Author:** [Ziv Chaba](https://github.com/Ziv33)

---


<p align="center">
  <img width="1487" height="455" alt="nb1_KITTI_DATASET_EXAMPLE" src="https://github.com/user-attachments/assets/c6987695-d8c5-4c63-85b6-8e3f4527dfc7" />
  <br>
  <em>KITTI Dataset - Example </em>
</p>

---

## Table of Contents

- [Background](#background)
- [About KITTI Dataset](#about-kitti-dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Analysis and Results](#analysis-and-results)
- [Presentation](#presentation)

---

## Background

This project delivers a rigorous, quantitative audit of how classical and deep-learning vision algorithms behave when the clean, well-lit imagery they were designed for gives way to the noise, compression artifacts, and darkness of real driving conditions.

At its core is a unified evaluation engine built around three tasks spanning the full abstraction hierarchy of computer vision — low-level geometric tracking (ORB), mid-level unsupervised structure (K-Means segmentation), and high-level learned semantics (YOLOv8 object detection). Rather than reporting isolated accuracy numbers, the engine subjects every task to dense, 13-point sweeps across three physically-grounded camera distortions (Gaussian noise, JPEG compression, low-light exposure) and expresses the resulting decay on a single, physically meaningful axis — Signal-to-Noise Ratio (SNR, dB) — so that fundamentally different failure modes can be compared apples-to-apples.

Around this evaluation core sits a two-pronged mitigation framework: an adaptive classical restoration pipeline (denoising, deblocking, CLAHE) that cleans images before inference, and a model-level fine-tuning pipeline — including seven distinct training strategies and a novel three-stage curriculum learning schedule — that adapts the detector's own weights to tolerate distortion directly. A hybrid configuration combines both, quantifying whether pre-processing and fine-tuning are complementary or redundant.

The result is a complete, statistically grounded robustness study — with every hyperparameter globally calibrated, every sweep run across a stratified sample of the validation set, and every finding backed by continuous performance-versus-SNR curves — built to inform how safety-critical automated systems should be engineered.

## About KITTI Dataset
 
[**KITTI**](http://www.cvlibs.net/datasets/kitti/) is one of the most widely used benchmarks in autonomous driving research, created by the Karlsruhe Institute of Technology and the Toyota Technological Institute at Chicago. It consists of real-world imagery captured from a moving vehicle driving through urban, residential, and highway environments in and around Karlsruhe, Germany — making it a far more representative testbed for perception robustness than curated, studio-style datasets.
 
This project uses the **KITTI 2D Object Detection** benchmark: **7,481 labeled color images**, split into **3,712 training** and **3,769 validation** frames, with dense bounding-box annotations across three safety-relevant object classes:
 
| Class      | Description                                              |
|------------|-----------------------------------------------------------|
| Car        | Passenger vehicles — the dominant, highest-recall class   |
| Pedestrian | Relatively Small-footprint                                |
| Cyclist    | Relatively Small-footprint                                |
 
KITTI's combination of unmodified sensor imagery, real driving-scene diversity, and precise 2D annotations makes it an ideal foundation for this study: it allows distortions to be layered *on top of* genuinely realistic road scenes, and lets high-level detection performance be measured against trustworthy ground truth rather than synthetic or pseudo-labels — while the low- and mid-level tasks (ORB, K-Means) are evaluated against clean-image anchors using the project's own calibration and pseudo-ground-truth protocols.

## Project Structure

```
TowardsRobustImageProcessing/
├── environment.yml                  # Environment configuration
├── data/
│   └── kitti/
│       ├── dataset.yaml             # YAML configuration for YOLOv8 training
│       ├── images/                  # Native training and validation color images
│       ├── labels/                  # YOLO format ground-truth labels
│       ├── masks/                   # Optional semantic ground-truth masks
│       └── processed/
│           ├── distorted/           # Saved degraded images (segregated by distortion and sweep level)
│           └── enhanced/            # Saved restored/enhanced images
├── src/
│   ├── dataset.py                   # KITTI annotations parser and validation loader
│   ├── distortions.py               # Mathematical degradation functions
│   ├── enhancements.py              # Classical adaptive restoration filters
│   ├── evaluation.py                # Metric calculators (SNR, IoU, recall, precision)
│   ├── train_yolo.py                # Script to execute multi-strategy YOLOv8 training
│   └── tasks/
│       ├── base_task.py             # Abstract base class interface
│       ├── registry.py              # Centralized task registration system
│       ├── orb_matching.py          # Low-level: ORB detection & matching task
│       ├── kmeans_segmentation.py   # Mid-level: K-Means pixel clustering task
│       └── yolov8_detection.py      # High-level: YOLOv8 inference wrapper task
├── notebooks/
│   ├── 1_data_exploration.ipynb     # Baseline exploration and hyperparameter calibration
│   ├── 2_distorted_evaluation.ipynb # Dataset-wide baseline degradation sweeps vs. SNR
│   ├── 3_enhanced_evaluation.ipynb  # Pre-processing adaptive restoration sweeps
│   └── 4_fine_tuning_yolo.ipynb     # Model-level multi-strategy fine-tuning and insightful evaluations
└── output/
    ├── checkpoints/
    │   ├── yolov8_clean.pt          # Model 1: Clean-trained baseline model weights
    │   ├── yolov8_finetuned_50_50.pt # Model 2: 50/50 Balanced robust model weights
    │   ├── yolov8_finetuned_80_20.pt # Model 3: 80/20 Target robust model weights
    │   ├── yolov8_finetuned_low.pt   # Model 4: Low-noise band model weights
    │   ├── yolov8_finetuned_mid.pt   # Model 5: Mid-noise band model weights
    │   ├── yolov8_finetuned_high.pt  # Model 6: High-noise band model weights
    │   └── yolov8_finetuned_curriculum.pt # Model 7: 3-Stage Curriculum transfer-learned weights
    ├── metrics/
    │   └── nb3_results.json         # Serialized database containing all evaluated metrics on-disk
    └── plots/                       # Graphs, inferences
```

## Installation
 
### 1. Clone the Repository
 
```bash
git clone https://github.com/Ziv33/TowardsRobustImageProcessing.git
cd TowardsRobustImageProcessing
```
 
### 2. Create the Environment
 
All dependencies are pinned in `environment.yml`. Create and activate the Conda environment with:
 
```bash
conda env create -f environment.yml
conda activate digitalImageProcessing
```

### 3. Run the Notebooks
 
The four notebooks form a sequential pipeline: each stage caches artifacts — calibrated hyperparameters, distorted images, evaluation metrics, and model checkpoints — that the following notebook depends on. They should therefore be executed **in order**, from a kernel running the environment created above.
 
```bash
jupyter lab
```
 
You can also open the notebooks with Visual Studio Code.
From the Jupyter interface, open each notebook under `notebooks/` and execute it top to bottom (**Kernel → Restart Kernel and Run All Cells** is recommended to guarantee a clean, reproducible state):
 
| Order | Notebook | What Running It Produces |
|:---:|---|---|
| 1 | `1_data_exploration.ipynb` | Downloads and splits KITTI, calibrates the K-Means and ORB hyperparameters, and establishes the clean-image baseline |
| 2 | `2_distorted_evaluation.ipynb` | Generates the degraded image sets and produces the dataset-wide degradation-vs-SNR sweeps |
| 3 | `3_enhanced_evaluation.ipynb` | Applies the adaptive restoration filters and evaluates the recovered performance |
| 4 | `4_fine_tuning_yolo.ipynb` | Trains all seven YOLOv8 fine-tuning strategies and evaluates them against the mitigation baselines |
 
While working through a notebook:
 
- Read the markdown cells before each code section — they document the methodology, the mathematical definition of each metric, and the reasoning behind every design decision.
- Inspect the inline plots and image grids as each cell executes; they are the same figures referenced later in the [Analysis and Results](#analysis-and-results) section.
- Let each notebook finish fully before moving to the next one, since generated artifacts (distorted images in `data/kitti/processed/`, metrics in `output/metrics/`, and checkpoints in `output/checkpoints/`) are required as inputs downstream.
> Notebook 1 will automatically download and extract the official KITTI Color Image dataset (~12 GB) and its 2D labels on first run. Ensure a stable internet connection and sufficient free disk space before starting.

## Analysis and Results

In this section, the analysis and the results are discussed.
Every real-world perception system is eventually asked to see through something it was never trained on: grain from a noisy sensor, blockiness from an overcompressed video feed, or the simple absence of light. This section follows the project through the four stages that turn that observation into a measurable, actionable study — first establishing what "good" looks like on clean imagery, then watching that performance erode under controlled distortion, and finally testing two different ways of winning it back. The narrative below moves through the notebooks in the order the project itself was built, and every figure produced along the way is discussed in the context of what it revealed.

### Establishing the Clean Baseline
 
Any claim about performance loss is only as credible as the baseline it is measured against, so the project begins by calibrating its own instruments rather than assuming textbook defaults are appropriate for KITTI. Both the K-Means cluster count and the ORB keypoint budget are swept across thirty representative validation scenes, with the resulting curves plotted alongside their standard deviation to confirm that a single global choice is genuinely representative of the dataset and not an artifact of one lucky frame.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b09595c9-9652-4f16-aff2-ea8680fed72d" width="49%"/>
  <img src="https://github.com/user-attachments/assets/f9555ced-f005-45ed-b5ad-c5e4d549f70a" width="49%"/>
</p>

The elbow curve on the left settles cleanly at four clusters — beyond that point, additional clusters buy only marginal reductions in reconstruction error while fragmenting otherwise coherent regions of road, vegetation, and sky. The ORB sweep on the right tells a similar story of diminishing returns: keypoint matching accuracy under moderate noise rises quickly and then plateaus well before a thousand features, so capping the detector at four hundred keypoints preserves essentially all of the achievable matching quality while cutting the brute-force matching cost by roughly a third. Neither number was chosen for convenience — both are the point where the curve itself says further complexity stops paying for itself.

<p align="center">
  <img src="https://github.com/user-attachments/assets/539bfc1a-215c-4e6c-adc3-b21ca201ea8c" width="100%"/>
</p>

With that budget fixed, the calibrated ORB detector is run once on a clean validation frame, and its keypoints are cached as the geometric anchor that every later distorted or restored version of the same scene will be measured against — a single, trustworthy reference rather than a moving target.

<p align="center">
  <img src="https://github.com/user-attachments/assets/539a5d64-8906-4b08-b722-fde18bc2c461" width="100%"/>
</p>

Segmentation posed a different problem: KITTI's detection split carries no pixel-level semantic labels, so there is no ground truth to compare against in the traditional sense. The project works around this by treating the clean-image K-Means output itself as a pseudo-ground-truth mask, then aligning the unsupervised cluster identities of every later frame back to this anchor by maximum spatial overlap before computing IoU. It is a pragmatic substitute for labeled data, and one that still allows segmentation stability to be tracked quantitatively rather than only inspected by eye.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1d698432-a2b2-4a13-927a-920e6c52c7e0" width="100%"/>
</p>
The three distortions that structure the rest of the study are then introduced side by side on this same anchor frame. Gaussian noise coarsens the image with sensor-like grain, severe JPEG compression breaks it into visible blocks and color banding, and the low-light transform plunges the scene into the kind of underexposure a camera would face at dusk or in a tunnel. Each was chosen because it corresponds to a distortion an actual vehicle camera encounters, not because it is convenient to simulate.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1e661764-b0f4-4c38-98f6-b9e83684e644" width="100%"/>
</p>

Paired against each distortion is the classical filter designed to undo it — adaptive bilateral denoising for the noisy frame, a deblocking filter for the compressed one, and CLAHE-based exposure correction for the dark one. Even at a glance, the restored frames are visibly closer to the clean original than the distorted inputs, which is the qualitative preview of the quantitative recovery that Part Three later puts a number on.

### Watching Performance Erode Across the SNR Sweep
 
Once the baseline was trustworthy, the real experiment could begin: pushing each of the three tasks through a dense, thirteen-level sweep of every distortion, run across a stratified sample of one hundred fifty-one validation frames so that no single scene's quirks could skew the result. Every measurement is expressed against Signal-to-Noise Ratio in decibels, which matters more than it might first appear — noise, compression, and exposure are physically unrelated phenomena, and SNR is what makes it possible to place them on the same axis and ask, fairly, which one hurts performance fastest.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bf455efe-4d97-4b98-a2b2-f8decdcb31bd" width="100%"/>
</p>

The resulting curves make the differences between abstraction levels immediately visible. ORB matching and K-Means segmentation both decline gradually and predictably across the entire sweep, tracing something close to a straight line from clean to severely degraded. YOLOv8's recall for cars tells a very different story: it holds almost perfectly flat near its ceiling for a wide range of moderate distortion, then falls away sharply once the signal crosses a critical threshold. Pedestrians and cyclists start from a visibly lower recall even on clean data, a consequence of how little of the image they occupy, and their decline is steeper still once distortion sets in.

<p align="center">
  <img src="https://github.com/user-attachments/assets/43aa2bce-c7f9-44f6-aa8c-15a92dc7821a" width="100%"/>
</p>

Normalizing each curve to its own clean-baseline score strips away the difference in starting points and isolates something more interesting: rate of decay. Seen this way, ORB and K-Means are the most resilient of the three tasks, retaining the largest share of their original performance even under heavy corruption. YOLOv8, and especially its detection of cyclists and pedestrians, loses the largest relative share — confirming that the richer, learned representations a deep detector relies on are also the ones most vulnerable to having their input signal corrupted.

### Recovering Performance Through Pre-Processing
 
The first attempt at recovering the lost performance works at the sensor level: distorted frames pass through the same adaptive restoration filters shown earlier before being handed to the three evaluation tasks, and the resulting curves are plotted directly against the unmitigated distorted baseline.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d8aa090e-be3c-4ed0-a37e-a0269ab9448d" width="100%"/>
</p>

For ORB, adaptive denoising recovers a meaningful share of the keypoint matches lost to Gaussian noise and JPEG compression, and its willingness to scale its own strength down on near-clean frames means restoration never costs accuracy where it isn't needed — a real advantage over a fixed-strength filter that would over-smooth clean imagery. Under low light, restoration initially trails the raw signal before overtaking it as the scene grows darker, which makes sense: CLAHE-style contrast correction has little to correct until the frame is genuinely underexposed.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fd3c9bc3-a8c6-454d-bc09-ac3dbe794b04" width="100%"/>
</p>

K-Means shows a comparable pattern. Deblocking closely tracks, and modestly improves on, the raw distorted curve under compression by smoothing away the blocky boundaries that would otherwise fracture a single color region into several spurious clusters, and CLAHE lifts segmentation quality substantially once low-light frames grow dark enough for exposure correction to matter, though a gap remains at the very darkest levels tested.

<p align="center">
  <img src="https://github.com/user-attachments/assets/08425eec-6f62-4e4d-9e41-8a47fc5dcedc" width="100%"/>
</p>

The clearest recovery in the entire project shows up in car detection. Restoration lifts recall well above the unmitigated curve across almost the whole noise and low-light range, and holds a steady edge under compression as well — solid evidence that a comparatively cheap pre-processing step can meaningfully widen the operating envelope of an already-trained detector for the object class that matters most by sheer frequency on the road.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d0d158d0-4a64-4b48-9588-36faad71644b" width="100%"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/685da14e-a751-4944-8a6e-44a099206465" width="100%"/>
</p>

Pedestrians and cyclists benefit from the same restoration pipeline, but the gain is narrower and noisier than it was for cars. These are small, low-pixel-count objects to begin with, and a classical filter that operates on the whole image without any notion of where the safety-critical objects actually are has less structural signal to preserve for them. That limitation is exactly what motivates the next stage of the project.

### Recovering Performance Through Model-Level Fine-Tuning
 
Rather than cleaning the pixels before inference, the second mitigation strategy changes the detector itself, retraining YOLOv8's weights directly on distorted data. Seven variants were trained to compare different philosophies of exposure to distortion: a clean-only baseline, two fixed-ratio mixtures of clean and distorted data, three models each trained on a single fixed noise band, and a three-stage curriculum that walks the model progressively from low to mid to high noise.

<p align="center">
  <img src="https://github.com/user-attachments/assets/90445280-841e-43ab-a107-a615b4c05903" width="100%"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/2b16772c-eb86-432a-adda-d42bb211812a" width="100%"/>
</p>

Before turning to the fine-tuned detector, the same anchor frame is run back through ORB and K-Means in its clean, distorted, and enhanced states, confirming visually that the restoration pipeline from the previous stage genuinely preserves the structures these classical tasks depend on rather than merely looking cleaner to the eye.

<p align="center">
  <img src="https://github.com/user-attachments/assets/40027cb8-ffe9-4378-85a4-d277bafa7da4" width="100%"/>
</p>

The same frame run through the clean-trained model on both distorted and enhanced inputs, alongside two of the fine-tuned models, offers a striking visual proof of concept: objects the clean model misses entirely on the distorted input are picked up correctly once the model itself has been retrained on distortion, independent of any pre-processing at all.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7fafc3ac-0bb4-4436-99e3-c21bacccfaa6" width="100%"/>
</p>

A natural worry with training on distorted data is that the model might quietly forget how to perform on clean images — a so-called clean tax. The results here are reassuring: every one of the seven robust-trained variants matches or slightly exceeds the clean-baseline model's own clean-domain recall, with the balanced and target-mixture strategies performing best of all. Robustness, in this case, was gained without giving anything up.

<p align="center">
  <img src="https://github.com/user-attachments/assets/51ea7ab6-81e6-4bd2-9fed-0ac5efac6975" width="100%"/>
</p>

The real test comes when each model is pushed into progressively harsher operational domains. The clean-baseline model collapses almost completely in the severe-noise domain, its recall falling to nearly zero, while every robust-trained strategy retains substantially more of its detection ability there — the balanced mixture, the target mixture, and the curriculum model hold up best of the seven. The message is unambiguous: a detector that has never seen distortion during training has no way to cope with it at inference time, no matter how well it performs on clean data.

<p align="center">
  <img src="https://github.com/user-attachments/assets/14f4d1a9-d15d-4255-b2e6-5eaa88f94543" width="80%"/>
</p>

Plotting precision against recall for all seven models under mixed distortion makes the trade-off concrete rather than abstract. The clean-baseline model sits confidently in the high-precision, low-recall corner — when it does detect something it is usually right, but it simply misses most objects once the input degrades. The robust-trained models cluster toward a more balanced region of the plot, with the balanced-mixture and curriculum models achieving the strongest joint position: high recall without sacrificing precision, which is exactly the combination a safety-critical system needs.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b8a07ffd-6958-4e2c-a393-c16213162459" width="100%"/>
</p>

Bringing every mitigation strategy together at a single severe noise level makes the comparison direct. Against the clean baseline, the unmitigated distorted signal, classical restoration, and the best fine-tuned model are compared class by class and in a consolidated summary. Fine-tuning consistently recovers more recall than pre-processing alone across all three classes, and the consolidated view shows model-level adaptation restoring the majority of the performance that severe distortion would otherwise erase — a materially larger recovery than restoration achieves by itself at this intensity.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2457bde5-7013-4bb4-9f5f-2fc2961d0944" width="100%"/>
</p>

The final comparison is the one the whole project has been building toward: does combining both mitigation strategies do better than either one alone? Four configurations are tracked across the full SNR sweep — the clean model on distorted input, the clean model on restored input, the fine-tuned model on distorted input, and the fine-tuned model on restored input together. The result is more interesting than a simple confirmation that more mitigation is always better. Both mitigated configurations comfortably outperform the clean, unmitigated model across every distortion, but the fine-tuned model given the raw distorted input directly — with no pre-processing at all — consistently sits above the hybrid curve, holding closer to the clean-baseline floor than any other configuration across noise, compression, and low light alike.

This is a genuinely useful finding. It suggests that once a detector has learned to tolerate distortion at the weight level, an upstream restoration step no longer adds value and may in fact introduce a small cost of its own — the same edge-softening that classical filters use to suppress noise can blur exactly the structural detail the fine-tuned model has learned to exploit directly from the raw signal. For a safety-critical, real-time system, this has a very practical implication: the restoration stage can be skipped entirely at inference time without giving up robustness, removing a full pre-processing pass from the perception pipeline's critical path and recovering the latency budget for other tasks. Fine-tuning, in other words, does not merely add another layer of defense on top of restoration — for this detector, it renders that layer largely redundant, and the simpler, faster, single-stage pipeline is also the better-performing one.

### Conclusions

Taken as a whole, this study makes a clear case for treating robustness as something that must be engineered and measured, not assumed. Establishing a properly calibrated clean baseline before introducing any distortion made every later comparison trustworthy rather than anecdotal, and expressing every result against a physically grounded Signal-to-Noise Ratio made it possible to compare three unrelated camera failure modes — noise, compression, and darkness — on genuinely equal terms. Across that comparison, a consistent hierarchy emerged: classical, hand-crafted tasks such as ORB matching and K-Means segmentation degrade gracefully and predictably, while the learned representations inside YOLOv8 hold their ground admirably under moderate distortion before falling away sharply once a critical threshold is crossed, with the smallest, safety-critical object classes proving the most fragile of all.

Of the two mitigation strategies tested, pre-processing restoration recovered meaningful performance cheaply and without touching the model at all, but its gains were structurally limited by the fact that it operates blindly on pixels rather than on the objects that actually matter. Fine-tuning the detector directly on distorted data closed that gap far more effectively, and did so without paying any measurable clean-domain penalty — the seven robust-trained models retained, and in most cases slightly improved on, their clean-baseline accuracy. Most notably, the final comparison showed that a well fine-tuned model, given the distorted signal directly, matched or exceeded the performance of the full hybrid pipeline — meaning the strongest and most efficient configuration uncovered by this project is not the most complex one. For a safety-critical automated system, where both reliability and latency are non-negotiable, this points toward a clear architectural recommendation: invest in training the perception model itself to tolerate the conditions it will actually encounter, rather than relying on a pre-processing stage to compensate for a model that was never taught to expect them.

# Presentation

A walkthrough of the project in PPT format is available here: [TowardsRobustImageProcessing.pptx](TowardsRobustImageProcessing.pptx).


