# Robust Multimodal Deepfake Detection
## A Critical Literature Survey of Visual, Audio, and Audio-Visual Methods

Literature review • IEEE numbered references • 13 October 2026

## Abstract

Deepfake detection requires more than recognising visible defects in a generated face. A recording may contain an altered face, synthetic speech, or a short manipulation within otherwise genuine material. Detection also becomes harder when media are compressed, recorded through unfamiliar devices, or produced by generators absent from the training set. This review examines the datasets, representations, fusion strategies, evaluation practices, and explanation methods relevant to a face-and-voice detection system. It traces the movement from image classifiers and generator-specific artefacts towards temporal modelling, speech countermeasures, and learned relationships between audio and video. Thirty selected references provide the basis for a critical narrative synthesis rather than an exhaustive systematic review. The evidence supports multimodal analysis as a way to broaden attack coverage, but does not establish that adding a second modality automatically improves generalisation. Dataset construction, label definitions, calibration, and treatment of missing inputs are equally consequential. The review identifies practical research gaps and relates them to the existing Robust Multimodal Deepfake Analysis project. Its recommended contribution is a reproducible comparison of unimodal and fusion baselines under controlled degradation and unseen-data conditions, with measured limitations and explanations assessed for their connection to model behaviour.

Keywords: deepfake detection; audio spoofing; multimodal fusion; generalisation; robustness; explainability.

## 1. Introduction and Review Objectives

The detection problem changes with the form of manipulation. Face replacement may preserve the original voice; voice synthesis may accompany an unchanged face; coordinated generation may alter both. A binary video label hides these distinctions. For a face-and-voice detector, the central question is which evidence remains useful when the manipulated component, generation method, and recording conditions change.

This review addresses five questions. What do commonly used datasets actually allow a detector to learn and an evaluator to claim? Which visual and audio representations have useful evidence of transfer beyond familiar attacks? How do score fusion, feature fusion, and explicit audio-visual correspondence differ? What constitutes a credible robustness or explanation result? Finally, which experiments would establish a defensible contribution for a modest research prototype?

The discussion focuses on human face and speech manipulation. It distinguishes whole-recording classification from temporal localisation and separates intrinsic media analysis from identity verification. General image generation, watermarking, provenance systems, and policy are outside the main scope. This boundary keeps the review relevant to a system that receives a recording and produces face, voice, and overall manipulation scores.

## 2. Review Method and Scope

### 2.1 Search and selection

The review uses a targeted narrative search conducted on 13 October 2026. Searches combined terms including “deepfake detection,” “audio spoofing,” “audio-visual fusion,” “cross-dataset,” “temporal localisation,” “robustness,” and “saliency sanity checks.” Discovery used web searches, followed by verification against primary publication pages, author-submitted arXiv records, conference proceedings, or official research resources. CVF, NeurIPS proceedings, and PMLR were used where accessible. Search results and reference links also guided the selection of related work.

Studies were selected for direct relevance to face-and-voice detection, a recognisable methodological contribution, or an important evaluation issue. Foundational architecture papers were retained to distinguish an inherited backbone from a detection-specific contribution. Recent work was included to show how the task has expanded beyond older face-swap benchmarks. Duplicated records, promotional summaries, unverifiable performance claims, and unrelated applications were excluded from the substantive evidence.

Thirty references are included. Publication metadata and available abstracts were checked; accessible full-text sections were consulted for selected recent multimodal studies. Access depth varied, so the review avoids detailed implementation claims where the available material did not support them. No exhaustive database export, independent screening process, or reproducible search-result count was produced. Accordingly, this document does not claim systematic-review completeness, a PRISMA process, or meta-analytic estimates.

### 2.2 Evidence interpretation

Studies are compared by task, modality, representation, dataset conditions, and evaluation boundaries. A limitation described here may be an author-reported observation or this review’s inference from the method’s assumptions; inferential points are identified in the discussion. Published metrics are retained only with their dataset context. Scores from different splits, sampling procedures, compression settings, or label definitions are not treated as a common leaderboard.

## 3. Datasets and the Meaning of the Detection Task

### 3.1 Visual benchmarks

FaceForensics++ established a shared setting for studying facial manipulation across DeepFakes, Face2Face, FaceSwap, and NeuralTextures, including compression conditions [1]. Its importance lies in controlled comparison: detectors can be tested against several manipulation families while recording quality is varied. Nevertheless, success within this collection establishes performance against its particular generation and processing pipeline. It does not establish coverage of every later synthesis method.

Celeb-DF was designed to increase the visual quality of synthetic celebrity videos and reduce conspicuous defects found in earlier collections. The paper describes 5,639 generated videos and evaluates existing detectors against the harder material [2]. This makes the dataset useful for examining whether a model relies on obvious defects. As an inference from its design, celebrity footage and its synthesis pipeline still define a restricted domain; transfer to ordinary users, different cameras, or new generators requires additional testing.

DFDC broadened scale and acquisition diversity, with more than 100,000 clips sourced from 3,426 paid actors and several manipulation procedures [3]. Its consent-based collection is a useful distinction from scraped celebrity material. Scale helps expose a learner to more variation, but cannot by itself prevent shortcuts. A detector could still exploit processing differences that correlate with the class. Larger data therefore complement, rather than replace, careful splits and external evaluation.

### 3.2 Audio and paired audio-video data

FakeAVCeleb introduces paired visual and synthetic speech material, including lip-synchronised generated audio [4]. Its four real/fake combinations permit a direct distinction between face manipulation, voice manipulation, and joint manipulation. This is essential for testing an overall label defined as “either component is fake.” However, the presence of several demographic groups does not prove equal performance across those groups. Fairness requires subgroup evaluation with adequate sample counts.

ASVspoof 2019 separates logical-access attacks involving synthesised or converted speech from physical-access replay conditions [5]. The challenge connects countermeasure evaluation to speaker verification through the tandem detection cost function, while retaining equal error rate. An audio detector trained on logical-access data should not be described as having demonstrated replay protection unless replay is actually evaluated. It also needs separate evidence before claims about background music, conversational recordings, or unfamiliar languages are made.

Table 1 summarises the complementary roles of these datasets. None supplies every test needed for a multimodal deployment claim.

Table 1. Dataset roles and evaluation boundaries.

| Dataset | Main evidence | Suitable use | Boundary of the claim |
|---|---|---|---|
| FaceForensics++ [1] | Manipulated faces under controlled compression | Visual baselines and manipulation transfer | Limited generator families and acquisition domain |
| Celeb-DF [2] | Higher-quality celebrity face swaps | External visual evaluation | Does not establish synthetic-voice detection |
| DFDC [3] | Large and varied face-swap collection | Training diversity and visual benchmarking | Pipeline cues and split design still matter |
| FakeAVCeleb [4] | Paired real/fake audio-video combinations | Branch attribution and fusion comparison | Generator coverage and subgroup reliability need testing |
| ASVspoof 2019 [5] | Speech synthesis, conversion, and replay protocols | Audio countermeasures with task-specific partitions | Logical and physical access are distinct conditions |

### 3.3 Standardisation and leakage

DeepfakeBench addresses inconsistent preprocessing, model implementations, and evaluation protocols that can make comparisons misleading [6]. The original benchmark integrates 15 detectors and nine datasets. Its methodological lesson is that changing the face crop, training input, or aggregation rule may change results independently of the proposed detector. A comparison should therefore disclose these choices before attributing gains to an architecture.

For the present project, identity separation is an important precaution, but it should be combined with grouping by original source recording. Frames or derived forgeries from the same source can share backgrounds, lighting, and capture characteristics. Official partitions should be preserved when evaluating a named benchmark. Any subset, balancing procedure, or departure from the official protocol must be reported, because each affects comparability.

## 4. Visual Deepfake Detection

### 4.1 Compact classifiers and pretrained backbones

MesoNet uses compact convolutional networks to capture mesoscopic image structure rather than relying entirely on fine forensic traces damaged by compression [7]. Its early experiments established the usefulness of a small learned detector for the manipulation types considered. The enduring contribution is a manageable baseline. Its reported detection rates should not be interpreted as evidence that the same model detects modern synthetic video under unrelated conditions.

Xception replaces conventional convolutional components with depthwise separable operations and was introduced for general image recognition [8]. Its later use in forensic pipelines should be distinguished from the original architecture paper. FaceForensics++ provides the detection setting in which such classifiers became important references. The resulting practical lesson is that a strong generic feature extractor can be useful, but its forensic reliability depends on the training distribution and processing choices.

EfficientNet studies coordinated scaling of network depth, width, and input resolution, with an emphasis on accuracy relative to computational cost [9]. EfficientNet-B0 is consequently a reasonable baseline for a project with limited GPU resources. That choice is justified by implementation cost and a known feature extractor, rather than by an assumption of superior forensic generalisation. It requires comparison against another suitable backbone under the same data protocol.

Frame classifiers also have an aggregation limitation. Averaging sampled frame outputs is simple, but can dilute evidence from a short edited interval. It discards motion order and may miss frames between sampling positions. This is a consequence of the aggregation design, not a measured result for the present project. A frame-based system should therefore state how many frames it inspects and whether its claim concerns full-video coverage.

### 4.2 Learning artefacts shared across manipulations

Face X-ray investigates the boundary introduced when a manipulated face is blended into an existing image [10]. Training can use constructed blends without requiring examples from every target generator. This moves the learning objective towards a shared operation rather than a catalogue of generator signatures. The assumption remains specific: a forgery must leave evidence related to blending. Fully generated content or a different editing pipeline may not preserve that cue.

Self-Blended Images construct synthetic training examples from pristine images to reproduce common inconsistencies between pseudo source and target regions [11]. The approach is valuable because it changes how training evidence is generated, not merely the classifier size. Its experiments support transfer under the paper’s cross-dataset and cross-manipulation settings. They do not establish that artificial blending captures every type of generated face or its temporal behaviour.

Together, these methods suggest an economical alternative to continually adding labelled fakes: create training variation around operations that several manipulation methods share. The risk is selecting an operation that future generators avoid. For this project, a blending-based training comparison would be useful only after a reproducible visual baseline exists; adding it before basic dataset and split checks would complicate interpretation.

### 4.3 Temporal and semantic evidence

LipForensics learns natural mouth motion through visual speech recognition before training a forgery detector using fixed mouth representations [12]. Its evaluation addresses unfamiliar manipulations and several distortions. Compared with texture-focused classification, it offers a reason to expect less reliance on fragile pixel artefacts: its representation is shaped by a meaningful temporal task. This is still a hypothesis about useful evidence, supported within the study’s tested conditions.

A practical limitation inferred from the method is dependence on visible, informative mouth motion. Occlusion, limited speech, or difficult viewing angles reduce that evidence. Temporal modelling also requires coherent sequences, so scattered independent frames cannot serve as an equivalent input. The trade-off is clear: frame models are easier to train and inspect, while temporal models preserve behaviour that may support transfer. Neither choice removes the need for external testing.

## 5. Audio Deepfake Detection

### 5.1 Spectrogram classifiers

Residual networks use shortcut connections to make deep feature learning easier to optimise [13]. Applying a ResNet to log-mel spectrograms offers a familiar training pipeline and a representation that can be inspected in time and frequency. However, the residual-learning paper establishes an image-recognition architecture, not the effectiveness of a particular speech-spoofing system. That effectiveness must come from the audio experiment.

Log-mel features provide a compact representation but omit waveform phase and compress spectral detail. Whether that loses useful spoofing evidence depends on the attack and front end. A spectrogram baseline should therefore be treated as one representation choice. Window length is equally important: a four-second crop reduces computation, but a central crop may miss an altered word elsewhere in a long recording. Recording-level claims require a windowing and aggregation policy.

### 5.2 Waveform and graph-based countermeasures

RawNet2 adapts a waveform model to anti-spoofing, allowing cues to be learned directly from audio rather than prescribing a conventional spectral representation [14]. The paper considers ASVspoof 2019 logical-access attacks and reports useful complementarity with baseline countermeasures. This supports waveform modelling as a serious comparison, especially when one spectral model performs unevenly across attacks. It does not imply that raw input is inherently robust to arbitrary codecs or microphones.

AASIST explicitly combines spectral and temporal relationships through graph attention [15]. Its design addresses the possibility that different attacks leave evidence in different domains. The paper also reports a lightweight variant, AASIST-L, with approximately 85,000 parameters. That makes it relevant to constrained experiments. Nevertheless, parameter count alone is not an inference-time comparison; preprocessing, input duration, hardware, and implementation must also be measured.

The contrast between these approaches and a generic spectrogram CNN is methodological. RawNet2 learns its audio front end, while AASIST structures interactions among learned spectral and temporal features. A baseline comparison can reveal whether an observed gain comes from representation, architecture, augmentation, or training exposure. Changing all four at once would make the cause of improvement difficult to establish.

### 5.3 Self-supervised representations and augmentation

Tak and colleagues combine a wav2vec 2.0 front end with fine-tuning and data augmentation for spoofing detection [16]. Their study reports improvements on the ASVspoof 2021 logical-access and deepfake databases. The useful principle is that representations learned from genuine speech can supply information that a small spoofing dataset alone may not provide. Pretraining nevertheless has language, channel, and corpus boundaries; these require testing in the intended application.

SpecAugment masks time and frequency regions and includes feature warping in its original speech-recognition formulation [17]. It is a useful regularisation candidate, but its original results are not spoof-detection results. The inference for detection is conditional: forcing reliance on several regions may improve transfer, while excessive masking may remove the brief artefact needed to identify an attack. Mask size and duration should therefore be evaluated, rather than inherited without justification.

Audio evaluation should include attack-level errors alongside aggregate AUC and EER. A model can rank most spoofed clips correctly while failing on one generator, and a threshold selected for familiar attacks may behave differently on unseen ones. This distinction becomes especially important when the branch score is passed into fusion: confidence and reliability are separate properties.

## 6. Multimodal Fusion and Audio-Visual Correspondence

### 6.1 What fusion can and cannot establish

The multimodal taxonomy of Baltrušaitis, Ahuja, and Morency separates representation, translation, alignment, fusion, and co-learning [18]. This is useful for deepfake research because a system can combine information without learning its correspondence. Averaging face and voice scores is score fusion. Concatenating embeddings allows feature interactions. Determining whether spoken content matches visible articulation requires aligned representations and a corresponding learning objective.

For an overall label meaning “at least one component is manipulated,” weighted averaging can suppress a strongly suspicious branch when the other branch reports a low score. Maximum fusion preserves the stronger score but also propagates isolated false alarms. Noisy-OR, expressed as 1 − (1 − p_video)(1 − p_audio), follows the event-union interpretation only when probabilities and dependence assumptions are appropriate. Otherwise, it is a scoring rule whose calibration must be measured.

Intermediate fusion can learn a more flexible decision from embeddings, but may exploit whichever modality is easiest in the training set. It may also learn dataset-specific correlations that disappear elsewhere. Its extra capacity is worthwhile only if it improves held-out performance under a common protocol. A small learned fusion head and simple score rules should precede a more complex cross-modal architecture in a limited-compute project.

### 6.2 Missing and unreliable inputs

ModDrop trains multimodal gesture models while randomly dropping channels, with the aim of retaining useful predictions when some signals are missing [19]. It provides a precedent for modality dropout, rather than direct evidence of deepfake-detection improvement. Applying the idea here requires explicit presence indicators and evaluation under audio-only and video-only conditions. A missing modality should not silently be interpreted as evidence of authenticity.

Absence also differs from low quality. A face may be detected but poorly resolved; audio may exist but contain little intelligible speech. A useful extension is to test quality-aware fusion, although the quality estimate itself can introduce another shortcut. When available evidence cannot support the requested decision, an inconclusive output is preferable to an unsupported authenticity judgement. Its frequency should be reported as coverage, not hidden from performance summaries.

### 6.3 Identity and learned cross-modal relationships

Audio-Visual Person-of-Interest DeepFake Detection learns identity-related audio and moving-face representations using real talking-face material [20]. It detects deviation from a person’s genuine reference characteristics, reducing reliance on a particular forgery generator. The operational boundary is the need for trustworthy reference material and an identifiable subject. It addresses identity-conditioned authentication, which is a different deployment setting from inspecting an arbitrary unfamiliar person.

AVFF uses self-supervised learning on real videos to learn audio-visual correspondence, then supervised learning for classification [21]. Its representation learning combines contrastive and autoencoding objectives with complementary masking and fusion. The paper reports 98.6% accuracy and 99.1% AUC on FakeAVCeleb. These are results under its evaluation setting; they are not predicted outcomes for this project or evidence of equivalent performance on circulating media.

The difference from embedding concatenation is substantive. AVFF’s training objective teaches relationships between modalities before the classification stage. By contrast, the present project’s fusion head receives already pooled face and voice features. It should not be called a lip-synchronisation detector without evidence that its inputs and objective preserve and test synchronised speech information.

### 6.4 Partial manipulation and temporal localisation

LAV-DF introduces content-driven edits that may occupy only part of a recording, together with boundary-aware temporal detection methods [22]. Its task exposes a weakness of whole-video labels: identifying a suspicious recording does not show which interval changed. Localisation requires temporal annotations and interval-level evaluation. A classification score or spatial face heatmap cannot substitute for that evidence.

AV-Deepfake1M extends content-driven audio, visual, and joint manipulation to more than one million videos involving over 2,000 subjects [23]. The authors report substantial difficulty for existing detection and localisation methods. Its scale and partial edits make it a useful later evaluation target. A resource-limited project can use a documented subset, but subset findings should not be represented as official full-benchmark results.

DiMoDif compares information from audio and visual speech representations to identify and localise discrepancies [24]. AuViRe reconstructs speech representations across modalities and uses reconstruction differences for temporal localisation [25]. They exemplify a shift towards whether the modalities convey compatible information. This review infers that legitimate dubbing, transmission delay, and limited mouth visibility require explicit controls, since disagreement alone is not sufficient evidence of malicious synthesis.

ExposeAnyone, a 2026 preprint, personalises an audio-to-expression diffusion model using subject reference material and evaluates suspected videos through reconstruction discrepancies [26]. It extends the identity-conditioned direction and reports robustness and unseen-generation experiments. Its reference requirement remains central to interpretation. The emerging lesson is to compare systems by available evidence and deployment assumptions before comparing scores.

## 7. Robustness, Explanation, and Decision Reliability

### 7.1 External validity and degradation

Deepfake-Eval-2024 collects circulating media from many websites and languages, and reports large performance losses for evaluated open-source detectors relative to earlier academic benchmarks [27]. The reviewed version was revised in May 2026. It supplies evidence that benchmark success does not necessarily transfer to contemporary media. Its outcomes concern the evaluated systems and collection; they should not be presented as a universal error estimate for all detectors.

Robustness has several distinct dimensions: signal corruption, unseen generators, unfamiliar identities, different acquisition conditions, and missing evidence. Compression augmentation mainly addresses the first. It cannot establish the others. For this project, a useful stress test varies JPEG or video compression, resizing, blur, noise, and audio encoding while keeping the test recordings fixed. Severity settings and per-condition scores should be disclosed.

Where a degradation is applied matters. Compressing extracted face crops evaluates the classifier’s input sensitivity. Compressing a full video before face detection also tests whether detection and cropping fail. Those experiments support different claims. End-to-end testing should include unsuccessful preprocessing cases, or report them separately, so that accuracy on surviving inputs does not conceal reduced coverage.

### 7.2 Explanations and their limits

Grad-CAM uses target gradients at a convolutional layer to produce a coarse map of regions associated with a model output [28]. It is useful for inspecting reliance on face boundaries or spectrogram regions. Its spatial resolution is limited by the feature map, and an upsampled display does not create additional localisation evidence. On spectrograms, highlighted regions indicate model sensitivity; they do not independently identify a physical synthesis mechanism.

Adebayo and colleagues show that visual appeal alone is insufficient for evaluating saliency methods and propose checks based on model and data randomisation [29]. The implication for deepfake analysis is to test whether explanations change when learned parameters or relevant inputs change. Correct predictions, false positives, and false negatives all belong in the explanation gallery. Selecting only persuasive examples would prevent a useful assessment of failure.

A practical evaluation can compare the effect of perturbing highly highlighted regions with matched low-highlight or random regions. Such tests need controls because masking itself can create unfamiliar inputs. They provide evidence about explanation behaviour, not complete causal proof. In fusion, separate explanations should also be checked against removal of each modality to see whether the apparent contribution agrees with the score change.

### 7.3 Calibration, thresholds, and fair comparison

Guo and colleagues document miscalibration in modern neural networks and study calibration methods including temperature scaling [30]. A detector’s output should therefore be called a score or estimated probability unless calibration has been assessed. For fusion, this matters before scores are averaged or combined: two branches with similar numerical outputs may have different error frequencies.

AUC assesses ranking across thresholds. EER describes the point at which false acceptance and false rejection rates coincide under the evaluation convention. Neither determines an appropriate operational threshold. Precision also depends on the prevalence of manipulated content. Thresholds should be chosen on separate validation data according to the intended costs, then held fixed for testing. A threshold derived from test labels is descriptive and should not be presented as a deployable result.

Subgroup and per-attack reporting are similarly necessary. Identity, language, recording channel, and modality combination can expose weaknesses hidden by an average. Estimates should include sample counts and, where feasible, uncertainty intervals based on recordings rather than correlated frames. Compute comparisons should report preprocessing, latency, memory, and hardware alongside model quality.

## 8. Comparative Synthesis

Table 2 compares selected contributions without ranking incompatible performance numbers. The final column records this review’s interpretation of each method’s principal evaluation boundary.

Table 2. Representative methods and their practical boundaries.

| Study | Evidence or representation | Main contribution | Principal boundary |
|---|---|---|---|
| MesoNet [7] | Mesoscopic face features | Compact visual baseline | Early manipulation settings |
| Face X-ray [10] | Blending boundaries | Shared forgery-operation cue | Requires relevant blending evidence |
| Self-Blended Images [11] | Constructed image inconsistencies | Generalisation-oriented training data | Synthetic artefacts do not cover all generators |
| LipForensics [12] | Mouth-motion representations | Temporal transfer from lipreading | Requires informative visible motion |
| RawNet2 [14] | Waveform features | Learned audio front end | Channel and attack transfer need testing |
| AASIST [15] | Spectral-temporal graph features | Integrated audio countermeasure | Benchmark success has a defined protocol |
| wav2vec 2.0 countermeasure [16] | Self-supervised speech features | Pretraining with augmentation | Corpus, language, and compute constraints |
| POI detection [20] | Identity-conditioned face and voice | Generator-independent reference comparison | Needs trustworthy subject references |
| AVFF [21] | Learned audio-visual correspondence | Self-supervision followed by classification | Evaluation depends on data and split conditions |
| LAV-DF / AV-Deepfake1M [22], [23] | Partially edited paired media | Localisation-oriented benchmarks | Subsets and clip labels change the task |
| DiMoDif / AuViRe [24], [25] | Speech discrepancy and reconstruction | Cross-modal temporal localisation | Alignment and legitimate mismatch controls |
| ExposeAnyone [26] | Personalised expression reconstruction | Recent self-supervised POI direction | Reference access; preprint evidence |

Three conclusions follow from the comparison. First, representation matters most when its assumptions fit the attack: blending, motion, waveform structure, identity, and cross-modal agreement are different evidence sources. Second, learning from genuine speech or video can reduce dependence on enumerating known generators, but introduces pretraining and reference-data assumptions. Third, much of the credible contribution lies in the evaluation protocol. A small detector assessed honestly across failure conditions can support clearer conclusions than a larger model with an ambiguous split.

## 9. Research Gaps and Implications for the Existing Project

### 9.1 Priority gaps

The first gap is reliable coverage of single-modality manipulation. An overall score should be evaluated separately on genuine pairs, fake-audio-only, fake-video-only, and jointly manipulated pairs. Otherwise, a favourable average could arise from detecting whichever component dominates the collection. A fusion improvement needs comparison with both branches on exactly the same held-out recordings.

The second gap is generalisation under combined changes. Testing one codec at a time is useful, but deployed inputs may combine unfamiliar generation, noise, compression, and a new recording channel. A staged evaluation can begin with individual degradations and then test a small, declared set of combinations. This is more interpretable than creating a large unstructured augmentation suite.

The third gap is separation of genuine disagreement from forgery. A correspondence-based detector needs benign asynchronous, dubbed, or weakly visible speech examples. The fourth is useful uncertainty: presence flags handle unavailable inputs, but additional evidence is needed for low-quality inputs and out-of-domain scores. The fifth is explanation assessment linked to actual decision changes rather than persuasive visual presentation.

These are project-level priorities inferred from the reviewed evidence, not claims that the entire literature has ignored them. Establishing a new state-of-the-art architecture would require broader experiments than the current scope supports.

### 9.2 Current evidence and defensible contribution

The repository uses an EfficientNet-B0 visual branch, a ResNet-18 log-mel audio branch, simple score fusion, and an embedding-based fusion head with modality dropout. Its stored ASVspoof experiment reports test AUC 0.9771 and EER 6.20% on a balanced 5,000-clip subset. At threshold 0.5, recall for spoofed speech is 83.24%; the per-attack file shows approximately 1.17% detection for A17. These figures were checked against the saved metric and per-category files; this review did not rerun training.

The discrepancy between aggregate AUC and A17 detection motivates attack-level reporting and threshold analysis. It does not demonstrate face detection, paired-data transfer, or successful fusion. The repository documentation identifies real-data visual and fusion evaluation as pending. Nor should the saved test-derived EER threshold be treated as a threshold selected independently for deployment.

The project’s credible contribution is therefore a controlled experimental assessment of complementary face and voice baselines, simple versus learned fusion, degradation sensitivity, and explanation behaviour. The architecture is a practical implementation of established ideas. Novelty, if demonstrated, would lie in the particular evaluation findings and reproducibility, rather than in combining two familiar networks.

### 9.3 Recommended experimental sequence

First, verify source and identity grouping, preserve official partitions, and document subset construction. Second, establish both branches on paired data with modality-specific labels. Third, compare weighted, maximum, noisy-OR, and intermediate fusion using one validation procedure and a common test set. Include all four manipulation categories, missing-modality cases, and inconclusive coverage.

Fourth, evaluate a declared set of degradations before and after robustness training, separating crop-level from end-to-end processing. Fifth, perform external visual and audio evaluations without tuning on their test labels. Sixth, calibrate where justified using separate data and examine how thresholds affect false alarms and missed attacks. Finally, inspect explanation failures and report compute costs. Advanced temporal or correspondence models should be added only when these baselines identify a specific unresolved weakness.

## 10. Limitations of the Review and Responsible Use

This is a selective narrative review. It provides a technical framework and verified source references, but cannot establish exhaustive coverage or statistically pool heterogeneous experiments. Some recent studies are cited through arXiv versions; their status and version dates are recorded. The review’s method-level limitations are reasoned assessments unless explicitly described as reported results. No published experiments were independently reproduced.

Research datasets may contain sensitive faces and voices, so their access conditions, permitted uses, and consent context remain relevant to reproducibility. A detector’s error can falsely implicate a genuine speaker or miss an altered recording. Reports should preserve the inspected interval, processing conditions, decision threshold, and uncertainty. An absence of detected artefacts is limited evidence, particularly when one modality or part of a recording was never examined.

## 11. Conclusion

The literature supports face-and-voice analysis because different attacks alter different components of a recording. It also shows why a second branch is insufficient by itself: useful fusion depends on label semantics, input quality, calibration, and the relationship learned between modalities. Temporal speech and reconstruction methods broaden the available evidence, while localisation benchmarks expose shortcomings of whole-recording classification. For the existing project, the strongest next contribution is a reproducible evaluation of simple and learned fusion across manipulation categories, unseen conditions, and preprocessing failures. Claims of robustness and explanation should follow those measurements and retain their experimental boundaries.

## References

[1] A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, “FaceForensics++: Learning to detect manipulated facial images,” in Proc. IEEE/CVF ICCV, 2019. [Online]. Available: https://arxiv.org/abs/1901.08971

[2] Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, “Celeb-DF: A large-scale challenging dataset for DeepFake forensics,” in Proc. IEEE/CVF CVPR, 2020, pp. 3207–3216. [Online]. Available: https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Celeb-DF_A_Large-Scale_Challenging_Dataset_for_DeepFake_Forensics_CVPR_2020_paper.html

[3] B. Dolhansky et al., “The DeepFake Detection Challenge (DFDC) dataset,” arXiv:2006.07397, 2020. [Online]. Available: https://arxiv.org/abs/2006.07397

[4] H. Khalid, S. Tariq, M. Kim, and S. Woo, “FakeAVCeleb: A novel audio-video multimodal deepfake dataset,” in Proc. NeurIPS Datasets and Benchmarks, vol. 1, 2021. [Online]. Available: https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/d9d4f495e875a2e075a1a4a6e1b9770f-Abstract-round2.html

[5] M. Todisco et al., “ASVspoof 2019: Future horizons in spoofed and fake audio detection,” in Proc. Interspeech, 2019. [Online]. Available: https://arxiv.org/abs/1904.05441

[6] Z. Yan, Y. Zhang, X. Yuan, S. Lyu, and B. Wu, “DeepfakeBench: A comprehensive benchmark of deepfake detection,” in Proc. NeurIPS Datasets and Benchmarks, 2023. [Online]. Available: https://arxiv.org/abs/2307.01426

[7] D. Afchar, V. Nozick, J. Yamagishi, and I. Echizen, “MesoNet: A compact facial video forgery detection network,” in Proc. IEEE WIFS, 2018, doi: 10.1109/WIFS.2018.8630761. [Online]. Available: https://arxiv.org/abs/1809.00888

[8] F. Chollet, “Xception: Deep learning with depthwise separable convolutions,” in Proc. IEEE CVPR, 2017. [Online]. Available: https://arxiv.org/abs/1610.02357

[9] M. Tan and Q. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in Proc. ICML, PMLR, vol. 97, 2019, pp. 6105–6114. [Online]. Available: https://proceedings.mlr.press/v97/tan19a.html

[10] L. Li et al., “Face X-ray for more general face forgery detection,” in Proc. IEEE/CVF CVPR, 2020. [Online]. Available: https://arxiv.org/abs/1912.13458

[11] K. Shiohara and T. Yamasaki, “Detecting deepfakes with self-blended images,” in Proc. IEEE/CVF CVPR, 2022. [Online]. Available: https://arxiv.org/abs/2204.08376

[12] A. Haliassos, K. Vougioukas, S. Petridis, and M. Pantic, “Lips don't lie: A generalisable and robust approach to face forgery detection,” in Proc. IEEE/CVF CVPR, 2021. [Online]. Available: https://arxiv.org/abs/2012.07657

[13] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE CVPR, 2016. [Online]. Available: https://arxiv.org/abs/1512.03385

[14] H. Tak, J. Patino, M. Todisco, A. Nautsch, N. Evans, and A. Larcher, “End-to-end anti-spoofing with RawNet2,” in Proc. IEEE ICASSP, 2021. [Online]. Available: https://arxiv.org/abs/2011.01108

[15] J. Jung et al., “AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks,” in Proc. IEEE ICASSP, 2022. [Online]. Available: https://arxiv.org/abs/2110.01200

[16] H. Tak, M. Todisco, X. Wang, J. Jung, J. Yamagishi, and N. Evans, “Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation,” arXiv:2202.12233, 2022. [Online]. Available: https://arxiv.org/abs/2202.12233

[17] D. S. Park et al., “SpecAugment: A simple data augmentation method for automatic speech recognition,” in Proc. Interspeech, 2019, pp. 2613–2617, doi: 10.21437/Interspeech.2019-2680. [Online]. Available: https://arxiv.org/abs/1904.08779

[18] T. Baltrušaitis, C. Ahuja, and L.-P. Morency, “Multimodal machine learning: A survey and taxonomy,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 41, no. 2, 2019. [Online]. Available: https://arxiv.org/abs/1705.09406

[19] N. Neverova, C. Wolf, G. W. Taylor, and F. Nebout, “ModDrop: Adaptive multi-modal gesture recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 38, no. 8, 2016. [Online]. Available: https://arxiv.org/abs/1501.00102

[20] D. Cozzolino, A. Pianese, M. Nießner, and L. Verdoliva, “Audio-visual person-of-interest DeepFake detection,” in Proc. IEEE/CVF CVPR, 2023. [Online]. Available: https://arxiv.org/abs/2204.03083

[21] T. Oorloff et al., “AVFF: Audio-visual feature fusion for video deepfake detection,” in Proc. IEEE/CVF CVPR, 2024, pp. 27102–27112. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2024/html/Oorloff_AVFF_Audio-Visual_Feature_Fusion_for_Video_Deepfake_Detection_CVPR_2024_paper.html

[22] Z. Cai, K. Stefanov, A. Dhall, and M. Hayat, “Do you really mean that? Content driven audio-visual deepfake dataset and multimodal method for temporal forgery localization,” arXiv:2204.06228, 2022. [Online]. Available: https://arxiv.org/abs/2204.06228

[23] Z. Cai et al., “AV-Deepfake1M: A large-scale LLM-driven audio-visual deepfake dataset,” arXiv:2311.15308, 2023. [Online]. Available: https://arxiv.org/abs/2311.15308

[24] C. Koutlis and S. Papadopoulos, “DiMoDif: Discourse modality-information differentiation for audio-visual deepfake detection and localization,” arXiv:2411.10193, ver. 2, Apr. 2025. [Online]. Available: https://arxiv.org/abs/2411.10193v2

[25] C. Koutlis and S. Papadopoulos, “AuViRe: Audio-visual speech representation reconstruction for deepfake temporal localization,” arXiv:2511.18993, Nov. 2025, author record identifies WACV 2026. [Online]. Available: https://arxiv.org/abs/2511.18993

[26] K. Shiohara, T. Yamasaki, and V. Golyanik, “ExposeAnyone: Personalized audio-to-expression diffusion models are robust zero-shot face forgery detectors,” arXiv:2601.02359, Jan. 2026, preprint. [Online]. Available: https://arxiv.org/abs/2601.02359

[27] N. A. Chandra et al., “Deepfake-Eval-2024: A multi-modal in-the-wild benchmark of deepfakes circulated in 2024,” arXiv:2503.02857, ver. 5, May 2026. [Online]. Available: https://arxiv.org/abs/2503.02857v5

[28] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. IEEE ICCV, 2017. [Online]. Available: https://arxiv.org/abs/1610.02391

[29] J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, “Sanity checks for saliency maps,” in Advances in Neural Information Processing Systems, vol. 31, 2018. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html

[30] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. ICML, PMLR, vol. 70, 2017, pp. 1321–1330. [Online]. Available: https://proceedings.mlr.press/v70/guo17a.html

All online references accessed 13 October 2026. Conference citations identify the publication year; linked arXiv records may have an earlier submission date.
