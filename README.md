# GraDeNAR: Graph-based DeNoising and Artifact Removal Network for Optical Coherence Tomograph
IEEE Transactions on Image ProcessingIEEE Transactions on Image Processing
[MANUSCRIPT UNDER REVIEW]

Abstract

Optical Coherence Tomography (OCT) imaging is rising for its significant advantages over traditional methods in studying the cross-sections of tissue microstructures. OCT imaging offers in-situ and real-time tissue imaging without the complications associated with excisional biopsy. However, the noise and artifacts induced during the imaging process warrant multiple scans, causing time delays and rendering OCT scan- based medical diagnosis less effective. While minor denoising can still be achieved at a single frame level, the reliability of reconstructed regions in a frame initially affected by artifacts based on single frame data remains a question. As OCT imaging is volumetric (3D) in nature, we propose a Graph-based De- Noising and Artifact Removal network (GraDeNAR) that takes advantage of features from neighboring scan frames. It exploits the local and non-local relations in the neighborhood aggregated latent features to effectively denoise and reconstruct regions affected by noise and artifacts. Qualitative and quantitative analysis of the network’s performance on our rat-colon OCT dataset proves that the network outperforms existing state- of-the-art models. Additionally, the network’s performance is quantitatively validated on other 3D medical and non-medical datasets, demonstrating the network’s robustness in denoising and artifact removal tasks.

The generalized terminal command with ablation choices to run the evaluation script are as follows:

python evaluate.py --width <> --step <> --noisy-temporal <> --pretrained <> --feature-extractor <> --ablation <> --input <> --patch-dim <> --pre <> --LPF <> --HPF <>

Note: Pretrained weights and training codes will be added on acceptance of our paper to journal.
