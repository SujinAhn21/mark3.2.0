# Mark3.2.0 (HSHKD A-Mark)

Mark3.2 is the final capstone model for inter-floor noise audio classification, validated via semi-supervised learning and hard–soft hybrid knowledge distillation (HSHKD).
This version consolidates knowledge from Mark2.5 – Mark2.9 and was presented as an oral paper at CSA 2025.

Maintainer: SujinAhn21


## 1) What this repository is

Real residential audio contains mixed daily-life sounds, and public labels often do not cleanly separate inter-floor noise from other everyday noise.
Mark3.2 addresses this by distilling knowledge from a teacher ensemble into a single student model using a hard–soft hybrid objective:

- Hard loss: cross-entropy with ground-truth labels (labeled set)
- Soft loss: KL divergence to teacher soft targets (labeled + unlabeled set)
- Semi-supervised setup: labeled + unlabeled audio are mixed during training


## 2) Paper / Conference

Conference: The 17th International Conference on Computer Science and its Applications (CSA 2025)  
Manuscript ID: 0051  
Presentation type: Oral  

Title: HSHKD A-Mark: An Audio Classification Model with Hard-Soft Hybrid Knowledge Distillation  
Authors: Su Jin Ahn, Young Min Jeon, Jea Pil Ko, Ji Su Park  
Presenter: Su Jin Ahn  
Corresponding Author: Ji Su Park  

Proceedings / DOI:  
- TBD  


## 3) Reported results (from the CSA 2025 paper)

Multi-class noise-type classification:
- Overall accuracy: ~0.71
- Macro-F1: ~0.69

Binary classification (inter-floor noise vs other noise):
- Accuracy: ~0.86
- Precision: ~0.99
- Recall: ~0.84
- F1-score: ~0.91

Interpretation:
- Very high precision in the binary setting indicates low false-alarm rates, which is desirable for real-world deployment scenarios.


## 4) Dataset and labels

This work uses the AI-Hub inter-floor noise dataset.

Classes used in the paper:
- thumping
- water_sound
- drill_sound
- hammer_sound
- human_voice
- others

Training setup:
- Labeled set + unlabeled set (semi-supervised)

Evaluation setup:
- Balanced test set across the major classes
- Additional binary evaluation by grouping inter-floor-related classes into one category


## License

This project is licensed under the PolyForm Noncommercial License 1.0.0.  
Commercial use is not permitted.  
See the [LICENSE](./LICENSE) file for details.


## Korean summary

Mark3.2는 실제 거주 환경의 층간소음/생활소음 분류를 목표로 한 오디오 분류 모델입니다.  
Mark2.5 ~ 2.9 에서 추출한 지식을 teacher 앙상블로 구성하고, hard label(정답)과 teacher soft output(확률 분포)을 함께 학습하는 hard–soft hybrid knowledge distillation(HSHKD)로 단일 student 모델을 학습했습니다.  
본 연구는 CSA 2025에서 Oral로 발표되었습니다(Manuscript ID: 0051).  
본 레포지토리는 연구 결과를 기반으로 한 모델 구현과 실험 맥락을 정리하는 데 목적이 있습니다.
