# Google Colab 클라우드 학습 가이드 (MixIT - FUSS)

이 디렉토리(`colab/`)는 **MixIT (`models/neurips2020_mixit`)** 음원 분리 모델을 **Google Colab** 클라우드 GPU 환경에서 학습, 평가 및 추론할 수 있도록 구성한 설정 파일과 노트북을 포함하고 있습니다.

---

## 📁 구성 파일 안내

| 파일 | 설명 |
| :--- | :--- |
| [`MixIT_Colab_Training.ipynb`](./MixIT_Colab_Training.ipynb) | Google Colab에서 바로 열어 실행할 수 있는 **통합 학습/모니터링/추론/평가 노트북** |
| [`setup_colab.sh`](./setup_colab.sh) | Colab 런타임에 필요한 시스템 패키지(`aria2`, `sox`) 및 Python 의존성 설치, TensorFlow v1 호환성 검증 스크립트 |
| [`download_fuss_local.sh`](./download_fuss_local.sh) | FUSS 데이터셋(`ssdata.tar.gz`, 약 6.4GB)을 Colab 로컬 디스크(`/content/fuss_data`)에 멀티커넥션(`aria2c`)으로 고속 다운로드 후 아카이브 원본을 자동 삭제하여 디스크 공간 확보 |
| [`create_dummy_fuss.py`](./create_dummy_fuss.py) | 6.4GB 전체 데이터를 받기 전, 5초 만에 파이프라인 정상 동작 여부를 테스트할 수 있는 합성 FUSS 데이터 생성 스크립트 |
| [`requirements_colab.txt`](./requirements_colab.txt) | Colab 환경용 Python 필수 패키지 목록 |

---

## 🔧 적용된 Colab 최신 환경(Python 3.11+ / TensorFlow 2.16+) 호환성 패치

기존 Google Research의 코드는 2020년 기준 **Python 3.6 + TensorFlow 1.15** 환경에 맞춰 작성되어 있어 현재 Google Colab에서 그대로 실행하면 여러 오류가 발생합니다. 이를 해결하기 위해 다음 패치가 코드베이스에 적용되었습니다.

1. **`tf.estimator` 제거 대응 ([`models/train/tf_estimator_compat.py`](../models/train/tf_estimator_compat.py))**:
   - TensorFlow 2.16+에서 삭제된 `tf.compat.v1.estimator`를 대체하는 세션 기반(`tf.compat.v1.Session`) 호환 레이어를 추가하여 최신 Colab TensorFlow에서도 다운그레이드 없이 학습/평가가 동작합니다.
   - Google Drive의 체크포인트 디렉토리에 기존 체크포인트가 있으면 **자동으로 마지막 스텝부터 이어서 학습(Auto-Resume)**합니다.
2. **Keras 3 충돌 방지 및 Legacy Keras(`tf_keras`) 연동**:
   - `TF_USE_LEGACY_KERAS=1` 및 `tf_keras` 폴백을 적용하고, `LayerNormalizationScalarParams`의 `_fused` 속성 충돌과 `.apply()` 호출 문제를 수정했습니다.
3. **Python 3.12 `distutils` 제거 대응 ([`models/tools/inference.py`](../models/tools/inference.py))**:
   - Python 3.12에서 삭제된 `distutils.util.strtobool` 의존성을 순수 Python 함수로 교체했습니다.
4. **Colab RAM(12.7GB) OOM 방지 및 CLI 인자 확장 ([`models/neurips2020_mixit/train_model_on_fuss.py`](../models/neurips2020_mixit/train_model_on_fuss.py))**:
   - 기존 코드에 하드코딩되어 있던 `parallel_readers=512`(약 2.6GB+ 프리페치 버퍼 점유)를 기본값 `16`으로 조정하고, `--batch_size`, `--train_steps`, `--save_checkpoints_secs`, `--save_summary_steps` 등을 CLI 옵션으로 제어할 수 있게 개선했습니다.

---

## 🚀 사용 방법 (3단계)

### 1단계: 코드 업로드 (GitHub 또는 Google Drive)
- **방법 A (GitHub 권장)**: 현재 수정된 프로젝트를 본인의 GitHub 저장소에 Push합니다.
- **방법 B (Google Drive)**: 현재 `sound-separation` 폴더를 Google Drive의 `내 드라이브/sound-separation`(`/content/drive/MyDrive/sound-separation`)에 업로드합니다 (단, `papers/` 폴더는 학습에 불필요하므로 제외하고 업로드하면 훨씬 빠릅니다).

### 2단계: Colab에서 노트북 열기
1. [Google Colab](https://colab.research.google.com/)에 접속합니다.
2. `colab/MixIT_Colab_Training.ipynb` 파일을 업로드하여 엽니다.
3. 상단 메뉴에서 **`런타임` -> `런타임 유형 변경` -> 하드웨어 가속기: `T4 GPU` (또는 `L4` / `A100 GPU`)**를 선택합니다.

### 3단계: 노트북 셀 순서대로 실행
1. **1~2번 섹션**: Google Drive를 마운트하고 코드 및 환경을 설정합니다.
2. **3번 섹션**:
   - 처음 테스트할 때는 `USE_DUMMY_DATA = True`로 설정하여 5초 만에 파이프라인을 검증해 보세요.
   - 본 학습 시 `USE_DUMMY_DATA = False`로 설정하면 `/content/fuss_data`에 FUSS 데이터셋이 고속 다운로드됩니다.
3. **4~5번 섹션**: TensorBoard를 켜고 MixIT 학습을 시작합니다. 체크포인트는 10분마다 Google Drive(`/content/drive/MyDrive/sound_separation/checkpoints/mixit_fuss_exp1`)에 자동 백업되며, 런타임이 초기화되어도 다시 실행하면 이어서 학습됩니다.

