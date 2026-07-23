안녕하세요 2조입니다.

## Setup

모델 가중치(`models/`)는 용량이 커서(30GB+) git에 올리지 않습니다. 각자 아래로 받으세요.

```bash
pip install -r requirements.txt
python scripts/download_models.py        # 전체 다운로드
python scripts/download_models.py qwen   # Qwen2.5-VL-7B-Instruct만
python scripts/download_models.py llava  # LLaVA-OneVision-7B-OV만
```

자세한 파이프라인/설계는 [docs/PRD.md](docs/PRD.md) 참고.
