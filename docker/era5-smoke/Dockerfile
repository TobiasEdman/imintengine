FROM python@sha256:5be45dbade29bebd6886af6b438fd7e0b4eb7b611f39ba62b430263f82de36d2

ARG BASE_GIT_SHA=602c86dee8d4dbdc191ab2750c261cf86b56ac50
ARG BASE_ARCHIVE_SHA256=85976165d13f07c0654e37be5f05be76926cc1a623bea11275a079a0ef7b5db1

RUN python -m pip install --no-cache-dir \
      torch==2.6.0 torchvision==0.21.0 \
      einops==0.8.2 huggingface_hub==0.30.2 numpy==2.2.4 \
      psutil==7.0.0 pyproj==3.7.1 requests==2.32.3 \
      scikit-learn==1.6.1 scipy==1.15.2 timm==1.0.15 \
      --index-url https://download.pytorch.org/whl/cu124 \
      --extra-index-url https://pypi.org/simple

ADD https://codeload.github.com/TobiasEdman/ImintEngine/tar.gz/602c86dee8d4dbdc191ab2750c261cf86b56ac50 /imintengine-source.tar.gz
RUN echo "${BASE_ARCHIVE_SHA256}  /imintengine-source.tar.gz" | sha256sum -c - && \
    tar -xzf /imintengine-source.tar.gz -C /opt && \
    mv "/opt/imintengine-${BASE_GIT_SHA}" /opt/imintengine && \
    printf '%s\n' "${BASE_GIT_SHA}" > /opt/imintengine/.base_git_sha && \
    rm /imintengine-source.tar.gz

RUN python -c "import sys; from pathlib import Path; sys.path.insert(0, '/opt/imintengine'); import einops, numpy, pyproj, requests, scipy, sklearn, timm, torch, torchvision; from imint.fm.prithvi_mae.prithvi_mae import PrithviMAE; from imint.fm.upernet import ViTUPerNetHead; assert Path('/opt/imintengine/.base_git_sha').read_text().strip() == '${BASE_GIT_SHA}'; torch.use_deterministic_algorithms(True, warn_only=False); x = torch.randn(2, 64, 64, requires_grad=True); projection = torch.randn(64, 32); (x @ projection).square().mean().backward(); assert x.grad is not None and torch.isfinite(x.grad).all(); print({'status': 'ok', 'device': 'cpu', 'torch': torch.__version__, 'cuda': torch.version.cuda})"

WORKDIR /opt/imintengine
CMD ["python", "-c", "import torch; print(torch.__version__, torch.version.cuda)"]
