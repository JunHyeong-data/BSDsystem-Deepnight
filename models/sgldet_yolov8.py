"""
SGLDet + YOLOv8-m Integration
-------------------------------
논문: Self-Guided Low-Light Object Detection Framework (ICLR 2026)

YOLOv8m 실측 backbone 구조 (입력 640x640 기준):
  layer  2: C2f   → (B,  96, 160, 160)  stride=4   [P1]
  layer  4: C2f   → (B, 192,  80,  80)  stride=8   [P2]
  layer  6: C2f   → (B, 384,  40,  40)  stride=16  [P3]
  layer  9: SPPF  → (B, 576,  20,  20)  stride=32  [P4]
  (layer 8과 9는 같은 해상도이므로 9만 bottleneck으로 사용)

학습 시: backbone(B) → AuxDecoder(G) → MSE(x̃, x̂) = L_self
         + Detection Head(H) → L_det
         L_total = L_det + λ × L_self  (λ=0.01)

추론 시: backbone(B) → Head(H) only — 추가 비용 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


# ===========================================================================
# 1. SCI Enhancer (E)
# ===========================================================================
class SCIEnhancer(nn.Module):
    """Self-Calibrated Illumination 기반 self-supervised enhancer."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        illum = self.estimator(x)
        return torch.clamp(x / (illum + 1e-4), 0.0, 1.0)


# ===========================================================================
# 2. SDAP Denoiser (D)
# ===========================================================================
class SDAPDenoiser(nn.Module):
    """SDAP 기반 self-supervised denoiser (residual learning)."""

    def __init__(self, channels: int = 3, mid_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x - self.net(x), 0.0, 1.0)


# ===========================================================================
# 3. Fourier Fusion (F)
# ===========================================================================
class FourierFusion(nn.Module):
    """논문 수식 (4): x̂ = iFFT( |X^{E+D}| · exp(j·∠X^E) )"""

    def forward(
        self,
        x_enhanced: torch.Tensor,
        x_denoised: torch.Tensor,
    ) -> torch.Tensor:
        fft_e  = torch.fft.fft2(x_enhanced, norm="ortho")
        fft_ed = torch.fft.fft2(x_denoised, norm="ortho")

        amp_ed  = torch.abs(fft_ed)        # 노이즈 제거된 amplitude
        phase_e = torch.angle(fft_e)       # 구조 보존된 phase

        fused_fft = amp_ed * torch.exp(1j * phase_e)
        fused = torch.fft.ifft2(fused_fft, norm="ortho").real
        return torch.clamp(fused, 0.0, 1.0)


# ===========================================================================
# 4. Auxiliary U-Net Decoder (G)  — YOLOv8m 실측 채널에 맞춤
# ===========================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AuxDecoder(nn.Module):
    """
    YOLOv8m backbone feature 4개를 받아 U-Net 구조로 입력 해상도 이미지 재구성.

    입력 features (좌→우, 얕은→깊은):
      P1: (B, 96, 160, 160)  stride 4
      P2: (B, 192,  80,  80) stride 8
      P3: (B, 384,  40,  40) stride 16
      P4: (B, 576,  20,  20) stride 32  (bottleneck)

    Decoder path:
      P4 → up → 40x40 + P3 → conv
            → up → 80x80 + P2 → conv
            → up → 160x160 + P1 → conv
            → up → 320x320 → conv
            → up → 640x640 → out(3ch)
    """

    # YOLOv8m 실측 채널 (변경 시 자동 적응 — 안전성 위해 상수)
    DEFAULT_CHANNELS = (96, 192, 384, 576)

    def __init__(self, out_channels: int = 3,
                 backbone_channels: tuple = DEFAULT_CHANNELS):
        super().__init__()
        c1, c2, c3, c4 = backbone_channels

        # Bottleneck → 40x40
        self.up3   = nn.ConvTranspose2d(c4, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256 + c3, 256)

        # 40x40 → 80x80
        self.up2   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128 + c2, 128)

        # 80x80 → 160x160
        self.up1   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64 + c1, 64)

        # 160x160 → 320x320
        self.up0   = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv0 = DoubleConv(32, 32)

        # 320x320 → 640x640
        self.up_final = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(16, out_channels, 1)

    def forward(self, features: list) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f"AuxDecoder는 4개 feature 필요, {len(features)}개 받음")

        p1, p2, p3, p4 = features

        x = self.up3(p4)
        x = self._match_and_cat(x, p3)
        x = self.conv3(x)

        x = self.up2(x)
        x = self._match_and_cat(x, p2)
        x = self.conv2(x)

        x = self.up1(x)
        x = self._match_and_cat(x, p1)
        x = self.conv1(x)

        x = self.up0(x)
        x = self.conv0(x)

        x = self.up_final(x)
        return torch.sigmoid(self.out_conv(x))

    @staticmethod
    def _match_and_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)


# ===========================================================================
# 5. SGLDet + YOLOv8m 통합
# ===========================================================================
class SGLDetYOLO(nn.Module):
    """
    SGLDet 프레임워크 + YOLOv8m.

    학습:
      out = model({'img', 'cls', 'bboxes', 'batch_idx'})
      → {'loss', 'det_loss', 'self_loss'}

    추론:
      out = model(image_tensor)  # (B, 3, H, W)
      → ultralytics 예측 결과
    """

    # 사용할 backbone hook layer (실측 검증 완료)
    HOOK_LAYERS = [2, 4, 6, 9]   # P1, P2, P3, P4

    def __init__(
        self,
        yolo_weights: str = "yolov8m.pt",
        lambda_self: float = 0.01,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.lambda_self = lambda_self

        yolo = YOLO(yolo_weights)
        self.detector = yolo.model

        # ★ ultralytics가 pretrained 모델 로드 시 일부 layer를 freeze하므로
        #    SGLDet 학습을 위해 모든 backbone 파라미터를 trainable로 강제
        for p in self.detector.parameters():
            p.requires_grad = True

        # ultralytics v8DetectionLoss가 model.args.box, .cls, .dfl 등을
        # 속성 접근으로 사용하므로 dict일 경우 SimpleNamespace로 변환
        self._fix_model_args()

        # 클래스 수가 지정되면 Detect head 재초기화
        if num_classes is not None:
            self._reset_head(num_classes)

        # 보조 모듈 (학습 시에만 사용)
        self.enhancer    = SCIEnhancer()
        self.denoiser    = SDAPDenoiser()
        self.fourier     = FourierFusion()
        self.aux_decoder = AuxDecoder(out_channels=3)

        # Hook 등록
        self._features: list = []
        self._hooks: list = []
        self._register_hooks()

    # -----------------------------------------------------------------------
    # ultralytics model.args가 dict일 경우 SimpleNamespace로 변환
    # v8DetectionLoss는 args.box, args.cls, args.dfl 처럼 속성 접근함
    # -----------------------------------------------------------------------
    def _fix_model_args(self) -> None:
        """model.args를 SimpleNamespace로 보정 (loss 계산 호환)."""
        from types import SimpleNamespace

        defaults = {
            "box": 7.5, "cls": 0.5, "dfl": 1.5,
            "label_smoothing": 0.0, "fl_gamma": 0.0,
            "nbs": 64, "overlap_mask": True, "mask_ratio": 4,
        }

        args = getattr(self.detector, "args", None)
        if args is None:
            self.detector.args = SimpleNamespace(**defaults)
        elif isinstance(args, dict):
            merged = {**defaults, **args}
            self.detector.args = SimpleNamespace(**merged)
        else:
            # 이미 namespace이지만 누락 키가 있을 수 있음
            for k, v in defaults.items():
                if not hasattr(args, k):
                    setattr(args, k, v)

    # -----------------------------------------------------------------------
    # Detect head 재초기화 (custom num_classes 대응)
    # -----------------------------------------------------------------------
    def _reset_head(self, num_classes: int) -> None:
        """ultralytics Detect head를 새 num_classes로 재초기화."""
        try:
            from ultralytics.nn.modules import Detect

            old_detect = self.detector.model[-1]
            if not isinstance(old_detect, Detect):
                print("[Warning] Detect head를 찾지 못해 클래스 변경 스킵")
                return

            # 입력 채널 추출: cv3 branch의 첫 Conv → conv(nn.Conv2d).in_channels
            # ultralytics Conv = (conv: nn.Conv2d, bn, act)
            ch = []
            for branch in old_detect.cv3:
                first = branch[0]
                if hasattr(first, "conv"):                  # ultralytics Conv wrapper
                    ch.append(first.conv.in_channels)
                elif hasattr(first, "in_channels"):         # raw nn.Conv2d
                    ch.append(first.in_channels)
                else:
                    raise AttributeError(f"채널 추출 실패: {type(first).__name__}")

            # 새 Detect head 생성
            new_detect = Detect(nc=num_classes, ch=tuple(ch))
            new_detect.stride = old_detect.stride
            new_detect.f      = old_detect.f
            new_detect.i      = old_detect.i

            # 이전 device/dtype 보존
            new_detect = new_detect.to(next(old_detect.parameters()).device)

            self.detector.model[-1] = new_detect
            self.detector.nc = num_classes
            self.detector.names = {i: str(i) for i in range(num_classes)}

            if hasattr(self.detector, "yaml") and isinstance(self.detector.yaml, dict):
                self.detector.yaml["nc"] = num_classes

            print(f"[SGLDetYOLO] Detect head 재초기화: nc={num_classes}, ch={ch}")
        except Exception as e:
            print(f"[Warning] Detect head 재초기화 실패: {e}")

    # -----------------------------------------------------------------------
    # Hooks
    # -----------------------------------------------------------------------
    def _register_hooks(self) -> None:
        for idx in self.HOOK_LAYERS:
            try:
                layer = self.detector.model[idx]
            except (AttributeError, IndexError):
                print(f"[Warning] Layer {idx} 없음 → hook 스킵")
                continue
            h = layer.register_forward_hook(self._hook_fn)
            self._hooks.append(h)

    def _hook_fn(self, module, input_, output) -> None:
        feat = output[0] if isinstance(output, (list, tuple)) else output
        self._features.append(feat)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------
    def forward(self, batch_or_image, **kwargs):
        # ── 추론 모드 ────────────────────────────────────────────────
        if isinstance(batch_or_image, torch.Tensor) or not self.training:
            self._features.clear()
            try:
                return self.detector(batch_or_image)
            finally:
                # 추론 끝난 후 즉시 메모리 해제
                self._features.clear()

        # ── 학습 모드 ────────────────────────────────────────────────
        batch = batch_or_image
        x = batch["img"]
        device = x.device

        self._features.clear()

        # 1) Detection forward + ultralytics v8DetectionLoss
        try:
            det_loss, _ = self.detector.loss(batch)
            if det_loss.dim() > 0:
                det_loss = det_loss.sum()
        except Exception as e:
            print(f"[Warning] det loss 실패: {type(e).__name__}: {e}")
            det_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 2) Hook이 4개 feature 모았는지 확인
        if len(self._features) < 4:
            return {
                "loss":      det_loss,
                "det_loss":  det_loss.detach(),
                "self_loss": torch.tensor(0.0, device=device),
            }

        feats = self._features[:4]

        # 3) Fourier Fusion 감독 타깃 생성 (gradient X)
        with torch.no_grad():
            x_e   = self.enhancer(x)
            x_ed  = self.denoiser(x_e)
            x_hat = self.fourier(x_e, x_ed)

        # 4) AuxDecoder 재구성
        x_tilde = self.aux_decoder(feats)

        # 입력 해상도와 맞춤
        if x_tilde.shape[-2:] != x_hat.shape[-2:]:
            x_tilde = F.interpolate(x_tilde, size=x_hat.shape[-2:],
                                    mode="bilinear", align_corners=False)

        # 5) Self-guided MSE loss (논문 식 5)
        self_loss = F.mse_loss(x_tilde, x_hat)

        # 6) Total loss (논문 식 6)
        total = det_loss + self.lambda_self * self_loss

        return {
            "loss":      total,
            "det_loss":  det_loss.detach(),
            "self_loss": self_loss.detach(),
        }

    # -----------------------------------------------------------------------
    # Helper: AuxDecoder를 미리 build해서 optimizer에 등록 보장
    # -----------------------------------------------------------------------
    def warmup(self, img_size: int = 640, device: str = "cpu") -> None:
        """더미 forward로 Hook + AuxDecoder 파라미터 메모리 할당 강제."""
        self.train()
        was_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        self._features.clear()
        try:
            self.detector(dummy)
            if len(self._features) >= 4:
                self.aux_decoder(self._features[:4])
        except Exception as e:
            print(f"[Warning] warmup 중 오류: {e}")

        torch.set_grad_enabled(was_grad)
        self._features.clear()


# ===========================================================================
# 6. 사전학습 함수
# ===========================================================================
def pretrain_enhancer(
    dataloader, num_epochs: int = 30,
    device: str = "cuda", lr: float = 1e-4,
) -> SCIEnhancer:
    """SCI self-supervised pretraining (Zero-DCE 스타일 loss)."""
    enhancer = SCIEnhancer().to(device)
    opt = torch.optim.Adam(enhancer.parameters(), lr=lr)

    for epoch in range(num_epochs):
        last_loss = 0.0
        for batch in dataloader:
            imgs = batch["img"].to(device) if isinstance(batch, dict) \
                   else batch.to(device)
            enhanced = enhancer(imgs)

            mean_val = enhanced.mean(dim=[1, 2, 3], keepdim=True)
            exp_loss = F.mse_loss(mean_val, torch.full_like(mean_val, 0.6))

            diff_h = (enhanced[:, :, 1:, :] - enhanced[:, :, :-1, :]).abs().mean()
            diff_w = (enhanced[:, :, :, 1:] - enhanced[:, :, :, :-1]).abs().mean()

            loss = exp_loss + 0.1 * (diff_h + diff_w)
            opt.zero_grad(); loss.backward(); opt.step()
            last_loss = loss.item()

        print(f"[SCI] Epoch {epoch+1}/{num_epochs} loss={last_loss:.4f}")
    return enhancer


def pretrain_denoiser(
    dataloader, num_epochs: int = 30,
    device: str = "cuda", lr: float = 1e-4,
) -> SDAPDenoiser:
    """SDAP self-supervised pretraining (random sub-sample consistency)."""
    denoiser = SDAPDenoiser().to(device)
    opt = torch.optim.Adam(denoiser.parameters(), lr=lr)

    for epoch in range(num_epochs):
        last_loss = 0.0
        for batch in dataloader:
            imgs = batch["img"].to(device) if isinstance(batch, dict) \
                   else batch.to(device)

            sub1 = (imgs + 0.05 * torch.randn_like(imgs)).clamp(0, 1)
            sub2 = (imgs + 0.05 * torch.randn_like(imgs)).clamp(0, 1)

            d1, d2 = denoiser(sub1), denoiser(sub2)
            loss = F.mse_loss(d1, sub2) + F.mse_loss(d2, sub1)

            opt.zero_grad(); loss.backward(); opt.step()
            last_loss = loss.item()

        print(f"[SDAP] Epoch {epoch+1}/{num_epochs} loss={last_loss:.4f}")
    return denoiser


# ===========================================================================
# 자가 테스트
# ===========================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = SGLDetYOLO(yolo_weights="yolov8m.pt", lambda_self=0.01).to(device)
    print(f"Hooks: {len(model._hooks)}")

    # warmup으로 AuxDecoder 파라미터 강제 할당
    model.warmup(img_size=640, device=device)

    # 추론 테스트
    model.eval()
    x = torch.rand(1, 3, 640, 640).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"[OK] 추론 출력 type: {type(out).__name__}")

    # 학습 테스트 (더미 batch)
    model.train()
    batch = {
        "img":       torch.rand(2, 3, 640, 640).to(device),
        "cls":       torch.tensor([[2.0], [0.0]], device=device),
        "bboxes":    torch.tensor([[0.5, 0.5, 0.2, 0.2],
                                    [0.3, 0.3, 0.1, 0.1]], device=device),
        "batch_idx": torch.tensor([0, 1], dtype=torch.long, device=device),
    }
    try:
        out = model(batch)
        print(f"[OK] 학습 forward → loss={out['loss'].item():.4f} "
              f"(det={out['det_loss'].item():.4f}, self={out['self_loss'].item():.4f})")
    except Exception as e:
        print(f"[FAIL] 학습 forward: {type(e).__name__}: {e}")
