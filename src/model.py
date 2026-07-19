"""Text, EEG, and multimodal sentiment classifiers."""

import torch
import torch.nn as nn
from transformers import AutoModel

from .config import NUM_CLASSES
from .features import N_FAMILIES


class EEGSetEncoder(nn.Module):
    """Encode electrodes per subject, then pool the available readers."""

    def __init__(self, n_channels, channel_dim=32, eeg_dim=64, dropout=0.3):
        super().__init__()
        self.n_channels = n_channels
        self.channel_norm = nn.LayerNorm(N_FAMILIES)
        self.channel_encoder = nn.Sequential(
            nn.Linear(N_FAMILIES, channel_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_dim, channel_dim),
            nn.GELU(),
        )
        self.channel_embedding = nn.Parameter(torch.empty(n_channels, channel_dim))
        nn.init.normal_(self.channel_embedding, std=0.02)
        self.channel_score = nn.Linear(channel_dim, 1)
        self.subject_projection = nn.Sequential(
            nn.LayerNorm(channel_dim),
            nn.Linear(channel_dim, eeg_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, eeg, subject_mask):
        batch, subjects, features = eeg.shape
        expected = self.n_channels * N_FAMILIES
        if features != expected:
            raise ValueError(f"expected {expected} EEG features, received {features}")

        # Feature extraction is family-major: [24 families, 105 channels].
        channels = eeg.view(batch, subjects, N_FAMILIES, self.n_channels)
        channels = channels.permute(0, 1, 3, 2)
        tokens = self.channel_encoder(self.channel_norm(channels))
        tokens = tokens + self.channel_embedding.view(1, 1, self.n_channels, -1)
        weights = torch.softmax(self.channel_score(tokens).squeeze(-1), dim=-1)
        subject_embeddings = (tokens * weights.unsqueeze(-1)).sum(dim=2)
        subject_embeddings = self.subject_projection(subject_embeddings)

        mask = subject_mask.unsqueeze(-1).type_as(subject_embeddings)
        summed = (subject_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts, weights


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        model_name,
        fusion,
        text_mode,
        n_channels,
        text_dim=128,
        channel_dim=32,
        eeg_dim=64,
        dropout=0.3,
        zero_eeg_contribution=False,
    ):
        super().__init__()
        self.fusion = fusion
        self.text_mode = text_mode
        self.zero_eeg_contribution = zero_eeg_contribution
        self.text_encoder = None
        self.eeg_encoder = None
        if zero_eeg_contribution and fusion != "gated":
            raise ValueError("zero_eeg_contribution is only valid for gated fusion")

        if fusion != "eeg":
            self.text_encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.text_encoder.config.hidden_size
            self.text_projection = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, text_dim),
                nn.GELU(),
            )
            if text_mode == "frozen":
                self.text_encoder.requires_grad_(False)

        if fusion != "text":
            self.eeg_encoder = EEGSetEncoder(
                n_channels=n_channels,
                channel_dim=channel_dim,
                eeg_dim=eeg_dim,
                dropout=dropout,
            )

        if fusion == "text":
            classifier_input = text_dim
        elif fusion == "eeg":
            classifier_input = eeg_dim
        elif fusion == "concat":
            classifier_input = text_dim + eeg_dim
        elif fusion == "gated":
            self.eeg_to_text = nn.Linear(eeg_dim, text_dim)
            self.gate_logits = nn.Parameter(torch.full((text_dim,), -2.0))
            classifier_input = text_dim
        else:
            raise ValueError(f"unknown fusion type {fusion!r}")

        if fusion == "concat":
            self.fusion_head = nn.Sequential(
                nn.LayerNorm(classifier_input),
                nn.Linear(classifier_input, text_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            classifier_input = text_dim
        else:
            self.fusion_head = nn.Identity()
        self.classifier = nn.Linear(classifier_input, NUM_CLASSES)

    def train(self, mode=True):
        super().train(mode)
        if self.text_encoder is not None and self.text_mode == "frozen":
            self.text_encoder.eval()
        return self

    def _text(self, input_ids, attention_mask):
        hidden = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0]
        return self.text_projection(hidden)

    def forward(
        self,
        input_ids,
        attention_mask,
        eeg,
        subject_mask,
        return_diagnostics=False,
        **_,
    ):
        text_embedding = None
        eeg_embedding = None
        channel_weights = None
        diagnostics = None
        if self.text_encoder is not None:
            text_embedding = self._text(input_ids, attention_mask)
        if self.eeg_encoder is not None:
            eeg_embedding, channel_weights = self.eeg_encoder(eeg, subject_mask)

        if self.fusion == "text":
            fused = text_embedding
        elif self.fusion == "eeg":
            fused = eeg_embedding
        elif self.fusion == "concat":
            fused = self.fusion_head(torch.cat([text_embedding, eeg_embedding], dim=-1))
        else:
            gate = torch.sigmoid(self.gate_logits)
            candidate_contribution = gate * self.eeg_to_text(eeg_embedding)
            if self.zero_eeg_contribution:
                eeg_contribution = torch.zeros_like(candidate_contribution)
            else:
                eeg_contribution = candidate_contribution
            fused = text_embedding + eeg_contribution

        logits = self.classifier(fused)
        if return_diagnostics:
            diagnostics = {}
            if self.fusion == "gated":
                logits_without_eeg = self.classifier(text_embedding)
                diagnostics = {
                    "text_embedding_norm": text_embedding.norm(dim=-1),
                    "eeg_embedding_norm": eeg_embedding.norm(dim=-1),
                    "candidate_eeg_contribution_norm": candidate_contribution.norm(
                        dim=-1
                    ),
                    "gated_eeg_contribution_norm": eeg_contribution.norm(dim=-1),
                    "logits_with_eeg": logits,
                    "logits_without_eeg": logits_without_eeg,
                }
            return logits, channel_weights, diagnostics
        return logits, channel_weights

    def gate_mean(self):
        if not hasattr(self, "gate_logits"):
            return None
        return float(torch.sigmoid(self.gate_logits).mean().detach().cpu())

    def gate_values(self):
        if not hasattr(self, "gate_logits"):
            return None
        return torch.sigmoid(self.gate_logits).detach().cpu().tolist()
