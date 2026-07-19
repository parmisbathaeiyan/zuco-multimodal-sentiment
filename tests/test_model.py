import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from src.features import N_FAMILIES
from src.engine import task_initialization_fingerprint
from src.model import EEGSetEncoder, MultimodalClassifier


class DummyTextEncoder(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(20, hidden_size)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class ModelTests(unittest.TestCase):
    def test_eeg_encoder_handles_missing_subjects(self):
        batch, subjects, channels = 3, 4, 5
        encoder = EEGSetEncoder(channels, channel_dim=8, eeg_dim=12, dropout=0.0)
        eeg = torch.randn(batch, subjects, channels * N_FAMILIES)
        mask = torch.tensor(
            [
                [True, True, True, True],
                [True, False, True, False],
                [False, True, False, False],
            ]
        )
        embedding, weights = encoder(eeg, mask)
        self.assertEqual(tuple(embedding.shape), (batch, 12))
        self.assertEqual(tuple(weights.shape), (batch, subjects, channels))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(batch, subjects))

    @patch("src.model.AutoModel.from_pretrained", return_value=DummyTextEncoder())
    def test_gated_fusion_forward(self, _):
        model = MultimodalClassifier(
            model_name="dummy",
            fusion="gated",
            text_mode="finetune",
            n_channels=5,
            text_dim=10,
            channel_dim=8,
            eeg_dim=6,
            dropout=0.0,
        )
        logits, weights, diagnostics = model(
            input_ids=torch.randint(0, 20, (4, 7)),
            attention_mask=torch.ones(4, 7, dtype=torch.long),
            eeg=torch.randn(4, 3, 5 * N_FAMILIES),
            subject_mask=torch.ones(4, 3, dtype=torch.bool),
            return_diagnostics=True,
        )
        self.assertEqual(tuple(logits.shape), (4, 3))
        self.assertEqual(tuple(weights.shape), (4, 3, 5))
        self.assertEqual(tuple(diagnostics["logits_without_eeg"].shape), (4, 3))
        self.assertEqual(len(model.gate_values()), 10)
        self.assertLess(model.gate_mean(), 0.2)

    @patch("src.model.AutoModel.from_pretrained", return_value=DummyTextEncoder())
    def test_zero_gated_contribution_preserves_text_only_logits(self, _):
        model = MultimodalClassifier(
            model_name="dummy",
            fusion="gated",
            text_mode="finetune",
            n_channels=5,
            text_dim=10,
            channel_dim=8,
            eeg_dim=6,
            dropout=0.0,
            zero_eeg_contribution=True,
        )
        logits, _, diagnostics = model(
            input_ids=torch.randint(0, 20, (4, 7)),
            attention_mask=torch.ones(4, 7, dtype=torch.long),
            eeg=torch.randn(4, 3, 5 * N_FAMILIES),
            subject_mask=torch.ones(4, 3, dtype=torch.bool),
            return_diagnostics=True,
        )
        torch.testing.assert_close(logits, diagnostics["logits_without_eeg"])
        torch.testing.assert_close(
            diagnostics["gated_eeg_contribution_norm"], torch.zeros(4)
        )
        self.assertTrue(
            torch.all(diagnostics["candidate_eeg_contribution_norm"] > 0)
        )

    @patch("src.model.AutoModel.from_pretrained", return_value=DummyTextEncoder())
    def test_gated_task_initialization_is_reproducible(self, _):
        kwargs = {
            "model_name": "dummy",
            "fusion": "gated",
            "text_mode": "finetune",
            "n_channels": 5,
            "text_dim": 10,
            "channel_dim": 8,
            "eeg_dim": 6,
            "dropout": 0.0,
        }
        torch.manual_seed(1042)
        aligned = MultimodalClassifier(**kwargs)
        aligned_fingerprint = task_initialization_fingerprint(aligned)
        torch.manual_seed(1042)
        zero = MultimodalClassifier(**kwargs, zero_eeg_contribution=True)
        self.assertEqual(
            aligned_fingerprint, task_initialization_fingerprint(zero)
        )


if __name__ == "__main__":
    unittest.main()
