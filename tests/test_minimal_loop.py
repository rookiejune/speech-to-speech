from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

import torch
from anydataset import AudioItem, AudioView, Modality, Role

from speech_to_speech.config import BPEConfig
from speech_to_speech.datamodule.batch_builder import CausalLMBatchBuilder
from speech_to_speech.datamodule.example import speech_pair_from_sample
from speech_to_speech.model.orchestrator import Orchestrator
from speech_to_speech.runtime import prepare_longcat_tokenizer
from speech_to_speech.types import AutoregressionExample
from helpers import MockQwen, MockTokenizer


class MinimalLoopTest(unittest.TestCase):
    def test_dataset_bpe_batch_model_backward(self) -> None:
        pair = speech_pair_from_sample(
            {
                (Role.SOURCE, Modality.AUDIO): AudioItem(
                    views={
                        AudioView.LONGCAT: {
                            "semantic_codes": torch.tensor([0, 1, 0, 1, 2]),
                            "acoustic_codes": torch.zeros((4, 5), dtype=torch.long),
                        }
                    }
                ),
                (Role.TARGET, Modality.AUDIO): AudioItem(
                    views={
                        AudioView.LONGCAT: {
                            "semantic_codes": torch.tensor([2, 1, 0, 1]),
                            "acoustic_codes": torch.zeros((4, 4), dtype=torch.long),
                        }
                    }
                ),
            }
        )
        config = BPEConfig(vocab_size=16, max_piece_frames=4)

        with TemporaryDirectory() as tmpdir:
            bpe = prepare_longcat_tokenizer([pair], config=config, cache_dir=tmpdir)

        tokenizer = MockTokenizer()
        model = Orchestrator(
            qwen3=MockQwen(),
            tokenizer=tokenizer,
            bpe_vocab_size=bpe.vocab_size,
            pretrained=False,
        )
        builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
        audio_ids = torch.tensor(bpe.encode_units(pair.source_ids.tolist()))
        batch = builder.autoregression(AutoregressionExample(audio_ids=audio_ids))

        output = model(batch)

        self.assertIsNotNone(output.loss)
        output.loss.backward()
        grads = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(grads)


if __name__ == "__main__":
    unittest.main()
