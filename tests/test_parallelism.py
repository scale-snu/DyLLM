import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dyllm.config import Config
from dyllm.distributed import tensor_parallel_cosine_mask
from dyllm.engine.sequence import Sequence, SequenceStatus
from dyllm.sampling_params import SamplingParams


class ParallelismTest(unittest.TestCase):
    def test_sequence_pickle_preserves_worker_state(self):
        seq = Sequence(
            [10, 11, 99, 99],
            SamplingParams(max_new_tokens=2, steps=2, num_full_steps=1, mask_id=99),
        )
        seq.processed_steps = 1
        seq.status = SequenceStatus.SPARSE
        seq.last_tokens = [12]
        seq.last_token_pos = [2]

        restored = pickle.loads(pickle.dumps(seq))

        self.assertEqual(restored.seq_id, seq.seq_id)
        self.assertEqual(restored.token_ids, seq.token_ids)
        self.assertEqual(restored.last_tokens, seq.last_tokens)
        self.assertEqual(restored.last_token_pos, seq.last_token_pos)
        self.assertEqual(restored.processed_steps, seq.processed_steps)
        self.assertEqual(restored.status, SequenceStatus.SPARSE)

    def test_additive_shard_stats_match_full_vector_cosine(self):
        old = torch.tensor([[1.0, 2.0, -3.0, 4.0], [1.0, 0.0, 1.0, 0.0]])
        new = torch.tensor([[1.5, 1.0, -2.0, 5.0], [-1.0, 0.0, -1.0, 0.0]])
        stats = torch.zeros(2, 3)
        for old_shard, new_shard in zip(old.chunk(2, dim=-1), new.chunk(2, dim=-1)):
            stats[:, 0] += (old_shard * new_shard).sum(dim=-1)
            stats[:, 1] += old_shard.square().sum(dim=-1)
            stats[:, 2] += new_shard.square().sum(dim=-1)

        expected = torch.nn.functional.cosine_similarity(old, new, dim=-1) < 0.5
        self.assertTrue(torch.equal(tensor_parallel_cosine_mask(stats, 0.5), expected))

    def test_expert_parallel_size_validation(self):
        moe_config = SimpleNamespace(
            model_type="llada",
            num_experts=64,
            max_position_embeddings=8192,
            mask_token_id=1,
        )
        with (
            patch("dyllm.config.os.path.isdir", return_value=True),
            patch("dyllm.config.AutoConfig.from_pretrained", return_value=moe_config),
        ):
            config = Config("model", tensor_parallel_size=2, expert_parallel_size=2)
            self.assertEqual(config.expert_parallel_size, 2)
            with self.assertRaisesRegex(ValueError, "must be 1 or equal"):
                Config("model", tensor_parallel_size=4, expert_parallel_size=2)


if __name__ == "__main__":
    unittest.main()
