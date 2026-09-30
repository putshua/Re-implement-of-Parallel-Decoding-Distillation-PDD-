import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_embeddings', ROOT / 'scripts/prepare_embeddings.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class TestPromptPreparation(unittest.TestCase):
    def encode(self, filename, contents, flag):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / filename
            source.write_text(contents, encoding='utf-8')
            output = root / 'embeddings'
            class Encoder:
                def __init__(self, *args, **kwargs):
                    pass
                def __call__(self, captions, device):
                    return [torch.full((2, 4), float(len(c))) for c in captions]
            class Module:
                T5EncoderModel = Encoder
            args = ['prepare_embeddings', flag, str(source), '--checkpoint', str(root / 'model'),
                    '--wan-root', str(root / 'wan'), '--output', str(output),
                    '--batch-size', '1', '--shard-size', '1', '--negative-prompt', 'blurry']
            with patch('sys.argv', args), patch.object(prepare, 'wan_module', return_value=Module), \
                 patch.object(torch.cuda, 'set_device'), patch.object(torch, 'autocast', return_value=nullcontext()), \
                 patch.dict('os.environ', {'RANK':'0', 'WORLD_SIZE':'1', 'LOCAL_RANK':'0'}):
                prepare.main()
                paths = sorted(output.glob('prompts_*.pt'))
                blobs = [torch.load(p, weights_only=True) for p in paths]
                self.assertEqual([b['prompts'][0] for b in blobs], ['A bird flies.', 'A boat sails.'])
                self.assertEqual(blobs[0]['t5_text_embeddings'].shape, (1, 512, 4))
                self.assertEqual(torch.load(output / 'negative_embeddings.pt', weights_only=True).shape, (512, 4))
                # A second invocation reuses valid shards; changing captions must be rejected.
                prepare.main()
                source.write_text(contents.replace('A bird flies.', 'A dog runs.'), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Stale embeddings'):
                    prepare.main()

    def test_text_prompts_and_negative_cache(self):
        self.encode('prompts.txt', 'A bird flies.\n\nA boat sails.\n', '--prompts')

    def test_jsonl_without_ids(self):
        contents = '\n'.join(json.dumps({'caption':s}) for s in ['A bird flies.', 'A boat sails.'])
        self.encode('prompts.jsonl', contents, '--manifest')


if __name__ == '__main__':
    unittest.main()
