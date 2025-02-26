# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os.path
from typing import (
    List,
    Sequence,
)

from transformers import AutoTokenizer

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.tools.logging import logger


class HfTokenizer(Tokenizer):
    """
    Tokenizing and encoding/decoding text using a hf tokenizer.

    Args:
        model_path (str): The path to the hf tokenizer model file.
    """

    def __init__(self, model_path: str):
        assert os.path.isdir(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        self._n_words: int = len(self.tokenizer)
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id

        logger.info(
            f"Hf tokenizer built: #words {self.n_words}, BOS ID {self.bos_id}, EOS ID {self.eos_id}"
        )

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
        Returns:
            list[int]: A list of token IDs.
        """
        assert type(s) is str

        input_ids = self.tokenizer(
            s, add_special_tokens=False, truncation=False,
        )['input_ids']

        if bos:
            input_ids.insert(0, self.bos_id)
        if eos:
            input_ids.append(self.eos_id)

        return input_ids

    def decode(self, ids: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            ids (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.tokenizer.decode(ids)
