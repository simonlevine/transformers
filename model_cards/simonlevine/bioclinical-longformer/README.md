---
language: 
- en
thumbnail: https://raw.githubusercontent.com/simonlevine/model_card/blob/master/3-Figure2-1%20(1).png
tags:
- NLI
- clinical
- roberta
- longformer
datasets:
- MIMIC-iii
- MIMIC-cxr
- MIMIC-iv
metrics:
- bleu
- sacrebleu
---

# bioclinical-longformer

## Model description

This is an elongated version of AllenAI's Biomedical RoBERTa.
In addition, it has been pretrained on ~2.3 million patient records for the clinical domain.

## Intended uses & limitations

For whole-document clinical text embeddings and downstream use, such as medical coding (ICD) automation.

#### How to use

```python

from transformers import RobertaTokenizer


class RobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)

class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)

model = RobertaLongForMaskedLM.from_pretrained('simonlevine/bioclinical-longformer',gradient_checkpointing=True)
tokenizer = RobertaTokenizer.from_pretrained('simonlevine/bioclinical-longformer')

```

#### Limitations and bias

- Limited to 4096-max-token-length documents.
- Not trained with global attention.
- Not trained with MIMIC-IV (yet)
- Requires instantiation with the following classes:


## Training

Designed for use on PhysioNet's MIMIC-III/MIMIC-CXR/MIMIC-IV or similar.
Trained on concatenated MIMIC-III (all notes) and MIMIC-CXR, *then* elongated.

## Eval results

-TBA
