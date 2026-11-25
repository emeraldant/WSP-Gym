"""Inkling 2 Mini needs Inkling 2 Small's tokenizer (h200k). tinker-cookbook 0.6.1 predates Mini and maps every
Inkling model except 2-Small to the first-generation tokenizer (o200k), which scrambles Mini's prompts: with it Mini
answers "17 + 25" with multilingual noise, with h200k it answers "42". Import this before building any renderer."""
from tinker_cookbook import tokenizer_utils as tu

FREE = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
for name in (FREE, "thinkingmachines/Inkling-2-Mini"):
    tu.register_tokenizer(name, lambda: tu.get_tokenizer("thinkingmachines/Inkling-2-Small"))
