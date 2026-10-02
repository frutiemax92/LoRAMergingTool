# Instruction

SSR-Merge proposes a method for merging multiple LoRA models to a checkpoint.
It seems to give better performances than other techniques.

The idea is to take the krea2 turbo model in the Checkpoint folder, and merge the Loras in Loras folder using this technique.
This is the paper: https://arxiv.org/pdf/2606.10617
This is the github of that paper: https://github.com/nagara214/SSR-Merge
This is the github for krea 2: https://github.com/krea-ai/krea-2

Make sure to use VRAM optimization techniques for it to work i.e. store the weights of the base model as nf4, and use bf16 whenever possible.