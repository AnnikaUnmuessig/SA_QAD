#Memory optimized version to fit on 2 T4 GPUs


#NVIDIA QAD (Quantization Aware Distillation) baseline implementation
#Datasets: GSM8K (very cheap dataset) after AIME24/25
#Model

"""
Dual-GPU (2x T4/P100) Quantization-Aware Distillation (QAD) for Kaggle.

Sized for a small model (~0.5B-1.5B params). The quantized "student" lives on
GPU 0 and does all the training (optimizer states, gradients, activations).
The full-precision "teacher" lives entirely on GPU 1 and only ever runs
inference (no_grad, no optimizer, no gradient checkpointing needed) -- this
avoids the OOM you get from cramming both models onto a single 14-16GB GPU.

Usage (inside a Kaggle notebook cell or as a script, with 2x T4 enabled):

    python kaggle_qad_train.py \
        --model_name Qwen/Qwen2.5-1.5B-Instruct \
        --recipe general/ptq/int4_blockwise_weight_only \
        --output_dir /kaggle/working/qad-int4 \
        --max_steps 200

If only a single GPU is visible, the script automatically falls back to
putting both models on the same device (you'll want smaller batch/seq
length in that case).

Notes on why each choice was made are inline as comments.
"""

import argparse
import gc
import os

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.distill.plugins.huggingface import DistillArgsWithTeacherModel
from modelopt.torch.quantization.plugins.transformers_trainer import QADTrainer

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Use Llama 3.2 Instruct so tokenizer/chat-template family matches the Nemotron dataset.",
    )
    p.add_argument(
        "--recipe",
        default="general/ptq/int4_blockwise_weight_only",
        help="NVFP4 needs Blackwell; use INT4 weight-only for T4/P100.",
    )
    p.add_argument("--data_source", choices=["wikitext", "nemotron_sft"], default="wikitext",
                    help="'nemotron_sft' streams real QAD training data from the paper's own "
                         "public release (nvidia/Llama-Nemotron-Post-Training-Dataset), matching "
                         "the setup used for Llama Nemotron Super V1 in the QAD report.")
    p.add_argument("--nemotron_domain", choices=["math", "code", "science", "chat"], default="math",
                    help="Domain split to pull, mirroring the paper's math-only/code-only ablation "
                         "(Table 4/5) that tests cross-domain transfer from partial coverage.")
    p.add_argument("--dataset_name", default="wikitext")
    p.add_argument("--dataset_config", default="wikitext-2-raw-v1") #QAD is remarkably robust to the training data's content (Paper citation)
    p.add_argument("--teacher_model", default=None,
                    help="Teacher model name for distillation. Defaults to the same model.")
    p.add_argument("--output_dir", default="/kaggle/working/qad-output")
    p.add_argument("--quantized_output_dir", default=None,
                    help="Directory for the quantized checkpoint saved with ModelOpt state.")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5,
                    help="Paper (Xin et al. 2026) recommends 1e-5 to 1e-6 for QAD; use the lower "
                         "end for SFT-converged models, the higher end for RL-trained models "
                         "where the cold-start SFT data is further from the model's current "
                         "distribution (see Section 4.2).")
    p.add_argument("--max_seq_length", type=int, default=512,
                    help="Kept short to control activation memory on 16GB.")
    return p.parse_args()


class DualGPUQADTrainer(QADTrainer):
    """QADTrainer variant that keeps the teacher on its own GPU.

    KDTrainer._compute_teacher_outputs (from modelopt.torch.distill.plugins.huggingface)
    calls `self._teacher_model(**inputs)` directly using whatever tensors the student's
    forward pass produced -- i.e. tensors already living on the student's device. If the
    teacher lives on a different device, that call fails with a device-mismatch error.
    This override moves inputs to the teacher's device before the call, and moves the
    resulting logits back to the student's device afterward so the KD loss (which
    compares student_logits vs teacher_logits) can be computed normally.
    """

    def _compute_teacher_outputs(self, inputs):
        teacher_device = next(self._teacher_model.parameters()).device
        student_device = next(self.model.parameters()).device

        moved_inputs = {
            k: (v.to(teacher_device) if torch.is_tensor(v) else v) for k, v in inputs.items()
        }

        with torch.no_grad(), self._ds_gather(self._teacher_model.parameters()):
            self._teacher_model.eval()
            outputs = self._teacher_model(**moved_inputs)

        if outputs.logits.device != student_device:
            outputs.logits = outputs.logits.to(student_device)
        return outputs


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()
    # Put the teacher on a second physical GPU if one is available; otherwise fall back
    # to sharing the single device (you'll want a smaller batch/seq length in that case).
    teacher_device = "cuda:1" if num_gpus > 1 else device
    # bf16, not fp16: loading weights directly in fp16 and then also using HF Trainer's
    # fp16=True (GradScaler-based AMP) crashes with "Attempting to unscale FP16 gradients",
    # because GradScaler assumes fp32 master weights. bf16 has the same exponent range as
    # fp32, so it needs no loss scaling and works fine with weights loaded directly in
    # bf16 -- no fp32 master-weight copy required, which matters on a 16GB T4. T4 (Turing)
    # has no bf16 tensor-core acceleration so matmuls are somewhat slower than fp16 would
    # be, but this is the config NVIDIA's own examples use (see examples/llm_qat/quantize.py).
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    teacher_model_name = args.teacher_model or args.model_name
    quantized_output_dir = args.quantized_output_dir or os.path.join(args.output_dir, "quantized")

    print(f"Detected {num_gpus} GPU(s). Student device: {device}. Teacher device: {teacher_device}.")

    mto.enable_huggingface_checkpointing()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the student once, quantize it in place, and save it with ModelOpt state attached.
    student = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch_dtype
    ).to(device)
    student.gradient_checkpointing_enable()  # essential to fit on 16GB

    # --- Training set ---
    if args.data_source == "nemotron_sft":
        streamed = load_dataset(
            "nvidia/Llama-Nemotron-Post-Training-Dataset",
            "SFT",
            split=args.nemotron_domain,
            streaming=True,
        )

        def to_text(example):
            prompt = example.get("input", "")
            response = example.get("output", "")
            return {"text": f"{prompt}{response}"}

        examples = []
        for i, ex in enumerate(streamed.map(to_text)):
            if i >= 2000:
                break
            if ex["text"].strip():
                examples.append(ex["text"])

        train_dataset = Dataset.from_dict({"text": examples})
    else:
        raw_dataset = load_dataset(args.dataset_name, args.dataset_config)
        train_dataset = raw_dataset["train"].filter(lambda x: len(x["text"].strip()) > 0)
        train_dataset = train_dataset.select(range(min(2000, len(train_dataset))))

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])

    def make_forward_loop(target_device):
        def forward_loop(model):
            for i in range(min(32, len(train_dataset))):
                batch = train_dataset[i]
                input_ids = torch.tensor([batch["input_ids"]]).to(target_device)
                attention_mask = torch.tensor([batch["attention_mask"]]).to(target_device)
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
        return forward_loop

    recipe = load_recipe(args.recipe)
    student = mtq.quantize(student, recipe.quantize, make_forward_loop(device))
    #mtq.print_quant_summary(student)

    os.makedirs(quantized_output_dir, exist_ok=True)
    student.save_pretrained(quantized_output_dir)
    tokenizer.save_pretrained(quantized_output_dir)
    #print(f"Saved quantized checkpoint to {quantized_output_dir}")

    # Free the pre-quantization student before reloading; frees a little headroom on GPU 0.
    del student
    gc.collect()
    torch.cuda.empty_cache()

    # Reload the quantized checkpoint so ModelOpt state is restored through the standard HF path.
    student = AutoModelForCausalLM.from_pretrained(
        quantized_output_dir, torch_dtype=torch_dtype
    ).to(device)
    student.gradient_checkpointing_enable()

    # --- Teacher model ---
    # Quantizing the student globally upgrades the model's attention class (e.g.
    # Qwen2Attention -> _QuantAttention) for every instance created in this process,
    # including a freshly-loaded teacher. _QuantAttention.forward() unconditionally
    # reaches for self.q_bmm_quantizer / k_bmm_quantizer / v_bmm_quantizer, which only
    # exist on instances that went through mtq.quantize()'s insertion pass. So the
    # teacher must go through the same insertion step, then have every quantizer
    # disabled so it runs at full precision. It's loaded directly onto its own GPU
    # (teacher_device) so it doesn't compete with the student for memory.
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name, torch_dtype=torch_dtype
    ).to(teacher_device)
    teacher = mtq.quantize(teacher, recipe.quantize, make_forward_loop(teacher_device))
    mtq.disable_quantizer(teacher, lambda name: True)
    teacher.eval()
    teacher.requires_grad_(False)

    gc.collect()
    torch.cuda.empty_cache()

    distill_args = DistillArgsWithTeacherModel(
        distill=True,
        teacher_model=teacher,
        criterion="logits_loss",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=(device.startswith("cuda")),
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to=[],
    )
    if num_gpus > 1:
        # Prevent HF Trainer from auto-wrapping the student in DataParallel across both
        # visible GPUs -- we're deliberately keeping the student on a single GPU (device)
        # and using the second GPU exclusively for the teacher.
        training_args._n_gpu = 1

    trainer = DualGPUQADTrainer(
        model=student,
        processing_class=tokenizer,
        args=training_args,
        distill_args=distill_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()
    print(f"Done. QAD checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()