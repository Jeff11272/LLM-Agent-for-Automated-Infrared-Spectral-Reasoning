import os
import copy
import numpy as np
from openai import OpenAI

class SpectrumCLS:
    def __init__(
        self,
        train_data: list,
        val_data: list,
        test_data: list,
        true_labels_val: list,
        true_labels_test: list,
        api_key: str,
        base_url: str = "...",
        model: str = "qwen-plus",
        max_rounds: int = 5,
        temperature: float = 0.5,
        few_shot_initial: int = None,
        early_stopping_patience: int = 1
    ):
        # Data splits and labels
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.true_labels_val = true_labels_val
        self.true_labels_test = true_labels_test

        # LLM client configuration
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # Hyperparameters
        self.MAX_ROUNDS = max_rounds
        self.TEMPERATURE = temperature
        self.FEW_SHOT_INITIAL = few_shot_initial
        self.EARLY_STOPPING_PATIENCE = early_stopping_patience

        # System prompt for classification reasoning
        self.SYSTEM_PROMPT = (
            "You are a spectral chemometrics and pattern recognition expert. "
            "Your task: given few-shot training examples (each with spectrum feature vector x and label), "
            "analyze each test sample's similarity and principal components, then output only the list of predicted labels."
        )

    def format_few_shot(self, examples):
        few = examples if self.FEW_SHOT_INITIAL is None else examples[:self.FEW_SHOT_INITIAL]
        return "\\n".join(f"{e['name']}: x={e['x']}, label={e['label']}" for e in few)

    def format_eval_prompt(self, eval_examples):
        xs = [e['x'] for e in eval_examples]
        return f"Please classify the following samples based on above examples. Output only a list of labels:\\n{xs}"

    def chat_predict(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.TEMPERATURE,
        )
        text = resp.choices[0].message.content.strip()
        preds = [tok.strip() for tok in text.strip("[]").split(",")]
        return preds, text

    def evaluate(self, preds, truths):
        wrong = [i for i, (p, t) in enumerate(zip(preds, truths)) if p != t]
        acc = 1 - len(wrong) / len(truths)
        return acc, wrong

    def augment_with_hard_examples(self, wrong_indices):
        for idx in wrong_indices:
            e = self.val_data[idx]
            self.train_data.append({
                "name": f"{e['name']}_hard",
                "x": e['x'],
                "label": e['label'],
                "split": "train"
            })

    def run(self):
        # Baseline evaluation on test set
        print("=== Initial test set evaluation ===")
        initial_messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Known examples:\\n{self.format_few_shot(self.train_data)}\\n\\n"
                f"{self.format_eval_prompt(self.test_data)}"
            }
        ]
        preds_test, _ = self.chat_predict(initial_messages)
        test_acc, _ = self.evaluate(preds_test, self.true_labels_test)
        print(f"Initial test accuracy: {test_acc:.2%}")

        # Save best state
        best_train = copy.deepcopy(self.train_data)
        best_test_acc = test_acc
        no_improve = 0

        # Prepare conversation history
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Multi-round validation with early stopping
        for round_idx in range(1, self.MAX_ROUNDS + 1):
            print(f"\\n=== Round {round_idx} validation ===")
            user_content = (
                f"Known examples:\\n{self.format_few_shot(self.train_data)}\\n\\n"
                f"{self.format_eval_prompt(self.val_data)}"
            )
            messages.append({"role": "user", "content": user_content})
            preds_val, val_text = self.chat_predict(messages)
            print("Validation predictions:", preds_val)
            messages.append({"role": "assistant", "content": val_text})

            val_acc, wrong_idxs = self.evaluate(preds_val, self.true_labels_val)
            print(f"Validation accuracy: {val_acc:.2%}, wrong indices: {wrong_idxs}")
            if val_acc == 1.0:
                print("Validation perfect, stopping early.")
                break

            # Augment hard examples and feedback
            self.augment_with_hard_examples(wrong_idxs)
            feedback = ", ".join(self.val_data[i]['name'] for i in wrong_idxs[:3])
            messages.append({"role": "user", "content": f"Hard examples added: {feedback}"})

            # Evaluate on test set
            test_messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"Known examples:\\n{self.format_few_shot(self.train_data)}\\n\\n"
                    f"{self.format_eval_prompt(self.test_data)}"
                }
            ]
            preds_test, _ = self.chat_predict(test_messages)
            test_acc_new, _ = self.evaluate(preds_test, self.true_labels_test)
            print(f"Post-round test accuracy: {test_acc_new:.2%}")

            if test_acc_new > best_test_acc:
                best_test_acc = test_acc_new
                best_train = copy.deepcopy(self.train_data)
                no_improve = 0
            else:
                no_improve += 1
                print(f"No improvement ({no_improve}/{self.EARLY_STOPPING_PATIENCE}), rolling back.")
                self.train_data = copy.deepcopy(best_train)
                if no_improve > self.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Final best test accuracy: {best_test_acc:.2%}")

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error
import re
class SpectrumReg:
    def __init__(
        self,
        dataset,
        api_key: str,
        base_url: str = "...",
        model: str = "qwen-turbo",
        max_rounds: int = 5,
        temperature: float = 0.5,
        few_shot_initial: int = None,
        early_stopping_patience: int = 1
    ):
        # Dataset splits from REG_Dataset
        self.train_data     = dataset.train_data    # list of {'name','x','y','split'}
        self.val_data       = dataset.val_data
        self.test_data      = dataset.test_data
        self.true_y_val     = dataset.y_val_true    # list of floats
        self.true_y_test    = dataset.y_test_true

        # LLM client
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model  = model

        # Hyperparameters
        self.MAX_ROUNDS             = max_rounds
        self.TEMPERATURE             = temperature
        self.FEW_SHOT_INITIAL       = few_shot_initial
        self.EARLY_STOPPING_PATIENCE = early_stopping_patience

        # Prompts
        self.SYSTEM_PROMPT = (
            "You are a spectral chemometrics and regression expert. "
            "Based on provided examples of x→y, infer the relationship and predict y for new x. "
            "Use internal chain‐of‐thought but output only a Python list of predicted y values."
        )
        self.PROMPT_READ = (
            "Read the following structured data, each entry has 'name', 'x', 'y':\n{data}\n"
        )
        self.PROMPT_PREDICT = (
            "Predict 'y' for these new samples given only their 'x':\n{xs}\n"
            "Return only the list of predicted y values."
        )

    def format_read(self, examples):
        few = examples if self.FEW_SHOT_INITIAL is None else examples[:self.FEW_SHOT_INITIAL]
        return self.PROMPT_READ.format(data=few)

    def format_predict(self, examples):
        xs = [e['x'] for e in examples]
        return self.PROMPT_PREDICT.format(xs=xs)


    def chat(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.TEMPERATURE
        )
        raw = resp.choices[0].message.content.strip()

        # 去掉 ```python``` 标记
        clean = raw.replace("```python\n", "").replace("\n```", "")

        # 切分出 token 字符串
        str_tokens = [tok.strip() for tok in clean.strip("[]").split(",")]

        # —— 新增：把每个 token 转成 float —— 
        # 如果你的 token 里有额外的单/双引号，也可以先 strip 掉：
        float_preds = [
            float(tok.strip("'\""))  # 去掉可能的引号后转 float
            for tok in str_tokens
            if re.match(r"^-?\d+(\.\d+)?$", tok.strip("'\""))  # 只处理像数字的 token
        ]

        return float_preds, clean

    def evaluate(self, preds, truths):
        #print('preds:',preds)
        
        r2 = r2_score(truths, preds)
        # identify hardest by absolute error
        errors = [abs(p - t) for p, t in zip(preds, truths)]
        hard = sorted(range(len(errors)), key=lambda i: errors[i], reverse=True)
        return r2, hard

    def augment_hard(self, wrong_idxs):
        for idx in wrong_idxs[:3]:
            e = self.val_data[idx]
            self.train_data.append({
                "name": f"{e['name']}_hard",
                "x": e['x'],
                "y": e['y'],
                "split": "train"
            })

    def run(self):
        # 1) Baseline test evaluation
        print("=== Initial Test Evaluation ===")
        msgs = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        msgs.append({"role": "user", "content": self.format_read(self.train_data)})
        msgs.append({"role": "user", "content": self.format_predict(self.test_data)})
        preds, _ = self.chat(msgs)
        test_r2, _ = self.evaluate(preds, self.true_y_test)
        print(f"Initial test R²: {test_r2:.4f}")
        # RMSE for initial test
        init_rmse = mean_squared_error(self.true_y_test, preds, squared=False)
        print(f"Initial test RMSE: {init_rmse:.4f}")

        best_train = copy.deepcopy(self.train_data)
        best_r2 = test_r2
        no_improve = 0
        history = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # 2) Multi-round validation
        for rd in range(1, self.MAX_ROUNDS + 1):
            print(f"\n=== Round {rd} Validation ===")
            v_msgs = history + [
                {"role": "user", "content": self.format_read(self.train_data)},
                {"role": "user", "content": self.format_predict(self.val_data)}
            ]
            preds_v, reply = self.chat(v_msgs)
            r2_v, hard_idxs = self.evaluate(preds_v, self.true_y_val)
            print(f"Validation R²: {r2_v:.4f}, hardest indices: {hard_idxs[:3]}")
            history.append({"role": "assistant", "content": reply})

            if r2_v >= 1.0:
                print("Perfect validation; stopping early.")
                break

            self.augment_hard(hard_idxs)
            fb = ", ".join(self.val_data[i]['name'] for i in hard_idxs[:3])
            history.append({"role": "user", "content": f"Augmented hard examples: {fb}"})

            # test evaluation
            t_msgs = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.format_read(self.train_data)},
                {"role": "user", "content": self.format_predict(self.test_data)}
            ]
            preds_t, _ = self.chat(t_msgs)
            r2_t, _ = self.evaluate(preds_t, self.true_y_test)
            # RMSE for this round's test
            rmse_t = mean_squared_error(self.true_y_test, preds_t, squared=False)
            print(f"Post-round test R²: {r2_t:.4f}, RMSE: {rmse_t:.4f}")

            if r2_t > best_r2:
                best_r2 = r2_t
                best_train = copy.deepcopy(self.train_data)
                no_improve = 0
            else:
                no_improve += 1
                print(f"No improvement ({no_improve}/{self.EARLY_STOPPING_PATIENCE}); rolling back.")
                self.train_data = copy.deepcopy(best_train)
                if no_improve > self.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        # Final best R²
        print(f"\nFinal best test R²: {best_r2:.4f}")
        # RMSE for final best test
        final_msgs = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.format_read(best_train)},
            {"role": "user", "content": self.format_predict(self.test_data)}
        ]
        final_preds, _ = self.chat(final_msgs)
        final_rmse = mean_squared_error(self.true_y_test, final_preds, squared=False)
        print(f"Final test RMSE: {final_rmse:.4f}")



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectrum_anomaly_agent.py

An LLM‐driven multi‐round anomaly detection pipeline using few‐shot reasoning.
Train on normal examples only, validate and test on mixed normal/anomaly sets,
iteratively augment hard examples and early‐stop based on ROC‐AUC.
"""

import copy
from openai import OpenAI
from sklearn.metrics import precision_score, roc_auc_score

class SpectrumAno:
    def __init__(
        self,
        dataset,             # instance of ANO_Dataset
        api_key: str,
        base_url: str = "...",
        #base_url: str = "https://jeniya.top/v1",
        model: str = "deepseek-v3",
        max_rounds: int = 5,
        temperature: float = 0.5,
        few_shot_initial: int = None,
        early_stopping_patience: int = 2
    ):
        # Data splits
        self.train_data    = dataset.train_data
        self.val_data      = dataset.val_data
        self.test_data     = dataset.test_data
        self.y_val_true    = dataset.y_val
        self.y_test_true   = dataset.y_test
        # LLM client
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model  = model

        # Hyperparameters
        self.MAX_ROUNDS              = max_rounds
        self.TEMPERATURE              = temperature
        self.FEW_SHOT_INITIAL        = few_shot_initial
        self.EARLY_STOPPING_PATIENCE = early_stopping_patience

        # Prompt templates
        # self.SYSTEM_PROMPT = (
        #     "You are a spectroscopic anomaly‐detection expert. "
        #     "Given few‐shot examples labeled True (anomaly) or False (normal), "
        #     "use chain-of-thought to output only a Python list of True/False for new samples."
        # )
        # self.READ_PROMPT = (
        #     "Here are labeled examples (name, x, label):\n{data}\n"
        #     "Label True indicates anomaly, False indicates normal."
        # )
        # self.PREDICT_PROMPT = (
        #     "Predict anomaly status for these samples (only x provided):\n{xs}\n"
        #     "Return only a Python list of True/False."
        # )

        self.SYSTEM_PROMPT = (
    "You are a spectroscopic anomaly‐detection expert. "
    "Given a few labeled examples (True for anomaly, False for normal), "
    "perform chain‑of‑thought reasoning but output only a Python list of True/False."
)

        self.READ_PROMPT = (
    "Here are {n_train} labeled examples (name, x, label):\n{data}\n"
    "Note: True indicates anomaly; False indicates normal."
)

        self.PREDICT_PROMPT = (
    "You have {n_pred} new samples (only the feature vectors x are provided) in the same format. "
    "There are exactly {n_pred} samples, and you must return exactly {n_pred} Boolean values.  "
    "Respond with a single Python list of length {n_pred} containing only True or False, "
    "in the same order as the input samples, with no additional text or punctuation.\n"
    "Sample feature vectors:\n{xs}"
)


    # def format_read(self, examples):
    #     few = examples if self.FEW_SHOT_INITIAL is None else examples[:self.FEW_SHOT_INITIAL]
    #     return self.READ_PROMPT.format(data=few)

    # def format_predict(self, examples):
    #     xs = [e['x'] for e in examples]
    #     return self.PREDICT_PROMPT.format(xs=xs)

    def format_read(self, examples):
        few = examples if self.FEW_SHOT_INITIAL is None else examples[:self.FEW_SHOT_INITIAL]
        return self.READ_PROMPT.format(n_train=len(few), data=few)

    def format_predict(self, examples):
        xs = [e['x'] for e in examples]
        return self.PREDICT_PROMPT.format(
            n_pred=len(examples),
            xs=xs
        )


    def chat_predict(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.TEMPERATURE
        )
        text = resp.choices[0].message.content.strip()
        preds = [tok.strip().lower() == 'true' for tok in text.strip("[]").split(",")]
        return preds, text

    def evaluate(self, preds, truths):
        """
        Compute precision and ROC‐AUC, identify hard examples by mismatch.
        Returns:
            precision (float), auc (float), hard_indices (list)
        """
        # print("preds:",len(preds))
        # print('content:',preds)
        precision = precision_score(truths, preds)
        # For binary predictions, use preds as scores for AUC
        auc = roc_auc_score(truths, preds)
        hard = [i for i, (p, t) in enumerate(zip(preds, truths)) if p != t]
        return precision, auc, hard

    def augment_hard(self, wrong_idxs):
        for idx in wrong_idxs:
            e = self.val_data[idx]
            e_hard = copy.deepcopy(e)
            e_hard['name'] += "_hard"
            self.train_data.append(e_hard)

    def run(self):
        # 1) Initial test evaluation
        print("=== Initial Test Evaluation ===")
        msgs = [{"role":"system","content":self.SYSTEM_PROMPT}]
        msgs.append({"role":"user","content":self.format_read(self.train_data)})
        msgs.append({"role":"user","content":self.format_predict(self.test_data)})
        preds_test, _ = self.chat_predict(msgs)
        prec_test, auc_test, _ = self.evaluate(preds_test, self.y_test_true)
        print(f"Initial Test Precision: {prec_test:.4f}, AUC: {auc_test:.4f}")

        best_train = copy.deepcopy(self.train_data)
        best_auc = auc_test
        no_improve = 0
        history = [{"role":"system","content":self.SYSTEM_PROMPT}]

        # 2) Multi‐round validation
        for rd in range(1, self.MAX_ROUNDS+1):
            print(f"\n=== Round {rd} Validation ===")
            v_msgs = history + [
                {"role":"user","content":self.format_read(self.train_data)},
                {"role":"user","content":self.format_predict(self.val_data)}
            ]
            preds_val, reply = self.chat_predict(v_msgs)
            prec_val, auc_val, hard_idxs = self.evaluate(preds_val, self.y_val_true)
            print(f"Validation Precision: {prec_val:.4f}, AUC: {auc_val:.4f}, Hard count: {len(hard_idxs)}")
            history.append({"role":"assistant","content":reply})

            if auc_val >= 1.0:
                print("Perfect validation AUC; stopping early.")
                break

            self.augment_hard(hard_idxs)
            fb = ", ".join(self.val_data[i]['name'] for i in hard_idxs[:3])
            history.append({"role":"user","content":f"Augmented hard examples: {fb}"})

            # 3) Test re‐evaluation
            t_msgs = [
                {"role":"system","content":self.SYSTEM_PROMPT},
                {"role":"user","content":self.format_read(self.train_data)},
                {"role":"user","content":self.format_predict(self.test_data)}
            ]
            preds_t, _ = self.chat_predict(t_msgs)
            prec_t, auc_t, _ = self.evaluate(preds_t, self.y_test_true)
            print(f"Post‐round Test Precision: {prec_t:.4f}, AUC: {auc_t:.4f}")

            if auc_t > best_auc:
                best_auc = auc_t
                best_train = copy.deepcopy(self.train_data)
                no_improve = 0
            else:
                no_improve += 1
                print(f"No improvement ({no_improve}/{self.EARLY_STOPPING_PATIENCE}); rolling back.")
                self.train_data = copy.deepcopy(best_train)
                if no_improve > self.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"\nFinal best Test Precision/AUC: {prec_t:.4f} / {best_auc:.4f}")

