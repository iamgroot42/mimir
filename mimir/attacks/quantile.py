"""
    Implementation of the attack proposed in 'Scalable Membership Inference Attacks via Quantile Regression'
    https://arxiv.org/pdf/2307.03694.pdf
"""
import torch as ch
from mimir.models import QuantileReferenceModel, Model
from transformers import TrainingArguments
from sklearn.metrics import mean_squared_error
from transformers import TrainingArguments, Trainer
from datasets import Dataset

from mimir.attacks.all_attacks import Attack


class CustomTrainer(Trainer):
    def __init__(
        self,
        alpha_fpr,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha_fpr = alpha_fpr

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = ch.mean(
            ch.max(
                self.alpha_fpr * (logits - labels),
                (1 - self.alpha_fpr) * (labels - logits),
            )
        )
        return (loss, outputs) if return_outputs else loss


class QuantileAttack(Attack):
    """
    Implementation of the attack proposed in 'Scalable Membership Inference Attacks via Quantile Regression'
    https://arxiv.org/pdf/2307.03694.pdf
    """

    def __init__(self, config, model: Model, alpha: float):
        """
        alpha (float): Desired FPR
        """
        ref_model = QuantileReferenceModel(
            config, name="Sreevishnu/funnel-transformer-small-imdb"
        )
        super().__init__(self, config, model, ref_model)
        self.alpha = alpha

    def _train_quantile_model(self, dataset):
        def tokenize_function(examples):
            return self.ref_model.tokenizer(
                examples["text"], padding="max_length", truncation=True
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        training_args = TrainingArguments(
            output_dir="quantile_ref_model",
            evaluation_strategy="epoch",
            num_train_epochs=1,
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            rmse = mean_squared_error(labels, predictions, squared=False)
            return {"rmse": rmse}

        trainer = CustomTrainer(
            alpha_fpr=self.alpha,
            model=self.ref_model.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset,
            compute_metrics=compute_metrics,
        )
        # Train quantile model
        trainer.train()

    def prepare(self, known_non_members):
        """
        Step 1: Use non-member dataset, collect confidence scores for correct label.
        Step 2: Train a quantile regression model that takes X as input and predicts quantile. Use pinball loss
        Step 3: Test by checking if member: score is higher than output of quantile regression model.
        """

        # Step 1: Use non-member dataset, collect confidence scores for correct label.
        # Get likelihood scores from target model for known_non_members
        # Note that these non-members should be different from the ones in testing
        scores = [self.target_model.get_ll(x) for x in known_non_members]
        # Construct a dataset out of this to be used in Huggingface, with
        # "text" containing the actual data, and "labels" containing the scores
        dataset = Dataset.from_dict({"text": known_non_members, "labels": scores})

        # Step 2: Train a quantile regression model that takes X as input and predicts quantile. Use pinball loss
        self._train_quantile_model(dataset)

    def attack(self, document, **kwargs):
        # Step 3: Test by checking if member: score is higher than output of quantile regression model.

        # Get likelihood score from target model for doc
        ll = self.target_model.get_ll(document)

        # Return ll - quantile_model(doc)
        tokenized = self.ref_model.tokenizer(document, return_tensors="pt")
        # Shift items in the dictionary to the correct device
        tokenized = {k: v.to(self.ref_model.model.device, non_blocking=True) for k, v in tokenized.items()}
        quantile_score = self.ref_model.model(**tokenized)
        print(quantile_score)
        quantile_score = quantile_score.logits.item()

        # We want higher score to be non-member
        return quantile_score - ll
