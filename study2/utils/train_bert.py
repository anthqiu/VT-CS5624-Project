import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup, AutoModelForSequenceClassification
from torch.optim import AdamW
from datasets import TopicSentenceDataset
from sklearn.metrics import accuracy_score, f1_score


def build_bert_classifier(model_name="bert-base-uncased", num_labels=2):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model


def train_val_test_split(dataset, train_ratio=0.8, val_ratio=0.1):
    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])


def train_one_topic(topic_df,
                    topic_name,
                    model_name="bert-base-uncased",
                    batch_size=16,
                    num_epochs=6,
                    lr=2e-5,
                    device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Training topic: {topic_name} on {device} ===")

    dataset = TopicSentenceDataset(topic_df, model_name=model_name)
    if len(dataset) < 100:  # 太少的数据可以先跳过
        print(f"[{topic_name}] too few samples: {len(dataset)}")
        return None

    train_ds, val_ds, test_ds = train_val_test_split(dataset)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = build_bert_classifier(model_name=model_name).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs} | loss={avg_loss:.4f}, "
              f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"[{topic_name}] Test acc={test_acc:.4f}, f1={test_f1:.4f}")

    return model, (test_acc, test_f1)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, f1
