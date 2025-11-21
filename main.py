import data as utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from merlin import LexGrouping, MeasurementStrategy, QuantumLayer
from merlin.builder import CircuitBuilder

from tqdm import tqdm

train_dataset, test_dataset, dates_train, dates_test = utils.create_custom_datasets(
    file_path="train.xlsx",
    forecast_horizon=6,
    context_lenght=12,
    technical_feats=False,
    split_ratio=0.2,
    lags=None
)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def run_experiment(model: nn.Module, dataloader: DataLoader, epochs: int = 60, lr: float = 0.05):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, target) in enumerate(loop):
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    model.eval()
    train_acc = 0
    test_acc = 0
    with torch.no_grad():
        for x_train, y_train in train_dataloader:
            train_preds = model(x_train).argmax(dim=1)
            train_acc += (train_preds == y_train).float().mean().item()
        for x_test, y_test in test_dataloader: 
            test_preds = model(x_test).argmax(dim=1)
            test_acc += (test_preds == y_test).float().mean().item()
    return train_acc / len(train_dataloader), test_acc / len(test_dataloader)



# builder = CircuitBuilder(n_modes=6)
# builder.add_entangling_layer(trainable=True, name="U1")
# builder.add_angle_encoding(
#     modes=list(range(224)),  # one mode per Iris feature
#     name="input",
#     scale=np.pi,
# )
# builder.add_rotations(trainable=True, name="theta")
# builder.add_superpositions(depth=1, trainable=True)

# quantum_core = QuantumLayer(
#     input_size=224,
#     builder=builder,
#     n_photons=3,                             # Equivalent to input_state=[1,1,1,0,0,0]
#     measurement_strategy=MeasurementStrategy.PROBABILITIES,
# )

model = nn.Sequential(
    nn.Conv2d(12, 3, kernel_size=3, padding=1),
    nn.AvgPool2d(2),
    nn.ReLU(),
    nn.Conv2d(3, 3, kernel_size=3, padding=1),
    nn.AvgPool2d(2),
    nn.ReLU(),
    nn.Conv2d(3, 3, kernel_size=3, padding=1),
    nn.AvgPool2d(2),
    nn.ReLU(),
    nn.Flatten(),
    QuantumLayer.simple(input_size=6, n_params=60),
    LexGrouping(QuantumLayer.simple(input_size=6, n_params=60).output_size, 6),
    nn.Linear(6, 36),
    nn.ReLU(),
    nn.Linear(36, 128),
    nn.ReLU(),
    nn.Linear(128, 224),
)

print("Starting training...")

train_acc, test_acc = run_experiment(model, train_dataloader, epochs=80, lr=0.05)
print(f"Train accuracy: {train_acc:.3f} – Test accuracy: {test_acc:.3f}")

