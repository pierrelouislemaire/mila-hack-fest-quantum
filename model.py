import torch 
import torch.nn as nn
import torch.nn.functional as F

from merlin import LexGrouping, MeasurementStrategy, QuantumLayer
from merlin.builder import CircuitBuilder

import numpy as np

class dense_model(nn.Module): #TODO
    def __init__(self, input_size=10, hidden_size=50, output_size=1,):
        super(dense_model, self).__init__()
        self.builder = CircuitBuilder(n_modes=6)
        self.builder.add_entangling_layer(trainable=True, name="U1")
        self.builder.add_angle_encoding(
            modes=list(range(6)),  # one mode per Iris feature
            name="input",
            scale=np.pi,
        )
        self.builder.add_entangling_layer(trainable=True, name="U2")
        # self.builder.add_rotations(trainable=True, name="theta")
        # self.builder.add_superpositions(depth=1, trainable=True)

        quantum_core = QuantumLayer(
            input_size=6,
            builder=self.builder,
            n_photons=3,                             # Equivalent to input_state=[1,1,1,0,0,0]
            measurement_strategy=MeasurementStrategy.PROBABILITIES,
        )
        
        self.enc = nn.Sequential(
                                nn.Conv2d(12, 12, kernel_size=3, padding=1),
                                nn.AvgPool2d(2),
                                nn.ReLU(),
                                nn.Conv2d(12, 6, kernel_size=3, padding=1),
                                nn.AvgPool2d(2),
                                nn.ReLU(),
                                nn.Conv2d(6, 3, kernel_size=3, padding=1),
                                nn.AvgPool2d(2),
                                nn.ReLU(),)
        
        self.quant_res = nn.Sequential(
                                #nn.BatchNorm1d(6),
                                quantum_core,
                                LexGrouping(quantum_core.output_size, 6),
                            )
        
        self.dec = nn.Sequential(
                                nn.Linear(6, 128),
                                nn.ReLU(),
                                nn.Linear(128, 224),
                                nn.Flatten()
,
)

    def forward(self, x):
        x = self.enc(x)
        x = nn.Flatten()(x)
        res = self.quant_res(x)
        out = x
        out = out #+ res
        out = self.dec(out)
        return out