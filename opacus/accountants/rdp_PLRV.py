# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

from .accountant import IAccountant
from .analysis import rdp_plrv as privacy_analysis


class RDP_PLRVAccountant(IAccountant):
    DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def __init__(self):
        super().__init__()
        self.history = []
        self.args = {
            "moment":1,
            "theta":2,
            'k':0,
            'mu':0,
            'sigma':0,
            'a':0,
            'b':0,
            'epsilon':1/60000,
            'max_grad_norm':2,
        }

    def step(self, noise_multiplier, sample_rate):
        if len(self.history) >= 1:
            last_args, num_steps = self.history.pop()
            if (
                last_args == self.args
            ):
                self.history.append(
                    (last_args, num_steps + 1)
                )
            else:
                self.history.append(
                    (last_args, num_steps)
                )
                self.history.append((self.args, 1))

        else:
            self.history.append((self.args, 1))

    def get_privacy_spent(
        self, *, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ) -> Tuple[float, float]:
        if not self.history:
            return 0, 0
        
        best_alpha = 0
        epsis = []
        print(self.history)
        for args, num_steps in self.history:
            _alpha, epsi = privacy_analysis.compute_rdp(
                  args = args,
                  num_steps=num_steps,
                )
            best_alpha = _alpha
            epsis.append(epsi)
            
        rdp = sum(epsis)
        eps, best_alpha = privacy_analysis.get_privacy_spent(
            orders=best_alpha, rdp=rdp, delta=delta)
        return float(eps), float(best_alpha)

    def get_epsilon(
        self, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ):
        """
        Return privacy budget (epsilon) expended so far.

        Args:
            delta: target delta
            alphas: List of RDP orders (alphas) used to search for the optimal conversion
                between RDP and (epd, delta)-DP
        """
        eps, _ = self.get_privacy_spent(delta=delta, alphas=alphas)
        return eps

    def __len__(self):
        return len(self.history)

    @classmethod
    def mechanism(cls) -> str:
        return "rdp_plrv"
