import random

from mindspore import ops, Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.common import dtype as mstype


class DynamicDecayLR(LearningRateSchedule):
    """Learning rate schedule for dynamic"""

    def __init__(self, lr, n_epochs, step_per_epoch, offset, decay_start_epoch):
        super().__init__()
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.lr = lr
        self.n_epochs = n_epochs
        self.step_per_epoch = step_per_epoch
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
        self.cast = ops.Cast()

    def construct(self, global_step):
        epoch = self.cast(global_step, mstype.float32) // self.step_per_epoch
        return self.lr * (1.0 - max(Tensor(0.0), epoch + self.offset - self.decay_start_epoch) / (
                self.n_epochs - self.decay_start_epoch))


class ReplayBuffer:
    """Replay buffer"""

    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data:
            element = ops.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self[i].copy())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return ops.cat(to_return)
