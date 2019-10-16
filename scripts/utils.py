from aegis_core.callbacks import TensorboardFieldCallback, TensorboardCallback
from aegis_core.callbacks import TensorboardActions, TensorboardPGETReward
from aegis_core.callbacks import TensorboardPGETWeights, TensorboardPGETTraces

def env_callbacks(summary_writer, env_name):
  cbs = [
    #log sum of rewards every episode
    TensorboardFieldCallback(summary_writer, "reward", name_format="{}/" + env_name,
      reduce="sum", interval="done", step_for_step=False),
    #log action distribution every episode
    TensorboardActions(summary_writer, env_name=env_name, interval="done",
      step_for_step=False),
  ]

  return cbs

def pget_callbacks(summary_writer, name, interval=100):
  cbs = [
    TensorboardPGETWeights(summary_writer, name, interval=interval,
      combine=False, step_for_step=True),
    TensorboardPGETTraces(summary_writer, name, interval=interval,
      combine=False, step_for_step=True),
    TensorboardPGETReward(summary_writer, name, interval=interval,
      step_for_step=True),
  ]

  return cbs

def curiosity_callbacks(summary_writer, name, interval=100):
  cbs = [
    TensorboardFieldCallback(summary_writer, "loss", name_format=name + " curiosity/{}",
      reduce="mean", interval=interval, step_for_step=True),
    TensorboardFieldCallback(summary_writer, "surprise", name_format=name + " curiosity/{}",
      reduce="mean", interval=interval, step_for_step=True),
  ]

  return cbs
