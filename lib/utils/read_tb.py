
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('T-3.5/tb_logs/events.out.tfevents.1570645252.pcjp.6096.0',
  size_guidance={
event_accumulator.SCALARS: 0,

})
ea.Reload()

print(ea.Tags())
print(len(ea.Scalars('train/acc')))

for tag in ea.Tags()['scalars']:
  print(tag, ea.Scalars(tag)[-1])


