### optimizer
- RMSP
```python
```

- SGD
```python
optimizer = dict(type='SGD', lr=0.001, momentum=0.99, weight_decay=5e-5)
```

- Adam
```python
optimizer = dict(type='Adam', lr=0.001, beta1=0.9,beta2=0.99, weight_decay=5e-5)
```


### lr_scheduler

- StepLR
```python
lr_scheduler = dict(type='StepLR', step_size=10, gamma=0.1)
```

- MultiStepLR
```python
lr_scheduler = dict(type='MultiStepLR', milestones=[30,50,80], gamma=0.1)
```

- MultiStepWarmup
```python
lr_scheduler = dict(type='MultiStepWarmup', milestones=[30,50,80], gamma=0.1,warm_up_epochs=5)
```

- CosineAnnealingLR
```python
lr_scheduler = dict(type='CosineAnnealingLR', T_max=20)
```

- CosineWarmup
```python
lr_scheduler = dict(type='CosineWarmup', warm_up_epochs=5,epochs=100)
```