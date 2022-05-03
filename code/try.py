from pytorch_modelsize import SizeEstimator
se = SizeEstimation(data.drln, input_size=(4,3,96,96))
estimate = se.estimate_size()
# Returns
# (Size in Megabytes, Total Bits)
print(estimate) # (0.5694580078125, 4776960)
