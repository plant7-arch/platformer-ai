import jax
import jax.numpy as jnp

print(jax.devices())                     # expect [CudaDevice(id=0)]
x = jnp.ones((4096, 4096))
print((x @ x).block_until_ready().sum()) # runs a matmul on the GPU
