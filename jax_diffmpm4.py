import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import IPython
import jax.nn as jnn
import jax.lax as jlax
import timeit
import jax



dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
max_steps = 2048
steps = 1024
gravity = 3.8
target = [0.8, 0.2]
mass = 1.0


'''
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss = scalar()

weights = scalar()
bias = scalar()
x_avg = vec()

actuation = scalar()
'''

n_sin_waves = 4
actuation_omega = 20
act_strength = 4
key = random.PRNGKey(0)


class Scene:
  def __init__(self):
    self.n_particles = 0
    self.n_solid_particles = 0
    self.x = []
    self.actuator_id = []
    self.particle_type = []
    self.offset_x = 0
    self.offset_y = 0

  def add_rect(self, x, y, w, h, actuation, ptype=1):
    if ptype == 0:
      assert actuation == -1
    global n_particles
    w_count = int(w / dx) * 2
    h_count = int(h / dx) * 2
    real_dx = w / w_count
    real_dy = h / h_count
    for i in range(w_count):
      for j in range(h_count):
        self.x.append([
          x + (i + 0.5) * real_dx + self.offset_x,
          y + (j + 0.5) * real_dy + self.offset_y
        ])
        self.actuator_id.append(actuation)
        self.particle_type.append(ptype)
        self.n_particles += 1
        self.n_solid_particles += int(ptype == 1)

  def set_offset(self, x, y):
    self.offset_x = x
    self.offset_y = y

  def finalize(self):
    global n_particles, n_solid_particles
    n_particles = self.n_particles
    n_solid_particles = self.n_solid_particles
    print('n_particles', n_particles)
    print('n_solid', n_solid_particles)

  def set_n_actuators(self, n_act):
    global n_actuators
    n_actuators = n_act


def robot(scene):
  scene.set_offset(0.1, 0.03)
  scene.add_rect(0.0, 0.1, 0.3, 0.1, -1)
  scene.add_rect(0.0, 0.0, 0.05, 0.1, 0)
  scene.add_rect(0.05, 0.0, 0.05, 0.1, 1)
  scene.add_rect(0.2, 0.0, 0.05, 0.1, 2)
  scene.add_rect(0.25, 0.0, 0.05, 0.1, 3)
  scene.set_n_actuators(4)



def allocate_arrays():
  global weights, bias, actuation, actuator_id, particle_type, x, v, C, F
  global grid_m_in, grid_v_in, grid_v_out, loss, x_avg, index_array, offset_array

  weights = random.normal(key, (n_actuators, n_sin_waves))
  bias = jnp.zeros((n_actuators,))
  actuation = jnp.zeros((n_actuators,))
  actuator_id = jnp.zeros((n_particles,))
  particle_type = jnp.zeros((n_particles,))
  x = jnp.zeros((n_particles, dim))
  v = jnp.zeros((n_particles, dim))


  C = jnp.zeros((n_particles, dim, dim))
  F = jnp.zeros((n_particles, dim, dim))

  grid_m_in = jnp.zeros((n_grid, n_grid))

  grid_v_in = jnp.zeros((n_grid, n_grid, dim))
  grid_v_out = jnp.zeros((n_grid, n_grid, dim))

  loss = jnp.zeros(())
  x_avg = jnp.zeros((dim,))
  index_array = np.zeros((n_grid, n_grid, dim))
  offset_array = np.zeros((9, 2))
  
  for i in range(n_grid):
    for j in range(n_grid):
      index_array[i, j] = np.array([i, j])
      
  idx = 0
  for i in range(3):
    for j in range(3):
      offset_array[0] = np.array([i, j])
      idx += 1
      
  index_array = jnp.array(index_array)
  offset_array = jnp.int32(jnp.array(offset_array))



def compute_actuation(t):
  act = 0.0
  for j in range(n_sin_waves):
    act += weights[:, j] * jnp.sin(actuation_omega * t * dt +
                    2 * math.pi / n_sin_waves * j)
  act += bias
  actuation = jnp.tanh(act)
    
  return actuation
 
 
 
def g2p_idx(x):
  base = jnp.int32(x * inv_dx - 0.5)
  return base + offset_array
  
def g2p_accum(x, grid_v_out):
  base = jnp.int32(x * inv_dx - 0.5)
  fx = x * inv_dx - base
  w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
  new_v = jnp.array([0.0, 0.0])
  new_C = jnp.array([[0.0, 0.0], [0.0, 0.0]])

  idx = 0
  for i in range(3):
    for j in range(3):
      dpos = jnp.array([i, j]) - fx
      g_v = grid_v_out[idx]
      weight = w[i][0] * w[j][1]
      new_v += weight * g_v
      new_C += 4 * weight * jnp.outer(g_v, dpos) * inv_dx
      idx += 1

  x += dt * new_v
  
  return x, new_v, new_C
  
 
def g2p(x, grid_v_out):
  base = jnp.int32(x * inv_dx - 0.5)
  fx = x * inv_dx - base
  w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
  new_v = jnp.array([0.0, 0.0])
  new_C = jnp.array([[0.0, 0.0], [0.0, 0.0]])

  for i in range(3):
    for j in range(3):
      dpos = jnp.array([i, j]) - fx
      g_v = grid_v_out[base[0] + i, base[1] + j]
      weight = w[i][0] * w[j][1]
      new_v += weight * g_v
      new_C += 4 * weight * jnp.outer(g_v, dpos) * inv_dx

  x += dt * new_v
  
  return x, new_v, new_C
  

def grid_op(index_tuple, grid_m_in, grid_v_in):
  bound = 3
  coeff = 0.5
  
  i = index_tuple[0]
  j = index_tuple[1]
  
  normal = jnp.array([0., 1.])
  
  inv_m = 1 / (grid_m_in + 1e-10)
  v_out = jnp.expand_dims(inv_m, -1) * grid_v_in
  v_out -= dt * gravity * jnp.array([0., 1.])
  
  v_out = jnp.where(jnp.logical_and(i < bound, v_out[0] < 0), jnp.zeros_like(v_out), v_out)
  
  
  v_out = jnp.where(jnp.logical_and(i > n_grid - bound, v_out[0] > 0), jnp.zeros_like(v_out), v_out)
  
  lin = (v_out.transpose() @ normal)
  
  vit = v_out - lin * normal
  lit = jnp.linalg.norm(vit + 1e-10)  + 1e-10
  
  v_out_gate_2 = jnp.where(lit + coeff * lin <= 0, jnp.zeros_like(v_out), (1 + coeff * lin / lit) * vit)
  v_out_gate_1 = jnp.where(lin < 0, v_out_gate_2, jnp.zeros_like(v_out))
  v_out = jnp.where(jnp.logical_and(j < bound, v_out[1] < 0), v_out_gate_1, v_out)
            
  v_out = jnp.where(jnp.logical_and(j > n_grid - bound, v_out[1] > 0), jnp.zeros_like(v_out), v_out)
      
  
  return v_out
  
  
  
def p2g_accum(x, v, new_F, affine, stress):

  
  grid_m_in = jnp.zeros((n_grid, n_grid))
  grid_v_in = jnp.zeros((n_grid, n_grid, dim))
  
  for p in range(n_particles):
    base = jnp.int32(jnp.floor(x[p] * inv_dx - 0.5))
    fx = x[p] * inv_dx - jnp.floor(base)
    w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
    
    for i in range(3):
      for j in range(3):
        offset = jnp.array([i, j])
        dpos = (jnp.array([i, j]) - fx) * dx
        weight = w[i][0] * w[j][1]
        idx = base + offset
        mask = jnn.one_hot(idx[0] * n_grid + idx[1], n_grid * n_grid)
        mask = jnp.expand_dims(jnp.reshape(mask, (n_grid, n_grid)), -1)
        a = jnp.expand_dims(jnp.expand_dims(weight * (mass * v[p] + affine[p] @ dpos), 0), 0)
        grid_v_in +=  a * mask
        grid_m_in += jnp.squeeze(weight * mass * mask)
        
    return grid_m_in, grid_v_in

def p2g_contrib(x, v, C, F, actuator_id, actuation):
  #ANDYTODO: Input args, output args (should also handle mutation), vectorize as much as possible
  
  new_F = (jnp.eye(dim) + dt * C) @ F
  
  J = jnp.linalg.det(new_F)
  
  #F.at[p].set(new_F) #ANDYTODO: we need to mutate this
  
  
  
  act_id = actuator_id
  act = actuation[jnp.maximum(0, act_id)] * act_strength
  act = jnp.where(act == -1, act, jnp.zeros_like(act))

  A = jnp.array([[0.0, 0.0], [0.0, 1.0]]) * act
  
  
  cauchy = jnp.array([[0.0, 0.0], [0.0, 0.0]])
  

  new_F_T = jnp.transpose(new_F)
  new_F_T_inv = jnp.linalg.inv(new_F_T)
  
  #cauchy = mu * (new_F - new_F_T_inv) + la * jnp.log(J) * jnp.eye(dim) @ new_F_T_inv
  cauchy = mu * (new_F - new_F_T_inv)
  
  cauchy += new_F @ A @ new_F.transpose()
  
  
  stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
  affine = stress + mass * C
  
  
  return new_F, affine, stress
  
  
  
  
#jit everything:

p1_j = jit(vmap(p2g_contrib, in_axes=(0, 0, 0, 0, 0, None)))
p2_j = jit(p2g_accum)
go_j = jit(vmap(vmap(grid_op)))
g1_j = jit(vmap(g2p_idx))
g2_j = jit(vmap(g2p_accum))
ca_j = jit(compute_actuation)

p1 = vmap(p2g_contrib, in_axes=(0, 0, 0, 0, 0, None))
p2 = p2g_accum
go = vmap(vmap(grid_op))
g1 = vmap(g2p_idx)
g2 = vmap(g2p_accum)
ca = compute_actuation
  
  
def advance(t, args):
  x = args[0]
  v = args[1]
  C = args[2]
  F = args[3]
  actuation = ca_j(t)
  F, affine, stress = p1_j(x, v, C, F, actuator_id, actuation)
  grid_m_in, grid_v_in = p2_j(x, v, F, affine, stress)
  grid_v_out = go_j(index_array, grid_m_in, grid_v_in)
  idxes = g1_j(x)
  grid_v_out = jnp.take(grid_v_out, idxes)
  x, v, C = g2_j(x, grid_v_out)
  
  return x, v, C, F
  
a = jit(advance)


def forward(x, v, C, F):
  x, v, C, F = jlax.fori_loop(0, 2048, advance, (x, v, C, F))
  #for i in range(2000):
  '''
    #p2g_contrib(x[0], v[0], C[0], F[0], actuator_id[0])
    actuation = ca(t)
    F, affine, stress = p1(x, v, C, F, actuator_id, actuation)
    #grid_m_in, grid_v_in = p2g_accum(x[0], v[0], new_F[0], affine[0], stress[0])
    grid_m_in, grid_v_in = p2(x, v, F, affine, stress)
    #v_out = grid_op(jnp.array([0, 0]), grid_m_in[0], grid_v_in[0])
    grid_v_out = go(index_array, grid_m_in, grid_v_in)
    x, v, C = g(x, grid_v_out)
    #x, v, C, F = advance(t, x, v, C, F, actuator_id)
    
  
    #print(timeit.timeit(lambda : g(x, grid_v_out), number=number) / number)
    #print(timeit.timeit(lambda : go(index_array, grid_m_in, grid_v_in), number=number) / number)
    #print(timeit.timeit(lambda : p2(x, v, new_F, affine, stress), number=number) / number)
    #print(timeit.timeit(lambda : p(x, v, C, F, actuator_id), number=number) / number)
  '''
    
    
  return jnp.mean(x[:, 0])
  
def main():
# initialization
  scene = Scene()
  robot(scene)
  scene.finalize()
  allocate_arrays()


  global x, v, C, F, actuator_id, particle_type #TODO: don't forget to reset these!
  x = jnp.array(scene.x)
  F = jnp.array(np.array([np.array([[1., 0.], [0., 1.]]) for _ in range(n_particles)]))

  actuator_id = jnp.array(scene.actuator_id, dtype=jnp.int32)
  particle_type = jnp.array(scene.particle_type)

  losses = []
  number = 10
  
  print('time advance')
  #IPython.embed()
  #print(timeit.timeit(lambda : a(0, x, v, C, F)[0].block_until_ready(), number=number) / number)
  #print(timeit.timeit(lambda : a(0, x, v, C, F)[0].block_until_ready(), number=number) / number)
  
  #for i in range(100):
  #  print(forward(x, v, C, F, actuator_id))
  f = jit(forward)
  forward_grad = jit(grad(forward)) #TODO: jit(grad(forward))
  
  #gradient = forward_grad(x, v, C, F, actuator_id)
  #val = forward(x, v, C, F)
  #dx = forward_grad(x, v, C, F)
  #print(jnp.sum(jnp.isnan(dx)))
  print('begin timeit')
  
  
  #print(f(x, v, C, F, actuator_id, actuation))
  print(timeit.timeit(lambda : f(x, v, C, F).block_until_ready(), number=number) / number)
  print(timeit.timeit(lambda : f(x, v, C, F).block_until_ready(), number=number) / number)
  print(timeit.timeit(lambda : forward_grad(x, v, C, F).block_until_ready(), number=number) / number)
  IPython.embed()
  print(timeit.timeit(lambda : forward_grad(x, v, C, F).block_until_ready(), number=number) / number)
  
  
  
  
if __name__ == "__main__":
  main()


