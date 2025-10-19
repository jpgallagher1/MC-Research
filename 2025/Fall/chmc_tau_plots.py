"""
Description:
    CHMC Implementation with AVF: FPI
    USE THE CORRECT ENVIRONMENT:  CHMC_FALL_2025
    YYYY-MM-DD

Author: John Gallagher
Created: 2025-09-28
Last Modified: 2025-10-16
Version: 1.0.0

"""

import jax
import jax.numpy as jnp
from jax import jit
import time

jax.config.update("jax_enable_x64", True)


def gauss_ndimf_jax(x, precision_matrix=None, cov=None, dim=2):
    """n-Dim Gaussian target distribution."""
    dim = len(x)

    # Error Classes
    class MultipleMatrices(Exception):
        pass

    # dealing with getting a precision matrix or cov matrix
    if precision_matrix is not None and cov is not None:
        raise MultipleMatrices(
            "Please supply either a Precision Matrix or a Covariance Matrix"
        )
    if precision_matrix is None and cov is not None:
        precision_matrix = jnp.linalg.inv(cov)
    if precision_matrix is None and cov is None:
        precision_matrix = jnp.eye(dim)
        # jnp.linalg.det(precision_matrix)**(-1/2)
        # (2*jnp.pi)**(-dim/2)*
    return jnp.exp(-0.5 * (x @ precision_matrix @ x))


def qex(qp):
    """
    qex: q extracted from qp state vector
    """
    dim = len(qp) // 2
    return qp[:dim]


def pex(qp):
    """
    pex: p extracted from qp state vector
    """
    dim = len(qp) // 2
    return qp[dim:]


def J_sym(vec):
    """
    J is the symplectic Jacobian matrix for Hamiltonians where J = ([[0, I]])
    """
    dim = len(vec) // 2
    return jnp.concatenate([vec[dim:], -vec[:dim]])


def qJ_sym(vec):
    """
    updates q side of vector with qdot = p
    Returns
    array([p], [0])
    """
    dim = len(vec) // 2
    return jnp.concatenate([vec[dim:], jnp.zeros(dim)])


def pJ_sym(vec):
    """
    qp with p = -qdot only
    Returns
    array([p], [0])
    """
    dim = len(vec) // 2
    return jnp.concatenate([jnp.zeros(dim), -vec[:dim]])


def draw_p(qp, key):
    q = qex(qp)
    p = jax.random.normal(key, shape=(dim,))
    return jnp.concatenate([q, p]), None


def gen_leapfrog(gradH, tau, N):
    def leapfrog(qp):
        """
        Requires gradH, tau, N
        Leapfrog integrator
        Takes state vector qp, and integrates it according to hamiltonian Ham

        """

        def lf_step(carry_in, _):
            qp0 = carry_in
            qhalf_p0 = qp0 + 0.5 * tau * qJ_sym(gradH(qp0))
            qhalf_pout = qhalf_p0 + tau * pJ_sym(gradH(qhalf_p0))
            qp_out = qhalf_pout + 0.5 * tau * qJ_sym(gradH(qhalf_pout))
            return qp_out, _

        qp_final, _ = jax.lax.scan(lf_step, qp, xs=None, length=N)
        return qp_final

    return leapfrog

def gen_midpointFPI(gradH, tau, N, tol,maxIter, solve = jnp.linalg.solve):
    """
    Generates midpointFPI function with appropriate statics: 
    tau, tol, maxIter, 
    """
    def midpointFPI(qp, _):
        """
        FPI_mid integrator
        Requries qp:statevector, and gradH defined before hand

        y(i+1) = y(i) + tau * J_sym GradH( 0.5*(y(i)+y(i+1)))

        """
        x0 = qp

        def G(y):
            """
            G(y) = x0 + tau * J_sym GradH( 0.5*(x+y))
            """
            midpoint = 0.5 * (x0 + y)
            return x0 + tau * J_sym(gradH(midpoint))

        def F(y):
            return y - G(y)

        def newton_step(qp):
            jacF = jax.jacobian(F)
            qpout = x0 - solve(jacF(qp), F(qp))
            return qpout

        def cond(carry):
            i, qp = carry
            Fqp = F(qp)
            err = jnp.linalg.norm(Fqp)
            return (err > tol) & (i < maxIter)

        def body_step(carry):
            i, qp = carry
            return [i + 1, newton_step(qp)]

        _, qp_out = jax.lax.while_loop(cond, body_step, [0, qp])
        return qp_out, qp_out
    def midpointFPI_T(qp):
        qp_out, _ = jax.lax.scan(midpointFPI, [qp, None], xs = None, length = N)
        return qp_out
    return midpointFPI_T

def accept(delta, key):
    alpha = jnp.minimum(1.0, jnp.exp(delta))
    u = jax.random.uniform(key, shape=())
    return u <= alpha


def gen_hmc_kernel(H, tau, N):
    gradH = jax.grad(H)
    integrator = gen_leapfrog(gradH, tau, N)

    def hmc_kernel(carry_in, key):
        carry, _, _ = carry_in
        qp0, _ = draw_p(carry, key)
        print(qp0)
        qp_star = integrator(qp0)
        deltaH = H(qp0) - H(qp_star)  # -(final - init) = init -final
        is_accepted = accept(deltaH, key)
        qp_out = jnp.where(is_accepted, qp_star, qp0)
        carry_out = [qp_out, deltaH, is_accepted]
        return carry_out, carry_out
    return hmc_kernel

def hmc_sampler(initial_sample, keys, H, tau, T):
    N = jnp.ceil(T/tau).astype(int)
    hmc_kernel = gen_hmc_kernel(H, tau, N)
    _, samples = jax.lax.scan(hmc_kernel, initial_sample, xs=keys)
    return samples

def gen_hamiltonian(Mass_inv, target):
    
    def hamiltonian(qp):
        q, p = qex(qp), pex(qp)
        return 0.5 * jnp.sum(p @ Mass_inv @ p) - jnp.log(target(q))
    return hamiltonian


# def J_H(gH):
#     """Same operation as Symplectic Jacobian"""
#     return jnp.concatenate([gH[dim:], -gH[:dim]])


dim = 2
Mass_inv = jnp.eye(dim)
target = gauss_ndimf_jax
hamiltonian = gen_hamiltonian(Mass_inv, target)
grad_target = jit(jax.grad(target))
jit_H = jit(hamiltonian)
gradH = jax.jit(jax.grad(hamiltonian))

# jit_integrator = jax.jit(leapfrog)
# jit_integrator = jit(midpointFPI)

# Set parameters
key = jax.random.PRNGKey(1)

initnum_samples = 1
mainnum_samples = 10000
keys_start = jax.random.split(key, initnum_samples)
keys_main = jax.random.split(key, mainnum_samples)
qp_init = jax.random.normal(key, shape=(2 * dim,))

# Structure of carry
# init_sample: [Array: sample, float: deltaH, bool: Accepted]
init_sample = [qp_init, 1, False]

tau = 0.2
T = 1
tol = 1e-4
max_iter = 100
# compile
start = time.time()
jhmc_sampler = jit(hmc_sampler)
sample_FPI = jhmc_sampler(init_sample, keys_start)
end = time.time()
print("1st run:", end - start)
# main run
start = time.time()
sample_FPI = jhmc_sampler(init_sample, keys_main)
end = time.time()
print(
    f"{mainnum_samples} runs: {end - start:.2f} \n 1 run:  {(end-start)/mainnum_samples}"
)
