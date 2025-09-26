module load python3/3.10.7
export PATH="${PATH}:/zhome/00/4/50173/.local/bin"

pip install --user --upgrade pip

pip install --user --upgrade optax tensorboard tensorboard_plugin_profile pytorch_lightning  matplotlib smt pydot graphviz h5py tqdm meshio "jax[cuda12-local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax orbax-checkpoint

# On MAC
pip install --user --upgrade jax-metal jax flax optax orbax-checkpoint tensorboard tensorboard_plugin_profile pytorch_lightning  matplotlib smt pydot graphviz h5py tqdm meshio