# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
from typing import Any
from matplotlib import pyplot as plt
import jax.numpy as jnp
import numpy as np
import jax
from jax import random, vmap, jit
import optax
import flax
from torch.utils.tensorboard import SummaryWriter
import os
import collections
from flax.training import checkpoints
import orbax.checkpoint
from deeponet_acoustics.models.datastructures import NetworkArchitectureType, NetworkContainer, TrainingSettings, TransferLearning
from deeponet_acoustics.utils.timings import TimingsWriter
from deeponet_acoustics.utils.utils import expandCnnData
from deeponet_acoustics.models.networks_flax import flattened_traversal, freezeLayersToKeys, freezeCnnLayersToKeys
from deeponet_acoustics.models import loss_functions
from deeponet_acoustics.datahandlers.datagenerators import DataInterface


from functools import partial
from tqdm import trange

LossLogger = collections.namedtuple('LossLogger', ['loss_train', 'loss_val', 'nIter'])

def exponential_decay(step_size, decay_steps, decay_rate, step_offset=0):
  def schedule(i):
    return step_size * decay_rate ** ((i + step_offset) / decay_steps)
  return schedule

TAG_BN = "bn"
TAG_TN = "tn"
TAG_B0 = "b0"
TAG_ADAPTIVE = "adaptive_weights"

# Define the model
class DeepONet:    
    is_bn_fnn: bool
    params: flax.core.FrozenDict
    branch_apply: Any
    trunk_apply: Any    

    def __init__(self, settings: TrainingSettings, dataset: DataInterface, module_bn: NetworkContainer, module_tn: NetworkContainer, 
                 log_dir, transfer_learning: TransferLearning=None):
        
        lr = settings.learning_rate
        if settings.use_adaptive_weights:
            adaptive_weights_shape = min(settings.batch_size_branch, dataset.N) * min(settings.batch_size_coord, dataset.P),
        else:
            adaptive_weights_shape = []

        decay_steps = settings.decay_steps
        decay_rate = settings.decay_rate

        self.loss_logger = LossLogger([],[],[])                    
        self.step_offset = 0

        self.log_dir = log_dir
        self.is_bn_fnn = module_bn.network.network_type != NetworkArchitectureType.RESNET        
        dim_bn = module_bn.in_dim
        dim_tn = module_tn.in_dim        

        if transfer_learning is None:
            

            branch_params = module_bn.network.init(
                random.PRNGKey(1234), 
                jnp.expand_dims(jnp.ones(dim_bn), axis=0) if self.is_bn_fnn else expandCnnData(np.ones(dim_bn))
            )
            
            trunk_params  = module_tn.network.init(random.PRNGKey(4321), 
                                            jnp.ones(dim_tn))
            if len(adaptive_weights_shape) > 0:
                self.params = flax.core.frozen_dict.freeze(
                    {TAG_BN: branch_params, TAG_TN: trunk_params, TAG_B0: 0.0,
                     TAG_ADAPTIVE: jnp.ones(adaptive_weights_shape)}
                )
            else:
                self.params = flax.core.frozen_dict.freeze(
                    {TAG_BN: branch_params, TAG_TN: trunk_params, TAG_B0: 0.0}
                )

            freeze_layers = set()
        else:            
            ckpt_dir = transfer_learning.transfer_model_path
            resume = transfer_learning.resume_learning
            freeze_layers = freezeLayersToKeys(transfer_learning.freeze_layers)

            if resume:
                self.step_offset, self.loss_logger = loadLosses(ckpt_dir)        

            # https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html            
            self.params = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)
            if self.params == None:
                raise Exception(f"Could not load model parameter checkpoint at {ckpt_dir}")
            
            if len(adaptive_weights_shape) == 0:
                self.params.pop(TAG_ADAPTIVE, None) # remove adaptive weights if existing
            elif TAG_ADAPTIVE in self.params:
                if adaptive_weights_shape != self.params[TAG_ADAPTIVE].shape:
                    raise Exception("Mismatch between the loaded adaptive weights and the batch dimensions. Reset the batch dimension to equal values as for the transfered model or disable/modify the self-adaptive weights")
            else:
                self.params[TAG_ADAPTIVE] = jnp.ones(adaptive_weights_shape)

            if not self.is_bn_fnn:
                freeze_layers_cnn = freezeCnnLayersToKeys(self.params[TAG_BN])
                freeze_layers.update(freeze_layers_cnn)

            self.params = flax.core.frozen_dict.freeze(self.params)

        #print(freeze_layers)
        #print(jax.tree_map(jnp.shape, self.params))

        self.branch_apply = module_bn.network.apply
        self.trunk_apply = module_tn.network.apply

        self.opt_scheduler = exponential_decay(
            lr, decay_steps=decay_steps, decay_rate=decay_rate, step_offset=self.step_offset
            )
            
        def optimizerSelector(path, _):
            if path in freeze_layers:
                return 'none'
            elif path == TAG_ADAPTIVE:
                return 'opt_adaptive_weights'
            else:
                return 'opt'

        # path consists of ('params', 'tag', 'bias'), where 'tag' is the layer tag to freeze
        # 'b0' and 'adaptive_weights' are a special cases, where path = ('b0')
        tags = [TAG_BN,TAG_TN,TAG_B0,TAG_ADAPTIVE] if len(adaptive_weights_shape) > 0 else [TAG_BN,TAG_TN,TAG_B0]
        label_fn = flattened_traversal(optimizerSelector, tags)

        self.optimizer = optax.chain(            
            optax.clip_by_global_norm(0.01), # ensure no extreme flucturation in loss
            optax.multi_transform(
               {'opt': optax.adamw(learning_rate=self.opt_scheduler), 
                'opt_adaptive_weights': optax.adam(1e-5), # hardcoded
                'none': optax.set_to_zero()
                }, 
                label_fn)
            )

        self.opt_state = self.optimizer.init(self.params)

    def train(self, dataloader, dataloader_val, nIter, save_every=200, do_timings=False):
        """Main train loop using dataloaders (3D data)."""
        writer = SummaryWriter(log_dir=self.log_dir)
        timer = TimingsWriter(log_dir=self.log_dir) if do_timings else None

        num_batches = np.ceil(dataloader.dataset.N/dataloader.batch_size)
        
        pbar_epochs = trange(np.ceil(nIter/num_batches).astype('int'))

        i = self.step_offset
        if i == 0:
            self.writeState(i, pbar_epochs, dataloader, dataloader_val, writer)
        
        timer.resetTimings() if do_timings else None
        timer.startTiming('total_iter') if do_timings else None
        timer.startTiming('dataloader') if do_timings else None
        for _ in pbar_epochs:             
            for _, data_batch in enumerate(dataloader):
                jax.block_until_ready(data_batch) if do_timings else None
                timer.endTiming('dataloader') if do_timings else None

                i += 1
                
                if do_timings:
                    timer.startTiming('backprop') if do_timings else None
                    self.params, self.opt_state, _ = self.step(self.params, self.opt_state, data_batch)
                    jax.block_until_ready(self.params) if do_timings else None
                    jax.block_until_ready(self.opt_state) if do_timings else None
                    timer.endTiming('backprop') if do_timings else None

                    timer.writeTimings({'total_iter': 'Total time iter:', 
                                        'dataloader': 'Dataloader:',
                                        'backprop': 'Back-propagation:'})
                    timer.resetTimings()
                    timer.startTiming('total_iter') if do_timings else None
                    timer.startTiming('dataloader') if do_timings else None
                else:
                    self.params, self.opt_state, _ = self.step(self.params, self.opt_state, data_batch)
                
                if i % save_every == 0:             
                    self.writeState(i, pbar_epochs, dataloader, dataloader_val, writer)
        
        # save final result
        if i % save_every != 0:
            self.writeState(i, pbar_epochs, dataloader, dataloader_val, writer)
    
    def trainFromDataset(self, dataset, dataset_val, nIter, save_every=100, do_timings=False):
        """Main train loop using dataset directly (currently for 1D/2D data)."""
        writer = SummaryWriter(log_dir=self.log_dir)
        timer = TimingsWriter(log_dir=self.log_dir) if do_timings else None

        data = iter(dataset)
        pbar = trange(nIter)

        i = self.step_offset
        if i == 0:
            self.writeState(i, pbar, dataset, dataset_val, writer)
        

        timer.resetTimings() if do_timings else None
        for _ in pbar:            
            timer.startTiming('total_iter') if do_timings else None
            i += 1

            timer.startTiming('dataloader') if do_timings else None
            data_batch = next(data)
            jax.block_until_ready(data_batch) if do_timings else None
            timer.endTiming('dataloader') if do_timings else None

            if do_timings:
                timer.startTiming('backprop')
                self.params, self.opt_state, _ = self.step(self.params, self.opt_state, data_batch)
                jax.block_until_ready(self.params)
                jax.block_until_ready(self.opt_state)
                timer.endTiming('backprop')
                timer.endTiming('total_iter')

                timer.writeTimings({'total_iter': 'Total time iter:', 
                                    'dataloader': 'Dataloader:',
                                    'backprop': 'Back-propagation:'})
                timer.resetTimings()
            else:
                self.params, self.opt_state, _ = self.step(self.params, self.opt_state, data_batch)

            if i % save_every == 0:
                self.writeState(i, pbar, dataset, dataset_val, writer)

        # save final result (if not already done)
        if i % save_every != 0:
            self.writeState(i, pbar, dataset, dataset_val, writer)


    def operator_net(self, params, B, y):
        trunk_params, b0 = params[TAG_TN], params[TAG_B0]
        T = self.trunk_apply(trunk_params, y)
        
        return jnp.sum(B * T) + b0

    def branch_net(self, params, u):
        branch_params = params[TAG_BN]
        if self.is_bn_fnn:
            return self.branch_apply(branch_params, u)
        else:
            return self.branch_apply(branch_params, expandCnnData(u), mutable=['batch_stats'])[0]
    
    # Define total loss
    def loss(self, params, batch):
        return loss_functions.loss(params, batch, 
                                   self.branch_net, self.operator_net, 
                                   apply_adaptive_weights=TAG_ADAPTIVE in params)
            
    # Define a compiled update step
    @partial(jit, static_argnums=(0))
    def step(self, params_all, opt_state, data_batch):
        def traverse_dict(fn, params):
            flat_dict = flax.traverse_util.flatten_dict(params)
            return flax.traverse_util.unflatten_dict(
                {k: fn(k, v) for k, v in flat_dict.items()})
        
        idx_coord = data_batch[2].flatten() # get coordinate indexes for the current batch
        
        # extract adaptive weights for the current batch
        params = traverse_dict(lambda k,v: v[idx_coord] if TAG_ADAPTIVE in k else v, params_all)
        params = flax.core.frozen_dict.freeze(params)

        # calculate gradients for network parameters and adaptive weights
        loss_value, grads = jax.value_and_grad(self.loss)(params, data_batch)
        
        # negate gradient for adaptive weights
        grads = traverse_dict(lambda k,v: -v if TAG_ADAPTIVE in k else v, grads)
        grads = flax.core.frozen_dict.freeze(grads)

        # update network parameters and adaptive weights
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        def clip_and_combine_adaptive_weights(k,v):
            if TAG_ADAPTIVE in k:
                adaptive_weights_all = params_all[TAG_ADAPTIVE]
                weights_clipped = jnp.clip(params[TAG_ADAPTIVE],0,1000)
                return adaptive_weights_all.at[idx_coord].set(weights_clipped)
            else:
                return v

        # re-insert the updated adaptive weights for this batch into the full list of weights (and clip)
        params = traverse_dict(clip_and_combine_adaptive_weights, params)
        params = flax.core.frozen_dict.freeze(params)
        
        return params, opt_state, loss_value

    def plotLosses(self, figs_dir=None):
        plt.figure(figsize = (6,5))
        plt.plot(self.loss_logger.nIter, self.loss_logger.loss_train, lw=2, label='Training loss')
        plt.plot(self.loss_logger.nIter, self.loss_logger.loss_val, lw=2, label='Validation loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        if figs_dir == None:
            plt.show()
        else:
            fig_path = os.path.join(figs_dir, 'loss.png')
            plt.savefig(fig_path,bbox_inches='tight',pad_inches=0)

    def writeState(self, it, pbar_epochs, dataloader_train, dataloader_val, writer):
        self.loss_logger.nIter.append(it)

        # training loss
        data_train_batch = next(iter(dataloader_train))
        loss_train_value = loss_functions.loss(self.params, data_train_batch, 
                                               self.branch_net, self.operator_net,
                                               apply_adaptive_weights=False)

        # validation loss
        data_val_batch = next(iter(dataloader_val))
        loss_val_value = loss_functions.loss(self.params, data_val_batch, 
                                             self.branch_net, self.operator_net, 
                                             apply_adaptive_weights=False)

        # Store losses
        self.loss_logger.loss_train.append(loss_train_value)
        self.loss_logger.loss_val.append(loss_val_value)

        # Print losses        
        pbar_epochs.set_postfix({'Train loss': loss_train_value, 'Val loss': loss_val_value})                

        # Save loss to disk
        self.writeSummary(writer, loss_train_value, loss_val_value, it)
                
        # Save model to disk
        self.writeModel(it) #, write_separate=True

    def writeModel(self, iter):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                            target=self.params,
                            step=iter,
                            overwrite=False,
                            orbax_checkpointer=orbax_checkpointer)

    def writeSummary(self, writer, loss_train, loss_val, iter):
        writer.add_scalar(f'Loss/train/loss', np.array(loss_train), iter)        
        writer.add_scalar(f'Loss/val/loss', np.array(loss_val), iter)
        writer.add_scalar(f'Loss/learning_rate', np.array(self.opt_scheduler(iter-self.step_offset)), iter)

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        branch_latent = self.branch_net(params, U_star)
        s_pred = vmap(self.operator_net, (None, None, 0))(params, branch_latent, Y_star)
        return s_pred

def loadLosses(path: str):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    
    step_offset = event_acc.Scalars('Loss/learning_rate')[-1].step        
    nIter = list(map(lambda e: e.step, event_acc.Scalars('Loss/train/loss')))
    loss_train = list(map(lambda e: e.value, event_acc.Scalars('Loss/train/loss')))
    loss_val = list(map(lambda e: e.value, event_acc.Scalars('Loss/val/loss')))

    return step_offset, LossLogger(loss_train,loss_val,nIter)