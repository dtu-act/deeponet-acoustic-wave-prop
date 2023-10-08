# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import os
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
import utils.dsp as dsp
import utils.utils as utils
import matplotlib.animation as anim
from pathlib import Path

figsize_x, figsize_y = 8, 4

def plotReceiver(t,p_pred,p_ref,x0_srcs,tmax,c_phys,figs_dir=None,plot_axis=True):
    """ Plot impulse response and transfer function for each source position """

    for i, x0 in enumerate(x0_srcs):
        p_ref_i = p_ref[i,:]
        p_pred_i = p_pred[i,:]        
        
        format_fn = lambda x: [str("%.2f" % e) for e in x]
        path_file_td = os.path.join(figs_dir, "td_t=%s_x0=%s.png" % (str("%.2f" % tmax), format_fn(x0)))
        path_file_tf = os.path.join(figs_dir, "tf_t=%s_x0=%s.png" % (str("%.2f" % tmax), format_fn(x0)))
        
        show_legends = False #i==0

        plotTimeDomain(t/c_phys, t/c_phys, p_ref_i, p_pred_i, show_legends=show_legends, path_file=path_file_td, plot_axis=plot_axis)
        plotTransferFunction(p_pred_i, p_ref_i, tmax/c_phys, freq_min_max=[20, 1000], show_legends=show_legends, path_file=path_file_tf, plot_axis=plot_axis)

def plotAtReceiverPosition(srcs,recvs,recvs_indxs,t_phys,p_pred,p_ref,tmax_phys,figs_dir=None,plot_axis=True, animate=False):
    """ Plot impulse response and transfer function for each source position """

    filepath = os.path.join(figs_dir, "err.txt")
    with open(filepath, 'w') as f:
        for i, x0 in enumerate(srcs):
            r0 = recvs[i,:]

            p_ref_i = p_ref[i,:,recvs_indxs[i]]
            p_pred_i = p_pred[i,:,recvs_indxs[i]]
            
            format_fn = lambda x: [str("%.2f" % e) for e in x]
            path_file_td = os.path.join(figs_dir, "td_t=%s_x0=%s_r0=%s.png" % (str("%.2f" % (tmax_phys*343)), format_fn(x0), format_fn(r0)))
            path_file_tf = os.path.join(figs_dir, "tf_t=%s_x0=%s_r0=%s.png" % (str("%.2f" % (tmax_phys*343)), format_fn(x0), format_fn(r0)))

            show_legends = False #i==0

            plotTimeDomain(t_phys, t_phys, p_ref_i, p_pred_i, show_legends=show_legends, path_file=path_file_td, plot_axis=plot_axis, animate=animate)
            plotTransferFunction(p_pred_i, p_ref_i, tmax_phys, freq_min_max=[20, 1000], show_legends=show_legends, path_file=path_file_tf, plot_axis=plot_axis)
            utils.calcErrors(p_pred_i, p_ref_i, x0, r0, f)

def plotTimeDomain(t1d_pred, t1d_ref, p_pred, p_ref, show_legends=False, path_file=None, show_plot=False, plot_axis=True, animate=False):
    indx0_max = max(enumerate(np.abs(p_ref.flatten())),key=lambda x: x[1])[0]
    indx1_max = max(enumerate(np.abs(p_pred.flatten())),key=lambda x: x[1])[0]
    p_max = max(np.abs(p_ref[indx0_max]), np.abs(p_pred[indx1_max]))

    fig = plt.figure(figsize=(figsize_x, figsize_y))      
    plt.plot(t1d_ref, p_ref/p_max, linestyle='-', linewidth=4, color='blue')
    plt.plot(t1d_pred, p_pred/p_max, linestyle='--', linewidth=4, color='red')
    if show_legends:
        plt.legend(['Pred', 'Ref'], loc='upper right')
    
    plt.grid()        
    #plt.ylim([min(min(p_ref),min(p_pred)),max(max(p_ref),max(p_pred))])
    plt.ylim([-1.0,1.0])
    if plot_axis:                    
        #fig.axes[0].get_yaxis().set_ticks([-0.005, 0.125, 0.25, 0.375, 0.505])
        #fig.axes[0].get_yaxis().set_ticklabels(['0', '0.125', '0.25', '0.375', '0.5'])
        plt.xlabel('t [sec]')
        plt.ylabel('pressure [Pa]')
    else:
        fig.axes[0].get_yaxis().set_visible(False)
        fig.axes[0].get_xaxis().set_visible(False)
    
    if path_file != None:
        plt.savefig(path_file,bbox_inches='tight',pad_inches=0)

    if show_plot:
        plt.show(block=True)

    plt.close()
    
    if animate:
        gifAnimate(t1d_pred, t1d_ref, p_pred/p_max, p_ref/p_max, path_file=path_file)

def gifAnimate(t1d_pred, t1d_ref, p_pred, p_ref, path_file=None):
    # PRED/REF ANIMATE
    fig = plt.figure(figsize=(figsize_x, figsize_y))
    ax = plt.subplot()

    ax.set_xlim([0, 0.05])
    ax.set_ylim([-1, 1])
    # ax.set_xlabel('t [sec]')
    # ax.set_ylabel('pressure [Pa]')
    ax.grid()

    legends = ['Pred', 'Ref']
    line_params = [[4,'-','blue'], [4,'--','red']]
    N=len(line_params)

    lines = [ax.plot([], [], linewidth=line_params[i][0], linestyle=line_params[i][1], color=line_params[i][2], 
        label=legends[i])[0] for i in range(N)] #lines to animate
    
    def init():
        for line in lines:
            line.set_data([], [])

        return lines #return everything that must be updated

    def animate(i):
        lines[0].set_data(t1d_ref[0:i], p_ref[0:i])
        lines[1].set_data(t1d_pred[0:i], p_pred[0:i])
        
        return lines
    
    ani = anim.FuncAnimation(fig, animate, init_func=init, frames=len(t1d_pred), blit=True, repeat=False)    

    if path_file != None:
        path_file_gif = Path(path_file)
        path_file_gif = path_file_gif.with_suffix('.gif')
        ani.save(path_file_gif)

    plt.close()

    # ERROR GIF
    fig = plt.figure(figsize=(figsize_x, figsize_y))
    ax = plt.subplot()

    ax.set_xlim([0, 0.05])
    ax.set_ylim([0, max(abs(p_ref - p_pred))])
    # ax.set_xlabel('t [sec]')
    # ax.set_ylabel('pressure [Pa]')
    ax.grid()

    line = ax.plot([], [], linewidth=4, linestyle='-', color='magenta')
    
    def init():
        line[0].set_data([], [])
        return lines #return everything that must be updated

    def animate(i):
        line[0].set_data(t1d_ref[0:i], abs(p_ref[0:i] - p_pred[0:i]))
        return lines
    
    ani = anim.FuncAnimation(fig, animate, init_func=init, frames=len(t1d_pred), blit=True, repeat=False)    

    if path_file != None:
        path_file_gif = Path(path_file)
        path_file_gif = path_file_gif.with_suffix('')
        path_file_gif = str(path_file_gif) + '_err'
        path_file_gif = Path(path_file_gif).with_suffix('.gif')
        ani.save(path_file_gif)

def plotTransferFunction(p_pred, p_ref, tmax, freq_min_max=[0,np.inf], show_legends=False, path_file=None, show_plot=False, plot_axis=True):
    N = len(p_pred)
    dt = tmax/N
    fs = 1/dt

    indx0_max = max(enumerate(np.abs(p_ref.flatten())),key=lambda x: x[1])[0]
    indx1_max = max(enumerate(np.abs(p_pred.flatten())),key=lambda x: x[1])[0]
    p_max = max(np.abs(p_ref[indx0_max]), np.abs(p_pred[indx1_max]))

    fft_values_pred, f_values_pred = dsp.calcFFT(p_pred.flatten()/p_max, fs, NFFT=1024)
    fft_values_ref, f_values_ref = dsp.calcFFT(p_ref.flatten()/p_max, fs, NFFT=1024)

    indx_min = np.where(freq_min_max[0] > f_values_pred)[0][-1]
    indx_max = np.where(freq_min_max[1] < f_values_pred)[0]
    indx_max = len(f_values_pred) if len(indx_max) == 0 else indx_max[0]

    f_values_pred = f_values_pred[indx_min:indx_max]
    fft_values_pred = fft_values_pred[indx_min:indx_max]
    f_values_ref = f_values_ref[indx_min:indx_max]
    fft_values_ref = fft_values_ref[indx_min:indx_max]

    ref0 = 2e-5 # SPL reference
    freq_rms_pred = fft_values_pred/np.sqrt(2)
    freq_rms_ref = fft_values_ref/np.sqrt(2)

    fig = plt.figure(figsize=(figsize_x, figsize_y))
    plt.plot(f_values_pred, 20*np.log10(freq_rms_pred/ref0), linestyle='-', linewidth=4, color='blue')
    plt.plot(f_values_ref, 20*np.log10(freq_rms_ref/ref0), linestyle='--', linewidth=4, color='red')
    # plt.plot(f_values_pred, fft_values_pred, linestyle='-', linewidth=4, color='blue')
    # plt.plot(f_values_ref, fft_values_ref, linestyle='--', linewidth=4, color='red')
    if show_legends:
        plt.legend(['Pred', 'Ref'], loc='upper right')
    
    plt.grid()
    plt.ylim([10,70])
    
    if plot_axis:        
        fig.axes[0].get_xaxis().set_ticks([20,200,600,1000])
        fig.axes[0].get_xaxis().set_ticklabels(['20','200','600','1000    '])
        plt.xlabel('Frequency [Hz]')
    
    else:
        #plt.axis('off')
        fig.axes[0].get_yaxis().set_visible(False)
        fig.axes[0].get_xaxis().set_visible(False)

    if path_file != None:
        plt.savefig(path_file,bbox_inches='tight',pad_inches=0)

    if show_plot:
        plt.show(block=True)
    plt.close()

## 1D ##
def plotPrediction1D(XX,TT,S_pred,S_test,x0,show_plot=False,path_file=None):
    colormap = cm.magma_r
    # Visualization
    fig = plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    plt.pcolor(XX,TT, S_test, cmap=colormap)
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title(f'Exact $s(x,t)$ for $x_0={x0}$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(1,3,2)
    plt.pcolor(XX,TT, S_pred, cmap=colormap)
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title(f'Predict $s(x,t)$ for $x_0={x0}$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(1,3,3)
    plt.pcolor(XX,TT, np.abs(S_pred - S_test), cmap=colormap)
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title(f'Absolute $L_1$ error for $x_0={x0}$')
    plt.colorbar()
    plt.tight_layout()

    if path_file != None:
        fig_path = os.path.join(path_file, f'results_srcs{x0}.png')
        plt.savefig(fig_path,bbox_inches='tight',pad_inches=0)

    if show_plot:
        plt.show(block=True)

def plotWaveFields1D(XX,TT,p_pred,p_ref,x0_srcs,tmax,c_phys,plot_cbar=False,figs_dir=None, plot_axis=True):
    """ Plot the prediction, reference and L1 error for each source position """

    r0_list, _ = utils.calcReceiverPositionsSimpleDomain(XX, x0_srcs)

    for i, x0 in enumerate(x0_srcs):
        p_pred_i = p_pred[i].reshape(-1,1)
        p_ref_i = p_ref[i].reshape(-1,1)
        x1d = XX.flatten()
        t1d = TT.flatten()

        r0 = r0_list[i]

        # all x0 values are the same for each source - extract first [0]
        path_file_ref = os.path.join(figs_dir, "p_ref_x0=%0.2f.png" % x0)
        path_file_pred = os.path.join(figs_dir, "p_pred_x0=%0.2f.png" % x0)
        path_file_err = os.path.join(figs_dir, "err_L1_x0=%0.2f.png" % x0)
        
        path_cbar_file_ref = os.path.join(figs_dir, "cbar_ref_x0=%0.2f.png" % x0) if plot_cbar else None
        path_cbar_file_pred = os.path.join(figs_dir, "cbar_pred_x0=%0.2f.png" % x0) if plot_cbar else None
        path_cbar_file_err = os.path.join(figs_dir, "cbar_err_L1_x0=%0.2f.png" % x0) if plot_cbar else None
        
        label_str = 'Receiver' if i==0 else None

        err_L1 = np.abs(p_pred_i - p_ref_i).flatten()

        plotData1D(x1d, t1d/c_phys, p_ref_i.flatten(), 
            v_minmax=[0,1], vline=[r0,'--', 4, 'red', label_str],
            path_file=path_file_ref, path_cbar_file=path_cbar_file_ref, plot_axis=plot_axis)
        plotData1D(x1d, t1d/c_phys, p_pred_i.squeeze().flatten(), 
            v_minmax=[0,1], vline=[r0,'-', 4, 'blue', label_str],
            path_file=path_file_pred, path_cbar_file=path_cbar_file_pred, plot_axis=plot_axis)
        plotData1D(x1d, t1d/c_phys, err_L1, 
            vline=[r0,'-', 4, 'orange', label_str], v_minmax=[0,0.02], 
            path_file=path_file_err, path_cbar_file=path_cbar_file_err, plot_axis=plot_axis)

def plotData1D(XX, TT, p, vline=None, path_file=None, path_cbar_file=None, v_minmax=[], show_plot=False, plot_axis=False, colormap=cm.magma_r):
    res = 160
    fig = plt.figure(figsize=(figsize_x, figsize_y))
    if v_minmax:
        cax = plt.tricontourf(XX, TT, p, res, cmap=colormap, vmin=v_minmax[0], vmax=v_minmax[1])
    else:
        cax = plt.tricontourf(XX, TT, p, res, cmap=colormap)

    if vline:
        plt.axvline(x=vline[0], linestyle=vline[1], linewidth=vline[2], color=vline[3], label=vline[4])
        if vline[4]:
            plt.legend()        
        
    plt.tight_layout()
    if plot_axis:
        plt.axis('off')
        fig.axes[0].get_xaxis().set_ticks([-1.0, 0.0, 1.0])
        fig.axes[0].get_xaxis().set_ticklabels(['      -1.0', '0.0', '1.0     '])
        plt.xlabel('x [m]')
        plt.ylabel('t [s]')
        plt.colorbar()
    else:
        plt.axis('off')
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)

    if path_file != None:
        plt.savefig(path_file,bbox_inches='tight',pad_inches=0)
    
    if path_cbar_file != None:
        fig,ax = plt.subplots(figsize=(4, 4))
        
        if v_minmax:
            prec = 3
            tick_low = v_minmax[0]
            tick_high = v_minmax[1]
            tick_mid = (v_minmax[1]-v_minmax[0])/2
            cbar = plt.colorbar(cax,ax=ax,ticks=[tick_low, tick_mid, tick_high])
            cbar.set_ticklabels([f'{round(tick_low,prec)}', f'{round(tick_mid,prec)}', f'>{round(tick_high,prec)}'])
        else:
            cbar = plt.colorbar(cax,ax=ax)
        
        cbar.ax.tick_params(labelsize=12)

        ax.remove()
        plt.savefig(path_cbar_file,bbox_inches='tight',pad_inches=0, dpi=800)

    if show_plot:
        plt.show(block=True)
    plt.close()