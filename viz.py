import matplotlib.pyplot as plt
import numpy as np
def plot_measures(results_all, noise_mults):

    R, C = 1, 5
    plt.figure(figsize=(C * 3, R * 3)) #, dpi=300)
    # for n in ns_orig:
    for i, noise_mult in enumerate(noise_mults):
        r = results_all[noise_mult]
        ps, ns, train_scores, test_scores, wnorms, pseudo_traces, cov_traces, nuclear_norms,  H_traces = \
        r['ps'], r['ns'], r['train_scores'], r['test_scores'], np.array(r['wnorms']), np.array(r['pseudo_traces']), np.array(r['cov_traces']), np.array(r['nuclear_norms']), np.array(r['H_traces'])

    #     n = ns_orig[0]
        n = ns[0]
        # select what to paint
        for color in range(2):
            idxs = (ps/n < 0.72) + (ps/n > 0.78)
            if color == 1:
                idxs = ~idxs
            idxs *= ~np.isnan(train_scores)
            idxs *= (ps/n) < 2

            num_points = ps.size
            plt.subplot(R, C, 1)
            plt.plot((ps / n)[idxs], train_scores[idxs], label=f'noise_mult={noise_mult}')
            plt.xlabel('p/n')
            plt.ylabel('train mse')
            plt.yscale('log')
            plt.xscale('log')
        #     plt.legend()

            plt.subplot(R, C, 2)
            plt.plot((ps / n)[idxs], test_scores[idxs], '-')
            print('num nan', np.sum(np.isnan(test_scores)))
            plt.xlabel('p/n')
            plt.ylabel('test mse')
            plt.yscale('log')
            plt.xscale('log')

            plt.subplot(R, C, 3)
        #     if i == 2:
            plt.plot(H_traces[idxs], test_scores[idxs], '.-', alpha=0.5) #, c=np.arange(num_points)) #'red')
    #         plt.plot(cov_traces[idxs], test_scores[idxs], '.', alpha=0.5) #, c=np.arange(num_points)) #'red')
    #         plt.plot(nuclear_norms[idxs], test_scores[idxs], '.', alpha=0.5) #, c=np.arange(num_points)) #'red')    

            plt.xlabel('$tr[X (X^TX)^{-1}X^T]$')
            plt.ylabel('test mse')
            plt.yscale('log')
            plt.xscale('log')    

            plt.subplot(R, C, 4)
            plt.plot(wnorms[idxs], test_scores[idxs], '.', alpha=0.5)
            plt.xlabel('$||\hat{w}||_2$')
            plt.ylabel('test mse')
            plt.yscale('log')
            plt.xscale('log')

            plt.subplot(R, C, 5)
            plt.plot(np.abs(np.array(wnorms) - 1)[idxs], test_scores[idxs], '.', alpha=0.5)
            plt.xlabel('$abs(||\hat{w}||_2 - ||w^*||_2)$')
            plt.ylabel('test mse')
            plt.yscale('log')
            plt.xscale('log') 

    plt.tight_layout()
    plt.show()
    
    
    
def plot_Htrace(results_all, noise_mults):
    R, C = 1, 5
    plt.figure(figsize=(C * 3, R * 3), dpi=300) #, dpi=300)
    # for n in ns_orig:
    for i, noise_mult in enumerate(noise_mults):
        print(results_all.keys())
        r = results_all[noise_mult]
        ps, ns, train_scores, test_scores, wnorms, H_traces = \
        r['ps'], r['ns'], r['train_scores'], r['test_scores'], np.array(r['wnorms']), np.array(r['H_traces'])
        bias_list, var_list = np.array(r['bias_list']), np.array(r['var_list'])
        
        train_scores = np.array([np.mean(train_scores[i]) for i in range(train_scores.shape[0])])
        test_scores = np.array([np.mean(test_scores[i]) for i in range(test_scores.shape[0])])
        H_traces = np.array([np.mean(H_traces[i]) for i in range(H_traces.shape[0])])

    #     n = ns_orig[0]
        n = ns[0]
        # select what to paint
        for color in range(3):
            if color == 0:
                idxs = (ps/n - 1 < -0.1)
            elif color == 1:
                idxs = (np.abs(ps/n - 1) < 0.1)
            elif color ==2:
                idxs = (ps/n - 1 > 0.1)
#                 idxs = ~idxs
#             idxs *= ~np.isnan(train_scores)
#             idxs *= (ps/n) < 2

            num_points = ps.size
            plt.subplot(R, C, 1)
            plt.plot((ps / n)[idxs], train_scores[idxs], '.-', label=f'noise_mult={noise_mult}', alpha=0.5)
            plt.xlabel('p/n')
            plt.ylabel('train mse')
            plt.yscale('log')
            plt.xscale('log')
        #     plt.legend()

            plt.subplot(R, C, 2)
            plt.plot((ps / n)[idxs], test_scores[idxs], '.-', alpha=0.5)
            print('num nan', np.sum(np.isnan(test_scores)))
            plt.xlabel('p/n')
            plt.ylabel('test mse')
            plt.yscale('log')
            plt.xscale('log')

            plt.subplot(R, C, 3)
        #     if i == 2:
            plt.plot((ps / n)[idxs], np.square(bias_list[idxs]), '.-', alpha=0.5) #, c=np.arange(num_points)) #'red')
            plt.xlabel('p/n')
            plt.ylabel('bias$^2$')
            plt.yscale('log')
            plt.xscale('log') 


            plt.subplot(R, C, 4)
        #     if i == 2:
            plt.plot((ps / n)[idxs], var_list[idxs], '.-', alpha=0.5) #, c=np.arange(num_points)) #'red')
            plt.xlabel('p/n')
            plt.ylabel('var')
            plt.yscale('log')
            plt.xscale('log')            
            
            plt.subplot(R, C, 5)
        #     if i == 2:
            plt.plot(var_list[idxs], test_scores[idxs], '.-', alpha=0.5) #, c=np.arange(num_points)) #'red')
            plt.xlabel('var')
            plt.ylabel('test mse')
            plt.yscale('log')
            plt.xscale('log')            
            
            '''
            plt.subplot(R, C, 3)
        #     if i == 2:
            plt.plot((ps / n)[idxs], H_traces[idxs], '.-', alpha=0.5) #, c=np.arange(num_points)) #'red')
    #         plt.plot(cov_traces[idxs], test_scores[idxs], '.', alpha=0.5) #, c=np.arange(num_points)) #'red')
    #         plt.plot(nuclear_norms[idxs], test_scores[idxs], '.', alpha=0.5) #, c=np.arange(num_points)) #'red')    

            plt.xlabel('p/n')
            plt.ylabel('$tr[H]$')
            plt.yscale('log')
            plt.xscale('log')                
            
            plt.subplot(R, C, 4)
        #     if i == 2:
            plt.plot(H_traces[idxs], test_scores[idxs], '.-', alpha=0.5) #, c=np.arange(num_points)) #'red')
            plt.xlabel('$tr[H]$')
            plt.ylabel('test mse')
            plt.yscale('log')
            plt.xscale('log')
            '''
            

    plt.tight_layout()
    plt.show()