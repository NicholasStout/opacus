import pandas as pd
import numpy as np
from check_privacy_and_statistics import compute_privacy_lmo, compute_usefulness_lmo


def take_Si(SS):
    # Function to fetch a set of parameters at one time. 
    times=1
    for item in SS:
        times=times*len(SS[item])

    Si={}
    move=True
    one_round=len(SS)-1
    pos=[int(i) for i in np.zeros(len(SS))]

    cnt=0
    for _ in range(times):
        for idx, ss in enumerate(SS):
            try:
                if ss=='G_theta_k':
                    Si["G_theta"]=SS[ss][pos[idx]][0]
                    Si["G_k"]=SS[ss][pos[idx]][1]
                elif ss=="U_b_a":
                    Si["U_b"]=SS[ss][pos[idx]][0]
                    Si["U_a"]=SS[ss][pos[idx]][1]
                else:
                    Si[ss]=SS[ss][pos[idx]]
            except:
                pos[idx]=0
                pos[int(idx+1)]=pos[int(idx+1)]+1
                
                if ss=='G_theta_k':
                    Si["G_theta"]=SS[ss][pos[idx]][0]
                    Si["G_k"]=SS[ss][pos[idx]][1]
                elif ss=="U_b_a":
                    Si["U_b"]=SS[ss][pos[idx]][0]
                    Si["U_a"]=SS[ss][pos[idx]][1]
                else:
                    Si[ss]=SS[ss][pos[idx]]
                
            if move:
                pos[idx]+=1 if pos[idx]<len(SS[ss]) else pos[idx]
                move=False
            if cnt<one_round:
                cnt=cnt+1
                continue
            else:
                yield Si
                Si={}
                move=True
                cnt=0


def search_epsilon(SS, epsilon_threshold, demo_cnt=1000):
    # The searched condition is epsilon_threshold.
    cnt = 0
    searched_parameters = {}
    searched_parameters["overall_epsilon"] = 0
    for lmo in take_Si(SS):
        privacy_lmo = compute_privacy_lmo(lmo)
        if privacy_lmo==[]:
            continue
        else:
            overall_epsilon, opt_order, rdp_lmo = privacy_lmo
            if overall_epsilon < epsilon_threshold and np.isreal(rdp_lmo[int(opt_order-2)]):
                if overall_epsilon > searched_parameters['overall_epsilon']:
                    searched_parameters = lmo
                    searched_parameters["overall_epsilon"] = overall_epsilon
        
        cnt = cnt + 1
        if demo_cnt == cnt:
            break
    
    return searched_parameters


def search_usefulness(SS, epsilon_threshold, lmo_gamma=0.9, demo_cnt=1000):
    # The searched condition is epsilon_threshold and usefulness.
    cnt = 0
    searched_parameters = {}
    searched_parameters["overall_epsilon"] = 0
    searched_parameters["usefulness"] = 0
    for lmo in take_Si(SS):
        privacy_lmo = compute_privacy_lmo(lmo)
        usefulness, _ = compute_usefulness_lmo(lmo, lmo_gamma=lmo_gamma)
        if privacy_lmo==[]:
            continue
        else:
            overall_epsilon, opt_order, rdp_lmo = privacy_lmo
            if overall_epsilon < epsilon_threshold and np.isreal(rdp_lmo[int(opt_order-2)]):
                if usefulness > searched_parameters["usefulness"]:
                    searched_parameters = lmo
                    searched_parameters["overall_epsilon"] = overall_epsilon
                    searched_parameters["usefulness"] = usefulness
        
        cnt = cnt + 1
        if demo_cnt == cnt:
            break
    
    return searched_parameters


def save_epsilon_usefulness(SS, epsilon_threshold, lmo_gamma=0.9, demo_cnt=1000):
    # Just save epsilon and usefulness of all parameters.
    cnt = 0
    for sdx, lmo in enumerate(take_Si(SS)):
        if sdx==0:
            columns = list(lmo.keys())
            columns.extend(["overall_epsilon", "usefulness"])
            df=pd.DataFrame([], columns=columns)
        privacy_lmo = compute_privacy_lmo(lmo)
        usefulness, _ = compute_usefulness_lmo(lmo, lmo_gamma=lmo_gamma)
        if privacy_lmo==[]:
            continue
        else:
            overall_epsilon, opt_order, rdp_lmo = privacy_lmo
            if overall_epsilon < epsilon_threshold and np.isreal(rdp_lmo[int(opt_order-2)]):
                added_parameters = list(lmo.values())
                added_parameters.extend([overall_epsilon, usefulness])
                df.loc[len(df.index)] = added_parameters
        
        cnt = cnt + 1
        if demo_cnt == cnt:
            break
    
    return df


if __name__ == '__main__':
    S = {
        "a1": np.linspace(0.1, 0.9, 9),
        "a3": np.linspace(0.1, 0.9, 9),
        "a4": np.linspace(0.1, 0.9, 9),
        "G_theta_k": [(1,2), (2,2), (3,2), (5,1), (9,0.5), (7.5,1), (0.5,1)],  # k>0; theta>0; t<1/theta
        "E_lambda": [0.1, 0.5, 1, 5],  # E_lambda>0; t<E_lambda;
        "U_b_a": [(1,0), (2,1)],  # b>a; when t=0: MGF=1;
        "delta": [1e-10],
    }
    epsilon_threshold = 1
    lmo_gamma = 0.9
    
    demo_cnt = False  # The options could be {False, "any numbers"(3000, ...)}; Choosing False will go through all the paramters.
    save_df = False
    
    searched_parameters_epsilon = search_epsilon(S, epsilon_threshold, demo_cnt=demo_cnt)
    print(f"When considering maximum the epsilon below {epsilon_threshold}, we found the parameters: {searched_parameters_epsilon}.")
    
    searched_parameters_usefulness = search_usefulness(S, epsilon_threshold, lmo_gamma=lmo_gamma, demo_cnt=demo_cnt)
    print(f"When considering maximum the usefulness below epsilon {epsilon_threshold}, we found the parameters: {searched_parameters_usefulness}.")
    
    ## another choice: saving the epsilon and the usefulness of all parameters and choosing parameters offline.
    df = save_epsilon_usefulness(S, epsilon_threshold, lmo_gamma=lmo_gamma, demo_cnt=demo_cnt)
    if save_df:
        df.to_csv(f"parameters_gamma{lmo_gamma}.csv")
    else:
        print(df)
    