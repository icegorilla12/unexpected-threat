import numpy as np
from player_functions import*
import scipy.stats

def flight_time_naive(initial_x,initial_y,end_x,end_y,ball_velocity=15):
    x1,y1,x2,y2 = initial_x,initial_y,end_x,end_y
    time = np.sqrt((x2-x1)**2 + (y2-y1)**2)/ball_velocity
    return np.round(time, decimals=2)

def check_offside(data_attack, data_defence, start_frame, attack_team, tol = 0.2):
    idx = data_attack.loc[data_attack['Frame']==start_frame].index[0]
    columns_players_attack = [c for c in data_attack.columns[:33] if c[-1]=='x' and c!='Ball_x']
    columns_players_defence = [c for c in data_defence.columns[:33] if c[-1]=='x'][1:]
    if attack_team=='Home':
        offside_line = np.nanmax(data_defence.loc[idx,columns_players_defence])
        offside_index = np.where(data_attack.loc[idx,columns_players_attack]>offside_line+0.2)
    else:
        offside_line = np.nanmin(data_defence.loc[idx,columns_players_defence])
        offside_index = np.where(data_attack.loc[idx,columns_players_attack]<offside_line-0.2)
    
    return offside_index[0]

def intercept_time(x1,x2,y1,y2,vx,vy,velocity):
    reaction_time = 0.7 
    x_final  = x1 + vx*reaction_time
    y_final = y1 + vy*reaction_time
#     t1 = (5-velocity)/7
#     dist_covered = velocity*t1 + 0.5*7*t1**2
    distance = np.sqrt((x2-x_final)**2 + (y2-y_final)**2)
    t2 = distance/5
    time = reaction_time +t2
    
    return time

    

def generate_PPCF(data_attack, data_defence, start_frame,attack_team,target,ball_velocity=15, targetx=None, targety=None,offside=True):
    offside_index = []
    if offside:
        offside_index = check_offside(data_attack,data_defence, start_frame, attack_team)
    idx = data_attack.loc[data_attack['Frame']==start_frame].index[0]
    columns_ball = [c for c in data_attack.columns[:33] if c[:4] == 'Ball']
    x1 = data_attack.loc[idx,columns_ball].to_list()[0]
    y1 = data_attack.loc[idx,columns_ball].to_list()[1]
    if target=='same':
        target_x = x1
        target_y = y1
    elif target=='different':
        target_x = targetx
        target_y = targety
    ball_time = flight_time_naive(x1,y1,target_x,target_y, ball_velocity = ball_velocity) 
    velocity_attack = player_velocity(data_attack,start_frame)[0,:]
    vx_attack = player_velocity(data_attack,start_frame)[1,:]
    vy_attack = player_velocity(data_attack,start_frame)[2,:]
    velocity_defence = player_velocity(data_defence,start_frame)[0,:]
    vx_defence = player_velocity(data_defence,start_frame)[1,:]
    vy_defence = player_velocity(data_defence,start_frame)[2,:]
    x1_attack = player_location(data_attack,start_frame)[0,:]
    y1_attack = player_location(data_attack,start_frame)[1,:]
    x1_defence = player_location(data_defence,start_frame)[0,:]
    y1_defence = player_location(data_defence,start_frame)[1,:]
    attack_intercept = intercept_time(x1_attack,target_x,y1_attack,target_y,vx_attack,vy_attack,velocity_attack)
    defence_intercept = intercept_time(x1_defence,target_x,y1_defence,target_y,vx_defence,vy_defence,velocity_defence)
    time_to_control_att = 3*np.log(10) * (np.sqrt(3)*0.45/np.pi + 1/4.3)
    time_to_control_def = 3*np.log(10) * (np.sqrt(3)*0.45/np.pi + 1/(4.3*1.72))
    dT = 0.04
    dT_array = np.arange(ball_time-dT,ball_time+10,dT) 
    PPCFatt = np.zeros(len(dT_array))
    PPCFdef = np.zeros(len(dT_array))
    ptot = 0.0
    i = 1
    PPCFatt_ind = np.zeros(len(attack_intercept))
    PPCFdef_ind = np.zeros(len(defence_intercept))
#     if (np.nanmin(attack_intercept)-max(ball_time,np.nanmin(defence_intercept)))>time_to_control_att:
#         return 0,1
#     elif (np.nanmin(defence_intercept)-max(ball_time,np.nanmin(attack_intercept)))>time_to_control_def:
#         return 1,0
    while 1-ptot>0.01 and i<dT_array.size:
            ball_time = dT_array[i]
            attack_prob = get_probability(ball_time,attack_intercept)
            for index in offside_index:
                attack_prob[index]=0
            defence_prob = get_probability(ball_time,defence_intercept)
            dPPCFdT_att = (1-ptot)*attack_prob*4.3
            PPCFatt_ind += dPPCFdT_att*dT
            PPCFatt[i] = np.nansum(PPCFatt_ind)
            dPPCFdT_def = (1-ptot)*defence_prob*4.3*1.72
            PPCFdef_ind += dPPCFdT_def*dT
            PPCFdef[i] = np.nansum(PPCFdef_ind)
            ptot = PPCFatt[i] + PPCFdef[i]
            i+= 1
        
    if i>=dT_array.size:
            print("Integration failed to converge: %1.3f" % (ptot) )
    return PPCFatt[i-1] , PPCFdef[i-1], PPCFatt_ind


def PPCF_field(data_attack, data_defence, start_frame, attack_team, x_grids = 40):
    y_grids = int(x_grids*3/4)
    dx = np.round(105/x_grids, decimals=2)
    dy = np.round(68/y_grids, decimals=2)
    x_divisions=np.linspace(dx,105-dx,x_grids)
    x_divisions=np.around(x_divisions,decimals=2)
    y_divisions=np.linspace(dy,68-dy,y_grids)
    y_divisions=np.around(y_divisions,decimals=2)
    PPCF = np.zeros((y_grids,x_grids))
    for i,y in enumerate(y_divisions):
        for j,x in enumerate(x_divisions):
            PPCF[i,j], temp , temp_tuple = generate_PPCF(data_attack, data_defence, start_frame=start_frame,attack_team=attack_team,target='different', targetx=x, targety=y,offside=True)
    return PPCF,x_divisions,y_divisions

# def get_dist_probability(distance):
# #     coefficient = -0.03400
# #     intercept = 2.336
#     coefficient = -0.0005526
#     intercept = 1.9638
#     prob  = 1/(1+np.exp(-((distance**2)*coefficient+intercept)))
#     return prob
    
def pass_probability(data_attack, data_defence, start_frame,  targetx, targety,player_in_possession,attack_team,target='different',offside=True):
    tol = 1
    kit = get_player_order(data_attack)
    possession_index  = [index for index,value in enumerate(kit) if value == player_in_possession]
    idx = data_attack.loc[data_attack['Frame']==start_frame].index[0]
    columns_ball = [c for c in data_attack.columns[:33] if c[:4] == 'Ball']
    x1 = data_attack.loc[idx,columns_ball].to_list()[0]
    y1 = data_attack.loc[idx,columns_ball].to_list()[1]
    PPCF_target,_,PPCFatt_ind = generate_PPCF(data_attack, data_defence, start_frame,target=target, attack_team=attack_team,targetx=targetx, targety=targety,offside=True)
#     _,_,PPCFatt_ind_dribble = generate_PPCF(data_attack, data_defence, start_frame,target=target, attack_team=attack_team,targetx=targetx, targety=targety,ball_velocity = 3,offside=True)
    for index in possession_index:
#                 PPCFcontrol = PPCFatt_ind_dribble[index]
                PPCFatt_ind[index]=0
    PPCF_pass = np.nansum(PPCFatt_ind)
#     _,_,control_PPCF = generate_PPCF(data_attack, data_defence, start_frame=start_frame,attack_team=attack_team,target='same', targetx=None, targety=None,offside=True)
#     control_PPCF = control_PPCF[index]
    r = np.sqrt((targety-y1)**2+(targetx-x1)**2)
#     dist_prob = normal_distribution(5,r)
    dist_prob = scipy.stats.norm(0, 23.9).cdf(r+0.01)-scipy.stats.norm(0, 23.9).cdf(r-0.01)
    decision_prob = (dist_prob*PPCF_pass**(1.04))
    
    
    return decision_prob,PPCF_target
    
    
def pass_prob_field(data_attack, data_defence, start_frame,player_in_possession,attack_team,x_grids=40,target='different',offside=True):
    y_grids = int(x_grids*3/4)
    dx = np.round(105/x_grids, decimals=2)
    dy = np.round(68/y_grids, decimals=2)
    x_divisions=np.linspace(dx,105-dx,x_grids)
    x_divisions=np.around(x_divisions,decimals=2)
    y_divisions=np.linspace(dy,68-dy,y_grids)
    y_divisions=np.around(y_divisions,decimals=2)
    decision_prob = np.zeros((y_grids,x_grids))
    control_prob =  np.zeros((y_grids,x_grids))
    for i,y in enumerate(y_divisions):
        for j,x in enumerate(x_divisions):
            decision_prob[i,j],control_prob[i,j]= pass_probability(data_attack, data_defence, start_frame=start_frame,  targetx=x, targety=y,player_in_possession=player_in_possession,attack_team=attack_team,target='different',offside=True)
    norm = 1/np.sum(decision_prob)
    decision_prob = decision_prob*norm
    prob = np.multiply(decision_prob, control_prob)
    return prob

def normal_distribution(variance, r, mean=0):
    value = np.exp(-(r**2)/(2*variance**2))/(2*np.pi*variance**2)
    return value


def assign_xT(xTmatrix, x_coord,y_coord,x_divisions=np.linspace(0,105,21),  y_divisions=np.linspace(0,68,16)):
    zones = np.arange(0,((len(x_divisions)-1)*(len(y_divisions)-1)))
    zones = np.reshape(zones,(20,15))
    xTmatrix = xTmatrix.flatten()
    for i in range(len(x_divisions)-1):
        if x_coord < x_divisions[i+1]:
            break
    for j in range(len(y_divisions)-1):
        if y_coord < y_divisions[j+1]:
            break
    zone1 = zones[i][j]
    zone1= zone1.astype('int')
    xTzone1 = xTmatrix[zone1]
    return xTzone1

# def pass_payoff(xTmatrix,data_attack, data_defence, start_frame,  targetx, targety,player_in_possession,attack_team,target='different',offside=True):
#     passprobability = pass_probability(data_attack, data_defence, start_frame,  targetx, targety,player_in_possession,attack_team,target='different',offside=True)
#     xT = assign_xT(xTmatrix,x_coord=targetx,y_coord=targety, x_divisions=np.linspace(0,105,21),  y_divisions=np.linspace(0,68,16))
#     payoff = xT*passprobability
#     return payoff

def pass_payoff_field(xTmatrix,data_attack, data_defence, start_frame, player_in_possession,attack_team,x_grids=40,offside=True):
    y_grids = int(x_grids*3/4)
    dx = np.round(105/x_grids, decimals=2)
    dy = np.round(68/y_grids, decimals=2)
    x_divisions=np.linspace(dx,105-dx,x_grids)
    x_divisions=np.around(x_divisions,decimals=2)
    y_divisions=np.linspace(dy,68-dy,y_grids)
    y_divisions=np.around(y_divisions,decimals=2)
#     idx = data_attack.loc[data_attack['Frame']==start_frame].index[0]
#     columns_ball = [c for c in data_attack.columns[:33] if c[:4] == 'Ball']
#     x1 = data_attack.loc[idx,columns_ball].to_list()[0]
#     y1 = data_attack.loc[idx,columns_ball].to_list()[1]
    prob = pass_prob_field(data_attack, data_defence, start_frame,player_in_possession,attack_team,x_grids=x_grids,target='different',offside=True)
#     on_ball_xT = assign_xT(xTmatrix, x1,y1,x_divisions=np.linspace(0,105,21),  y_divisions=np.linspace(0,68,16))
    for i,y in enumerate(y_divisions):
        for j,x in enumerate(x_divisions):
            xT=assign_xT(xTmatrix,x_coord=x,y_coord=y, x_divisions=np.linspace(0,105,21),  y_divisions=np.linspace(0,68,16))
            prob[i,j] = prob[i,j]*xT
    return prob