import numpy as np
import scipy.signal as signal
import warnings
warnings.filterwarnings("ignore")
import math

def calculate_velocities(data, windowlength = 9, maxspeed=12, smooth = 'Least-Squares'):
    columns_x = [c for c in data.columns[:33] if c[-1]=='x']
    columns_players_x = [c for c in data.columns[:33] if c[-1]=='x' and c!='Ball_x']
    columns_y = [c for c in data.columns[:33] if c[-1]=='y']
    columns_players_y = [c for c in data.columns[:33] if c[-1]=='y' and c!='Ball_y']
    columns  = columns_x+columns_y
    if smooth=='Least-Squares':
        for i,j,k in zip(columns_x,columns_y,columns):
            data[i[:-1]+'vx'] = signal.savgol_filter(data[i], window_length = windowlength,mode='nearest', polyorder = 1, deriv=1, delta=0.04)
            data[j[:-1]+'vy'] = signal.savgol_filter(data[j], window_length = windowlength, mode='nearest', polyorder = 1, deriv=1, delta=0.04)
#             data.loc[data[i[:-1]+'vx']<0.001,i[:-1]+'vx'] = 0
#             data.loc[data[j[:-1]+'vy']<0.001,j[:-1]+'vy'] = 0
            data[k[:-1]+'velocity'] = np.sqrt(data[i[:-1]+'vx']**2+data[j[:-1]+'vy']**2)
        for i,j in zip(columns_players_x,columns_players_y):
            data.loc[data[i[:-1]+'velocity']>maxspeed,i[:-1]+'velocity'] = np.nan
            data.loc[data[j[:-1]+'velocity']>maxspeed,j[:-1]+'velocity'] = np.nan
    return data
    
def calculate_acceleration(data,windowlength = 9, maxacceleration = 7, smooth = 'Least-Squares'):
    columns_players_x = [c for c in data.columns[:33] if c[-1]=='x' and c!='Ball_x']
    columns_players_y = [c for c in data.columns[:33] if c[-1]=='y' and c!='Ball_y']
    columns  = columns_players_x+columns_players_y
    if smooth=='Least-Squares':
        for i,j,k in zip(columns_players_x,columns_players_y,columns):
            data[i[:-1]+'ax'] = signal.savgol_filter(data[i], window_length = windowlength,mode='nearest', polyorder = 2, deriv=2, delta=0.04)
            data[j[:-1]+'ay'] = signal.savgol_filter(data[j], window_length = windowlength, mode='nearest',polyorder = 2, deriv=2, delta=0.04)
#             data.loc[data[i[:-1]+'ax']<0.001,i[:-1]+'ax'] = 0
#             data.loc[data[j[:-1]+'ay']<0.001,j[:-1]+'ay'] = 0
            data[k[:-1]+'acceleration'] = np.sqrt(data[i[:-1]+'ax']**2+data[j[:-1]+'ay']**2)
            data.loc[data[i[:-1]+'acceleration']>maxacceleration,i[:-1]+'acceleration'] = np.nan
            data.loc[data[j[:-1]+'acceleration']>maxacceleration,j[:-1]+'acceleration'] = np.nan
            
    return data

def pass_velocity(data, start_frame):
    idx = data.loc[data['Frame']==start_frame].index[0]
    velocity = np.mean(data['Ball_velocity'][idx:idx+10])
    return velocity

def get_event_coordinates(events_data,start_frame):
    x1 = events_data.loc[events_data['Start Frame']==start_frame]['Start X'].to_list()[0]*105
    y1 = events_data.loc[events_data['Start Frame']==start_frame]['Start Y'].to_list()[0]*68
    x2 = events_data.loc[events_data['Start Frame']==start_frame]['End X'].to_list()[0]*105
    y2 = events_data.loc[events_data['Start Frame']==start_frame]['End Y'].to_list()[0]*68
    return x1,y1,x2,y2

def flight_time(passes_events_data,tracking_data,start_frame,end_x=None,end_y=None):
    x1,y1,x2,y2 = get_event_coordinates(passes_events_data,start_frame)
    if end_x and end_y:
        x2 = end_x
        y2 = end_y
   
    ball_velocity = pass_velocity(tracking_data,start_frame)
    time = np.sqrt((x2-x1)**2 + (y2-y1)**2)/ball_velocity
    return time

def player_velocity(data, start_frame):
    idx = data.loc[data['Frame']==start_frame].index[0]
    columns_velocity = [c for c in data.columns if c[-8:]=='velocity' and c!='Ball_velocity']
    columns_vx = [c for c in data.columns if c[-2:]=='vx' and c!='Ball_vx']
    columns_vy = [c for c in data.columns if c[-2:]=='vy' and c!='Ball_vy']
    m = len(columns_velocity)
    velocities = []
    vxs = []
    vys = []
    for a,b,c in zip(columns_velocity,columns_vx,columns_vy):
        velocities.append(np.mean(data[a][idx:idx+10]))
        vxs.append(np.mean(data[b][idx:idx+10]))
        vys.append(np.mean(data[c][idx:idx+10]))
    velocity = np.concatenate((velocities,vxs,vys),axis=0)
    return np.reshape(velocity,(3,m))

def player_location(data, start_frame):
    idx = data.loc[data['Frame']==start_frame].index[0]
    columns_x = [c for c in data.columns[:33] if c[-1:]=='x' and c!='Ball_x']
    columns_y = [c for c in data.columns[:33] if c[-1:]=='y' and c!='Ball_y']
    m = len(columns_x)
    xs = []
    ys = []
    for a,b in zip(columns_x,columns_y):
        xs.append(np.mean((data[a][idx:idx+5])))
        ys.append(np.mean((data[b][idx:idx+5])))
    location = np.concatenate((xs,ys),axis=0)
    return np.reshape(location,(2,m))

def get_player_order(data):
    kit = []
    columns = [c for c in data.columns[:33] if c[-1]=='x' and c!='Ball_x']
    for string in columns:
            kit.append(''.join(char for char in string if char.isdigit()))
    return kit

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_probability(ball_time, intercept_times):
    prob = []
    for i in intercept_times:
        prob.append(1/(1+np.exp(-(ball_time-i)*np.pi/(np.sqrt(3)*0.45))))
    prob = np.asarray(prob)
    return prob

def intercept_probability(tracking_data,start_frame,events_data=None,flighttime=None,player_number=None,end_x = None, end_y = None):
    kit = get_player_order(tracking_data)
    if player_number:
        player_idx = [idx for idx, s in enumerate(kit) if str(player_number) in s][0]
        velocity = player_velocity(tracking_data,start_frame)[0,player_idx].to_list()
        vx = player_velocity(tracking_data,start_frame)[1,player_idx].to_list()
        vy = player_velocity(tracking_data,start_frame)[2,player_idx].to_list()
        x1 = player_location(tracking_data,start_frame)[0,player_idx].to_list()
        y1 = player_location(tracking_data,start_frame)[1,player_idx].to_list()     
    else:
        velocity = player_velocity(tracking_data,start_frame)[0,:]
        vx = player_velocity(tracking_data,start_frame)[1,:]
        vy = player_velocity(tracking_data,start_frame)[2,:]
        x1 = player_location(tracking_data,start_frame)[0,:]
        y1 = player_location(tracking_data,start_frame)[1,:]      
    if end_x and end_y:
        x2 = end_x
        y2 = end_y
    else:
        _,_,x2,y2 = get_event_coordinates(events_data,start_frame)
    if flighttime:
        T = flighttime
    else:
        T = flight_time(events_data,tracking_data,start_frame,end_x,end_y)
    reaction_time = 0.7 
    x_final  = x1 + vx*reaction_time
    y_final = y1 + vy*reaction_time
    t1 = (5-velocity)/7
    dist_covered = velocity*t1 + 0.5*7*t1**2
    distance = np.sqrt((x2-x_final)**2 + (y2-y_final)**2)
    t2 = (distance-dist_covered)/5
    time = reaction_time + t1+t2
    prob = get_probability(T,time)
    return prob


    
    
    
    
    


# def PPCF(tracking_home,tracking_away):
    
    

# def pass_difficulty(data,start_frame,player_in_possession,end_x = None,end_y = None):
#     kit = get_player_order(tracking_data)
#     velocity = player_velocity(tracking_data,start_frame)[0,player_idx].to_list()
#     vx = player_velocity(tracking_data,start_frame)[1,player_idx].to_list()
#     vy = player_velocity(tracking_data,start_frame)[2,player_idx].to_list()
#     x1 = player_location(tracking_data,start_frame)[0,player_idx].to_list()
#     y1 = player_location(tracking_data,start_frame)[1,player_idx].to_list()
#     if end_x and end_y:
#         x2 = end_x
#         y2 = end_y
#     else:
#         _,_,x2,y2 = get_event_coordinates(events_data,start_frame)
#     v1 =  ((x2-x1[i]),(y2-y1[i]))
#     v2 =  (vx[i],vy[i])
#     angle = angle_between(v1,v2)
    
    
    
    
    
    

## Average retardation of a ball, modelled by taking all the passes and averaging the acceleration for them. (Wrong because a player controls the ball and that is averaged as well leading to erroneous values of acceleration)
# def calculate_retardation_ball(events_data,tracking_data):
#     start_frame = np.asarray(events_data.loc[(events_data['Type']=='PASS')&(events_data['Team']=='Home')]['Start Frame'])
#     end_frame = np.asarray(events_data.loc[(events_data['Type']=='PASS')&(events_data['Team']=='Home')]['End Frame'])
#     average_acceleration = 0
#     for a, b in zip(start_frame, end_frame):
#         dt = tracking_data.loc[tracking_data['Frame']==b]['Time [s]'].to_list()[0] - tracking_data.loc[tracking_data['Frame']==a]['Time [s]'].to_list()[0]
#         dv = tracking_data.loc[tracking_data['Frame']==b]['Ball_velocity'].to_list()[0] - tracking_data.loc[tracking_data['Frame']==a]['Ball_velocity'].to_list()[0]
#         if math.isnan(dv):
#             continue
#         elif dt>0:
#             average_acceleration += dv/dt
#     return np.round(average_acceleration/len(start_frame), decimals=2)