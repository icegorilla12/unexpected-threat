import numpy as np
import pandas as pd
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PPCF import*

def change_headers(tracking_data):
    columns = tracking_data.columns
    temp = []
    for j in columns:
        temp.append(j)
    for i in range(len(columns)):
        if columns[i][:6] == 'Player':
            temp[i] = columns[i] +'_x'
        if columns[i][:7] == 'Unnamed':
            temp[i] = columns[i-1] + '_y'
    temp[-2] = 'Ball_x'
    tracking_data.columns = temp
    return tracking_data

def standardize_data(tracking_data, pitch_dimensions = (105,68)):
    tracking_data['Time [m]'] = tracking_data['Time [s]']/60
    columns_x = [c for c in tracking_data.columns if c[-1]=='x']
    columns_y = [c for c in tracking_data.columns if c[-1]=='y']
    second_half_index = tracking_data.Period.idxmax(2)
    tracking_data.loc[:second_half_index,columns_x] = tracking_data.loc[:second_half_index,columns_x]*pitch_dimensions[0]
    tracking_data.loc[:second_half_index,columns_y] = tracking_data.loc[:second_half_index,columns_y]*pitch_dimensions[1]
    tracking_data.loc[second_half_index:,columns_x] = (1-tracking_data.loc[second_half_index:,columns_x])*pitch_dimensions[0]
    tracking_data.loc[second_half_index:,columns_y] = (1-tracking_data.loc[second_half_index:,columns_y])*pitch_dimensions[1]
    return tracking_data


def merge_data(data_home, data_away):
    tracking_data = data_home.drop(['Ball_x','Ball_y'], axis=1).merge(data_away, left_index = True, right_index = True, suffixes=['','_away'])
    return tracking_data

def visualize_frame(data_home, data_away, time,attack_team, pitch_dimensions=(105,68), player_size = 200, ball_size = 100,x_grids=40,PPCF=True,show_velocity=True,annotate=True, home_color ='blue', away_color ='red',ax = None): ## Time should be given in seconds
    if ax == None:
        pitch = Pitch(pitch_type='custom',half=False, goal_type='box',  pitch_color='#f5f5f5', line_color='#696969', linewidth=2,pitch_length=pitch_dimensions[0],pitch_width=pitch_dimensions[1],constrained_layout=True)
        fig,ax  = pitch.draw(figsize=(8,6))
    frame = data_home.loc[data_home['Time [s]']==time].Frame.to_list()[0]
    columns_x_home = [c for c in data_home.columns[:33] if c[-1]=='x' and c!='Ball_x']
    columns_y_home = [c for c in data_home.columns[:33] if c[-1]=='y' and c!='Ball_y']
    columns_x_away = [c for c in data_away.columns[:33] if c[-1]=='x' and c!='Ball_x']
    columns_y_away = [c for c in data_away.columns[:33] if c[-1]=='y' and c!='Ball_y']
    ax.invert_yaxis()
    ax.scatter(x = data_home.loc[frame,columns_x_home] , y  = data_home.loc[frame,columns_y_home], color = home_color, edgecolors = 'black', s=player_size, zorder=1, alpha = 0.8)
    ax.scatter(x = data_away.loc[frame,columns_x_away] , y  = data_away.loc[frame,columns_y_away], color = away_color, edgecolors = 'black', s=player_size, zorder=1, alpha = 0.8)
    ax.scatter(x = data_home.loc[frame,['Ball_x']] , y = data_home.loc[frame,['Ball_y']], color = 'white' , edgecolors = 'black', s=ball_size,zorder=2,alpha=0.8)
    if show_velocity:
        columns_vx_home = [c for c in data_home.columns if c[-2:]=='vx' and c!='Ball_vx']
        columns_vy_home = [c for c in data_home.columns if c[-2:]=='vy' and c!='Ball_vy']
        columns_vx_away = [c for c in data_away.columns if c[-2:]=='vx' and c!='Ball_vx']
        columns_vy_away = [c for c in data_away.columns if c[-2:]=='vy' and c!='Ball_vy']
        ax.quiver(data_home.loc[frame,columns_x_home] , data_home.loc[frame,columns_y_home], data_home.loc[frame,columns_vx_home], data_home.loc[frame,columns_vy_home], color='black', units='inches', angles='xy', scale=20 ,width=0.015,headlength=3,headwidth=2,alpha=0.8)
        ax.quiver(data_away.loc[frame,columns_x_away] , data_away.loc[frame,columns_y_away], data_away.loc[frame,columns_vx_away], data_away.loc[frame,columns_vy_away], color='black', units='inches' , angles='xy', scale=20,width=0.015,headlength=3,headwidth=2,alpha=0.8)
    if annotate:
        kit_home = []
        kit_away = []
        for string in columns_x_home:
            kit_home.append(''.join(char for char in string if char.isdigit()))
        for string in columns_x_away:
            kit_away.append(''.join(char for char in string if char.isdigit()))
        for a,x,y in zip(kit_home,data_home.loc[frame,columns_x_home],data_home.loc[frame,columns_y_home]):
            ax.annotate(a , xy = (x-0.9,y+0.7))
        for a,x,y in zip(kit_away,data_away.loc[frame,columns_x_away],data_away.loc[frame,columns_y_away]):
            ax.annotate(a , xy = (x-0.9,y+0.7))
    if PPCF:
        start_frame = data_home.loc[data_home['Time [s]']==time]['Frame'].to_list()[0]
        if attack_team=='Home':
            PPCFa,_,_ = PPCF_field(data_home, data_away,start_frame , attack_team, x_grids = x_grids)
            c_map='bwr_r'
        else:
            PPCFa,_,_ = PPCF_field(data_away, data_home,start_frame , attack_team, x_grids = x_grids)
            c_map='bwr'
        ax.imshow(np.flipud(PPCFa), cmap=c_map,interpolation='spline36',alpha=0.5,vmin=0.0, vmax=1.0,extent = (0, 105,0,68))
    
#     ax.set_title(f'Match Situation at Time-{time} seconds'  )

    
def make_movie(output_file,data_home, data_away,time_start, time_end,attack_team,pitch_dimensions = (105,68), show_velocity=True,clear_frame = False, fps = 25, home_color = 'blue', away_color='red', ball_color='white',player_size=200,ball_size=100, annotate = True,ax = None,PPCF=True,x_grids=16):
    video_file = output_file
    clear_frames = clear_frame    # Should it clear the figure between each frame?
    fps = 25

    # Output video writer
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Metrica Movie', artist='Matplotlib', comment='Footbal Bloody Hell')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    if ax == None:
        pitch = Pitch(pitch_type='custom',half=False, goal_type='box',  pitch_color='#f5f5f5', line_color='#696969', linewidth=2,pitch_length=pitch_dimensions[0],pitch_width=pitch_dimensions[1],constrained_layout=True)
        fig,ax  = pitch.draw(figsize=(8,6))
    frame_start = data_home.loc[data_home['Time [s]']==time_start].Frame.to_list()[0]
    frame_end = data_home.loc[data_home['Time [s]']==time_end].Frame.to_list()[0]
    columns_x_home = [c for c in data_home.columns[:33] if c[-1]=='x' and c!='Ball_x']
    columns_y_home = [c for c in data_home.columns[:33] if c[-1]=='y' and c!='Ball_y']
    columns_x_away = [c for c in data_away.columns[:33] if c[-1]=='x' and c!='Ball_x']
    columns_y_away = [c for c in data_away.columns[:33] if c[-1]=='y' and c!='Ball_y']
    frames = np.arange(frame_start,frame_end,1)
    ax.invert_yaxis()
    with writer.saving(fig, video_file, 100):
        for j,frame in enumerate(frames):
            figobjs = [] 
            objs = ax.scatter(x = data_home.loc[frame,columns_x_home] , y  = data_home.loc[frame,columns_y_home], color = home_color, edgecolors = 'black', s=player_size, zorder=1, alpha = 0.8)
            figobjs.append(objs)
            objs = ax.scatter(x = data_away.loc[frame,columns_x_away] , y  = data_away.loc[frame,columns_y_away], color = away_color, edgecolors = 'black', s=player_size, zorder=1, alpha = 0.8)
            figobjs.append(objs)
            objs = ax.scatter(x = data_home.loc[frame,['Ball_x']] , y = data_home.loc[frame,['Ball_y']], color = ball_color , edgecolors = 'black', s=ball_size,zorder=2,alpha=0.8)
            figobjs.append(objs)
            if annotate:
                kit_home = []
                kit_away = []
                for string in columns_x_home:
                    kit_home.append(''.join(char for char in string if char.isdigit()))
                for string in columns_x_away:
                    kit_away.append(''.join(char for char in string if char.isdigit()))
                for a,x,y in zip(kit_home,data_home.loc[frame,columns_x_home],data_home.loc[frame,columns_y_home]):
                    objs = ax.annotate(a , xy = (x-0.9,y+0.7))
                    figobjs.append(objs)
                for a,x,y in zip(kit_away,data_away.loc[frame,columns_x_away],data_away.loc[frame,columns_y_away]):
                    objs = ax.annotate(a , xy = (x-0.9,y+0.7))
                    figobjs.append(objs)
            if show_velocity:
                    columns_vx_home = [c for c in data_home.columns if c[-2:]=='vx' and c!='Ball_vx']
                    columns_vy_home = [c for c in data_home.columns if c[-2:]=='vy' and c!='Ball_vy']
                    columns_vx_away = [c for c in data_away.columns if c[-2:]=='vx' and c!='Ball_vx']
                    columns_vy_away = [c for c in data_away.columns if c[-2:]=='vy' and c!='Ball_vy']
                    objs = ax.quiver(data_home.loc[frame,columns_x_home] , data_home.loc[frame,columns_y_home], data_home.loc[frame,columns_vx_home], data_home.loc[frame,columns_vy_home], color='black', units='inches', angles='xy', scale=20 ,width=0.015,headlength=3,headwidth=2,alpha=0.8)
                    figobjs.append(objs)
                    objs = ax.quiver(data_away.loc[frame,columns_x_away] , data_away.loc[frame,columns_y_away], data_away.loc[frame,columns_vx_away], data_away.loc[frame,columns_vy_away], color='black', units='inches' , angles='xy', scale=20,width=0.015,headlength=3,headwidth=2,alpha=0.8)
                    figobjs.append(objs)
            if PPCF:
                    if attack_team=='Home':
                        PPCFa,_,_ = PPCF_field(data_home, data_away,frame , attack_team, x_grids = x_grids)
                        c_map='bwr_r'
                    else:
                        PPCFa,_,_ = PPCF_field(data_away, data_home,frame , attack_team, x_grids = x_grids)
                        c_map='bwr'
                    objs = ax.imshow(np.flipud(PPCFa), cmap=c_map,interpolation='spline36',alpha=0.5,vmin=0.0, vmax=1.0,extent = (0, 105,0,68))
                    figobjs.append(objs)
            if clear_frames:
                    fig.clear()
            ax.set_title(f'Time-{"{:.2f}".format(time_start + 0.04*j)} seconds'  )
            print(np.round(time_start + 0.04*j,decimals=2))
            writer.grab_frame()
            for figobj in figobjs:
                    figobj.remove()
    plt.clf()
    